"""
Fly against the BFM expert or a trained SAC tactic in FlightGear, at real time.

Both jets are env.py's: the same JSBSim f16.xml the tactics trained on, stepped here in Python under
the same rules. A kill is 0.5 s inside the other's gun cone (+/-15 deg, 150-1000 m), 200 m is the
ground and 12 km apart is a disengage. FlightGear is only your cockpit, stick and screen:

  your controls  FlightGear -> Python over native-ctrls (FGNetCtrls v27, UDP 5551), read every sim step
  your jet       Python -> FlightGear over native-fdm, with its speeds, so the HUD and gauges read
  the bandit     a multiplayer aircraft, as in fg_replay.py

  python fg_play.py --launch                                                you vs the BFM expert
  python fg_play.py --launch --vs models/angle_tactic_E/policy_best.pt      you vs the energy tactic
  python fg_play.py --vs straight --start tail_chase                        gunnery practice
  python fg_play.py --vs models/angle_tactic_W/policy_final.pt --start defensive --speed 0.5 --loop

--start is one of fly_policy's Table 1 geometries seen from your seat (tail_chase: you are 3 km
behind the bandit), or random. --vs takes a tactic policy .pt or an opponent from adversary.LADDER
(straight, turn4, turn6, bfm_energy, bfm_angles). Against a policy you fly the env's adversary seat
and the policy decides every 0.1 s, as it trained; against a bot you fly the agent's seat.

Controls are FlightGear's own: a joystick plugged in before FlightGear starts, the keyboard (numpad or
arrows for the stick, numpad 5 centres it, PgUp/PgDn throttle) or the mouse (right-click toggles the
mouse yoke). Trim and speed brake reach the jet too. P pauses. FlightGear must have been started with
--native-ctrls, which --launch does. Every fight is saved as an .acmi in --out, for Tacview or
fg_replay.py.
"""

import argparse
import os
import socket
import struct
import sys
import time

import numpy as np
import torch

from adversary import LADDER
from env import DT, INIT_THROTTLE, OBS_DIMS, AngleTacticEnv
from export import write_acmi
from fg_replay import (FG_AIRCRAFT, FGFS, HOST, LOD_PROP, TELNET_PORT, VIEWS, Link, Props, _listening,
                       _wrap180, launch)
from fly_policy import NAMED_STARTS, load_frozen_policy

CTRLS_PORT = 5551
FGFS_EXTRA = [f"--native-ctrls=socket,out,50,{HOST},{CTRLS_PORT},udp"]
# FGNetCtrls v27, 744 bytes: byte offsets of the doubles used here, found by setting each control
# over telnet and searching the packet; bit 0 of the uint32 at FREEZE_AT is FlightGear's pause
CTRLS_LEN, CTRLS_VERSION = 744, 27
CTRLS = {"aileron": 8, "elevator": 16, "rudder": 24, "aileron_trim": 32, "elevator_trim": 40,
         "rudder_trim": 48, "speedbrake": 72, "throttle": 160}
FREEZE_AT = 640
# what FlightGear's instruments read (fg_replay.KIN), straight from your jet's JSBSim
JSB_KIN = {"alpha": "aero/alpha-rad", "beta": "aero/beta-rad", "phidot": "velocities/phidot-rad_sec",
           "thetadot": "velocities/thetadot-rad_sec", "psidot": "velocities/psidot-rad_sec",
           "vcas": "velocities/vc-kts", "climb_rate": "velocities/h-dot-fps",
           "v_north": "velocities/v-north-fps", "v_east": "velocities/v-east-fps",
           "v_down": "velocities/v-down-fps", "v_body_u": "velocities/u-fps", "v_body_v": "velocities/v-fps",
           "v_body_w": "velocities/w-fps", "A_X_pilot": "accelerations/a-pilot-x-ft_sec2",
           "A_Y_pilot": "accelerations/a-pilot-y-ft_sec2", "A_Z_pilot": "accelerations/a-pilot-z-ft_sec2"}
COUNTDOWN_S, END_HOLD_S = 3.0, 3.0
# the env's outcome (the agent seat's view) -> yours, per seat
YOURS = {"self": {"win": "win", "loss": "loss", "crash": "crash", "adversary_crash": "bandit-crash"},
         "adv": {"win": "loss", "loss": "win", "crash": "bandit-crash", "adversary_crash": "crash"}}
SAY = {"win": "WIN: guns kill on {vs}", "loss": "LOSS: {vs} gunned you down", "crash": "CRASH: you hit the deck",
       "bandit-crash": "{vs} hit the deck", "timeout": "DRAW: time is up", "disengaged": "DISENGAGED: over 12 km apart"}


class Controls:
    """Your controls, from FlightGear's native-ctrls stream; the newest packet wins."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((HOST, CTRLS_PORT))
        self.sock.setblocking(False)
        self.c = dict.fromkeys(CTRLS, 0.0)
        self.c["throttle"] = INIT_THROTTLE
        self.paused = self.seen = False

    def poll(self):
        while True:
            try:
                pkt = self.sock.recv(2048)
            except (BlockingIOError, ConnectionResetError):       # nothing waiting
                return
            if len(pkt) == CTRLS_LEN and struct.unpack_from("!I", pkt)[0] == CTRLS_VERSION:
                self.c = {k: struct.unpack_from("!d", pkt, at)[0] for k, at in CTRLS.items()}
                self.paused = bool(struct.unpack_from("!I", pkt, FREEZE_AT)[0] & 1)
                self.seen = True

    def stick(self):
        """[aileron, rudder, elevator, throttle 0..1] in JSBSim's signs: its rudder is FlightGear's reversed."""
        c = self.c
        return np.array([c["aileron"], -c["rudder"], c["elevator"], c["throttle"]], dtype=np.float32)

    def trim(self, fdm):
        """Trims and speed brake go straight to your jet; the env sets stick and throttle."""
        fdm["fcs/roll-trim-cmd-norm"] = self.c["aileron_trim"]
        fdm["fcs/pitch-trim-cmd-norm"] = self.c["elevator_trim"]
        fdm["fcs/yaw-trim-cmd-norm"] = -self.c["rudder_trim"]
        fdm["fcs/speedbrake-cmd-norm"] = self.c["speedbrake"]


class Cockpit:
    """FlightGear as your cockpit. tick() runs once per 50 Hz sim step: it waits for the step's slot
    on the wall clock, shows both jets as they are and returns your stick for the step."""

    def __init__(self, env, seat, props, speed=1.0):
        self.env, self.seat, self.props, self.speed = env, seat, props, speed
        self.link, self.ctl = Link(), Controls()
        self.clock = None

    @property
    def fdm(self):
        return self.env.fdm_self if self.seat == "self" else self.env.fdm_adv

    def show(self):
        self.link.send(self.env.trajectory[-1], {k: self.fdm[p] for k, p in JSB_KIN.items()}, own=self.seat)

    def say(self, text, color="yellow"):
        self.props.set(f"/sim/screen/{color}", text)

    def wait(self, secs):
        """Shows the current frame for secs of wall time; the fight clock starts again after."""
        end = time.perf_counter() + secs
        while time.perf_counter() < end:
            self.show()
            self.ctl.poll()
            time.sleep(DT)
        self.clock = None

    def tick(self):
        self.ctl.poll()
        while self.ctl.paused:                              # P in FlightGear
            self.wait(DT)
            self.ctl.poll()
        now = time.perf_counter()
        if self.clock is None or now > self.clock + 0.1:    # first step, after a pause, or fell behind
            self.clock = now
        self.clock += DT / self.speed
        time.sleep(max(0.0, self.clock - time.perf_counter()))
        self.show()
        self.ctl.poll()
        self.ctl.trim(self.fdm)
        return self.ctl.stick()


class HumanSeat:
    """The env's adversary seat, flown by you: env.step asks it for controls every sim step."""
    current = "you"

    def __init__(self, cockpit):
        self.cockpit = cockpit

    def reset(self):
        pass

    def act(self, ego_state, enm_state):
        return self.cockpit.tick()


def start_condition(name, seat, rng):
    """(Range, AA, ATA) for env.reset from a start named from your seat. In the adversary seat the
    same picture is mirrored: the bandit's ATA is 180 - your AA, its AA -180 - your ATA."""
    if name == "random":
        r, aa, ata = rng.uniform(1900, 5100), rng.uniform(-180, 180), rng.uniform(-180, 180)
    else:
        r, aa, ata = dict(NAMED_STARTS)[name]
    return (r, aa, ata) if seat == "self" else (r, _wrap180(-180.0 - ata), _wrap180(180.0 - aa))


def fight(env, cockpit, pi, ic, seed, vs):
    """One fight from start condition ic; returns (your outcome, seconds, the env's outcome)."""
    if hasattr(env.adversary, "_rng"):
        env.adversary._rng = np.random.default_rng(seed)    # seeded bot randomness, as fly_policy
    obs, _ = env.reset(seed=seed, options={"init_condition": ic})
    cockpit.say(f"Bandit: {vs}. Fight's on in {COUNTDOWN_S:.0f} s")
    cockpit.wait(COUNTDOWN_S)
    cockpit.say("FIGHT'S ON", "green")
    with torch.no_grad():
        while True:
            if pi is None:                                  # you fly the agent's seat
                action = cockpit.tick()
                action[3] = 2.0 * action[3] - 1.0           # the env maps [-1, 1] to throttle 0..1
            else:                                           # the policy does; you are env.adversary
                action = pi.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[2].squeeze(0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return YOURS[cockpit.seat].get(info["outcome"], info["outcome"]), env.step_count * DT, info["outcome"]


def label(vs):
    """A policy as <model folder>/<file>, e.g. angle_tactic_E/policy_best; a bot by its LADDER name."""
    if not vs.endswith(".pt"):
        return vs
    path = os.path.splitext(os.path.abspath(vs))[0]
    return f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"


def save(env, out, seat, vs, start, yours, outcome, secs):
    """The fight as an .acmi; the outcome in the name is yours ('you-win' keeps export.py's
    agent-side outcome parser off it)."""
    os.makedirs(out, exist_ok=True)
    tag = vs.replace("/", "-")
    path = os.path.join(out, f"{time.strftime('%Y%m%d-%H%M%S')}_{start}_vs_{tag}_you-{yours}.acmi")
    title = f"You vs {vs} from {start}: {yours.upper()} at {secs:.1f} s"
    if seat == "adv":                                       # you are the adversary object, the policy the agent
        write_acmi(env.trajectory, path, title=title, opponent="you", outcome=outcome)
    else:                                                   # you sit where export.py writes "SAC agent"
        write_acmi(env.trajectory, path, title=title, opponent=vs, outcome=outcome)
        with open(path, encoding="utf-8-sig") as f:
            text = f.read().replace("the SAC agent", "you").replace("Pilot=SAC agent", "Pilot=You")
        with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
            f.write(text)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--vs", default="bfm_angles", help=f"a tactic policy .pt, or one of {', '.join(LADDER)}")
    p.add_argument("--start", default="head_on", choices=[n for n, _ in NAMED_STARTS] + ["random"],
                   help="start geometry from your seat")
    p.add_argument("--launch", action="store_true", help="start FlightGear (or reuse a running one)")
    p.add_argument("--speed", type=float, default=1.0, help="game speed, e.g. 0.5 for slow motion")
    p.add_argument("--loop", action="store_true", help="fight again after each one, until Ctrl+C")
    p.add_argument("--seconds", type=float, default=120.0, help="time limit of a fight")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--view", choices=list(VIEWS), default="cockpit")
    p.add_argument("--out", default="logs/fg_play", help="folder for the fights' .acmi files")
    p.add_argument("--fgfs", default=FGFS)
    p.add_argument("--fg-aircraft", default=FG_AIRCRAFT, help="FGAddon Aircraft folder holding f16/")
    args = p.parse_args()

    vs_policy = args.vs.endswith(".pt")
    if not vs_policy and args.vs not in LADDER:
        p.error(f"--vs must be a .pt policy or one of {', '.join(LADDER)}")
    seat = "adv" if vs_policy else "self"
    steps = int(round(args.seconds / DT))
    if vs_policy:
        env = AngleTacticEnv(max_steps=steps, log_trajectory=True, obs_mode="extended")
        pi = load_frozen_policy(args.vs, OBS_DIMS["extended"])
    else:
        env = AngleTacticEnv(max_steps=steps, log_trajectory=True, obs_mode="extended", action_repeat=1,
                             adversary_type="ladder", opponent_level=LADDER.index(args.vs))
        pi = None

    proc = None
    if args.launch and not _listening(TELNET_PORT):
        proc, props = launch(args.fgfs, args.fg_aircraft, FGFS_EXTRA)
    else:
        try:
            props = Props()
        except OSError:
            sys.exit(f"no FlightGear on telnet port {TELNET_PORT}: run with --launch")
    try:
        cockpit = Cockpit(env, seat, props, args.speed)
    except OSError:
        sys.exit(f"UDP port {CTRLS_PORT} is taken: is another fg_play.py running?")
    if vs_policy:
        env.adversary = HumanSeat(cockpit)
    end = time.perf_counter() + 5.0
    while not cockpit.ctl.seen and time.perf_counter() < end:
        cockpit.ctl.poll()
        time.sleep(0.05)
    if not cockpit.ctl.seen:
        sys.exit(f"no controls from FlightGear on UDP {CTRLS_PORT}: it runs without {FGFS_EXTRA[0]}. "
                 "Close it and start again with --launch")
    props.set("/sim/current-view/view-number", VIEWS[args.view])
    lod = props.get(LOD_PROP)
    props.set(LOD_PROP, -1)                                 # the bandit at full detail at any range

    rng, score, n, vs = np.random.default_rng(args.seed), {}, 0, label(args.vs)
    try:
        while True:
            ic = start_condition(args.start, seat, rng)
            print(f"fight {n + 1}: you vs {vs} from {args.start} (R {ic[0]:.0f} m)", flush=True)
            yours, secs, outcome = fight(env, cockpit, pi, ic, args.seed + n, vs)
            msg = SAY.get(yours, yours).format(vs=vs)
            cockpit.say(msg, "green" if yours in ("win", "bandit-crash") else "red")
            path = save(env, args.out, seat, vs, args.start, yours, outcome, secs)
            score[yours] = score.get(yours, 0) + 1
            n += 1
            print(f"   {msg} at {secs:.1f} s  -> {path}", flush=True)
            cockpit.wait(END_HOLD_S)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        try:
            props.set(LOD_PROP, lod)
        except OSError:
            pass
    if n:
        print("score: " + ", ".join(f"{k} {v}" for k, v in sorted(score.items())))
    if proc:
        print(f"FlightGear is still running (pid {proc.pid}); close its window when done")


if __name__ == "__main__":
    main()
