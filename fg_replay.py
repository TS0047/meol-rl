"""
Replay saved fights in FlightGear 2024.1 with both F-16s, at real time. FlightGear only displays
them: its F-16 flight model (f16b52) is not jsbsim's f16.xml, so training stays in Python.

  agent jet     FlightGear's own aircraft, flown with --fdm=null and driven over native-fdm
                (FGNetFDM v24, 408 bytes, UDP port 5550)
  opponent jet  a multiplayer aircraft fed position packets (T_MsgHdr + T_PositionMsg, 232 bytes,
                UDP port 5000), drawn with FlightGear's AI F-16 model

Positions come from the .acmi files export.py writes and sit at 0N 0E, open ocean at sea level, as
in Tacview. A 50 Hz wall clock samples the fight and interpolates between its rows, so 0.5x stays
smooth and 2x skips rows. Each fight starts after the first frame is held for 1 s and ends with the
last frame held for 2 s. Then a telnet self-check reads both jets back from FlightGear and compares
them with that last frame.

  python fg_replay.py --launch visualizations/E_best/01_tail_chase_win.acmi       start FlightGear, replay
  python fg_replay.py --speed 0.5 --loop visualizations/E_best/07_defensive_loss.acmi
  python fg_replay.py --launch visualizations/O_meol_final                        each fight in turn
  python fg_replay.py --launch --live models/angle_tactic_E/policy_best.pt --start head_on

Without --launch, FlightGear must already be running with the flags in FGFS_ARGS. --launch reuses
one that already answers on the telnet port, and switches a new one to the chase view (--view picks
another; V cycles them in FlightGear). FlightGear stays open afterwards. While the script runs, the
opponent is drawn at full detail at any range (LOD_PROP).
"""

import argparse
import math
import os
import socket
import struct
import subprocess
import sys
import time

from env import DT
from export import _lonlat, read_acmi

FGFS = r"C:\Program Files\FlightGear 2024.1\bin\fgfs.exe"
FG_AIRCRAFT = os.path.join(os.path.expanduser("~"), "FlightGear", "Downloads", "Aircraft",
                           "org.flightgear.fgaddon.stable_2024", "Aircraft")
HOST, FDM_PORT, MP_PORT, TELNET_PORT = "127.0.0.1", 5550, 5000, 5501
FGFS_ARGS = ["--aircraft=f16-block-52", "--fdm=null", f"--native-fdm=socket,in,50,,{FDM_PORT},udp",
             f"--multiplay=in,50,{HOST},{MP_PORT}", f"--multiplay=out,10,{HOST},5002", "--callsign=SAC",
             f"--telnet={TELNET_PORT}", "--lat=0", "--lon=0", "--in-air", "--timeofday=noon",
             "--disable-terrasync", "--disable-real-weather-fetch", "--disable-ai-traffic"]
VIEWS = {"cockpit": 0, "helicopter": 1, "chase": 2, "tower": 3, "flyby": 6}   # /sim/view[n]
MP_MODEL = "AI/Aircraft/f16/Models/f-16.xml"
MP_CALLSIGN = "BANDIT"
# AI/MP level of detail: by default a jet under 400 px on screen is drawn as a low-detail glider, which
# in a fight at 0.15-5 km is nearly always. -1 is "High Detail only". FlightGear saves this preference
# (userarchive), so the user's value is put back on exit.
LOD_PROP = "/sim/rendering/static-lod/aimp-detailed"
LEAD_S = 1.0             # first frame held: FlightGear loads the MP model before the fight moves
HOLD_S = 2.0             # last frame held: the MP jet's smoothing settles before the self-check
FT = 0.3048
WGS_A, WGS_F = 6378137.0, 1.0 / 298.257223563
WGS_E2 = WGS_F * (2 - WGS_F)
SEED, RANDOM_STARTS = 2026, 30   # fly_policy's start set, as flown for visualizations/


# ---------- native-fdm: FGNetFDM v24, 408 bytes, network byte order ----------
def net_fdm(lat_deg, lon_deg, alt_m, roll_deg, pitch_deg, yaw_deg):
    r = math.radians
    head = struct.pack("!II3d", 24, 0, r(lon_deg), r(lat_deg), alt_m)
    floats = struct.pack("!22f", alt_m, r(roll_deg), r(pitch_deg), r(yaw_deg), *([0.0] * 18))  # agl: sea level
    engines = struct.pack("!I4I36f", 1, 2, 0, 0, 0, *([0.0] * 36))     # one engine, running
    tanks = struct.pack("!I4f", 1, 3000.0, 0, 0, 0)
    wheels = struct.pack("!I3I9f", 3, 0, 0, 0, *([0.0] * 9))           # gear up
    tail = struct.pack("!Iif10f", 0, 0, 20000.0, *([0.0] * 10))        # time, warp, visibility, surfaces
    return head + floats + engines + tanks + wheels + tail


# ---------- multiplayer: T_MsgHdr + T_PositionMsg, 232 bytes ----------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def q_lonlat(lon, lat):                          # SGQuat::fromLonLatRad: ECEF -> local NED
    zd2, yd2 = 0.5 * lon, -0.25 * math.pi - 0.5 * lat
    sz, sy, cz, cy = math.sin(zd2), math.sin(yd2), math.cos(zd2), math.cos(yd2)
    return (cz * cy, -sz * sy, cz * sy, sz * cy)


def q_euler(z, y, x):                            # SGQuat::fromEulerRad(yaw, pitch, roll): NED -> body
    sz, sy, sx = math.sin(z / 2), math.sin(y / 2), math.sin(x / 2)
    cz, cy, cx = math.cos(z / 2), math.cos(y / 2), math.cos(x / 2)
    return (cx * cz * cy + sx * sz * sy, sx * cz * cy - cx * sz * sy,
            cx * cz * sy + sx * sz * cy, cx * sz * cy - sx * cz * sy)


def angle_axis(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    ang = math.acos(max(-1.0, min(1.0, w / n)))
    s = math.sin(ang)
    if abs(s) < 1e-12:
        return (0.0, 0.0, 0.0)
    k = 2 * ang / (s * n)
    return (x * k, y * k, z * k)


def ecef(lat, lon, alt):                          # WGS-84, radians and metres
    n = WGS_A / math.sqrt(1 - WGS_E2 * math.sin(lat) ** 2)
    return ((n + alt) * math.cos(lat) * math.cos(lon), (n + alt) * math.cos(lat) * math.sin(lon),
            (n * (1 - WGS_E2) + alt) * math.sin(lat))


def mp_packet(t, lat_deg, lon_deg, alt_m, roll_deg, pitch_deg, yaw_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    q = qmul(q_lonlat(lon, lat), q_euler(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)))
    body = (MP_MODEL.encode().ljust(96, b"\0") + struct.pack("!2d", t, 0.1)     # time, lag
            + struct.pack("!3d", *ecef(lat, lon, alt_m)) + struct.pack("!3f", *angle_axis(q))
            + struct.pack("!12f", *([0.0] * 12)) + struct.pack("!I", 0))       # velocities, accels, pad
    # magic "FGFS", protocol 1.1, message 7 (position), length, range 100 nm, reply port, callsign
    head = (struct.pack("!6I", 0x46474653, 0x00010001, 7, 32 + len(body), 100, 0)
            + MP_CALLSIGN.encode().ljust(8, b"\0"))
    return head + body


class Link:
    """UDP to FlightGear: the agent's jet over native-fdm, the opponent over multiplayer."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.epoch = time.perf_counter()          # MP time must keep rising across fights and loops

    def send(self, frame):
        lon, lat = _lonlat(frame["self_pos"])
        self.sock.sendto(net_fdm(lat, lon, frame["self_pos"][2], *frame["self_att"]), (HOST, FDM_PORT))
        lon, lat = _lonlat(frame["adv_pos"])
        self.sock.sendto(mp_packet(time.perf_counter() - self.epoch, lat, lon, frame["adv_pos"][2],
                                   *frame["adv_att"]), (HOST, MP_PORT))


# ---------- pacing ----------
def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def lerp(a, b, s):
    """The fight at sim time s between rows a and b. Angles take the short way round; a pass through
    the vertical (roll and yaw flip 180 deg in one row) snaps instead of sweeping."""
    f = min(1.0, max(0.0, (s - a["t"]) / max(b["t"] - a["t"], 1e-9)))
    out = {"t": s}
    for k in ("self", "adv"):
        da = [_wrap180(y - x) for x, y in zip(a[f"{k}_att"], b[f"{k}_att"])]
        fa = f if max(abs(d) for d in da) < 90.0 else round(f)
        out[f"{k}_pos"] = tuple(x + f * (y - x) for x, y in zip(a[f"{k}_pos"], b[f"{k}_pos"]))
        out[f"{k}_att"] = tuple(x + fa * d for x, d in zip(a[f"{k}_att"], da))
    return out


def hold(link, row, secs):
    end = time.perf_counter() + secs
    while time.perf_counter() < end:
        link.send(row)
        time.sleep(DT)


def play(rows, link, speed=1.0):
    """Streams rows (any iterable, so a live fight works too) at speed x real time: a 50 Hz wall
    clock samples the fight between rows. Returns the last row, which was held HOLD_S."""
    it = iter(rows)
    prev = nxt = next(it)
    hold(link, prev, LEAD_S)
    start = tick = time.perf_counter()
    t0 = prev["t"]
    while nxt is not None:
        s = t0 + (time.perf_counter() - start) * speed
        while nxt is not None and nxt["t"] <= s:
            prev, nxt = nxt, next(it, None)
        link.send(prev if nxt is None else lerp(prev, nxt, s))
        tick = max(tick + DT, time.perf_counter() - DT)      # fell behind: drop ticks, never burst
        time.sleep(max(0.0, tick - time.perf_counter()))
    hold(link, prev, HOLD_S)
    return prev


# ---------- FlightGear's property tree over telnet ----------
class Props:
    """'data' mode telnet: `get` answers with the bare value, lines end in CRLF."""

    def __init__(self, wait_s=0.0, proc=None):
        end = time.time() + wait_s
        while True:
            try:
                self.sock = socket.create_connection((HOST, TELNET_PORT), timeout=5)
                break
            except OSError:
                if time.time() > end or (proc is not None and proc.poll() is not None):
                    raise
                time.sleep(2)
        self.sock.settimeout(120)                 # the port accepts during start-up, before the main loop answers
        self.f = self.sock.makefile("r", encoding="ascii", newline="\r\n")
        self.sock.sendall(b"data\r\n")

    def get(self, *paths):
        """One value, or a list for several paths. They go out in one send: FlightGear polls telnet
        about twice a second, and answers every line waiting at each poll."""
        self.sock.sendall("".join(f"get {p}\r\n" for p in paths).encode())
        values = [self.f.readline().strip() for _ in paths]
        return values[0] if len(paths) == 1 else values

    def set(self, path, value):                    # no reply in data mode
        self.sock.sendall(f"set {path} {value}\r\n".encode())


def launch(fgfs, aircraft_dir):
    """Starts fgfs and waits until its scenery is loaded; returns (process, telnet)."""
    proc = subprocess.Popen([fgfs, f"--fg-aircraft={aircraft_dir}", *FGFS_ARGS],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"started FlightGear (pid {proc.pid}), waiting for its scenery ...", flush=True)
    try:
        props = Props(wait_s=180, proc=proc)
    except OSError:
        sys.exit(f"no telnet on port {TELNET_PORT}"
                 + (f": fgfs exited ({proc.returncode})" if proc.poll() is not None else ""))
    end = time.time() + 300
    while props.get("/sim/sceneryloaded") not in ("true", "1"):
        if proc.poll() is not None or time.time() > end:
            sys.exit("FlightGear exited or never loaded its scenery")
        time.sleep(1)
    print(f"FlightGear {props.get('/sim/version/flightgear')} up, aircraft {props.get('/sim/aircraft')}")
    return proc, props


def self_check(props, row):
    """Reads both jets back and prints the worst |FlightGear - sent| per jet."""
    nodes = [f"/ai/models/multiplayer[{i}]" for i in range(8)]
    seen = props.get(*[f"{n}/{k}" for n in nodes for k in ("callsign", "valid")])
    mp = next((n for n, c, v in zip(nodes, seen[::2], seen[1::2]) if c == MP_CALLSIGN and v == "true"), None)
    jets = [("agent", "", "heading-deg", row["self_pos"], row["self_att"])]
    if mp is None:
        print(f"   check: opponent {MP_CALLSIGN} not found in /ai/models")
    else:
        jets.append(("opponent", mp, "true-heading-deg", row["adv_pos"], row["adv_att"]))
    for name, node, heading, pos, att in jets:
        lon, lat = _lonlat(pos)
        try:
            got = [float(v) for v in props.get(*[node + p for p in (
                "/position/latitude-deg", "/position/longitude-deg", "/position/altitude-ft",
                "/orientation/roll-deg", "/orientation/pitch-deg", f"/orientation/{heading}")])]
        except ValueError:
            print(f"   check {name:8s} no reading from FlightGear")
            continue
        d_pos = max(abs(got[0] - lat), abs(got[1] - lon))
        d_alt = abs(got[2] * FT - pos[2])
        d_att = max(abs(_wrap180(g - a)) for g, a in zip(got[3:], att))
        ok = d_pos < 1e-6 and d_alt < 0.01 and d_att < 1e-4
        print(f"   check {name:8s} lat/lon {d_pos:.1e} deg, alt {d_alt:.1e} m, attitude {d_att:.1e} deg  "
              f"{'OK' if ok else 'MISMATCH'}")


# ---------- live: fly a policy and stream it as it is simulated ----------
def live(policy, start):
    """Yields the rows of `policy` flying fly_policy's `start` against the full BFM expert, as they are
    simulated: the same fight fly_policy saves for that start (deterministic actions, seeded BFM)."""
    import numpy as np
    import torch
    from adversary import LADDER
    from env import AngleTacticEnv, OBS_DIMS
    from fly_policy import NAMED_STARTS, load_frozen_policy

    rng = np.random.default_rng(SEED)
    starts = list(NAMED_STARTS) + [
        (f"random_{i + 1:02d}", (float(rng.uniform(1900, 5100)), float(rng.uniform(-180, 180)),
                                 float(rng.uniform(-180, 180)))) for i in range(RANDOM_STARTS)]
    names = [n for n, _ in starts]
    if start not in names:
        sys.exit(f"--start must be one of {', '.join(names)}")
    i = names.index(start)
    env = AngleTacticEnv(max_steps=6000, log_trajectory=True, obs_mode="extended",
                         adversary_type="ladder", opponent_level=LADDER.index("bfm_angles"))
    pi = load_frozen_policy(policy, OBS_DIMS["extended"])
    env.adversary._rng = np.random.default_rng(SEED + i)
    obs, _ = env.reset(seed=SEED + i, options={"init_condition": starts[i][1]})
    sent = 0
    with torch.no_grad():
        while True:
            yield from env.trajectory[sent:]
            sent = len(env.trajectory)
            action = pi.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[2].squeeze(0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"   {info['outcome'].upper()} at {env.step_count / 50.0:.1f} s", flush=True)
                yield from env.trajectory[sent:]
                return


def fights(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(os.path.join(p, n) for n in os.listdir(p) if n.endswith(".acmi"))
        else:
            yield p


def _listening(port):
    try:
        socket.create_connection((HOST, port), timeout=1).close()
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("paths", nargs="*", help=".acmi files or folders of them, replayed in turn")
    p.add_argument("--launch", action="store_true", help="start FlightGear (or reuse a running one)")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed, e.g. 0.5 or 2")
    p.add_argument("--loop", action="store_true", help="start over after the last fight")
    p.add_argument("--live", metavar="POLICY", help="fly this tactic policy live instead of replaying files")
    p.add_argument("--start", default="head_on", help="--live start: a fly_policy name, e.g. tail_chase, random_07")
    p.add_argument("--view", choices=list(VIEWS),
                   help="FlightGear view to switch to (default: chase after --launch, else left as it is)")
    p.add_argument("--fgfs", default=FGFS)
    p.add_argument("--fg-aircraft", default=FG_AIRCRAFT, help="FGAddon Aircraft folder holding f16/")
    args = p.parse_args()
    if bool(args.paths) == bool(args.live):
        p.error("give .acmi files or folders, or --live <policy.pt>")
    files = list(fights(args.paths)) or [None]               # None: the live fight

    proc = props = None
    if args.launch and not _listening(TELNET_PORT):
        proc, props = launch(args.fgfs, args.fg_aircraft)
    else:
        try:
            props = Props()
        except OSError:
            print(f"no FlightGear telnet on port {TELNET_PORT}: sending anyway, without the self-check")
    view = args.view or ("chase" if proc else None)
    lod = None
    if props:
        if view:
            props.set("/sim/current-view/view-number", VIEWS[view])
        lod = props.get(LOD_PROP)
        props.set(LOD_PROP, -1)
    link = Link()
    try:
        while True:
            for path in files:
                if path is None:
                    print(f"live: {args.live} from {args.start} vs bfm_angles", flush=True)
                    rows = live(args.live, args.start)
                else:
                    rows, meta = read_acmi(path)
                    if not rows:
                        print(f"{os.path.basename(path)}: no frames with both jets, skipped")
                        continue
                    print(f"{os.path.basename(path)}: {rows[-1]['t']:.1f} s vs {meta.get('opponent', '?')}",
                          flush=True)
                last = play(rows, link, args.speed)
                if props:
                    self_check(props, last)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if lod:
            try:
                props.set(LOD_PROP, lod)
            except OSError:
                pass                                         # FlightGear already closed
    if proc:
        print(f"FlightGear is still running (pid {proc.pid}); close its window when done")


if __name__ == "__main__":
    main()
