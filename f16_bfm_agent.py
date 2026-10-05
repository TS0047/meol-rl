"""
f16_bfm_agent.py
================
Rule-based Basic-Fighter-Maneuver (BFM) agent for a JSBSim F-16, designed to be
stepped at 50 Hz.  The tactical rules are taken from R. L. Shaw,
"Fighter Combat: Tactics and Maneuvering" (USNIP, 1985):

  Ch.1  gun envelope (≈500-3000 ft), lead / snapshot / tracking shots
  Ch.2  pursuit curves (lead / pure / lag), lag-displacement roll, high & low
        yo-yo, lead turn, nose-to-nose / nose-to-tail turns, flat scissors
  Ch.3  1v1 similar aircraft: angles fight vs energy fight, guns defence,
        defensive maneuvering (break turn, extension, reversal), and the
        "assess angular / energy advantage first" philosophy
  Appx  corner speed, sustained vs instantaneous G, energy (Es / Ps)

Architecture (three layers, all evaluated every control tick):

  1. Perception        : relative geometry (range, closure, ATA, AOT, TCA, Es...)
  2. Tactical rules    : posture (OFFENSIVE / NEUTRAL / DEFENSIVE) -> maneuver
                         -> (lift-vector direction, load factor, speed, trigger)
  3. Flight control    : lift-vector roll-to-direction + closed-loop Nz +
                         speed/throttle, output = JSBSim FCS commands

Only observable quantities of the *target* are used (position, velocity,
attitude) - i.e. what a radar/visual track could give.

All vectors are NED (x=North, y=East, z=Down), feet and ft/s.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

G = 32.174
FT_PER_NM = 6076.115
D2R = math.pi / 180.0
R2D = 180.0 / math.pi
DOWN = np.array([0.0, 0.0, 1.0])
UP = -DOWN


# --------------------------------------------------------------------------- #
# small math helpers
# --------------------------------------------------------------------------- #
def unit(v, eps=1e-9):
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros(3)


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def angle(a, b):
    return math.acos(clip(float(np.dot(a, b)), -1.0, 1.0))


def rot_about(v, axis, a):
    """Rodrigues rotation of v about unit `axis` by angle a (rad, right-hand)."""
    k = unit(axis)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * float(np.dot(k, v)) * (1 - math.cos(a))


def perp_part(v, axis):
    """component of v perpendicular to unit(axis)"""
    k = unit(axis)
    return v - k * float(np.dot(v, k))


_NZ_TAB = [-3.15, -1.18, 0.5, 1.3, 2.1, 2.8, 4.0, 5.4, 6.2, 9.5]
_U_TAB = [0.44, 0.20, 0.0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.0, -1.0]


def n_cap(kcas):
    """Load factor the *stick-limited* JSBSim F-16 FCS actually delivers at full
    back stick (measured, 5-25 kft).  Used for turn-rate / radius predictions."""
    return float(np.interp(kcas, [120, 200, 250, 300, 350, 450, 550],
                           [1.0, 3.0, 3.8, 4.75, 5.8, 7.6, 8.8]))


# --------------------------------------------------------------------------- #
# data containers
# --------------------------------------------------------------------------- #
@dataclass
class AcState:
    pos: np.ndarray          # NED ft (z = -altitude MSL)
    vel: np.ndarray          # NED ft/s
    phi: float               # roll  rad
    theta: float             # pitch rad
    psi: float               # yaw   rad
    alpha: float             # rad
    nz: float                # load factor along -z body (g)
    kcas: float
    mach: float
    # derived (filled in __post_init__)
    xb: np.ndarray = field(init=False)
    yb: np.ndarray = field(init=False)
    zb: np.ndarray = field(init=False)
    speed: float = field(init=False)
    h: float = field(init=False)

    def __post_init__(self):
        cp, sp = math.cos(self.phi), math.sin(self.phi)
        ct, st = math.cos(self.theta), math.sin(self.theta)
        cy, sy = math.cos(self.psi), math.sin(self.psi)
        self.xb = np.array([ct * cy, ct * sy, -st])
        self.yb = np.array([sp * st * cy - cp * sy, sp * st * sy + cp * cy, sp * ct])
        self.zb = np.array([cp * st * cy + sp * sy, cp * st * sy - sp * cy, cp * ct])
        self.speed = float(np.linalg.norm(self.vel))
        self.h = -float(self.pos[2])


@dataclass
class Controls:
    aileron: float = 0.0     # roll-rate command  (-1..1, +ve = roll right)
    elevator: float = 0.0    # stick (-1 = full back / pull, +0.44 = full push)
    rudder: float = 0.0
    throttle: float = 0.6    # 0..0.5 = idle..MIL, 0.5..1 = AB
    speedbrake: float = 0.0
    trigger: bool = False


@dataclass
class Cmd:
    """Output of the tactical layer."""
    d: np.ndarray | None = None      # desired velocity-vector direction (steer_to)
    lift: np.ndarray | None = None   # OR explicit lift-vector direction
    n: float = 1.0                   # load factor for explicit lift
    n_max: float = 9.0               # cap for steer_to
    kp: float = 3.0                  # steering gain (1/s)
    v_des: float = 400.0             # KCAS
    thr: float | None = None         # explicit throttle override
    brake: float = 0.0
    fire: bool = False
    note: str = ""


# --------------------------------------------------------------------------- #
# ballistics shared by agent (aim solution) and sim (bullet flight)
# --------------------------------------------------------------------------- #
V_MUZ = 3300.0     # ft/s, M61A1 (Shaw Table 1-1)
K_DRAG = 0.25      # 1/s, bullet velocity decay (ground frame)
GUN_MIN, GUN_MAX = 500.0, 3000.0   # ft, Shaw ch.1 "reasonable" gun envelope


def _f(t):
    return (1.0 - math.exp(-K_DRAG * t)) / K_DRAG


def gun_solution(me: AcState, tg: AcState, tg_acc: np.ndarray | None = None):
    """Return (pointing direction (unit NED), time of flight, miss-free range)
    that makes a bullet fired along the returned direction meet the target
    (constant-velocity or constant-acceleration prediction), incl. gravity drop
    and bullet drag."""
    r0 = tg.pos - me.pos
    t = float(np.linalg.norm(r0)) / 2800.0
    d = unit(r0)
    for _ in range(6):
        p_int = tg.pos + tg.vel * t
        if tg_acc is not None:
            p_int = p_int + 0.5 * tg_acc * t * t
        f = _f(t)
        rhs = p_int - me.pos - me.vel * f - 0.5 * G * t * t * DOWN
        dist = float(np.linalg.norm(rhs))
        d = rhs / max(dist, 1e-6)
        y = dist / (V_MUZ * 1.0)          # = f(t) for exact solution
        y = min(y, 0.95 / K_DRAG)
        t = -math.log(1.0 - K_DRAG * y) / K_DRAG
    return d, t


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #
class BFMAgent:
    """
    style = 'angles'  : Shaw ch.3 "angles fight" - aggressive nose-low max-G turns,
                        pass from below, high-G lead turns.
    style = 'energy'  : Shaw ch.3 "energy fight" - best-sustained-turn speed,
                        ~2/3 sustained G, keep altitude/speed, use extension.
    """

    def __init__(self, style="angles", dt=0.02, seed=None, name="agent", hard_deck_ft=3000.0,
                 aggression=1.0, lock=True):
        self.style = style
        self.dt = dt
        self.name = name
        self.rng = random.Random(seed)
        self.hard_deck = hard_deck_ft
        self.aggr = aggression
        self.lock = lock          # True: aim the NOSE at the target (LOS lock, ATA->0) instead of a ballistic lead point

        # tactical state
        self.posture = "NEUTRAL"
        self.posture_t = -9.0
        self.mode = "ENTRY"
        self.mode_t0 = 0.0
        self.man = None            # active multi-phase maneuver (dict)
        self.cool = {}             # maneuver cooldown timers
        self.pass_side = self.rng.choice([-1, 1])
        self.jink_sign = self.rng.choice([-1, 1])
        self.jink_t = 0.0
        self.def_t0 = None
        self.t = 0.0
        self.last_los = None
        self.rc_f = 0.0
        self.tg_acc_f = np.zeros(3)
        self.tg_vel_prev = None
        self.d_prev = None
        self.dd_f = np.zeros(3)
        self.extend_until = -1.0

        # low-level state
        self.n_int = 0.0
        self.n_ref = 1.0
        self.nz_prev = 1.0
        self.nzd_f = 0.0
        self.recover = False
        self.log_mode = ""
        self.dbg_err_ft = -1.0
        self.sep_dir = 0
        # speed policy (see notes): the JSBSim F-16 FCS is stick-limited, so turn RATE is ~flat vs speed while
        # turn RADIUS ~ V^2 -> slower & tighter wins nose-to-nose (Shaw ch.2/3)
        self.v_floor = 250.0 if style == "angles" else 300.0
        self.v_turn = 290.0
        self.simple_merge = True
        self.gk = 0.6             # Nz-loop feedback gain multiplier

    # ------------------------------------------------------------------ #
    # main entry: called every control tick (50 Hz)
    # ------------------------------------------------------------------ #
    def act(self, me: AcState, tg: AcState, t: float) -> Controls:
        self.t = t
        dt = self.dt
        r = tg.pos - me.pos
        R = float(np.linalg.norm(r))
        los = r / max(R, 1.0)
        vrel = tg.vel - me.vel
        rc_raw = -float(np.dot(r, vrel)) / max(R, 1.0)          # +ve = closing
        self.rc_f += 0.25 * (rc_raw - self.rc_f)
        Rc = self.rc_f

        # target acceleration estimate (for lead prediction)
        if self.tg_vel_prev is not None:
            a_raw = (tg.vel - self.tg_vel_prev) / dt
            self.tg_acc_f += 0.04 * (a_raw - self.tg_acc_f)
        self.tg_vel_prev = tg.vel.copy()
        a_t = self.tg_acc_f.copy()
        a_t[2] += 0.0
        na = float(np.linalg.norm(a_t))
        if na > 8 * G:
            a_t *= 8 * G / na

        ata_m = angle(me.xb, los) * R2D                      # my nose off target
        ata_t = angle(tg.xb, -los) * R2D                     # his nose off me
        aot_t = 180.0 - ata_t                                # angle off HIS tail
        tca = angle(unit(me.vel), unit(tg.vel)) * R2D
        es_m = me.h + me.speed ** 2 / (2 * G)
        es_t = tg.h + tg.speed ** 2 / (2 * G)
        geo = dict(R=R, los=los, Rc=Rc, ata_m=ata_m, ata_t=ata_t, aot_t=aot_t, tca=tca,
                   dEs=es_m - es_t, dh=tg.h - me.h, a_t=a_t, r=r)

        self._update_posture(geo)
        cmd = self._tactics(me, tg, geo)
        cmd = self._safety(me, tg, geo, cmd)
        return self._flight_control(me, cmd)

    # ------------------------------------------------------------------ #
    # posture:  who has the angular advantage  (Shaw ch.3 "Defensive Maneuvering")
    # ------------------------------------------------------------------ #
    def _update_posture(self, g):
        am, at = g["ata_m"], g["ata_t"]
        new = self.posture
        if self.posture == "NEUTRAL":
            if at > 115 and am < 100:
                new = "OFFENSIVE"
            elif am > 115 and at < 100:
                new = "DEFENSIVE"
        elif self.posture == "OFFENSIVE":
            if at < 95 or am > 120:
                new = "NEUTRAL"
        elif self.posture == "DEFENSIVE":
            if am < 95 or at > 120:
                new = "NEUTRAL"
        if new != self.posture and (self.t - self.posture_t) > 0.4:
            self.posture = new
            self.posture_t = self.t
            self.man = None
            if new == "DEFENSIVE":
                self.def_t0 = self.t
            else:
                self.def_t0 = None

    def _set_mode(self, m):
        if m != self.mode:
            self.mode = m
            self.mode_t0 = self.t
        return m

    def _cool(self, name, secs):
        self.cool[name] = self.t + secs

    def _ready(self, name):
        return self.t >= self.cool.get(name, -1)

    # ------------------------------------------------------------------ #
    # tactical layer
    # ------------------------------------------------------------------ #
    def _tactics(self, me, tg, g):
        if self.posture == "OFFENSIVE":
            cmd = self._offensive(me, tg, g)
        elif self.posture == "DEFENSIVE":
            cmd = self._defensive(me, tg, g)
        else:
            cmd = self._neutral(me, tg, g)
        if self.mode not in ("GUNS_JINK",):
            cmd = self._energy_governor(me, cmd, g)
        return self._limit_dive(me, cmd)

    def _limit_dive(self, me, cmd):
        """Combat floor: the allowed dive angle shrinks linearly from 45 deg (>=10.5 kft) to 0 (<=4.5 kft).
        Without this both fighters spiral into the ground chasing each other (Shaw ch.2: defensive spiral)."""
        if cmd.d is None:
            return cmd
        max_dive = 45.0 * D2R * clip((me.h - 4500.0) / 6000.0, 0.0, 1.0)
        d = unit(cmd.d)
        pitch = math.asin(clip(float(d[2]), -1.0, 1.0))        # +ve = nose-down
        if pitch > max_dive:
            hz = math.hypot(float(d[0]), float(d[1]))
            hv = np.array([d[0] / hz, d[1] / hz, 0.0]) if hz > 1e-6 else unit(perp_part(me.vel, DOWN))
            cmd.d = hv * math.cos(max_dive) + DOWN * math.sin(max_dive)
        return cmd

    def _energy_governor(self, me, cmd, g):
        """Shaw ch.3: never trade away the speed needed for vertical maneuvering; an angles
        fighter that bleeds below useful maneuvering speed has nothing left to win with."""
        floor = self.v_floor
        if cmd.d is not None:
            if me.kcas < floor:
                deficit = min((floor - me.kcas) / 80.0, 1.6)
                cmd.n_max = min(cmd.n_max, max(2.6, cmd.n_max * (1.0 - 0.55 * deficit)))
                cmd.d = unit(cmd.d + DOWN * 0.22 * deficit)
                cmd.thr = 1.0
            # do not follow the bogey up a climb that costs the energy we need
            if me.theta > 22 * D2R and me.kcas < 420:
                cmd.d = unit(cmd.d + DOWN * 0.35)
        return cmd

    # speed targets ---------------------------------------------------- #
    def _fight_speed(self):
        # F-16 JSBSim: full stick gives ~5.8 g @350 KCAS, ~7.6 g @450.
        return 420.0 if self.style == "angles" else 380.0

    # ---------------------------- NEUTRAL ----------------------------- #
    def _neutral(self, me, tg, g):
        R, Rc, am, at = g["R"], g["Rc"], g["ata_m"], g["ata_t"]
        los = g["los"]
        cmd = Cmd(v_des=self._fight_speed())

        approaching = Rc > 0 and am < 100 and at < 130
        if approaching and R > 5500:
            # --- ENTRY: closing for a forward-quarter pass --------------------
            # Shaw ch.2/3: do not merge neutral - build flight-path separation
            # (lateral for the lead turn, vertical per style) before the pass.
            self._set_mode("ENTRY")
            right = unit(np.cross(DOWN, los))                    # horizontal, to my right of LOS
            off = 2200.0 * self.pass_side
            vert = (-700.0 if self.style == "angles" else +300.0)  # angles: pass from below
            t_go = R / max(Rc, 200.0)
            aim = tg.pos + tg.vel * t_go * 0.35 + right * off + DOWN * (-vert)
            cmd.d = unit(aim - me.pos)
            cmd.n_max = 4.0
            cmd.kp = 2.0
            return cmd

        if approaching and R > 1300 and not self.simple_merge:
            # --- LEAD TURN / merge  (Shaw ch.2 'Lead Turn') --------------------
            self._set_mode("LEAD_TURN")
            # turn in toward the bogey's flight path early enough to gain angles,
            # but not so early that we cross his nose inside gun range
            t_go = R / max(Rc, 200.0)
            aim = tg.pos + tg.vel * min(t_go, 3.0) * 0.5
            cmd.d = unit(aim - me.pos)
            fps = abs(float(np.dot(g["r"], unit(np.cross(tg.vel, DOWN)))))   # lateral flight-path separation
            cmd.n_max = n_cap(me.kcas) * (1.0 if fps > 800 else 0.8)
            cmd.kp = 4.0
            if R > 3200 and am < 25:
                cmd.n_max = 5.0
            return cmd

        # --- after the pass / non-converging: turning fight -----------------
        return self._turning_fight(me, tg, g, cmd)

    def _turning_fight(self, me, tg, g, cmd):
        """Nose-to-nose / nose-to-tail turning fight (Shaw ch.3 angles vs energy)."""
        R, am = g["R"], g["ata_m"]
        self._set_mode("TURN_FIGHT")
        cap = n_cap(me.kcas)
        if self.style == "angles":
            # max-G turn toward the bogey, nose-low assist to preserve speed
            vert = 0.18 * R
            aim = tg.pos + DOWN * vert + tg.vel * 0.0
            cmd.d = unit(aim - me.pos)
            cmd.n_max = cap
            cmd.kp = 4.0
            cmd.v_des = self.v_turn
        else:
            # energy: ~2/3 of sustained G, climb with remaining Ps, stay near best-rate speed
            aim = tg.pos + UP * (0.10 * R)
            cmd.d = unit(aim - me.pos)
            cmd.n_max = max(2.5, 0.7 * cap)
            cmd.kp = 3.0
            cmd.v_des = 390.0
            if am > 100:               # bogey far off the nose: pull harder
                cmd.n_max = cap
        # slow & turning -> unload a bit to keep vertical-maneuvering speed (Shaw ch.2)
        if me.kcas < 260:
            cmd.n_max = min(cmd.n_max, 3.5)
            cmd.v_des = 330.0
        return cmd

    # ---------------------------- OFFENSIVE --------------------------- #
    def _offensive(self, me, tg, g):
        R, Rc, am, aot = g["R"], g["Rc"], g["ata_m"], g["aot_t"]
        cap = n_cap(me.kcas)
        cmd = Cmd(v_des=self._fight_speed())

        # active multi-phase out-of-plane maneuver ---------------------------
        if self.man is not None:
            c = self._run_maneuver(me, tg, g, cmd)
            if c is not None:
                return c

        # gun solution / tracking ---------------------------------------------
        d_sol, tof = gun_solution(me, tg, g["a_t"] * 0.7)
        err = angle(me.xb, d_sol)
        err_ft = err * R
        in_env = (R <= 3300.0) if self.lock else (GUN_MIN * 0.8 <= R <= GUN_MAX * 1.15)
        if self.lock:
            err = am * D2R
            err_ft = err * R
        if in_env and err * R2D < 30 and am < 60 and (R < 2600 or err * R2D < 10):
            self._set_mode("GUN_TRACK")
            # steer so the *nose* (not velocity) lands on the pipper
            if self.lock:
                # nose-on-target (ATA -> 0): velocity must lead the nose by alpha along body +z
                cmd.d = unit(g["los"] + me.alpha * me.zb)
            else:
                cmd.d = unit(d_sol + me.alpha * me.zb)
            cmd.n_max = cap
            cmd.kp = 6.0
            self.dbg_err_ft = err_ft
            cmd.fire = (GUN_MIN <= R <= GUN_MAX) and err_ft < 16.0
            # hold closure ~ 0-100 ft/s at tracking range
            cmd.v_des = me.kcas + clip((60.0 - Rc) * 0.25, -25, 25)
            if R < 900 and Rc > 150:
                cmd.brake = 1.0
            # imminent overshoot at close range -> go to lag roll / yo-yo
            if R < 1100 and Rc > 220 and aot < 40 and self._ready("lagroll"):
                self._start_maneuver("LAG_ROLL")
            return cmd

        # overshoot triggers (Shaw ch.2) ------------------------------------------
        if R < 2600 and Rc > 130 and aot < 35 and self._ready("lagroll"):
            self._start_maneuver("LAG_ROLL")
            return self._run_maneuver(me, tg, g, cmd) or cmd
        if R < 5000 and Rc > 110 and 28 <= aot < 85 and am < 65 and self._ready("hiyoyo") \
                and me.kcas > 260 and (me.h - tg.h) < 3500:
            self._start_maneuver("HIGH_YOYO")
            return self._run_maneuver(me, tg, g, cmd) or cmd
        pitch_ok = abs(me.theta) < 25 * D2R
        if R > 2300 and 18 < am < 85 and Rc < 130 and aot < 80 and self._ready("loyoyo") \
                and pitch_ok and me.h > self.hard_deck + 4000:
            self._start_maneuver("LOW_YOYO")
            return self._run_maneuver(me, tg, g, cmd) or cmd

        # default pursuit: choose lead / pure / lag by desired closure (Shaw ch.2) -------
        self._set_mode("PURSUIT")
        Rc_des = clip((R - 2300.0) * 0.07, -40.0, 260.0)
        if Rc < Rc_des - 60:
            t_lead, name = min(1.5, 0.4 + (R / 6000.0)), "LEAD"
        elif Rc > Rc_des + 90:
            t_lead, name = -1.2, "LAG"
        else:
            t_lead, name = 0.0, "PURE"
        aim = tg.pos + tg.vel * t_lead
        cmd.d = unit(aim - me.pos)
        cmd.n_max = cap
        cmd.kp = 4.0
        cmd.v_des = clip(me.kcas + (Rc_des - Rc) * 0.35 / 1.688, 260, 520)
        cmd.note = name
        return cmd

    # multi-phase out-of-plane maneuvers --------------------------------------
    def _start_maneuver(self, name):
        self.man = dict(name=name, t0=self.t, phase="UP", side=0)

    def _run_maneuver(self, me, tg, g, cmd):
        m = self.man
        R, Rc, am, aot = g["R"], g["Rc"], g["ata_m"], g["aot_t"]
        el = self.t - m["t0"]
        cap = n_cap(me.kcas)
        name = m["name"]
        self._set_mode(name)

        if name in ("HIGH_YOYO", "LAG_ROLL"):
            if m["phase"] == "UP":
                # wings-level pull-up out of the bogey's plane: reduces closure
                pitch = me.theta * R2D
                done = (Rc < 25) or el > (2.8 if name == "HIGH_YOYO" else 2.2) or pitch > 60 \
                    or (R > 4800 and name == "HIGH_YOYO")
                if not done:
                    # lift vector straight up, but keep a small bias toward the bogey side
                    # so we can keep him in sight (slow roll toward him)
                    towards = unit(perp_part(g["los"], me.vel))
                    lift = unit(UP + 0.25 * towards)
                    cmd.lift = lift
                    cmd.n = min(4.5 if name == "HIGH_YOYO" else 3.5, cap)
                    cmd.v_des = me.kcas
                    cmd.brake = 0.5 if name == "LAG_ROLL" else 0.0
                    return cmd
                m["phase"] = "DOWN"
                m["t1"] = self.t
            # DOWN: roll toward bogey, pull to lead (high yo-yo) or lag (lag roll)
            el2 = self.t - m["t1"]
            t_lead = 0.9 if name == "HIGH_YOYO" else -0.6
            aim = tg.pos + tg.vel * t_lead
            if g["R"] < 1000 and Rc > 0 and name == "LAG_ROLL":
                aim = tg.pos - tg.vel * 1.0
            cmd.d = unit(aim - me.pos)
            cmd.n_max = cap
            cmd.kp = 4.0
            cmd.v_des = self._fight_speed() + 20
            ending = el2 > 3.5 or (am < 18 and 500 < R < 3300) or (aot > 95)
            if ending:
                self._cool("hiyoyo" if name == "HIGH_YOYO" else "lagroll", 3.0)
                self.man = None
                return None
            return cmd

        if name == "LOW_YOYO":
            # pull the nose down inside the turn to gain closure & lead (Shaw ch.2 low yo-yo)
            vert = 0.55 * R
            aim = tg.pos + tg.vel * 0.7 + DOWN * vert
            cmd.d = unit(aim - me.pos)
            cmd.n_max = cap
            cmd.kp = 4.0
            cmd.v_des = 450.0
            ending = el > 3.0 or am < 12 or R < 1700 or me.theta < -35 * D2R \
                or me.h < self.hard_deck + 2500
            if ending:
                self._cool("loyoyo", 5.0)
                self.man = None
                return None
            return cmd
        self.man = None
        return None

    # ---------------------------- DEFENSIVE --------------------------- #
    def _defensive(self, me, tg, g):
        R, Rc, am, at = g["R"], g["Rc"], g["ata_m"], g["ata_t"]
        cap = n_cap(me.kcas)
        cmd = Cmd(v_des=self._fight_speed())
        los_to_me = -g["los"]
        since = self.t - (self.def_t0 or self.t)

        # --- GUNS DEFENCE (Shaw ch.3): break in-plane until he nears firing
        #     parameters, then out-of-plane jink to spoil the solution.
        threat = at < 40 and R < 4200
        if threat and R < 2300:
            self._set_mode("GUNS_JINK")
            if self.t - self.jink_t > 1.0:
                self.jink_sign *= -1
                self.jink_t = self.t
            # attacker's plane normal; move perpendicular to it
            nrm = unit(np.cross(los_to_me, tg.vel))
            if np.linalg.norm(nrm) < 0.2:
                nrm = me.yb
            lift = unit(self.jink_sign * nrm + 0.3 * UP)
            cmd.lift = lift
            cmd.n = min(cap, 6.0)
            cmd.v_des = 450.0
            return cmd

        # --- EXTENSION / bug-out when energy-superior and he is not yet shooting
        if (R > 3500 or at > 55) and me.kcas > 420 and me.h > 11000 and g["dEs"] > 800 and Rc < 150 \
                and at > 70:
            if self.t > self.extend_until + 0.0 and (self.extend_until < 0 or self.t - self.extend_until > 6):
                self.extend_until = self.t + 4.0
        if self.t < self.extend_until:
            self._set_mode("EXTENSION")
            away = unit(perp_part(-g["los"] * 1.0, DOWN) + DOWN * 0.15)
            cmd.d = away
            cmd.n_max = 2.2
            cmd.kp = 1.5
            cmd.v_des = 540.0
            cmd.thr = 1.0
            return cmd

        # --- BREAK TURN: hard in-plane turn toward the attacker, lift vector slightly
        #     below the bogey so the break also makes a look-down & saves speed.
        self._set_mode("BREAK")
        aim = tg.pos + DOWN * (0.10 * R)
        cmd.d = unit(aim - me.pos)
        cmd.n_max = cap
        cmd.kp = 5.0
        cmd.v_des = 430.0 if me.kcas < 430 else me.kcas
        # attacker overshooting (closing fast, short range): keep pulling nose-to-nose to
        # force a flat scissors where our slower speed/tighter radius wins (Shaw ch.2)
        if R < 2200 and Rc > 120:
            cmd.v_des = 330.0
            cmd.brake = 0.4
        return cmd

    # ------------------------------------------------------------------ #
    # safety layer (always evaluated, overrides tactics)
    # ------------------------------------------------------------------ #
    def _safety(self, me, tg, g, cmd):
        V = max(me.speed, 200.0)
        dive = max(0.0, math.asin(clip(me.vel[2] / V, -1, 1)))      # +ve = descending
        n_po = 5.0
        loss = (V * V / (G * (n_po - 1.0))) * (1.0 - math.cos(dive)) + 0.3 * V * dive * 0.0
        floor = self.hard_deck
        # ground avoidance: wings-level (upright) max pull
        if me.h - loss - 600.0 < floor and dive > 0.0:
            self._set_mode("GROUND_AVOID")
            c = Cmd(lift=UP, n=min(7.0, n_cap(me.kcas)), v_des=450, thr=1.0)
            return c
        if me.h < floor and me.vel[2] > -50:
            c = Cmd(d=unit(perp_part(me.vel, DOWN) + UP * 0.5 * me.speed), n_max=4.0, v_des=420, thr=1.0)
            self._set_mode("CLIMB_OUT")
            return c
        # low-energy recovery: unload, nose down, full power
        if me.kcas < 170 or me.alpha * R2D > 24:
            self._set_mode("RECOVER")
            nd = unit(perp_part(me.vel, DOWN) + DOWN * 0.25 * max(me.speed, 200))
            return Cmd(d=nd, n_max=1.8, v_des=400, thr=1.0, kp=1.5)
        # vertical separation before a close head-on pass (Shaw ch.2: pass above/below, never co-altitude)
        R, Rc = g["R"], g.get("Rc", 0.0)
        rel, vrel = tg.pos - me.pos, tg.vel - me.vel
        vv = float(np.dot(vrel, vrel))
        t_ca = -float(np.dot(rel, vrel)) / vv if vv > 1.0 else 99.0
        miss = float(np.linalg.norm(rel + vrel * max(t_ca, 0.0))) if 0.0 < t_ca < 99 else 1e5
        trig = R < 3500 and 0.0 < t_ca < 3.5 and miss < 500 and Rc > 40
        if self.sep_dir != 0 and (R > 3800 or Rc < 0):
            self.sep_dir = 0                      # pass is over -> release latch
        if trig and self.sep_dir == 0:
            # who will be above whom at closest approach?  (NED z is down)
            z_rel = float((tg.pos + tg.vel * t_ca)[2] - (me.pos + me.vel * t_ca)[2])
            if me.h < 5500:
                self.sep_dir = +1
            elif abs(z_rel) > 25.0:
                self.sep_dir = +1 if z_rel > 0 else -1      # he will be lower -> I go over him
            else:
                self.sep_dir = +1 if self.name == "BLUE" else -1
        if self.sep_dir != 0:
            self._set_mode("SEPARATE")
            if self.sep_dir > 0:
                return Cmd(lift=UP, n=min(5.0, n_cap(me.kcas)), v_des=cmd.v_des, thr=cmd.thr)
            return Cmd(lift=UP, n=-1.0, v_des=cmd.v_des, thr=cmd.thr)
        return cmd

    # ------------------------------------------------------------------ #
    # flight-control layer: lift-vector -> FCS commands
    # ------------------------------------------------------------------ #
    def _steer_to(self, me, cmd):
        """Convert a desired velocity direction into (lift direction, load factor)."""
        V = max(me.speed, 150.0)
        vh = unit(me.vel)
        d = unit(cmd.d)
        cth = clip(float(np.dot(vh, d)), -1, 1)
        th = math.acos(cth)
        e = perp_part(d, vh)
        if np.linalg.norm(e) < 1e-4:
            e = -me.zb * 1.0 if th < 1.0 else me.yb           # arbitrary perpendicular
        e_hat = unit(e)
        # feed-forward: rate of change of the commanded direction (LOS-rate term).
        # Discontinuities (mode switches) are rejected, the rest is low-passed and
        # clamped so it can never dominate the proportional term.
        if self.d_prev is not None:
            jump = angle(d, self.d_prev)
            if jump < 1.5 * D2R:
                dd = (d - self.d_prev) / self.dt
                self.dd_f += 0.08 * (dd - self.dd_f)
            else:
                self.dd_f *= 0.5
        self.d_prev = d.copy()
        ff = perp_part(self.dd_f, vh)
        ffn = float(np.linalg.norm(ff))
        if ffn * V * 0.8 > 2.0 * G:
            ff *= 2.0 * G / (ffn * V * 0.8)
        n_lim = max(cmd.n_max, 1.05)
        w_max = G * math.sqrt(max(n_lim ** 2 - 1.0, 0.01)) / V
        w = clip(0.4 * cmd.kp * th, 0.0, w_max * 1.3)   # 0.4: gain scale keeps k*tau_plant < 1
        a_cmd = V * w * e_hat + V * 0.8 * ff
        L = a_cmd - G * DOWN                 # required lift vector (specific force)
        n = float(np.linalg.norm(L)) / G
        n = min(n, n_lim)
        return unit(L), n

    def _flight_control(self, me, cmd: Cmd) -> Controls:
        out = Controls()
        if cmd.lift is not None:
            l_des, n_des = unit(cmd.lift), cmd.n
        else:
            l_des, n_des = self._steer_to(me, cmd)

        # ---- roll: put the lift vector (-z body) on l_des ---------------------
        ly = float(np.dot(l_des, me.yb))
        lz = float(np.dot(l_des, -me.zb))
        if math.hypot(ly, lz) < 0.08:
            phi_err = 0.0
        else:
            phi_err = math.atan2(ly, lz)
        p_des = clip(3.8 * phi_err, -3.0, 3.0)                     # rad/s
        out.aileron = clip(0.3182 * p_des, -1.0, 1.0)

        # ---- pull only as far as the lift vector is on the desired side -----------
        c = math.cos(phi_err)
        if c > 0:
            n_eff = 1.0 + (n_des - 1.0) * c ** 1.5 if n_des > 1.0 else n_des
        else:
            n_eff = 0.6 if n_des > 1.0 else n_des
        n_eff = clip(n_eff, -2.5, 8.8)
        if me.alpha * R2D > 21:
            n_eff = min(n_eff, 1.0 + max(0.0, (24 - me.alpha * R2D)) * 0.4)

        # ---- Nz loop (FF + PI) on the JSBSim F-16 stick->g map -------------------
        # reference shaping (tau 0.25 s) + FF + PI + Nz-rate damping.  Gains from a step-response
        # sweep on the JSBSim F-16 (kp .4, ki .05, kd .025, tau .25).
        self.n_ref += (self.dt / (0.25 + self.dt)) * (n_eff - self.n_ref)
        e = self.n_ref - me.nz
        nzd = (me.nz - self.nz_prev) / self.dt
        self.nz_prev = me.nz
        self.nzd_f += 0.3 * (nzd - self.nzd_f)
        u_ff = float(np.interp(self.n_ref, _NZ_TAB, _U_TAB))     # measured stick->Nz map @ ~400 KCAS
        u_ff *= clip((400.0 / max(me.kcas, 150.0)) ** 1.6, 0.6, 2.2) if self.n_ref > 0.5 else 1.0
        sat = (u_ff <= -0.98 and e > 0) or (u_ff >= 0.43 and e < 0)
        if not sat:
            self.n_int = clip(self.n_int + e * self.dt, -1.2, 1.2)
        gs = clip((400.0 / max(me.kcas, 150.0)) ** 2.0, 0.25, 2.5)      # feedback gain ~ 1/q
        u = u_ff - self.gk * gs * (0.4 * e + 0.05 * self.n_int + 0.025 * self.nzd_f)
        out.elevator = clip(u, -1.0, 0.44)

        # ---- throttle (speed hold) -----------------------------------------------
        if cmd.thr is not None:
            out.throttle = cmd.thr
        else:
            err = cmd.v_des - me.kcas
            thr = 0.52 + 0.022 * err if err > 0 else 0.52 + 0.012 * err
            out.throttle = clip(thr, 0.0, 1.0)
        out.speedbrake = cmd.brake
        out.trigger = bool(cmd.fire)
        out.rudder = 0.0
        self.log_mode = f"{self.posture[:3]}/{self.mode}{('/' + cmd.note) if cmd.note else ''}"
        return out
