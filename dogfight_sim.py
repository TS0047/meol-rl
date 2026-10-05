"""
dogfight_sim.py
===============
Two JSBSim F-16s fighting each other at 50 Hz (dt = 0.02 s), guns only.

  * every control tick: read both aircraft -> agent.act() -> write FCS commands -> step JSBSim
  * M61A1 model: 6000 rpm, 3300 ft/s muzzle velocity, bullet drag + gravity drop, dispersion,
    511 rounds, sphere hit-test in the target frame, N hits = kill
  * crash / mid-air collision / hard-deck detection
  * Tacview .acmi export and CSV log

Run:  python dogfight_sim.py --scenario neutral --blue angles --red energy --seed 1
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

import jsbsim

from f16_bfm_agent import (AcState, Controls, BFMAgent, FT_PER_NM, G, D2R, R2D, DOWN, UP,
                           V_MUZ, K_DRAG, unit, clip, n_cap, gun_solution)

R_EARTH_FT = 20925524.9
LAT0, LON0 = 30.0, 0.0


# --------------------------------------------------------------------------- #
# F-16 wrapper
# --------------------------------------------------------------------------- #
class F16:
    def __init__(self, name, north_ft, east_ft, alt_ft, heading_deg, kcas, dt=0.02, substeps=1):
        self.name = name
        self.substeps = substeps
        f = jsbsim.FGFDMExec(None)
        f.set_debug_level(0)
        f.load_model("f16")
        f.set_dt(dt / substeps)
        lat = LAT0 + (north_ft / R_EARTH_FT) * R2D
        lon = LON0 + (east_ft / (R_EARTH_FT * math.cos(LAT0 * D2R))) * R2D
        f["ic/h-sl-ft"] = alt_ft
        f["ic/lat-gc-deg"] = lat
        f["ic/long-gc-deg"] = lon
        f["ic/psi-true-deg"] = heading_deg
        f["ic/vc-kts"] = kcas
        f["ic/gamma-deg"] = 0.0
        f["ic/theta-deg"] = 2.0
        f.run_ic()
        f["propulsion/set-running"] = -1
        f["fcs/throttle-cmd-norm"] = 0.7
        self.fdm = f
        # settle: ~3 s of wings-level 1-g flight with the same low-level controller so the
        # fight starts from a trimmed, stable state (FCS integrators alive)
        self.fdm = f
        self.alive = True
        _a = BFMAgent(dt=dt)
        from f16_bfm_agent import Cmd
        for _ in range(int(3.0 / dt)):
            s_ = self.state()
            hold = Cmd(lift=UP, n=1.0 + 0.004 * (alt_ft - s_.h) - 0.002 * s_.vel[2] * -1.0, v_des=kcas)
            c_ = _a._flight_control(s_, hold)
            self.apply(c_)
            self.step()
        self.alive = True
        self.hits = 0
        self.ammo = 511
        self.cause = ""

    def state(self) -> AcState:
        f = self.fdm
        lat, lon, h = f["position/lat-gc-deg"], f["position/long-gc-deg"], f["position/h-sl-ft"]
        n = (lat - LAT0) * D2R * R_EARTH_FT
        e = (lon - LON0) * D2R * R_EARTH_FT * math.cos(LAT0 * D2R)
        return AcState(
            pos=np.array([n, e, -h]),
            vel=np.array([f["velocities/v-north-fps"], f["velocities/v-east-fps"], f["velocities/v-down-fps"]]),
            phi=f["attitude/phi-rad"], theta=f["attitude/theta-rad"], psi=f["attitude/psi-rad"],
            alpha=f["aero/alpha-rad"], nz=f["accelerations/Nz"],
            kcas=f["velocities/vc-kts"], mach=f["velocities/mach"])

    def apply(self, c: Controls):
        f = self.fdm
        f["fcs/aileron-cmd-norm"] = c.aileron
        f["fcs/elevator-cmd-norm"] = c.elevator
        f["fcs/rudder-cmd-norm"] = c.rudder
        f["fcs/throttle-cmd-norm"] = c.throttle
        f["fcs/speedbrake-cmd-norm"] = c.speedbrake

    def step(self):
        for _ in range(self.substeps):
            self.fdm.run()


# --------------------------------------------------------------------------- #
# gun / bullets
# --------------------------------------------------------------------------- #
class Gun:
    RPS = 100.0              # 6000 rpm
    SIGMA = 5e-3             # rad, 1-sigma dispersion per axis (≈5 mil, Shaw ch.1)

    def __init__(self, rng, hit_radius=11.0):
        self.rng = rng
        self.acc = 0.0
        self.b_p0 = np.zeros((0, 3))
        self.b_v0 = np.zeros((0, 3))
        self.b_t = np.zeros(0)
        self.hit_r = hit_radius
        self.prev_rel = None
        self.shots = 0

    def fire(self, me: AcState, dt, ammo):
        self.acc += self.RPS * dt
        n = int(self.acc)
        self.acc -= n
        n = min(n, ammo)
        if n <= 0:
            return 0
        d = np.tile(me.xb, (n, 1))
        d = d + self.rng.normal(0, self.SIGMA, (n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        v0 = me.vel + V_MUZ * d
        # muzzle slightly ahead of the airframe, spawn spread inside this tick
        frac = (np.arange(n) / max(n, 1)) * dt
        p0 = me.pos + me.xb * 15.0 - me.vel * 0.0
        self.b_p0 = np.vstack([self.b_p0, np.tile(p0, (n, 1))])
        self.b_v0 = np.vstack([self.b_v0, v0])
        self.b_t = np.concatenate([self.b_t, -frac])
        self.shots += n
        return n

    def _pos(self, t):
        f = (1.0 - np.exp(-K_DRAG * np.maximum(t, 0))) / K_DRAG
        return self.b_p0 + self.b_v0 * f[:, None] + 0.5 * G * (np.maximum(t, 0) ** 2)[:, None] * DOWN

    def advance(self, dt, tgt_prev: AcState, tgt_now: AcState):
        """advance bullets by dt; return number of hits on target during this tick"""
        if len(self.b_t) == 0:
            return 0
        p_old = self._pos(self.b_t)
        self.b_t = self.b_t + dt
        p_new = self._pos(self.b_t)
        # relative to target (linear interpolation of target pos over the tick)
        rel_old = p_old - tgt_prev.pos
        rel_new = p_new - tgt_now.pos
        seg = rel_new - rel_old
        L2 = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-9)
        s = np.clip(-np.einsum("ij,ij->i", rel_old, seg) / L2, 0.0, 1.0)
        closest = rel_old + seg * s[:, None]
        dist = np.linalg.norm(closest, axis=1)
        hit = dist < self.hit_r
        nh = int(hit.sum())
        keep = (~hit) & (self.b_t < 4.5)
        self.b_p0, self.b_v0, self.b_t = self.b_p0[keep], self.b_v0[keep], self.b_t[keep]
        return nh


# --------------------------------------------------------------------------- #
# baseline opponents for sanity tests
# --------------------------------------------------------------------------- #
class PurePursuitBot:
    """Dumb reference opponent: pure pursuit, max-ish G, fires in the envelope."""

    def __init__(self, name="bot", dt=0.02):
        self.inner = BFMAgent(style="angles", dt=dt, name=name)
        self.name = name

    def act(self, me, tg, t):
        from f16_bfm_agent import Cmd
        r = tg.pos - me.pos
        R = float(np.linalg.norm(r))
        cmd = Cmd(d=unit(r), n_max=n_cap(me.kcas), kp=3.0, v_des=400)
        d_sol, _ = gun_solution(me, tg)
        err = math.acos(clip(float(np.dot(me.xb, d_sol)), -1, 1))
        cmd.fire = 500 <= R <= 3000 and err * R < 16
        cmd = self.inner._safety(me, tg, dict(R=R), cmd)
        return self.inner._flight_control(me, cmd)


class StraightBot:
    """Non-maneuvering target (wings level, constant speed)."""

    def __init__(self, name="target", dt=0.02):
        self.inner = BFMAgent(dt=dt, name=name)
        self.name = name

    def act(self, me, tg, t):
        from f16_bfm_agent import Cmd
        cmd = Cmd(lift=UP, n=1.0, v_des=380)
        return self.inner._flight_control(me, cmd)


# --------------------------------------------------------------------------- #
# engagement runner
# --------------------------------------------------------------------------- #
SCENARIOS = {
    # (blue N, E, alt, hdg), (red N, E, alt, hdg), kcas blue, kcas red
    "neutral":   dict(sep=9.0, lat=2500.0, dalt=0.0, bk=400, rk=400),     # head-on, 9 nm
    "neutral_hi": dict(sep=9.0, lat=2500.0, dalt=1500.0, bk=400, rk=400),
}


def build(scn, rng, dt):
    if scn in SCENARIOS:
        s = SCENARIOS[scn]
        sep = s["sep"] * FT_PER_NM
        alt = 15000.0
        jitter = rng.uniform(-800, 800)
        blue = F16("BLUE", 0.0, 0.0, alt, 0.0, s["bk"], dt)
        red = F16("RED", sep, s["lat"] + jitter, alt + s["dalt"], 180.0, s["rk"], dt)
        return blue, red
    if scn == "blue_offensive":        # blue 3000 ft behind red, both 400 KCAS
        blue = F16("BLUE", 0.0, 0.0, 15000.0, 0.0, 420, dt)
        red = F16("RED", 3500.0, 300.0, 15000.0, 0.0, 400, dt)
        return blue, red
    if scn == "blue_defensive":
        blue = F16("BLUE", 3500.0, 300.0, 15000.0, 0.0, 400, dt)
        red = F16("RED", 0.0, 0.0, 15000.0, 0.0, 420, dt)
        return blue, red
    if scn == "perch":                  # blue high 45deg off the tail, classic perch setup
        blue = F16("BLUE", -3000.0, 4500.0, 17000.0, 330.0, 420, dt)
        red = F16("RED", 0.0, 0.0, 15000.0, 0.0, 380, dt)
        return blue, red
    raise ValueError(scn)


class TurnBot:
    """Constant-G level turn (non-reactive maneuvering target) - used to test tracking."""

    def __init__(self, name="turner", dt=0.02, g=4.0, side=1):
        self.inner = BFMAgent(dt=dt, name=name)
        self.g, self.side, self.name = g, side, name

    def act(self, me, tg, t):
        from f16_bfm_agent import Cmd
        bank = math.acos(1.0 / self.g)
        l = np.array([0.0, 0.0, -1.0])
        h = np.array([-math.sin(math.atan2(me.vel[1], me.vel[0])), math.cos(math.atan2(me.vel[1], me.vel[0])), 0.0])
        lift = l * math.cos(bank) + self.side * h * math.sin(bank)
        return self.inner._flight_control(me, Cmd(lift=lift, n=self.g, v_des=400))


def make_agent(kind, name, dt, seed):
    if kind in ("angles", "energy"):
        return BFMAgent(style=kind, dt=dt, name=name, seed=seed)
    if kind == "pure":
        return PurePursuitBot(name, dt)
    if kind == "straight":
        return StraightBot(name, dt)
    if kind == "turn4":
        return TurnBot(name, dt, 4.0)
    if kind == "turn6":
        return TurnBot(name, dt, 6.0)
    raise ValueError(kind)


def acmi_line(oid, s: AcState, f):
    lat = f["position/lat-gc-deg"]
    lon = f["position/long-gc-deg"]
    alt_m = f["position/h-sl-ft"] * 0.3048
    return (f"{oid},T={lon:.6f}|{lat:.6f}|{alt_m:.1f}|{s.phi * R2D:.1f}|{s.theta * R2D:.1f}|"
            f"{(s.psi * R2D) % 360:.1f}")


def run(scn="neutral", blue="angles", red="energy", seed=0, t_max=180.0, hz=50, realtime=False,
        out_prefix=None, verbose=True, hits_to_kill=3, substeps=1):
    dt = 1.0 / hz
    rng = np.random.default_rng(seed)
    bf, rf = build(scn, np.random.default_rng(seed), dt)
    if substeps != 1:
        pass
    ag = {"BLUE": make_agent(blue, "BLUE", dt, seed), "RED": make_agent(red, "RED", dt + 0.0, seed + 100)}
    fr = {"BLUE": bf, "RED": rf}
    guns = {"BLUE": Gun(np.random.default_rng(seed + 1)), "RED": Gun(np.random.default_rng(seed + 2))}
    acmi = None
    if out_prefix:
        acmi = open(out_prefix + ".acmi", "w")
        acmi.write("FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2026-10-04T12:00:00Z\n")
        acmi.write("100,Name=F-16C,Type=Air+FixedWing,Coalition=Blue,Color=Blue,Callsign=BLUE\n")
        acmi.write("200,Name=F-16C,Type=Air+FixedWing,Coalition=Red,Color=Red,Callsign=RED\n")
    log = []
    t = 0.0
    outcome = "TIMEOUT"
    prev = {"BLUE": bf.state(), "RED": rf.state()}
    steps = int(t_max * hz)
    wall0 = time.perf_counter()
    overruns = 0
    max_loop = 0.0
    first_shot = {"BLUE": None, "RED": None}
    for k in range(steps):
        tick0 = time.perf_counter()
        sb, sr = bf.state(), rf.state()
        R = float(np.linalg.norm(sr.pos - sb.pos))
        # ---- decisions -------------------------------------------------------
        cb = ag["BLUE"].act(sb, sr, t) if bf.alive else Controls(throttle=0.5, elevator=-0.1)
        cr = ag["RED"].act(sr, sb, t) if rf.alive else Controls(throttle=0.5, elevator=-0.1)
        bf.apply(cb)
        rf.apply(cr)
        # ---- guns --------------------------------------------------------------
        if cb.trigger and bf.alive and bf.ammo > 0:
            n = guns["BLUE"].fire(sb, dt, bf.ammo)
            bf.ammo -= n
            first_shot["BLUE"] = first_shot["BLUE"] or t
        if cr.trigger and rf.alive and rf.ammo > 0:
            n = guns["RED"].fire(sr, dt, rf.ammo)
            rf.ammo -= n
            first_shot["RED"] = first_shot["RED"] or t
        # ---- physics ------------------------------------------------------------
        bf.step()
        rf.step()
        t += dt
        nb, nr = bf.state(), rf.state()
        hb = guns["BLUE"].advance(dt, sr, nr)
        hr = guns["RED"].advance(dt, sb, nb)
        if rf.alive:
            rf.hits += hb
        if bf.alive:
            bf.hits += hr
        # ---- referee ---------------------------------------------------------------
        if rf.alive and rf.hits >= hits_to_kill:
            rf.alive = False
            rf.cause = "SHOT"
        if bf.alive and bf.hits >= hits_to_kill:
            bf.alive = False
            bf.cause = "SHOT"
        for f_, s_ in ((bf, nb), (rf, nr)):
            if f_.alive and s_.h < 300:
                f_.alive = False
                f_.cause = "CRASH"
        if bf.alive and rf.alive and float(np.linalg.norm(nr.pos - nb.pos)) < 60:
            bf.alive = rf.alive = False
            bf.cause = rf.cause = "COLLISION"
        loop = time.perf_counter() - tick0
        max_loop = max(max_loop, loop)
        if realtime:
            slack = dt - loop
            if slack > 0:
                time.sleep(slack)
            else:
                overruns += 1
        # ---- log ---------------------------------------------------------------------
        if k % 5 == 0:
            Rn = float(np.linalg.norm(nr.pos - nb.pos))
            log.append((t, nb.pos[0], nb.pos[1], nb.h, nb.kcas, nb.nz, nr.pos[0], nr.pos[1], nr.h, nr.kcas, nr.nz,
                        Rn, ag["BLUE"].log_mode if hasattr(ag["BLUE"], "log_mode") else "",
                        ag["RED"].log_mode if hasattr(ag["RED"], "log_mode") else "", bf.hits, rf.hits))
            if acmi:
                acmi.write(f"#{t:.2f}\n{acmi_line(100, nb, bf.fdm)}\n{acmi_line(200, nr, rf.fdm)}\n")
        if not bf.alive or not rf.alive:
            break
    if not bf.alive or not rf.alive:
        if not bf.alive and not rf.alive:
            outcome = f"DRAW ({bf.cause}/{rf.cause})"
        elif not rf.alive:
            outcome = f"BLUE WINS (red {rf.cause})"
        else:
            outcome = f"RED WINS (blue {bf.cause})"
    if acmi:
        acmi.close()
    if out_prefix:
        import csv
        with open(out_prefix + ".csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "bN", "bE", "bAlt", "bKCAS", "bNz", "rN", "rE", "rAlt", "rKCAS", "rNz", "range",
                        "bMode", "rMode", "bHits", "rHits"])
            w.writerows(log)
    res = dict(outcome=outcome, t=round(t, 1), blue_hits=bf.hits, red_hits=rf.hits,
               blue_shots=guns["BLUE"].shots, red_shots=guns["RED"].shots,
               max_loop_ms=round(max_loop * 1000, 2), overruns=overruns, log=log)
    if verbose:
        print(f"[{scn} | blue={blue} red={red} seed={seed}] {outcome} at t={res['t']}s  "
              f"hits B/R = {bf.hits}/{rf.hits}  rounds B/R = {guns['BLUE'].shots}/{guns['RED'].shots}  "
              f"max tick {res['max_loop_ms']} ms")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="neutral", choices=list(SCENARIOS) + ["blue_offensive", "blue_defensive", "perch"])
    ap.add_argument("--blue", default="angles", choices=["angles", "energy", "pure", "straight", "turn4", "turn6"])
    ap.add_argument("--red", default="energy", choices=["angles", "energy", "pure", "straight", "turn4", "turn6"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=180.0)
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--realtime", action="store_true", help="pace the loop to wall-clock 50 Hz")
    ap.add_argument("--out", default=None, help="prefix for .acmi/.csv output")
    a = ap.parse_args()
    run(a.scenario, a.blue, a.red, a.seed, a.tmax, a.hz, a.realtime, a.out)
