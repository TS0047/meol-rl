"""
Angle-tactic training environment (Gymnasium API).
Two JSBSim F-16 instances (self + rule-based adversary) in a shared flat-earth
local NEU frame. Implements Eq.(8) geometry, Section 2.4 WEZ kill logic,
Eq.(38) angle-tactic reward, and Table 1 parameterized initial conditions.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from geometry import body_axes_enu, body_x_enu, combat_geometry, proximity, reset_geometry

from adversary import BFMAdversary, LadderAdversary, TOP as LADDER_TOP
from reward import R_angle

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Table 1 domain
RANGE_BOUNDS = (1900.0, 5100.0)
AA_BOUNDS = (-180.0, 180.0)
ATA_BOUNDS = (-180.0, 180.0)

# Win/loss judgment -- lenient, taken from the paper's Sec 2.4 WEZ:
# 150 m <= range <= 1000 m, +/-15 deg cone. Symmetric for both sides:
#   self shoots      : in range band and ATA <= SHOOT_ATA_DEG
#   adversary shoots : in range band and AA  >= BEHIT_AA_DEG (adversary nose on self)
# (was: range < 914 m, ATA <= 1 deg / AA >= 179 deg, no minimum range.)
# The minimum range keeps collision-range overlaps (LOS ~ 0, ATA/AA undefined)
# from counting as hits; set SHOOT_MIN_RANGE_M = 0.0 to remove it.
SHOOT_RANGE_M = 1000.0
SHOOT_MIN_RANGE_M = 150.0
SHOOT_ATA_DEG = 15.0
BEHIT_AA_DEG = 165.0   # 180 - 15
M2FT = 1.0 / 0.3048

DT = 1.0 / 50.0                 # sim step, 50 Hz (Sec 5.1)
INIT_THROTTLE = 0.7             # initial throttle command, both aircraft
CRASH_PENALTY = 1000.0          # flat terminal reward on self crash = -CRASH_PENALTY.
                                # Must exceed the discounted cost of surviving, worst_step_loss / (1 - gamma):
                                # a bad-but-flyable step (deck, slow, high AoA, sideslip, bandit on our tail,
                                # diving at the deck so P_ground is maxed at -3, pushing negative g so
                                # P_negg is maxed at -1) scores ~-9.5, and gamma = 0.99 (Table 2) gives
                                # ~950, so crashing never pays. Not scaled by remaining steps:
                                # discounting caps the value of the future at ~100 agent decisions, and
                                # -10 * remaining (up to -60000) would swamp the Q-targets.

# Agent decision rate. The sim (and the adversary) run at 50 Hz; the agent picks an
# action every ACTION_REPEAT sim steps and holds it. At 50 Hz, gamma = 0.99 sees only
# ~2 s ahead (0.99^1000 ~ 4e-5 for an event 20 s out), so a dive that ends in a crash
# 10 s later is invisible when it starts. At 10 Hz the same gamma sees ~10 s ahead.
# The per-decision reward is the MEAN of the per-sim-step rewards, so every reward
# constant keeps its per-step meaning; the crash penalty is added once, unaveraged.
ACTION_REPEAT = 5

# Observation layout. "paper" = Eq.(43) only. "extended" appends what the policy needs
# to fly raw control surfaces: its own attitude, body rates, vertical speed, and the
# bandit's direction/heading in body axes (Eq.43's AA/ATA are unsigned, so they cannot
# say whether the bandit is left or right, above or below).
OBS_DIMS = {"paper": 8, "extended": 21}

# Small dense bonus (+/-) added on every sim step in which one side ALONE satisfies its
# shoot condition. Deliberately minor next to c0=10 shaping; it warns that a kill is
# building up before the terminal outcome below arrives.
GOAL_BONUS = 1.0

# Gun kill = sparse combat result, R_goal of Eq.(27). A side wins when it alone holds
# the other inside its WEZ (Sec 2.4) for KILL_DWELL_S without a break -- a tracking
# solution, so a 20 ms graze on a head-on pass is not a kill -- and the episode ENDS,
# as a shoot-down ends the paper's engagements.
#   win  +WIN_REWARD   must beat the discounted shaping an agent could collect by
#                      loitering just outside the WEZ instead of shooting: up to
#                      ~8 / step / (1 - gamma) ~ 800, so 1500. (When episodes never
#                      ended, a cautious draw out-earned a win: timeouts averaged
#                      +150..+226 in run A, wins -150..-1173.)
#   loss -LOSS_PENALTY equal to CRASH_PENALTY: both lose the aircraft, so neither may
#                      look better than the other, and both exceed the discounted
#                      cost of surviving (~850, see CRASH_PENALTY).
#   timeout / adversary crash: 0; disengage: see DISENGAGE_PENALTY.
# terminate_on_kill=False restores the old rule (first one-sided step labels the
# episode, nothing ends it, no terminal reward) for comparison with run A.
KILL_DWELL_S = 0.5
KILL_DWELL_STEPS = int(round(KILL_DWELL_S / DT))
WIN_REWARD = 1500.0
LOSS_PENALTY = CRASH_PENALTY

# Leaving the fight (range beyond DISENGAGE_RANGE_M) ends the episode at the price of a
# loss. At 0 it was a free exit: once losses cost -1000, an agent that loses most fights
# simply ran away (run C3: 16 of 20 episodes disengaged, including every BFM episode).
# 12 km rather than 9 km: a straight-flying target from a bad start used to pass 9 km
# while the agent was still turning around to chase it.
DISENGAGE_RANGE_M = 12000.0
DISENGAGE_PENALTY = LOSS_PENALTY

FT2M = 0.3048
KTS2MPS = 0.514444
DEG2RAD = np.pi / 180.0
R_EARTH_FT = 20925524.9  # matches dogfight_sim.py's flat-earth NED approximation


def _make_fdm():
    import jsbsim
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model("f16")
    fdm.set_dt(DT)  # 50 Hz per Section 5.1
    return fdm


def _start_engine(fdm, throttle=INIT_THROTTLE):
    """Must be called AFTER run_ic(): run_ic() re-initialises the FCS and engine, which
    zeroes the throttle command and leaves the engine off. Also zeroes the stick/rudder
    commands so nothing from the previous episode survives a reset."""
    fdm["fcs/aileron-cmd-norm"] = 0.0
    fdm["fcs/rudder-cmd-norm"] = 0.0
    fdm["fcs/elevator-cmd-norm"] = 0.0
    fdm["fcs/throttle-cmd-norm[0]"] = float(throttle)
    fdm["propulsion/set-running"] = -1   # -1 = all engines


def _init_aircraft(fdm, pos_ned, psi_deg, speed_mps):
    lat = pos_ned[1] / 111320.0
    lon = pos_ned[0] / 111320.0
    fdm["ic/lat-gc-deg"] = lat
    fdm["ic/long-gc-deg"] = lon
    fdm["ic/h-sl-ft"] = pos_ned[2] / FT2M
    fdm["ic/psi-true-deg"] = psi_deg
    fdm["ic/u-fps"] = speed_mps / FT2M  # ft/s
    fdm["ic/v-fps"] = 0.0
    fdm["ic/w-fps"] = 0.0
    fdm["ic/phi-deg"] = 0.0
    fdm["ic/theta-deg"] = 0.0
    # NOT run_ic(): run_ic() leaves the control-surface actuator states and sim time from
    # the previous episode untouched (verified on JSBSim 1.3.1 -- a dirty episode leaked
    # aileron/elevator/rudder positions into the next). Setting the ICs and then calling
    # reset_to_initial_conditions(0) re-initialises everything, and gives bit-identical
    # state whether the FDM is brand new or has just flown a violent episode.
    fdm.reset_to_initial_conditions(0)
    _start_engine(fdm)  # reset also zeroes throttle command / stops the engine


def _read_state(fdm, origin_ll=(0.0, 0.0)):
    lat = fdm["position/lat-gc-deg"]
    lon = fdm["position/long-gc-deg"]
    north = (lat - origin_ll[0]) * 111320.0
    east = (lon - origin_ll[1]) * 111320.0
    up = fdm["position/h-sl-ft"] * FT2M
    pos = np.array([east, north, up])
    psi = fdm["attitude/psi-deg"]
    e_hat = body_x_enu(psi * DEG2RAD, fdm["attitude/theta-rad"])  # 3D nose vector, Eq.(8)
    v_ned = np.array([
        fdm["velocities/v-east-fps"] * FT2M,
        fdm["velocities/v-north-fps"] * FT2M,
        -fdm["velocities/v-down-fps"] * FT2M,
    ])
    return {
        "pos": pos, "psi_deg": psi, "e_hat": e_hat, "v": v_ned,
        "vt_mps": fdm["velocities/vtrue-kts"] * KTS2MPS,
        "vias_mps": fdm["velocities/vc-kts"] * KTS2MPS,  # Eq.(32)/(34) use V_IAS, not V_true
        "alt_m": pos[2],
        "roll_deg": fdm["attitude/phi-deg"],
        "pitch_deg": fdm["attitude/theta-deg"],
        "alpha_deg": fdm["aero/alpha-deg"],
        "beta_deg": fdm["aero/beta-deg"],
        # Raw feet/radian fields for f16_bfm_agent.AcState (NED ft, z=-alt),
        # mirroring dogfight_sim.py's F16.state() exactly so the BFM agent
        # sees state in the units/frame it was designed for.
        "pos_ned_ft": np.array([
            (fdm["position/lat-gc-deg"] - origin_ll[0]) * DEG2RAD * R_EARTH_FT,
            (fdm["position/long-gc-deg"] - origin_ll[1]) * DEG2RAD * R_EARTH_FT
            * np.cos(origin_ll[0] * DEG2RAD),
            -fdm["position/h-sl-ft"],
        ]),
        "vel_ned_fps": np.array([
            fdm["velocities/v-north-fps"], fdm["velocities/v-east-fps"], fdm["velocities/v-down-fps"],
        ]),
        "phi_rad": fdm["attitude/phi-rad"],
        "theta_rad": fdm["attitude/theta-rad"],
        "psi_rad": fdm["attitude/psi-rad"],
        "pqr_rad_s": np.array([
            fdm["velocities/p-rad_sec"], fdm["velocities/q-rad_sec"], fdm["velocities/r-rad_sec"],
        ]),
        "alpha_rad": fdm["aero/alpha-rad"],
        "nz": fdm["accelerations/Nz"],
        "kcas": fdm["velocities/vc-kts"],
        "mach": fdm["velocities/mach"],
        "v_body_mps": np.array([
            fdm["velocities/u-fps"] * FT2M,
            fdm["velocities/v-fps"] * FT2M,
            fdm["velocities/w-fps"] * FT2M,
        ]),
    }


def _energy(state):
    g = 9.81
    return state["alt_m"] + (state["vias_mps"] ** 2) / (2 * g)  # Eq.(32): V_IAS, not V_true


class AngleTacticEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=6000, log_trajectory=False, adversary_type="bfm",
                 obs_mode="extended", action_repeat=ACTION_REPEAT, terminate_on_kill=True,
                 opponent_level=None):
        """max_steps counts 50 Hz sim steps (6000 = 120 s), whatever action_repeat is.
        obs_mode="paper", action_repeat=1 reproduces the paper's interface exactly;
        terminate_on_kill=False reproduces run A's never-ending, first-graze outcome rule."""
        super().__init__()
        if obs_mode not in OBS_DIMS:
            raise ValueError(f"unknown obs_mode {obs_mode!r}, expected one of {list(OBS_DIMS)}")
        self.max_steps = max_steps
        self.log_trajectory = log_trajectory
        self.obs_mode = obs_mode
        self.action_repeat = int(action_repeat)
        self.terminate_on_kill = bool(terminate_on_kill)
        self._dwell_self = self._dwell_adv = 0
        self.observation_space = spaces.Box(low=-1e4, high=1e4, shape=(OBS_DIMS[obs_mode],),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.fdm_self = _make_fdm()
        self.fdm_adv = _make_fdm()
        self.adversary_type = adversary_type
        if adversary_type == "bfm":
            self.adversary = BFMAdversary()
        elif adversary_type == "ladder":
            # opponent curriculum; set self.adversary.level between episodes
            self.adversary = LadderAdversary(level=LADDER_TOP if opponent_level is None else opponent_level)
        elif adversary_type == "lag_baseline":
            # lag_adversary.py (LAG pretrained net) is no longer in the repo -- BFMAdversary
            # superseded it. Imported here so the default path does not depend on it.
            from lag_adversary import LAGBaselineAdversary
            self.adversary = LAGBaselineAdversary()
        else:
            raise ValueError(f"unknown adversary_type {adversary_type!r}")
        self.step_count = 0
        self.trajectory = []
        self.first_goal = None

    def _obs(self, geom, prox, s_self, s_adv):
        paper = [                                   # Eq.(43)
            geom["Range"] / 5000.0,
            prox / 300.0,
            geom["AA"] / 180.0,
            geom["ATA"] / 180.0,
            s_self["vias_mps"] / 300.0,
            s_self["alt_m"] / 6000.0,
            s_self["alpha_deg"] / 30.0,
            s_self["beta_deg"] / 30.0,
        ]
        if self.obs_mode == "paper":
            return np.array(paper, dtype=np.float32)
        xb, yb, zb = body_axes_enu(s_self["phi_rad"], s_self["theta_rad"], s_self["psi_rad"])
        los, e_adv = geom["los_hat"], s_adv["e_hat"]
        extended = [
            np.sin(s_self["phi_rad"]), np.cos(s_self["phi_rad"]),   # roll: which way is up
            np.sin(s_self["theta_rad"]),                             # pitch
            *(s_self["pqr_rad_s"] / 3.0),                            # body rates p, q, r
            los @ xb, los @ yb, los @ zb,       # bandit direction: ahead / right wing / below floor
            e_adv @ xb, e_adv @ yb, e_adv @ zb,  # bandit's nose direction in our body axes
            s_self["v"][2] / 100.0,             # vertical speed, + = climbing
        ]
        return np.array(paper + extended, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if options and "init_condition" in options:
            r0, aa0, ata0 = options["init_condition"]
        else:
            r0 = rng.uniform(*RANGE_BOUNDS)
            aa0 = rng.uniform(*AA_BOUNDS)
            ata0 = rng.uniform(*ATA_BOUNDS)
        self.last_init_condition = (float(r0), float(aa0), float(ata0))
        alt0 = 4500.0
        p_self, psi_self, p_adv, psi_adv = reset_geometry(r0, aa0, ata0, alt0)

        _init_aircraft(self.fdm_self, p_self, psi_self, 200.0)
        _init_aircraft(self.fdm_adv, p_adv, psi_adv, 200.0)

        s_self = _read_state(self.fdm_self)
        s_adv = _read_state(self.fdm_adv)
        geom = combat_geometry(s_self["pos"], s_self["e_hat"], s_adv["pos"], s_adv["e_hat"])
        prox = proximity(s_self["pos"], s_self["v"], s_adv["pos"], s_adv["v"])

        self.step_count = 0
        self.trajectory = []
        self.first_goal = None
        self._dwell_self = self._dwell_adv = 0
        self.adversary.reset()
        self._log_step(s_self, s_adv, geom, prox)
        return self._obs(geom, prox, s_self, s_adv), {}

    def step(self, action):
        """One agent decision: hold `action` for action_repeat 50 Hz sim steps (fewer if
        the episode ends). The adversary still acts every sim step."""
        action = np.clip(action, -1.0, 1.0)
        self.fdm_self["fcs/aileron-cmd-norm"] = float(action[0])
        self.fdm_self["fcs/rudder-cmd-norm"] = float(action[1])
        self.fdm_self["fcs/elevator-cmd-norm"] = float(action[2])
        self.fdm_self["fcs/throttle-cmd-norm[0]"] = float(np.clip((action[3] + 1) / 2, 0.0, 1.0))

        rewards, terminal = [], 0.0
        for _ in range(self.action_repeat):
            s_self_prev = _read_state(self.fdm_self)
            s_adv_prev = _read_state(self.fdm_adv)
            adv_action = self.adversary.act(s_adv_prev, s_self_prev)
            self.fdm_adv["fcs/aileron-cmd-norm"] = float(adv_action[0])
            self.fdm_adv["fcs/rudder-cmd-norm"] = float(adv_action[1])
            self.fdm_adv["fcs/elevator-cmd-norm"] = float(adv_action[2])
            self.fdm_adv["fcs/throttle-cmd-norm[0]"] = float(adv_action[3])

            self.fdm_self.run()
            self.fdm_adv.run()
            self.step_count += 1

            s_self = _read_state(self.fdm_self)
            s_adv = _read_state(self.fdm_adv)
            geom = combat_geometry(s_self["pos"], s_self["e_hat"], s_adv["pos"], s_adv["e_hat"])
            prox = proximity(s_self["pos"], s_self["v"], s_adv["pos"], s_adv["v"])

            bonus, terminal, terminated, outcome = self._check_outcome(geom, s_self, s_adv)
            truncated = self.step_count >= self.max_steps

            state_dict = {
                "Range": geom["Range"], "Proximity": prox, "AA": geom["AA"], "ATA": geom["ATA"],
                "E_self": _energy(s_self), "E_adv": _energy(s_adv),
                "altitude": s_self["alt_m"], "VIAS": s_self["vias_mps"],
                "alpha_AoA": s_self["alpha_deg"], "beta": s_self["beta_deg"],
                "v_up": s_self["v"][2], "V_true": s_self["vt_mps"], "Nz": s_self["nz"],
            }
            rewards.append(R_angle(state_dict, list(action), bonus))
            self._log_step(s_self, s_adv, geom, prox)
            if terminated or truncated:
                break
        reward = float(np.mean(rewards)) + terminal   # terminal outcome paid once, not averaged away

        if truncated and not terminated:
            outcome = "timeout"
        if terminated or truncated:
            outcome = self._label_outcome(outcome)
        info = {"outcome": outcome, "goal": self.first_goal, "HCA": geom["HCA"],
                "sim_steps": len(rewards), "opponent": getattr(self.adversary, "current", "")}
        return self._obs(geom, prox, s_self, s_adv), reward, terminated, truncated, info

    def _check_outcome(self, geom, s_self, s_adv):
        """Returns (bonus, terminal, terminated, outcome). bonus is averaged into the
        per-decision reward with the shaping; terminal is paid once, unaveraged."""
        in_band = SHOOT_MIN_RANGE_M <= geom["Range"] <= SHOOT_RANGE_M
        self_goal = in_band and geom["ATA"] <= SHOOT_ATA_DEG
        adv_goal = in_band and geom["AA"] >= BEHIT_AA_DEG
        # Only a one-sided hit counts. With the lenient cones a head-on pass puts both
        # aircraft inside each other's cone at once -- a merge, not a decision -- so it
        # neither builds a kill nor earns a bonus.
        one_self, one_adv = self_goal and not adv_goal, adv_goal and not self_goal
        self._dwell_self = self._dwell_self + 1 if one_self else 0
        self._dwell_adv = self._dwell_adv + 1 if one_adv else 0
        bonus = GOAL_BONUS if one_self else -GOAL_BONUS if one_adv else 0.0
        if s_self["alt_m"] <= 200.0:
            return 0.0, -CRASH_PENALTY, True, "crash"
        if self.terminate_on_kill:
            if self._dwell_self >= KILL_DWELL_STEPS:
                self.first_goal = "self"
                return bonus, WIN_REWARD, True, "win"
            if self._dwell_adv >= KILL_DWELL_STEPS:
                self.first_goal = "adversary"
                return bonus, -LOSS_PENALTY, True, "loss"
        elif self.first_goal is None and (one_self or one_adv):
            self.first_goal = "self" if one_self else "adversary"   # run A's rule: label only
        if s_adv["alt_m"] <= 200.0:
            # Neither this nor the base paper (Sec 2.4) scores "opponent
            # crashed" as a win -- terminate (avoids the Phi_energy
            # instability as E_adv collapses) but treat as a neutral draw.
            return bonus, 0.0, True, "adversary_crash"
        if geom["Range"] >= DISENGAGE_RANGE_M:
            return bonus, -DISENGAGE_PENALTY, True, "disengaged"
        return bonus, 0.0, False, "ongoing"

    def _label_outcome(self, outcome):
        """Final episode label. Win/loss = who satisfied the WEZ condition first.
        A self-crash always stays 'crash' (a loss)."""
        if outcome == "crash":
            return "crash"
        return {"self": "win", "adversary": "loss", "draw": "draw"}.get(self.first_goal, outcome)

    def _log_step(self, s_self, s_adv, geom, prox):
        if not self.log_trajectory:
            return
        self.trajectory.append({
            "t": self.step_count / 50.0,
            "self_pos": s_self["pos"].copy(), "adv_pos": s_adv["pos"].copy(),
            "Range": geom["Range"], "AA": geom["AA"], "ATA": geom["ATA"], "HCA": geom["HCA"],
            "Proximity": prox, "E_self": _energy(s_self), "E_adv": _energy(s_adv),
            "self_att": (s_self["roll_deg"], s_self["pitch_deg"], s_self["psi_deg"]),   # deg, for ACMI
            "adv_att": (s_adv["roll_deg"], s_adv["pitch_deg"], s_adv["psi_deg"]),
        })