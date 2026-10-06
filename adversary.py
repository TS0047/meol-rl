"""
Adapter for f16_bfm_agent.BFMAgent (uploaded, Shaw-doctrine rule-based BFM
controller) as a drop-in adversary, replacing the earlier placeholder
(adversary.py's proportional controller) and the LAG pretrained net
(lag_adversary.py). This is the closest adversary yet to the base paper's
own Section 5 "rule-based expert adversary ... from the doctrines and
instructions in [11]" [11]=Shaw -- BFMAgent's docstring cites the exact
same book, chapter by chapter (gun envelope, pursuit curves, yo-yos,
angles/energy fights, defensive maneuvering).

Only the (ego_state, enm_state) state-dict interface established by the
other two adversary adapters is used; BFMAgent itself is untouched.

LadderAdversary adds an opponent curriculum on top: a fixed ladder of
increasingly capable scripted fighters, from a straight-and-level target up to
the full BFMAgent, selected by `level` before each episode.
"""

import numpy as np

from f16_bfm_agent import UP, AcState, BFMAgent, Cmd
from dogfight_sim import TurnBot

DT = 1.0 / 50.0  # matches env.py's fdm.set_dt(1/50)

# Opponent curriculum, easiest first. Every rung is a coherent, ground-safe pilot:
#   straight    non-manoeuvring target at STRAIGHT_KCAS: teaches closing and tracking
#   turn4/turn6 constant-g level turn, random side (dogfight_sim.TurnBot): lead pursuit
#   bfm_energy  BFMAgent, sustained-g energy fight
#   bfm_angles  BFMAgent, max-g angles fight -- the paper's rule-based expert (Sec 5)
# A blend of BFMAgent's commands with wings-level flight was tried first and rejected:
# half a hard roll plus half "wings level" gave incoherent attitudes and the opponent
# crashed itself in 3 of 8 test episodes at the midpoint.
LADDER = ("straight", "turn4", "turn6", "bfm_energy", "bfm_angles")
TOP = len(LADDER) - 1

# The bottom rung must be catchable by a beginner. dogfight_sim.StraightBot cruises at 380
# KCAS (~230 m/s true at 4500 m), faster than our 200 m/s start, so an agent not yet using
# afterburner could never close: 43 of 53 episodes disengaged and the rest ended in
# spirals while chasing (run D, epochs 0-5, no wins). 250 KCAS (~160 m/s true) is slower
# than we start, so pointing at it is enough to close.
STRAIGHT_KCAS = 250.0


def _to_ac_state(s):
    """Build f16_bfm_agent.AcState from env._read_state()'s raw feet/radian
    fields (pos_ned_ft, vel_ned_fps, phi_rad, theta_rad, psi_rad, alpha_rad,
    nz, kcas, mach) -- added to _read_state specifically for this adapter."""
    return AcState(
        pos=s["pos_ned_ft"], vel=s["vel_ned_fps"],
        phi=s["phi_rad"], theta=s["theta_rad"], psi=s["psi_rad"],
        alpha=s["alpha_rad"], nz=s["nz"], kcas=s["kcas"], mach=s["mach"],
    )


def _controls_to_action(c):
    """BFMAgent's Controls is aileron/elevator/rudder/throttle; this project's
    action order is aileron/rudder/elevator/throttle. Throttle is already 0..1 and
    elevator -1..0.44, directly usable on fcs/*-cmd-norm."""
    return np.array([c.aileron, c.rudder, c.elevator, c.throttle], dtype=np.float32)


class BFMAdversary:
    """style: 'angles' (Shaw ch.3 aggressive max-G angles fight, BFMAgent's
    default) or 'energy' (sustained-G energy fight). randomize_style=True
    picks one of the two uniformly at random each episode (reset())."""

    def __init__(self, style="angles", seed=None, randomize_style=False):
        self.style = style
        self.seed = seed
        self.randomize_style = randomize_style
        self._agent = None
        self._t = 0.0
        self._rng = np.random.default_rng(seed)
        self.current = f"bfm_{style}"

    def reset(self):
        style = self._rng.choice(["angles", "energy"]) if self.randomize_style else self.style
        ep_seed = int(self._rng.integers(0, 2**31 - 1)) if self.seed is None else self.seed
        self._agent = BFMAgent(style=style, dt=DT, seed=ep_seed, name="adversary")
        self.current = f"bfm_{style}"
        self._t = 0.0

    def act(self, ego_state, enm_state):
        """ego_state/enm_state: dicts from env._read_state(). Returns
        [aileron, rudder, elevator, throttle] in this project's action order."""
        c = self._agent.act(_to_ac_state(ego_state), _to_ac_state(enm_state), self._t)
        self._t += DT
        return _controls_to_action(c)


class _StraightTarget:
    """Wings-level, constant-speed target: dogfight_sim.StraightBot at a chosen speed."""

    def __init__(self, v_kcas=STRAIGHT_KCAS):
        self.inner = BFMAgent(dt=DT, name="adversary")
        self.v_kcas = v_kcas

    def act(self, me, tg, t):
        return self.inner._flight_control(me, Cmd(lift=UP, n=1.0, v_des=self.v_kcas))


class LadderAdversary:
    """Opponent curriculum: plays LADDER[level], re-read at every reset(), so the
    trainer can move it between episodes. level=TOP is plain BFMAdversary."""

    def __init__(self, level=TOP, seed=None):
        self.level = int(level)
        self._rng = np.random.default_rng(seed)
        self._bot = None
        self._t = 0.0
        self.current = LADDER[self.level]

    def reset(self):
        self.current = name = LADDER[self.level]
        seed = int(self._rng.integers(0, 2**31 - 1))
        if name == "straight":
            self._bot = _StraightTarget()
        elif name in ("turn4", "turn6"):
            self._bot = TurnBot("adversary", DT, g=4.0 if name == "turn4" else 6.0,
                                side=int(self._rng.choice([-1, 1])))
        else:
            self._bot = BFMAgent(style=name.split("_")[1], dt=DT, seed=seed, name="adversary")
        self._t = 0.0

    def act(self, ego_state, enm_state):
        c = self._bot.act(_to_ac_state(ego_state), _to_ac_state(enm_state), self._t)
        self._t += DT
        return _controls_to_action(c)
