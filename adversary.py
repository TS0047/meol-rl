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
"""

import numpy as np

from f16_bfm_agent import AcState, BFMAgent

DT = 1.0 / 50.0  # matches env.py's fdm.set_dt(1/50)


def _to_ac_state(s):
    """Build f16_bfm_agent.AcState from env._read_state()'s raw feet/radian
    fields (pos_ned_ft, vel_ned_fps, phi_rad, theta_rad, psi_rad, alpha_rad,
    nz, kcas, mach) -- added to _read_state specifically for this adapter."""
    return AcState(
        pos=s["pos_ned_ft"], vel=s["vel_ned_fps"],
        phi=s["phi_rad"], theta=s["theta_rad"], psi=s["psi_rad"],
        alpha=s["alpha_rad"], nz=s["nz"], kcas=s["kcas"], mach=s["mach"],
    )


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

    def reset(self):
        style = self._rng.choice(["angles", "energy"]) if self.randomize_style else self.style
        ep_seed = int(self._rng.integers(0, 2**31 - 1)) if self.seed is None else self.seed
        self._agent = BFMAgent(style=style, dt=DT, seed=ep_seed, name="adversary")
        self._t = 0.0

    def act(self, ego_state, enm_state):
        """ego_state/enm_state: dicts from env._read_state(). Returns
        [aileron, rudder, elevator, throttle] matching this project's action
        ordering (BFMAgent's Controls is aileron/elevator/rudder/throttle;
        reordered here). Controls' throttle is already 0..1 and elevator
        already -1..0.44, directly usable on fcs/*-cmd-norm with no rescale,
        matching how env.py applies the adversary's raw action array."""
        me = _to_ac_state(ego_state)
        tg = _to_ac_state(enm_state)
        c = self._agent.act(me, tg, self._t)
        self._t += DT
        return np.array([c.aileron, c.rudder, c.elevator, c.throttle], dtype=np.float32)