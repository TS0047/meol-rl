"""
Reward-shaping functions for MEOL intra-option tactics (angle, snapshot, energy).
Implements Eq. (28)-(41) from Li et al., Drones 2025, 9, 384.

Constants not specified in the paper (c1, d_r, c_r, c_AA, c_ATA, altitude_deck,
V_corner, c_VIAS, alpha_stall, beta_max, c_beta) are set below as documented
assumptions -- see DEFAULT_CONSTANTS. Only c0=10 is taken from Table 2.
"""

import math
from dataclasses import dataclass


@dataclass
class Constants:
    c0: float = 10        # Table 2 (given)
    c1: float = 0.25           # NOT given in paper -- assumed
    d_r: float = 500.0        # target range (m), ~mid of WEZ [150,1000] -- assumed
    c_r: float = 400.0        # range potential scale (m) -- assumed
    c_AA: float = 30.0        # deg, AA potential scale -- assumed
    c_ATA: float = 30.0       # deg, ATA potential scale -- assumed
    altitude_deck: float = 500.0   # m, hard deck -- assumed
    V_corner: float = 120.0        # m/s, ~1.4-1.5x stall speed (Shaw p.408) -- assumed
    c_VIAS: float = 30.0           # m/s, penalty scale -- assumed
    alpha_stall: float = 25.0      # deg, mid of Shaw's 20-30deg usable AoA -- assumed
    beta_max: float = 20.0         # deg, sideslip limit -- assumed
    c_beta: float = 20.0       # deg, sideslip penalty scale -- assumed. Keep c_beta == beta_max:
                               # Eq.(36) is -(beta/c_beta)^2 inside the limit and -1 outside it, so
                               # any c_beta < beta_max makes the penalty jump UP (less negative) at the limit.


C = Constants()


# ---- Potential terms (Eq. 28-32) --------------------------------------

def phi_range(range_m: float, c: Constants = C) -> float:
    return math.exp(-abs(range_m - c.d_r) / c.c_r)


def phi_AA(AA_deg: float, c: Constants = C) -> float:
    return math.exp(-abs(AA_deg) / c.c_AA)


def phi_ATA(ATA_deg: float, c: Constants = C) -> float:
    return math.exp(-abs(ATA_deg) / c.c_ATA)


def phi_energy(E_self: float, E_adv: float) -> float:
    return E_self / E_adv if E_adv != 0 else 0.0


def r_proximity(proximity: float) -> float:
    return -proximity / 1000.0


# ---- Regularization terms (Eq. 33-37) ----------------------------------

def P_deck(altitude: float, c: Constants = C) -> float:
    return -1.0 if altitude <= c.altitude_deck else 0.0


def P_VIAS(vias: float, c: Constants = C) -> float:
    if vias <= c.V_corner:
        return -((vias - c.V_corner) / c.c_VIAS) ** 2
    return 0.0


def P_alpha_AoA(alpha_aoa: float, c: Constants = C) -> float:
    return -1.0 if abs(alpha_aoa) >= c.alpha_stall else 0.0


def P_beta(beta: float, c: Constants = C) -> float:
    if abs(beta) <= c.beta_max:
        return -(beta / c.c_beta) ** 2
    return -1.0


def C_action(action: list[float]) -> float:
    return -0.1 * sum(a ** 2 for a in action)


def regularization(state: dict, action: list[float], c: Constants = C) -> float:
    return (
        P_deck(state["altitude"], c)
        + P_VIAS(state["VIAS"], c)
        + P_alpha_AoA(state["alpha_AoA"], c)
        + P_beta(state["beta"], c)
        + C_action(action)
    )


# ---- Tactic reward assembly (Eq. 38, 40, 41) ----------------------------
# R_goal: sparse combat-outcome term, NOT defined numerically in the paper.
# Supplied externally per episode; see caller.

def R_angle(state: dict, action: list[float], R_goal: float, c: Constants = C) -> float:
    shaping = (
        c.c0 * phi_ATA(state["ATA"], c) * phi_AA(state["AA"], c) * phi_range(state["Range"], c)
        + c.c1 * phi_energy(state["E_self"], state["E_adv"])
        + r_proximity(state["Proximity"])
    )
    return R_goal + shaping + regularization(state, action, c)


def R_snapshot(state: dict, action: list[float], R_goal: float, c: Constants = C) -> float:
    shaping = (
        c.c0 * phi_ATA(state["ATA"], c) * phi_range(state["Range"], c)
        + phi_AA(state["AA"], c)
        + c.c1 * phi_energy(state["E_self"], state["E_adv"])
        + r_proximity(state["Proximity"])
    )
    return R_goal + shaping + regularization(state, action, c)


def R_energy(state: dict, action: list[float], R_goal: float, c: Constants = C) -> float:
    shaping = (
        c.c0 * phi_ATA(state["ATA"], c) * phi_range(state["Range"], c) * phi_energy(state["E_self"], state["E_adv"])
        + c.c1 * phi_AA(state["AA"], c)
        + r_proximity(state["Proximity"])
    )
    return R_goal + shaping + regularization(state, action, c)