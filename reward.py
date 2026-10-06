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
    # Potential scales (not given in the paper). They must give a usable gradient over the
    # whole engagement, not just inside the WEZ: Table 1 starts at 1900-5100 m and any AA/ATA.
    # With the earlier 400 m / 30 deg, c0*phi_ATA*phi_AA*phi_range was 0.003 when pointing at
    # the bandit from 3500 m vs 2.2 in the WEZ -- flat everywhere the agent actually starts,
    # and the trained policy learned to point (ATA ~0) but never closed into the WEZ.
    c_r: float = 2000.0       # range potential scale (m) -- assumed; phi_range(3500 m) = 0.22
    c_AA: float = 60.0        # deg, AA potential scale -- assumed; phi_AA(180) = 0.05
    c_ATA: float = 60.0       # deg, ATA potential scale -- assumed; phi_ATA(90) = 0.22
    altitude_deck: float = 500.0   # m, hard deck -- assumed
    V_corner: float = 120.0        # m/s, ~1.4-1.5x stall speed (Shaw p.408) -- assumed
    c_VIAS: float = 30.0           # m/s, penalty scale -- assumed
    alpha_stall: float = 25.0      # deg, mid of Shaw's 20-30deg usable AoA -- assumed
    beta_max: float = 20.0         # deg, sideslip limit -- assumed
    c_beta: float = 20.0       # deg, sideslip penalty scale -- assumed. Keep c_beta == beta_max:
                               # Eq.(36) is -(beta/c_beta)^2 inside the limit and -1 outside it, so
                               # any c_beta < beta_max makes the penalty jump UP (less negative) at the limit.
    # Ground safety (NOT in paper -- our addition), modelled on the F-16's Auto-GCAS: predict the
    # height a recovery would cost (react/roll upright, then pull n_pullout g) and compare it with
    # the height available above the deck. rho = needed / available; 1 = cannot recover.
    n_pullout: float = 5.0     # g, pull-out load factor; JSBSim F-16 gives ~4.8 g at 300 KCAS, more faster
    t_react: float = 1.5       # s, roll-upright + reaction delay, flown at the current sink rate
    rho_safe: float = 0.2      # rho at or below this: fully safe (gate 1, no P_ground)
    rho_unsafe: float = 0.6    # rho at or above this: gate 0, P_ground = -k_ground. On a logged fatal
                               # 230 m/s pursuit dive this fades the gate out over the ~9 s before the
                               # point of no return (rho = 1), leaving the agent time to react.
    k_ground: float = 3.0      # P_ground weight; 0.0 disables it (paper-faithful)
    gate_shaping: bool = True  # gate positive shaping by ground safety; False = paper-faithful ungated
    k_negg: float = 1.0        # P_negg weight (penalty capped at -k_negg); 0.0 disables -- NOT in paper


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


G0 = 9.81


def recovery_height(v_true: float, v_up: float, c: Constants = C) -> float:
    """Height (m) an Auto-GCAS-style recovery from the current dive would cost: fly
    t_react s at the current sink rate (roll upright), then a constant n_pullout g
    pull-up through the dive angle gamma: R * (1 - cos gamma), R = V^2 / (g (n - 1)).
    0 when level or climbing. Grows with V^2 and with dive angle -- which a plain
    time-to-impact measure misses, so it fires too late in fast, steep dives."""
    if v_up >= 0.0 or v_true <= 1.0:
        return 0.0
    sin_g = max(-1.0, v_up / v_true)
    radius = v_true ** 2 / (G0 * (c.n_pullout - 1.0))
    return radius * (1.0 - math.sqrt(1.0 - sin_g ** 2)) - c.t_react * v_up


def ground_safety(state: dict, c: Constants = C) -> float:
    """1 = safe .. 0 = unrecoverable, from rho = recovery_height / height above the deck:
    1 at rho <= rho_safe, 0 at rho >= rho_unsafe, linear between; 0 at or below the deck.
    Level flight is always safe (rho = 0) at any altitude above the deck."""
    avail = state["altitude"] - c.altitude_deck
    if avail <= 0.0:
        return 0.0
    rho = recovery_height(state.get("V_true", 0.0), state.get("v_up", 0.0), c) / avail
    return min(1.0, max(0.0, (c.rho_unsafe - rho) / (c.rho_unsafe - c.rho_safe)))


def P_ground(state: dict, c: Constants = C) -> float:
    """Anticipatory ground avoidance -- our addition, not in the paper. P_deck only fires
    once the aircraft is already below the deck: in a 300 m/s dive that is about a second
    before impact, past the point of no return and too late for gamma=0.99 to credit the
    manoeuvre that started the dive. This ramps from 0 to -k_ground as a recovery from the
    current dive starts to need most of the height left."""
    if c.k_ground == 0.0:
        return 0.0
    return -c.k_ground * (1.0 - ground_safety(state, c))


def P_negg(nz: float, c: Constants = C) -> float:
    """Negative-g penalty -- our addition, not in the paper. 0 for nz >= 0, ramping to
    -k_negg at -1 g and capped there. Policies settled into flying INVERTED, holding
    altitude on push (about -1.15 g) with full rudder: stable and survivable, but an
    F-16 pushes only ~-3 g against ~+9 g pulling, so it can barely turn its nose and
    never builds a gun solution (2 min behind a straight-and-level target without
    one). Manoeuvres that pass through inverted -- split-S, barrel roll -- are pulled
    at positive g and are not penalised; only sustained push-flight is."""
    return -c.k_negg * min(1.0, max(0.0, -nz))


def safety_gate(state: dict, c: Constants = C) -> float:
    """Multiplier in [0, 1] on POSITIVE shaping -- our addition, not in the paper.
    Safety must outrank offence, not just compete with it: once the potentials pay
    for closing on the bandit everywhere, following a diving bandit earns more than
    any bounded ground penalty costs, and the agent rides the chase into the ground.
    Gating by ground_safety means that as a dive becomes hard to recover from, only
    recovering can pay."""
    return ground_safety(state, c) if c.gate_shaping else 1.0


def _gated(shaping: float, state: dict, c: Constants) -> float:
    if shaping <= 0.0:
        return shaping   # never soften a negative shaping term
    return shaping * safety_gate(state, c)


def regularization(state: dict, action: list[float], c: Constants = C) -> float:
    return (
        P_deck(state["altitude"], c)
        + P_VIAS(state["VIAS"], c)
        + P_alpha_AoA(state["alpha_AoA"], c)
        + P_beta(state["beta"], c)
        + C_action(action)
        + P_ground(state, c)
        + P_negg(state.get("Nz", 1.0), c)
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
    return R_goal + _gated(shaping, state, c) + regularization(state, action, c)


def R_snapshot(state: dict, action: list[float], R_goal: float, c: Constants = C) -> float:
    shaping = (
        c.c0 * phi_ATA(state["ATA"], c) * phi_range(state["Range"], c)
        + phi_AA(state["AA"], c)
        + c.c1 * phi_energy(state["E_self"], state["E_adv"])
        + r_proximity(state["Proximity"])
    )
    return R_goal + _gated(shaping, state, c) + regularization(state, action, c)


def R_energy(state: dict, action: list[float], R_goal: float, c: Constants = C) -> float:
    shaping = (
        c.c0 * phi_ATA(state["ATA"], c) * phi_range(state["Range"], c) * phi_energy(state["E_self"], state["E_adv"])
        + c.c1 * phi_AA(state["AA"], c)
        + r_proximity(state["Proximity"])
    )
    return R_goal + _gated(shaping, state, c) + regularization(state, action, c)