"""Air-combat geometry, Eq. (8) of Li et al., Drones 2025, 9, 384."""

import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def body_x_enu(psi_rad, theta_rad):
    """Unit body-X (nose) vector e in (east, north, up), from yaw psi and pitch
    theta. Roll does not move the X axis. This is the e of Eq.(8)."""
    ct = np.cos(theta_rad)
    return np.array([ct * np.sin(psi_rad), ct * np.cos(psi_rad), np.sin(theta_rad)])


def body_axes_enu(phi_rad, theta_rad, psi_rad):
    """Body X (nose), Y (right wing) and Z (floor, body-down) unit vectors in
    (east, north, up). Same DCM as f16_bfm_agent.AcState, re-expressed from NED to
    ENU. Projecting a world vector onto these gives it in the aircraft's own frame."""
    cp, sp = np.cos(phi_rad), np.sin(phi_rad)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    cy, sy = np.cos(psi_rad), np.sin(psi_rad)
    xb = np.array([ct * sy, ct * cy, st])
    yb = np.array([sp * st * sy + cp * cy, sp * st * cy - cp * sy, -sp * ct])
    zb = np.array([cp * st * sy - sp * cy, cp * st * cy + sp * sy, -cp * ct])
    return xb, yb, zb


def heading_vector_enu(psi_rad):
    """Horizontal heading (pitch = 0), (east, north, up). Only for initialisation,
    where both aircraft start level."""
    return body_x_enu(psi_rad, 0.0)


heading_vector_ned = heading_vector_enu   # old name kept so existing imports still work


def combat_geometry(p_self, e_self, p_adv, e_adv):
    """p_*: position (east, north, up) in m. e_*: unit body-x (heading) vector.
    Returns dict with LOS, Range, ATA, AA (all self->adv convention, Eq.8)."""
    LOS = p_adv - p_self
    rng = np.linalg.norm(LOS)
    los_hat = unit(LOS)
    ata = np.degrees(np.arccos(np.clip(np.dot(e_self, los_hat), -1.0, 1.0)))
    aa = np.degrees(np.arccos(np.clip(np.dot(e_adv, los_hat), -1.0, 1.0)))
    hca = np.degrees(np.arccos(np.clip(np.dot(e_adv, e_self), -1.0, 1.0)))
    return {"Range": rng, "ATA": ata, "AA": aa, "HCA": hca, "LOS": LOS, "los_hat": los_hat}


def proximity(p_self, v_self, p_adv, v_adv):
    """Closure rate: (v_adv - v_self) . LOS_hat. Negative = closing."""
    los_hat = unit(p_adv - p_self)
    return float(np.dot(v_adv - v_self, los_hat))


def reset_geometry(range_m, AA_deg, ATA_deg, alt_m):
    """Place self at origin heading north; derive adversary position/heading
    consistent with Eq.(8) sign convention. Horizontal-plane initialization."""
    psi_self = 0.0
    e_self = heading_vector_enu(psi_self)
    ata = np.radians(ATA_deg)
    # rotate self heading by ATA to get LOS (self->adv) direction
    los_hat = np.array([
        e_self[0] * np.cos(ata) + e_self[1] * np.sin(ata),
        -e_self[0] * np.sin(ata) + e_self[1] * np.cos(ata),
        0.0,
    ])
    p_self = np.array([0.0, 0.0, alt_m])
    p_adv = p_self + range_m * los_hat
    aa = np.radians(AA_deg)
    e_adv = np.array([
        los_hat[0] * np.cos(aa) + los_hat[1] * np.sin(aa),
        -los_hat[0] * np.sin(aa) + los_hat[1] * np.cos(aa),
        0.0,
    ])
    psi_adv = np.degrees(np.arctan2(e_adv[0], e_adv[1]))
    return p_self, np.degrees(psi_self), p_adv, psi_adv