"""
Shared magnetic-control helper utilities.

IMPORTANT:
This file intentionally contains ONLY shared low-level helpers
during Phase A architecture preparation.

No nadir steering logic should be added yet.
"""

import numpy as np
from Basilisk.architecture import messaging


def normalize_safe(v, eps=1e-12):
    """
    Safely normalize a vector.

    Returns zero vector if norm is too small.
    """
    arr = np.asarray(v, dtype=float).reshape(3)

    norm = np.linalg.norm(arr)

    if norm < eps or not np.isfinite(norm):
        return np.zeros_like(arr,dtype=float)

    return arr / norm


def write_zero_dipole(out_msg, current_sim_nanos):
    """
    Write zero dipole command.

    Shared helper to avoid duplicated payload logic.
    """
    payload = messaging.DipoleRequestBodyMsgPayload()
    payload.dipole_B = [0.0, 0.0, 0.0]

    out_msg.write(payload, current_sim_nanos)


def saturate_dipole_uniform(m, max_dipole):
    """
    Uniform per-axis dipole saturation.

    Preserves dipole direction while scaling uniformly
    if any axis exceeds its limit.

    IMPORTANT:
    This preserves the exact behavior previously used
    inside BdotController.
    """
    m = np.asarray(m, dtype=float).reshape(3)

    max_dipole = np.asarray(max_dipole, dtype=float).reshape(3)

    if (
        not np.all(np.isfinite(m))
        or not np.all(np.isfinite(max_dipole))
    ):
        return np.zeros(3, dtype=float)

    ratios = np.divide(
        np.abs(m),
        max_dipole,
        out=np.full_like(m, np.inf),
        where=max_dipole > 0.0
    )

    max_ratio = np.max(ratios)

    if max_ratio > 1.0:
        m = m / max_ratio

    if not np.all(np.isfinite(m)):
        return np.zeros(3, dtype=float)

    return m