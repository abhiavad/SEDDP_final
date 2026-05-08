"""
Mass property utilities.

FINAL MASS PROPERTY PHILOSOPHY
------------------------------
- MATLAB-style explicit geometry is the authoritative geometry source.
- geometry_utils.py defines:
    * explicit panel geometry
    * explicit panel centers
    * explicit panel normals
    * explicit topology

IMPORTANT:
- Bus inertia is NOT derived from cuboid geometry.
- Bus inertia is hardcoded engineering data.
- Solar panels ARE modeled as thin rectangular plates.
- Total spacecraft MOI depends on panel deployment angle.

This file computes:
- total spacecraft mass
- total spacecraft CG
- total spacecraft inertia tensor

FRAME CONVENTION
----------------
All quantities are expressed in spacecraft body frame B.

RETURNS
-------
compute_mass_properties():

- total_mass_kg
- I_Bc_B_kgm2
- cg_B_m
"""

from typing import Tuple

import numpy as np

from simulation_config import (
    BUS_CG_B_M,
    BUS_INERTIA_B_KGM2,
    BUS_MASS_KG,
    PANEL_MASS_KG,
    PANEL_THICKNESS_M,
)

from geometry_utils import (

    # ======================================================
    # PANEL GEOMETRY
    # ======================================================

    panel_plus_x,
    panel_plus_y,

    panel_minus_x,
    panel_minus_y,

    # ======================================================
    # PANEL CENTERS
    # ======================================================

    center_panel_plus_x,
    center_panel_plus_y,

    center_panel_minus_x,
    center_panel_minus_y,

    # ======================================================
    # PANEL NORMALS
    # ======================================================

    normal_front_panel_plus_x,
    normal_front_panel_plus_y,

    normal_front_panel_minus_x,
    normal_front_panel_minus_y,
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Convert vector into unit vector.
    """
    v = np.asarray(v, dtype=float)

    norm_v = np.linalg.norm(v)

    if norm_v <= 1e-12:
        raise ValueError("Zero vector cannot be normalized")

    return v / norm_v


def compute_panel_length(panel: np.ndarray) -> float:
    """
    Panel deployment direction length.

    MATLAB topology:
    point_1 -> point_2 equivalent edge.
    """
    return np.linalg.norm(panel[1] - panel[0])


def compute_panel_width(panel: np.ndarray) -> float:
    """
    Panel attachment edge width.
    """
    return np.linalg.norm(panel[3] - panel[0])


def compute_panel_local_frame(
    panel: np.ndarray,
    normal_B: np.ndarray,
) -> np.ndarray:
    """
    Construct orthonormal panel frame.

    Local frame:
    - x: panel deployment/span direction
    - y: panel width direction
    - z: panel normal
    """

    x_B = unit_vector(panel[1] - panel[0])

    z_B = unit_vector(normal_B)

    y_B = np.cross(z_B, x_B)
    y_B = unit_vector(y_B)

    # Re-orthogonalize x

    x_B = np.cross(y_B, z_B)
    x_B = unit_vector(x_B)

    return np.column_stack((x_B, y_B, z_B))


def thin_plate_inertia_local(
    mass_kg: float,
    length_m: float,
    width_m: float,
    thickness_m: float,
) -> np.ndarray:
    """
    Thin rectangular plate inertia tensor
    about its own center in local frame.

    Local axes:
    - x : deployment/span direction
    - y : panel width direction
    - z : panel normal
    """

    Ixx = (mass_kg / 12.0) * (width_m**2 + thickness_m**2)

    Iyy = (mass_kg / 12.0) * (length_m**2 + thickness_m**2)

    Izz = (mass_kg / 12.0) * (length_m**2 + width_m**2)

    return np.array([
        [Ixx, 0.0, 0.0],
        [0.0, Iyy, 0.0],
        [0.0, 0.0, Izz],
    ])


def parallel_axis_shift(
    inertia_cg: np.ndarray,
    mass_kg: float,
    r_B_m: np.ndarray,
) -> np.ndarray:
    """
    Shift inertia tensor using parallel-axis theorem.
    """

    r_B_m = np.asarray(r_B_m, dtype=float)

    r2 = float(r_B_m @ r_B_m)

    return (
        inertia_cg
        +
        mass_kg
        *
        (
            r2 * np.eye(3)
            -
            np.outer(r_B_m, r_B_m)
        )
    )


# ==========================================================
# BUS MASS PROPERTIES
# ==========================================================

bus_mass_kg = float(BUS_MASS_KG)

bus_cg_B_m = np.asarray(BUS_CG_B_M, dtype=float)

bus_inertia_B_kgm2 = np.asarray(
    BUS_INERTIA_B_KGM2,
    dtype=float,
)

# ==========================================================
# PANEL LENGTHS
# ==========================================================

panel_plus_x_length_m = compute_panel_length(panel_plus_x)
panel_plus_y_length_m = compute_panel_length(panel_plus_y)

panel_minus_x_length_m = compute_panel_length(panel_minus_x)
panel_minus_y_length_m = compute_panel_length(panel_minus_y)

# ==========================================================
# PANEL WIDTHS
# ==========================================================

panel_plus_x_width_m = compute_panel_width(panel_plus_x)
panel_plus_y_width_m = compute_panel_width(panel_plus_y)

panel_minus_x_width_m = compute_panel_width(panel_minus_x)
panel_minus_y_width_m = compute_panel_width(panel_minus_y)

# ==========================================================
# PANEL LOCAL FRAMES
# ==========================================================

R_panel_plus_x = compute_panel_local_frame(
    panel_plus_x,
    normal_front_panel_plus_x,
)

R_panel_plus_y = compute_panel_local_frame(
    panel_plus_y,
    normal_front_panel_plus_y,
)

R_panel_minus_x = compute_panel_local_frame(
    panel_minus_x,
    normal_front_panel_minus_x,
)

R_panel_minus_y = compute_panel_local_frame(
    panel_minus_y,
    normal_front_panel_minus_y,
)

# ==========================================================
# PANEL LOCAL INERTIAS
# ==========================================================

I_panel_plus_x_local = thin_plate_inertia_local(
    PANEL_MASS_KG,
    panel_plus_x_length_m,
    panel_plus_x_width_m,
    PANEL_THICKNESS_M,
)

I_panel_plus_y_local = thin_plate_inertia_local(
    PANEL_MASS_KG,
    panel_plus_y_length_m,
    panel_plus_y_width_m,
    PANEL_THICKNESS_M,
)

I_panel_minus_x_local = thin_plate_inertia_local(
    PANEL_MASS_KG,
    panel_minus_x_length_m,
    panel_minus_x_width_m,
    PANEL_THICKNESS_M,
)

I_panel_minus_y_local = thin_plate_inertia_local(
    PANEL_MASS_KG,
    panel_minus_y_length_m,
    panel_minus_y_width_m,
    PANEL_THICKNESS_M,
)

# ==========================================================
# PANEL BODY-FRAME INERTIAS
# ==========================================================

I_panel_plus_x_B = (
    R_panel_plus_x
    @
    I_panel_plus_x_local
    @
    R_panel_plus_x.T
)

I_panel_plus_y_B = (
    R_panel_plus_y
    @
    I_panel_plus_y_local
    @
    R_panel_plus_y.T
)

I_panel_minus_x_B = (
    R_panel_minus_x
    @
    I_panel_minus_x_local
    @
    R_panel_minus_x.T
)

I_panel_minus_y_B = (
    R_panel_minus_y
    @
    I_panel_minus_y_local
    @
    R_panel_minus_y.T
)

# ==========================================================
# TOTAL SPACECRAFT CG
# ==========================================================

total_mass_kg = (
    BUS_MASS_KG
    +
    4.0 * PANEL_MASS_KG
)

first_moment = (

    BUS_MASS_KG * bus_cg_B_m

    +

    PANEL_MASS_KG * center_panel_plus_x

    +

    PANEL_MASS_KG * center_panel_plus_y

    +

    PANEL_MASS_KG * center_panel_minus_x

    +

    PANEL_MASS_KG * center_panel_minus_y
)

cg_B_m = first_moment / total_mass_kg

# ==========================================================
# SHIFTED PANEL INERTIAS
# ==========================================================

r_panel_plus_x = center_panel_plus_x - cg_B_m

r_panel_plus_y = center_panel_plus_y - cg_B_m

r_panel_minus_x = center_panel_minus_x - cg_B_m

r_panel_minus_y = center_panel_minus_y - cg_B_m

I_panel_plus_x_shifted = parallel_axis_shift(
    I_panel_plus_x_B,
    PANEL_MASS_KG,
    r_panel_plus_x,
)

I_panel_plus_y_shifted = parallel_axis_shift(
    I_panel_plus_y_B,
    PANEL_MASS_KG,
    r_panel_plus_y,
)

I_panel_minus_x_shifted = parallel_axis_shift(
    I_panel_minus_x_B,
    PANEL_MASS_KG,
    r_panel_minus_x,
)

I_panel_minus_y_shifted = parallel_axis_shift(
    I_panel_minus_y_B,
    PANEL_MASS_KG,
    r_panel_minus_y,
)

# ==========================================================
# SHIFT BUS INERTIA TO TOTAL CG
# ==========================================================

r_bus = bus_cg_B_m - cg_B_m

I_bus_shifted = parallel_axis_shift(
    bus_inertia_B_kgm2,
    BUS_MASS_KG,
    r_bus,
)

# ==========================================================
# TOTAL SPACECRAFT INERTIA
# ==========================================================

I_Bc_B_kgm2 = (

    I_bus_shifted

    +

    I_panel_plus_x_shifted

    +

    I_panel_plus_y_shifted

    +

    I_panel_minus_x_shifted

    +

    I_panel_minus_y_shifted
)

# ==========================================================
# VALIDATION
# ==========================================================

if not np.all(np.isfinite(cg_B_m)):
    raise ValueError("Computed CG contains invalid values")

if not np.all(np.isfinite(I_Bc_B_kgm2)):
    raise ValueError("Computed inertia contains invalid values")

if not np.allclose(
    I_Bc_B_kgm2,
    I_Bc_B_kgm2.T,
    atol=1e-12,
):
    raise ValueError("Inertia matrix is not symmetric")

# ==========================================================
# MAIN API
# ==========================================================


def compute_mass_properties() -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    Returns:
    - total_mass_kg
    - I_Bc_B_kgm2
    - cg_B_m
    """

    return (
        float(total_mass_kg),
        np.asarray(I_Bc_B_kgm2, dtype=float),
        np.asarray(cg_B_m, dtype=float),
    )


# ==========================================================
# EXPORTS
# ==========================================================

__all__ = [

    "compute_mass_properties",

    "total_mass_kg",

    "cg_B_m",

    "I_Bc_B_kgm2",
]