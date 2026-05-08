"""
Explicit spacecraft geometry utilities.

FINAL GEOMETRY PHILOSOPHY
-------------------------
This file is the Python implementation of the trusted MATLAB
spacecraft geometry model.

IMPORTANT:
- Geometry is EXPLICITLY defined.
- Points, faces, panels, normals, centers, and vectors
  are all explicitly constructed.
- This intentionally mirrors the MATLAB structure for:
    * traceability
    * debugging
    * torque validation
    * MOI validation
    * deployment-angle reasoning

BASILISK INTEGRATION
--------------------
At the END of this file:
- Explicit geometry is converted into Basilisk-compatible
  aerodynamic facets.

This preserves:
- existing Basilisk disturbance architecture
- existing AeroTorqueCalculator structure
- existing addFacet() workflow

GEOMETRY SOURCE
---------------
All geometry dimensions/constants come from:

simulation_config.py

REFERENCE
---------
This file mirrors the MATLAB geometry model closely.
"""

from typing import Dict
from typing import List

import numpy as np

from simulation_config import (
    BUS_CD,
    BUS_CG_B_M,
    LX_M,
    LY_M,
    LZ_M,
    LP_M,
    PANEL_CD,
    PANEL_MASS_KG,
    THETA_PANEL_DEG,
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


def compute_center(surface: np.ndarray) -> np.ndarray:
    """
    Compute geometric center of rectangular surface.
    """
    return np.mean(surface, axis=0)


def compute_normal(surface: np.ndarray) -> np.ndarray:
    """
    Compute rectangle normal using symmetric diagonal cross product.

    MATLAB-equivalent:
    cross(
        surface(3,:) - surface(1,:),
        surface(4,:) - surface(2,:)
    )
    """
    return unit_vector(
        np.cross(
            surface[2] - surface[0],
            surface[3] - surface[1]
        )
    )


def compute_rectangle_area(surface: np.ndarray) -> float:
    """
    Compute rectangle area from adjacent edges.
    """
    edge_1 = surface[1] - surface[0]

    edge_2 = surface[3] - surface[0]

    return np.linalg.norm(
        np.cross(edge_1, edge_2)
    )


# ==========================================================
# GEOMETRY PARAMETERS
# ==========================================================

lx = LX_M
ly = LY_M
lz = LZ_M
lp = LP_M
theta_deg = THETA_PANEL_DEG

theta_rad = np.deg2rad(theta_deg)
panel_z = -lz/2 - lp*np.cos(np.deg2rad(180-theta_deg))

# ==========================================================
# COG VECTORS
# ==========================================================

"""
NOTE:
These vectors are primarily for visualization/debugging.

The physically authoritative spacecraft CG used for:
- torque calculations
- aerodynamic disturbance calculations
- inertia calculations

comes from:
mass_properties.py
"""

cog = np.asarray(BUS_CG_B_M, dtype=float)

# ==========================================================
# CORNER POINTS
# ==========================================================

point_1 = np.array([ lx/2, -ly/2,  lz/2 ])
point_2 = np.array([ lx/2,  ly/2,  lz/2 ])
point_3 = np.array([-lx/2,  ly/2,  lz/2 ])
point_4 = np.array([-lx/2, -ly/2,  lz/2 ])

point_5 = np.array([ lx/2, -ly/2, -lz/2 ])
point_6 = np.array([ lx/2,  ly/2, -lz/2 ])
point_7 = np.array([-lx/2,  ly/2, -lz/2 ])
point_8 = np.array([-lx/2, -ly/2, -lz/2 ])

point_9 = np.array([
    lx/2 + lp*np.sin(theta_rad),
    -ly/2,
    panel_z
])

point_10 = np.array([
    lx/2 + lp*np.sin(theta_rad),
    ly/2,
    panel_z
])

point_11 = np.array([
    lx/2,
    ly/2 + lp*np.sin(theta_rad),
    panel_z
])

point_12 = np.array([
    -lx/2,
    ly/2 + lp*np.sin(theta_rad),
    panel_z
])

point_13 = np.array([
    -lx/2 - lp*np.sin(theta_rad),
    ly/2,
    panel_z
])

point_14 = np.array([
    -lx/2 - lp*np.sin(theta_rad),
    -ly/2,
    panel_z
])

point_15 = np.array([
    -lx/2,
    -ly/2 - lp*np.sin(theta_rad),
    panel_z
])

point_16 = np.array([
    lx/2,
    -ly/2 - lp*np.sin(theta_rad),
    panel_z
])

# ==========================================================
# BODY FACES
# ==========================================================

face_plus_z = np.array([
    point_1,
    point_2,
    point_3,
    point_4
])

face_minus_z = np.array([
    point_5,
    point_8,
    point_7,
    point_6
])

face_plus_x = np.array([
    point_1,
    point_5,
    point_6,
    point_2
])

face_minus_x = np.array([
    point_3,
    point_7,
    point_8,
    point_4
])

face_plus_y = np.array([
    point_3,
    point_2,
    point_6,
    point_7
])

face_minus_y = np.array([
    point_1,
    point_4,
    point_8,
    point_5
])

# ==========================================================
# PANELS
# ==========================================================

panel_plus_x = np.array([
    point_5,
    point_9,
    point_10,
    point_6
])

panel_plus_y = np.array([
    point_6,
    point_11,
    point_12,
    point_7
])

panel_minus_x = np.array([
    point_7,
    point_13,
    point_14,
    point_8
])

panel_minus_y = np.array([
    point_8,
    point_15,
    point_16,
    point_5
])

# ==========================================================
# FACE CENTERS
# ==========================================================

center_face_plus_z = compute_center(face_plus_z)
center_face_minus_z = compute_center(face_minus_z)

center_face_plus_x = compute_center(face_plus_x)
center_face_minus_x = compute_center(face_minus_x)

center_face_plus_y = compute_center(face_plus_y)
center_face_minus_y = compute_center(face_minus_y)

# ==========================================================
# PANEL CENTERS
# ==========================================================

center_panel_plus_x = compute_center(panel_plus_x)
center_panel_plus_y = compute_center(panel_plus_y)

center_panel_minus_x = compute_center(panel_minus_x)
center_panel_minus_y = compute_center(panel_minus_y)

# ==========================================================
# BODY NORMALS
# ==========================================================

normal_face_plus_z = compute_normal(face_plus_z)
normal_face_minus_z = compute_normal(face_minus_z)

normal_face_plus_x = compute_normal(face_plus_x)
normal_face_minus_x = compute_normal(face_minus_x)

normal_face_plus_y = compute_normal(face_plus_y)
normal_face_minus_y = compute_normal(face_minus_y)

# ----------------------------------------------------------
# Enforce outward directions
# ----------------------------------------------------------

if np.dot(normal_face_plus_z, [0,0,1]) < 0:
    normal_face_plus_z *= -1

if np.dot(normal_face_minus_z, [0,0,-1]) < 0:
    normal_face_minus_z *= -1

if np.dot(normal_face_plus_x, [1,0,0]) < 0:
    normal_face_plus_x *= -1

if np.dot(normal_face_minus_x, [-1,0,0]) < 0:
    normal_face_minus_x *= -1

if np.dot(normal_face_plus_y, [0,1,0]) < 0:
    normal_face_plus_y *= -1

if np.dot(normal_face_minus_y, [0,-1,0]) < 0:
    normal_face_minus_y *= -1

# ==========================================================
# PANEL FRONT NORMALS
# ==========================================================

normal_front_panel_plus_x = compute_normal(panel_plus_x)
normal_front_panel_plus_y = compute_normal(panel_plus_y)

normal_front_panel_minus_x = compute_normal(panel_minus_x)
normal_front_panel_minus_y = compute_normal(panel_minus_y)

# ==========================================================
# PANEL BACK NORMALS
# ==========================================================

normal_back_panel_plus_x = -normal_front_panel_plus_x
normal_back_panel_plus_y = -normal_front_panel_plus_y

normal_back_panel_minus_x = -normal_front_panel_minus_x
normal_back_panel_minus_y = -normal_front_panel_minus_y

# ==========================================================
# AREAS
# ==========================================================

area_face_plus_z = compute_rectangle_area(face_plus_z)
area_face_minus_z = compute_rectangle_area(face_minus_z)

area_face_plus_x = compute_rectangle_area(face_plus_x)
area_face_minus_x = compute_rectangle_area(face_minus_x)

area_face_plus_y = compute_rectangle_area(face_plus_y)
area_face_minus_y = compute_rectangle_area(face_minus_y)

area_panel_plus_x = compute_rectangle_area(panel_plus_x)
area_panel_plus_y = compute_rectangle_area(panel_plus_y)

area_panel_minus_x = compute_rectangle_area(panel_minus_x)
area_panel_minus_y = compute_rectangle_area(panel_minus_y)

# ==========================================================
# COG VECTORS
# ==========================================================

vector_face_plus_z_to_cog = cog - center_face_plus_z
vector_face_minus_z_to_cog = cog - center_face_minus_z

vector_face_plus_x_to_cog = cog - center_face_plus_x
vector_face_minus_x_to_cog = cog - center_face_minus_x

vector_face_plus_y_to_cog = cog - center_face_plus_y
vector_face_minus_y_to_cog = cog - center_face_minus_y

vector_panel_plus_x_to_cog = cog - center_panel_plus_x
vector_panel_plus_y_to_cog = cog - center_panel_plus_y

vector_panel_minus_x_to_cog = cog - center_panel_minus_x
vector_panel_minus_y_to_cog = cog - center_panel_minus_y

# ==========================================================
# BASILISK FACET EXPORT
# ==========================================================

"""
These facet dictionaries preserve compatibility with:
- AeroTorqueCalculator
- Basilisk addFacet()
- existing disturbance architecture
"""

facet_face_plus_z = {
    "name": "face_plus_z",
    "area": area_face_plus_z,
    "Cd": BUS_CD,
    "normal": normal_face_plus_z,
    "location": center_face_plus_z,
}

facet_face_minus_z = {
    "name": "face_minus_z",
    "area": area_face_minus_z,
    "Cd": BUS_CD,
    "normal": normal_face_minus_z,
    "location": center_face_minus_z,
}

facet_face_plus_x = {
    "name": "face_plus_x",
    "area": area_face_plus_x,
    "Cd": BUS_CD,
    "normal": normal_face_plus_x,
    "location": center_face_plus_x,
}

facet_face_minus_x = {
    "name": "face_minus_x",
    "area": area_face_minus_x,
    "Cd": BUS_CD,
    "normal": normal_face_minus_x,
    "location": center_face_minus_x,
}

facet_face_plus_y = {
    "name": "face_plus_y",
    "area": area_face_plus_y,
    "Cd": BUS_CD,
    "normal": normal_face_plus_y,
    "location": center_face_plus_y,
}

facet_face_minus_y = {
    "name": "face_minus_y",
    "area": area_face_minus_y,
    "Cd": BUS_CD,
    "normal": normal_face_minus_y,
    "location": center_face_minus_y,
}

# ----------------------------------------------------------
# PANEL FRONT FACETS
# ----------------------------------------------------------

facet_front_panel_plus_x = {
    "name": "front_panel_plus_x",
    "area": area_panel_plus_x,
    "Cd": PANEL_CD,
    "normal": normal_front_panel_plus_x,
    "location": center_panel_plus_x,
    "mass_kg": PANEL_MASS_KG,
}

facet_front_panel_plus_y = {
    "name": "front_panel_plus_y",
    "area": area_panel_plus_y,
    "Cd": PANEL_CD,
    "normal": normal_front_panel_plus_y,
    "location": center_panel_plus_y,
    "mass_kg": PANEL_MASS_KG,
}

facet_front_panel_minus_x = {
    "name": "front_panel_minus_x",
    "area": area_panel_minus_x,
    "Cd": PANEL_CD,
    "normal": normal_front_panel_minus_x,
    "location": center_panel_minus_x,
    "mass_kg": PANEL_MASS_KG,
}

facet_front_panel_minus_y = {
    "name": "front_panel_minus_y",
    "area": area_panel_minus_y,
    "Cd": PANEL_CD,
    "normal": normal_front_panel_minus_y,
    "location": center_panel_minus_y,
    "mass_kg": PANEL_MASS_KG,
}

# ----------------------------------------------------------
# PANEL BACK FACETS
# ----------------------------------------------------------

facet_back_panel_plus_x = {
    "name": "back_panel_plus_x",
    "area": area_panel_plus_x,
    "Cd": PANEL_CD,
    "normal": normal_back_panel_plus_x,
    "location": center_panel_plus_x,
    "mass_kg": PANEL_MASS_KG,
}

facet_back_panel_plus_y = {
    "name": "back_panel_plus_y",
    "area": area_panel_plus_y,
    "Cd": PANEL_CD,
    "normal": normal_back_panel_plus_y,
    "location": center_panel_plus_y,
    "mass_kg": PANEL_MASS_KG,
}

facet_back_panel_minus_x = {
    "name": "back_panel_minus_x",
    "area": area_panel_minus_x,
    "Cd": PANEL_CD,
    "normal": normal_back_panel_minus_x,
    "location": center_panel_minus_x,
    "mass_kg": PANEL_MASS_KG,
}

facet_back_panel_minus_y = {
    "name": "back_panel_minus_y",
    "area": area_panel_minus_y,
    "Cd": PANEL_CD,
    "normal": normal_back_panel_minus_y,
    "location": center_panel_minus_y,
    "mass_kg": PANEL_MASS_KG,
}

# ==========================================================
# COMPLETE FACET LIST
# ==========================================================

ALL_AERO_FACETS: List[Dict] = [

    # ------------------------------------------------------
    # BUS
    # ------------------------------------------------------

    facet_face_plus_z,
    facet_face_minus_z,

    facet_face_plus_x,
    facet_face_minus_x,

    facet_face_plus_y,
    facet_face_minus_y,

    # ------------------------------------------------------
    # PANELS
    # ------------------------------------------------------

    facet_front_panel_plus_x,
    facet_back_panel_plus_x,

    facet_front_panel_plus_y,
    facet_back_panel_plus_y,

    facet_front_panel_minus_x,
    facet_back_panel_minus_x,

    facet_front_panel_minus_y,
    facet_back_panel_minus_y,
]

# ==========================================================
# EXPORTS
# ==========================================================

__all__ = [

    # ------------------------------------------------------
    # POINTS
    # ------------------------------------------------------

    "point_1",
    "point_2",
    "point_3",
    "point_4",

    "point_5",
    "point_6",
    "point_7",
    "point_8",

    "point_9",
    "point_10",
    "point_11",
    "point_12",

    "point_13",
    "point_14",
    "point_15",
    "point_16",

    # ------------------------------------------------------
    # FACES
    # ------------------------------------------------------

    "face_plus_z",
    "face_minus_z",

    "face_plus_x",
    "face_minus_x",

    "face_plus_y",
    "face_minus_y",

    # ------------------------------------------------------
    # PANELS
    # ------------------------------------------------------

    "panel_plus_x",
    "panel_plus_y",

    "panel_minus_x",
    "panel_minus_y",

    # ------------------------------------------------------
    # CENTERS
    # ------------------------------------------------------

    "center_face_plus_z",
    "center_face_minus_z",

    "center_face_plus_x",
    "center_face_minus_x",

    "center_face_plus_y",
    "center_face_minus_y",

    "center_panel_plus_x",
    "center_panel_plus_y",

    "center_panel_minus_x",
    "center_panel_minus_y",

    # ------------------------------------------------------
    # NORMALS
    # ------------------------------------------------------

    "normal_face_plus_z",
    "normal_face_minus_z",

    "normal_face_plus_x",
    "normal_face_minus_x",

    "normal_face_plus_y",
    "normal_face_minus_y",

    "normal_front_panel_plus_x",
    "normal_back_panel_plus_x",

    "normal_front_panel_plus_y",
    "normal_back_panel_plus_y",

    "normal_front_panel_minus_x",
    "normal_back_panel_minus_x",

    "normal_front_panel_minus_y",
    "normal_back_panel_minus_y",

    # ------------------------------------------------------
    # AREAS
    # ------------------------------------------------------

    "area_face_plus_z",
    "area_face_minus_z",

    "area_face_plus_x",
    "area_face_minus_x",

    "area_face_plus_y",
    "area_face_minus_y",

    "area_panel_plus_x",
    "area_panel_plus_y",

    "area_panel_minus_x",
    "area_panel_minus_y",

    # ------------------------------------------------------
    # FACETS
    # ------------------------------------------------------

    "ALL_AERO_FACETS",
]