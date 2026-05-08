"""
Simulation-side configuration for the ADCS/Basilisk model.

FINAL GEOMETRY / MASS PROPERTY ARCHITECTURE

PHILOSOPHY
----------
- MATLAB geometry model is the authoritative geometry/topology source.
- Geometry is explicitly constructed in geometry_utils.py.
- This file stores ONLY:
    * configuration parameters
    * dimensions
    * masses
    * deployment angle
    * aerodynamic coefficients
    * hardcoded spacecraft bus inertia

IMPORTANT
---------
- Bus inertia is NOT derived from cuboid geometry.
- Spacecraft bus is not assumed to be a uniform solid block.
- Solar panels ARE modeled as thin rectangular plates.
- Panel MOI and total spacecraft MOI depend on deployment angle.

FRAME CONVENTION
----------------
- All geometry is defined in spacecraft body frame B.
- Origin is spacecraft reference point.
- Total spacecraft CG is computed in mass_properties.py.
"""

# ==========================================================
# TIMING
# ==========================================================

SIMULATION_TIME_S = 86400*3

DYN_DT_S = 0.05

# ==========================================================
# ORBIT
# ==========================================================

ORBIT_ELEMENTS = {
    "a": 6678e3,            # [m]
    "e": 0.0,               # [-]
    "i_deg": 97.7882,       # [deg]
    "Omega_deg": 48.2,      # [deg]
    "omega_deg": 347.8,     # [deg]
    "f_deg": 85.3           # [deg]
}

# ==========================================================
# ENVIRONMENT
# ==========================================================

ATMOSPHERE_PLANET_RADIUS_M = 6378137.0

ATMOSPHERE_BASE_DENSITY_KG_M3 = 4.39e-11

ATMOSPHERE_SCALE_HEIGHT_M = 300000.0

ATMOSPHERE_ENV_MIN_REACH_M = -200000.0

ATMOSPHERE_ENV_MAX_REACH_M = 1000000.0

MAG_FIELD_EPOCH = "2019 June 27, 10:23:0.0 (UTC)"

# ==========================================================
# INITIAL CONDITIONS / DISTURBANCE TRUTH
# ==========================================================

INITIAL_SIGMA_BN = [0.0, 0.0, 0.0]

# [rad/s]
INITIAL_OMEGA_BN_B_RADPS = [3.1415, 3.1415, 3.1415]

# [A*m^2]
RESIDUAL_DIPOLE_B_AM2 = [0.001, 0.001, 0.001]

# ==========================================================
# SPACECRAFT GEOMETRY PARAMETERS
# ==========================================================

"""
These values define the canonical spacecraft geometry.

IMPORTANT:
- geometry_utils.py constructs the explicit geometry
  directly from these parameters.
- This mirrors the MATLAB geometry model.

MATLAB REFERENCE:

lx = 0.05
ly = 0.05
lz = 0.15

lp = 0.15
"""

# ----------------------------------------------------------
# SPACECRAFT BODY DIMENSIONS [m]
# ----------------------------------------------------------

LX_M = 0.05

LY_M = 0.05

LZ_M = 0.17

# ----------------------------------------------------------
# SOLAR PANEL LENGTH [m]
# ----------------------------------------------------------

LP_M = 0.15

# ----------------------------------------------------------
# PANEL DEPLOYMENT ANGLE [deg]
# ----------------------------------------------------------

THETA_PANEL_DEG = 135.0

# ==========================================================
# SPACECRAFT MASS PROPERTIES
# ==========================================================

"""
IMPORTANT:
- Bus inertia tensor is hardcoded.
- It is NOT derived from cuboid geometry.
- These values should eventually come from:
    * CAD
    * engineering estimates
    * measured inertia
"""

# ----------------------------------------------------------
# BUS
# ----------------------------------------------------------

BUS_MASS_KG = 0.55

BUS_CD = 2.2

"""
Hardcoded spacecraft bus inertia tensor about
the BUS CG in body frame B.

UNITS:
[kg*m^2]
"""

BUS_INERTIA_B_KGM2 = [
    [0.00161, 0.0,    0.0],
    [0.0,    0.00161, 0.0],
    [0.0,    0.0,    0.000297],
]

"""
Bus-only CG location in body frame B.

IMPORTANT:
- This is NOT total spacecraft CG.
- Total spacecraft CG is computed in:
    mass_properties.py

using:
- bus CG
- bus mass
- panel masses
- deployment-angle-dependent panel locations
"""

BUS_CG_B_M = [0.0, 0.0, 0.025]

# ----------------------------------------------------------
# SOLAR PANELS
# ----------------------------------------------------------

PANEL_MASS_KG = 0.05

PANEL_CD = 2.2

"""
Approximate panel thickness.

Used conceptually for:
- structural interpretation
- future higher-fidelity models

Current inertia model assumes:
- thin rectangular plates
"""

PANEL_THICKNESS_M = 0.002

# ==========================================================
# EXPORT
# ==========================================================

__all__ = [

    # ======================================================
    # TIMING
    # ======================================================

    "SIMULATION_TIME_S",

    "DYN_DT_S",

    # ======================================================
    # ORBIT
    # ======================================================

    "ORBIT_ELEMENTS",

    # ======================================================
    # ENVIRONMENT
    # ======================================================

    "ATMOSPHERE_PLANET_RADIUS_M",

    "ATMOSPHERE_BASE_DENSITY_KG_M3",

    "ATMOSPHERE_SCALE_HEIGHT_M",

    "ATMOSPHERE_ENV_MIN_REACH_M",

    "ATMOSPHERE_ENV_MAX_REACH_M",

    "MAG_FIELD_EPOCH",

    # ======================================================
    # INITIAL CONDITIONS
    # ======================================================

    "INITIAL_SIGMA_BN",

    "INITIAL_OMEGA_BN_B_RADPS",

    "RESIDUAL_DIPOLE_B_AM2",

    # ======================================================
    # GEOMETRY PARAMETERS
    # ======================================================

    "LX_M",

    "LY_M",

    "LZ_M",

    "LP_M",

    "THETA_PANEL_DEG",

    # ======================================================
    # BUS
    # ======================================================

    "BUS_MASS_KG",

    "BUS_CD",

    "BUS_CG_B_M",

    "BUS_INERTIA_B_KGM2",

    # ======================================================
    # PANELS
    # ======================================================

    "PANEL_MASS_KG",

    "PANEL_CD",

    "PANEL_THICKNESS_M",
]