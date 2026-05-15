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
import numpy as np

from fsw_config import (
    FSW_STEP_TIME_S,
    ACTIVE_CONTROLLER,
    BDOT_GAIN,
    BDOT_GAIN_NADIR,
    KP_NADIR,
)

from Basilisk.utilities import (
    orbitalMotion,
    RigidBodyKinematics as rbk,
)

# ==========================================================
# TIMING
# ==========================================================

SIMULATION_TIME_S = 0.5*86400

# ==========================================================
# Dynamics timestep
#
# Must be an integer subdivision of the FSW timestep.
# Using 2x faster dynamics updates ensures the TAM sensor
# updates multiple times during each FSW sensing interval,
# preventing repeated B samples and zero Bdot estimates.
# ==========================================================

DYN_DT_S = FSW_STEP_TIME_S / 2.0

# ==========================================================
# LOGGING
# ==========================================================

# Engineering-analysis logging cadence.
#
# 54 sec ≈ 1% of a 90-minute orbit.
#
# IMPORTANT:
# This affects ONLY:
# - CSV export
# - plots
# - logged recorder samples
#
# Dynamics and FSW still run at full resolution.
LOGGING_DT_S = 54

# ==========================================================
# ORBIT
# ==========================================================

ORBIT_ELEMENTS = {
    "a": 6678e3,            # [m] Semi-major axis - 6678e3 (300km), 6928e3 (550km), 6978e3 (600km)
    "e": 0.0,               # [-] Eccentricity
    "i_deg": 96.6725,       # [deg] Inclination - 96.6725 deg (300km), 97.67 deg (550km), 97.7882 deg (600km)
    "Omega_deg": 48.2,      # [deg] RAAN (Right Ascension of Ascending Node)
    "omega_deg": 347.8,     # [deg] Argument of Periapsis (AoP)
    "f_deg": 85.3           # [deg] True Anomaly
}

# ==========================================================
# ENVIRONMENT
# ==========================================================

ATMOSPHERE_PLANET_RADIUS_M = 6378137.0

ATMOSPHERE_BASE_DENSITY_KG_M3 = 4.39e-11  # 4.39e-11 (300km) , 1.05e-12 (550km) , 5.63e-13 (600km)

ATMOSPHERE_SCALE_HEIGHT_M = 300000.0 # 300000.0 (300km), 550000.0 (550km), 600000.0 (600km)

ATMOSPHERE_ENV_MIN_REACH_M = -200000.0

ATMOSPHERE_ENV_MAX_REACH_M = 1000000.0

MAG_FIELD_EPOCH = "2019 June 27, 10:23:0.0 (UTC)"

# ==========================================================
# INITIAL CONDITIONS / DISTURBANCE TRUTH
# ==========================================================

"""
Initial attitude construction.

Nominal reference frame:
+X_B -> nadir
+Z_B -> velocity

A body-frame perturbation rotation is then applied.
"""

# ----------------------------------------------------------
# BODY-FRAME PERTURBATION
# ----------------------------------------------------------

INITIAL_PERTURBATION_AXIS_B = np.array([
    0.0,
    0.0,
    1.0,
])

INITIAL_PERTURBATION_ANGLE_DEG = 180.0

# ----------------------------------------------------------
# ORBITAL STATE
# ----------------------------------------------------------

MU_EARTH = 3.986004418e14

oe = orbitalMotion.ClassicElements()

oe.a = ORBIT_ELEMENTS["a"]
oe.e = ORBIT_ELEMENTS["e"]

oe.i = np.deg2rad(
    ORBIT_ELEMENTS["i_deg"]
)

oe.Omega = np.deg2rad(
    ORBIT_ELEMENTS["Omega_deg"]
)

oe.omega = np.deg2rad(
    ORBIT_ELEMENTS["omega_deg"]
)

oe.f = np.deg2rad(
    ORBIT_ELEMENTS["f_deg"]
)

r_N, v_N = orbitalMotion.elem2rv(
    MU_EARTH,
    oe
)

r_N = np.array(r_N, dtype=float)

v_N = np.array(v_N, dtype=float)

if (
    not np.all(np.isfinite(r_N))
    or not np.all(np.isfinite(v_N))
):
    raise ValueError(
        "Invalid orbital state vectors."
    )

r_norm = np.linalg.norm(r_N)

v_norm = np.linalg.norm(v_N)

if r_norm < 1e-12:
    raise ValueError(
        "Orbital position vector is too small."
    )

if v_norm < 1e-12:
    raise ValueError(
        "Orbital velocity vector is too small."
    )

# ----------------------------------------------------------
# NOMINAL BODY FRAME
# ----------------------------------------------------------

# +X_B -> nadir
xB_N = -r_N / r_norm

# +Z_B -> velocity
zB_N = v_N / v_norm

# Complete right-handed frame
yB_N = np.cross(zB_N, xB_N)

y_norm = np.linalg.norm(yB_N)

if y_norm < 1e-12:
    raise ValueError(
        "Velocity and nadir vectors are degenerate."
    )

yB_N /= y_norm

# Re-orthogonalize
zB_N = np.cross(xB_N, yB_N)

zB_N /= np.linalg.norm(zB_N)

# ----------------------------------------------------------
# NOMINAL DCM
# ----------------------------------------------------------

C_BN_nominal = np.vstack([
    xB_N,
    yB_N,
    zB_N,
])

# ----------------------------------------------------------
# BODY-FRAME PERTURBATION QUATERNION
# ----------------------------------------------------------

axis_B = np.array(
    INITIAL_PERTURBATION_AXIS_B,
    dtype=float
)

axis_norm = np.linalg.norm(axis_B)

if axis_norm < 1e-12:
    raise ValueError(
        "INITIAL_PERTURBATION_AXIS_B must be non-zero."
    )

axis_B /= axis_norm

angle_rad = np.deg2rad(
    INITIAL_PERTURBATION_ANGLE_DEG
)

q0 = np.cos(angle_rad / 2.0)

q_vec = axis_B * np.sin(angle_rad / 2.0)

quat = np.hstack([
    q_vec,
    q0
])

# ----------------------------------------------------------
# QUATERNION -> DCM
# ----------------------------------------------------------

C_perturb = rbk.EP2C(quat)

# ----------------------------------------------------------
# FINAL ATTITUDE
# ----------------------------------------------------------

C_BN = C_perturb @ C_BN_nominal
if not np.all(np.isfinite(C_BN)):
    raise ValueError(
        "Invalid initial attitude DCM."
    )

INITIAL_SIGMA_BN = rbk.C2MRP(
    C_BN
).tolist()

if not np.all(np.isfinite(INITIAL_SIGMA_BN)):
    raise ValueError(
        "Invalid INITIAL_SIGMA_BN."
    )

# [rad/s]
INITIAL_OMEGA_BN_B_RADPS = [0.025, 0.025, 0.025]

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

THETA_PANEL_DEG = 180.0 #90 deg to 180 deg

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
# AUTOMATIC OUTPUT FOLDER NAMING
# ==========================================================

controller_mode = ACTIVE_CONTROLLER.strip().upper()

# ----------------------------------------------------------
# CONTROLLER TAG
# ----------------------------------------------------------

controller_tag_map = {
    "NADIR_POINTING": "NP",
    "BDOT": "BDOT",
}

controller_tag = controller_tag_map.get(
    controller_mode,
    controller_mode
)

# ----------------------------------------------------------
# ALTITUDE TAG
# ----------------------------------------------------------

altitude_tag = (
    f"{int(ATMOSPHERE_SCALE_HEIGHT_M / 1000.0)}KM"
)

# ----------------------------------------------------------
# SOLAR PANEL ANGLE TAG
# ----------------------------------------------------------

panel_angle_tag = (
    f"{int(THETA_PANEL_DEG)}DEG_SPA"
)

# ----------------------------------------------------------
# SIMULATION DURATION TAG
# ----------------------------------------------------------

simulation_days = (
    SIMULATION_TIME_S / 86400.0
)

simulation_tag = (
    f"{simulation_days:.1f}"
    .replace(".", "_")
    + "DAYS"
)

# ----------------------------------------------------------
# INITIAL ANGULAR RATE MAGNITUDE TAG
# ----------------------------------------------------------

omega0_mag = np.linalg.norm(
    np.array(
        INITIAL_OMEGA_BN_B_RADPS,
        dtype=float
    )
)

omega0_tag = (
    f"{omega0_mag:.4f}"
    .replace(".", "_")
    + "_RPS"
)

# ----------------------------------------------------------
# GAIN TAGS
# ----------------------------------------------------------

if controller_mode == "BDOT":

    gain_tag = (
        f"BG_{BDOT_GAIN:.6g}"
        .replace(".", "_")
    )

elif controller_mode == "NADIR_POINTING":

    gain_tag = (
        f"BG_{BDOT_GAIN_NADIR:.6g}"
        .replace(".", "_")
        + "__"
        + f"KPN_{KP_NADIR:.6g}".replace(".", "_")
    )

else:

    gain_tag = "UNKNOWN_GAIN"

# ----------------------------------------------------------
# FINAL OUTPUT FOLDER NAME
# ----------------------------------------------------------

OUTPUT_FOLDER_NAME = "__".join([

    controller_tag,

    altitude_tag,

    panel_angle_tag,

    simulation_tag,

    omega0_tag,

    gain_tag
])

# ==========================================================
# EXPORT
# ==========================================================

__all__ = [

    # ======================================================
    # TIMING
    # ======================================================

    "OUTPUT_FOLDER_NAME",
    "SIMULATION_TIME_S",

    "DYN_DT_S",
    "LOGGING_DT_S",

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