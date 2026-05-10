"""
Flight Software (FSW) configuration.

Single source of truth for:
- timing
- controller selection
- gains
- actuator limits
- scheduler
- sensor tuning
"""
import math

# ==========================================================
# TIMING (FROM adcs_config.FSW_CONFIG)
# ==========================================================

# ==========================================================
# FSW CONTROL LOOP TIMING
# ==========================================================

"""
CONTROL ARCHITECTURE
--------------------

The ADCS control loop is divided into two phases:

1. SENSING
    - collect magnetometer samples
    - estimate B-dot

2. ACTUATION
    - hold commanded MTB dipole constant

The total control loop duration is:

    FSW_CONTROL_LOOP_DT_S

Internally, the FSW executes at a smaller
discrete scheduler interval:

    FSW_STEP_TIME_S

where:

    FSW_STEP_TIME_S
        = FSW_CONTROL_LOOP_DT_S / (NS + NA)

To maintain symmetric sensing/actuation timing:

    NA = NS

Therefore:

    FSW_STEP_TIME_S
        = FSW_CONTROL_LOOP_DT_S / (2 * NS)
"""

# Total sensing + actuation cycle duration [s]
FSW_CONTROL_LOOP_DT_S = 0.18


# ==========================================================
# CONTROLLER SELECTION
# ==========================================================
# Active flight-control mode.
#
# Supported modes:
#   "BDOT"
#   "NADIR_POINTING"
ACTIVE_CONTROLLER = "BDOT"

# ==========================================================
# CONTROLLER GAINS
# ==========================================================

# ----------------------------------------------------------
# B-DOT DETUMBLE
# ----------------------------------------------------------

# Existing validated detumble gain
BDOT_GAIN = 0.00625

# Future nadir-mode damping gain.
#
# IMPORTANT:
# Currently intentionally identical to BDOT_GAIN
# so that Phase A architecture preparation does
# NOT change existing behavior.
BDOT_GAIN_NADIR = BDOT_GAIN

# Ignore extremely small estimated angular rates
OMEGA_DEADBAND_RADPS = 1e-4


# ----------------------------------------------------------
# NADIR POINTING (PHASE A PLACEHOLDERS)
# ----------------------------------------------------------

# Geometric nadir-pointing proportional gain.
#
# Shared by:
#   - Y-axis steering
#   - Z-axis steering
#
# Torque law:
#
#   tau_align =
#       Kp * [0, Iyy*y_ang, Izz*z_ang]
#
# Initially kept at zero during
# architecture validation.
KP_NADIR = 0.00025

# ----------------------------------------------------------
# RECOVERY MODE THRESHOLDS
# ----------------------------------------------------------

"""
Recovery-mode hysteresis thresholds.

Recovery logic is based on the angle between:
    body +X axis
and:
    nadir vector.

Recovery mode is entered when:
    +X points sufficiently away from nadir.

Recovery mode is exited only after:
    +X safely returns toward nadir.

Using separate thresholds prevents
mode-chatter near 90-degree pointing error.
"""

# Enter recovery mode if:
# angle(+X, nadir) exceeds this value.
RECOVERY_ENTER_ANGLE_DEG = 95.0

# Exit recovery mode if:
# angle(+X, nadir) falls below this value.
RECOVERY_EXIT_ANGLE_DEG = 85.0

# No nadir-pointing torque applied if:
# angle(+X, nadir) is below this threshold.
NADIR_POINTING_DEADBAND_DEG = 5.0

# ==========================================================
# ACTUATOR LIMITS
# ==========================================================

# From get_max_dipole()
MAX_DIPOLE_AM2 = (0.065, 0.065, 0.065)


# ==========================================================
# SCHEDULER DISCRETIZATION
# ==========================================================

"""
NS:
    Number of sensing samples collected
    during each sensing phase.

NA:
    Number of actuation-hold intervals.

Architecture constraint:
    actuation duration = sensing duration

Therefore:
    NA is always forced equal to NS.
"""

NS = 3

# Keep symmetric sensing/actuation timing
NA = NS

# ==========================================================
# DERIVED FSW EXECUTION STEP
# ==========================================================

FSW_STEP_TIME_S = (
    FSW_CONTROL_LOOP_DT_S / (NS + NA)
)

"""
Digital timing sanity check.

Avoid awkward repeating/floating scheduler periods
such as:
    1/3
    5/7
    etc.

This improves:
- deterministic scheduling
- numerical consistency
- easier debugging
- future hardware implementation
"""

# enforce nanosecond-convertible timing cleanly
fsw_step_ns = FSW_STEP_TIME_S * 1e9

if abs(fsw_step_ns - round(fsw_step_ns)) > 1e-6:
    raise ValueError(
        "FSW_STEP_TIME_S does not map cleanly to integer nanoseconds."
    )

# ==========================================================
# ACTUATION HOLD DURATION
# ==========================================================

"""
Total duration for which the MTB dipole
command is held constant.
"""

TOTAL_ACTUATION_TIME = (
    NA * FSW_STEP_TIME_S
)

# ==========================================================
# CONTROLLER MODE MAPPING (REPLACES adcs_config.get_controller_mode)
# ==========================================================

def get_controller_mode_int() -> int:
    """
    Returns integer mode for compatibility with existing modules.

    0 → BDOT
    1 → NADIR_POINTING
    """

    mapping = {
        "BDOT": 0,
        "NADIR_POINTING": 1,
    }

    if ACTIVE_CONTROLLER not in mapping:
        raise ValueError(
            f"Unsupported ACTIVE_CONTROLLER: {ACTIVE_CONTROLLER}"
        )

    return mapping[ACTIVE_CONTROLLER]


# ==========================================================
# SENSOR CONFIGURATION
# ==========================================================

# From ENABLE_SENSOR_NOISE (default True in adcs_config)
HORIZON_USE_NOISE = True

# No explicit noise model defined previously → keep zero
HORIZON_NOISE_STD = [math.radians(0.5), math.radians(0.5), 0.0] # 0.5 deg max error in roll, pitch
HORIZON_BIAS = [0.0, 0.0, 0.0]

# ==========================================================
# MAGNETOMETER CONFIGURATION
# ==========================================================

# Ideal scale factor
MAGNETOMETER_SCALE_FACTOR = 1.0

# White measurement noise standard deviation [Tesla]
# 15 nT RMS
MAGNETOMETER_NOISE_STD = [
    15e-9,
    15e-9,
    15e-9
]

# Gauss-Markov walk bounds [Tesla]
# Approximately 3-sigma drift envelope
MAGNETOMETER_WALK_BOUNDS = [
    45e-9,
    45e-9,
    45e-9
]

# Constant sensor bias [Tesla]
# Keep zero initially
MAGNETOMETER_BIAS = [
    0.0,
    0.0,
    0.0
]

# Sensor saturation limits [Tesla]
# ±800 uT
MAGNETOMETER_MAX_OUTPUT = 800e-6
MAGNETOMETER_MIN_OUTPUT = -800e-6

ESTIMATION_BUFFER_SIZE = NS
assert ESTIMATION_BUFFER_SIZE == NS, \
    "ESTIMATION_BUFFER_SIZE must equal NS (sensing window length)"


# ==========================================================
# MTB CONFIGURATION
# ==========================================================

NUM_MTB = 3

GT_MATRIX_B = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
]

STEERING_MATRIX = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
]


# ==========================================================
# QUANTIZATION
# ==========================================================

# Magnetorquer dipole-command quantization step
DIPOLE_QUANTIZATION_STEP = 0.0005

# ==========================================================
# EXPORT
# ==========================================================

__all__ = [
    "FSW_STEP_TIME_S",
    "FSW_CONTROL_LOOP_DT_S",
    "ACTIVE_CONTROLLER",
    "BDOT_GAIN",
    "BDOT_GAIN_NADIR",
    "OMEGA_DEADBAND_RADPS",
    "KP_NADIR",
    "RECOVERY_ENTER_ANGLE_DEG",
    "RECOVERY_EXIT_ANGLE_DEG",
    "MAX_DIPOLE_AM2",
    "NS",
    "NA",
    "TOTAL_ACTUATION_TIME",
    "HORIZON_USE_NOISE",
    "HORIZON_NOISE_STD",
    "HORIZON_BIAS",
    "MAGNETOMETER_SCALE_FACTOR",
    "MAGNETOMETER_NOISE_STD",
    "MAGNETOMETER_WALK_BOUNDS",
    "MAGNETOMETER_BIAS",
    "MAGNETOMETER_MAX_OUTPUT",
    "MAGNETOMETER_MIN_OUTPUT",
    "NUM_MTB",
    "GT_MATRIX_B",
    "STEERING_MATRIX",
    "DIPOLE_QUANTIZATION_STEP",
    "get_controller_mode_int",
    "ESTIMATION_BUFFER_SIZE",
]