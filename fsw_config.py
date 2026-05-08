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

FSW_DT_S = 0.1


# ==========================================================
# CONTROLLER SELECTION
# ==========================================================

# Converted to uppercase for Code.py compatibility
ACTIVE_CONTROLLER = "BDOT"   # "BDOT" or "NADIR_POINTING"


# ==========================================================
# CONTROLLER GAINS
# ==========================================================

# From get_bdot_gain()
BDOT_GAIN = 0.05
OMEGA_DEADBAND_RADPS = 1e-4

# ==========================================================
# ACTUATOR LIMITS
# ==========================================================

# From get_max_dipole()
MAX_DIPOLE_AM2 = (0.065, 0.065, 0.065)


# ==========================================================
# MODE SCHEDULER (FROM FSW_CONFIG)
# ==========================================================

NS = 4   # sensing steps
NA = 4   # actuation steps


# ==========================================================
# ACTUATION TIME (CRITICAL FOR CONTROL LAW)
# ==========================================================

TOTAL_ACTUATION_TIME = NA * FSW_DT_S

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
    return mapping.get(ACTIVE_CONTROLLER, 0)

# ==========================================================
# SENSOR CONFIGURATION
# ==========================================================

# From ENABLE_SENSOR_NOISE (default True in adcs_config)
HORIZON_USE_NOISE = True

# No explicit noise model defined previously → keep zero
HORIZON_NOISE_STD = [math.radians(0.5), math.radians(0.5), 0.0] # 0.5 deg max error in roll, pitch
HORIZON_BIAS = [0.0, 0.0, 0.0]

# Magnetometer (unchanged)
MAGNETOMETER_SCALE_FACTOR = 1.0
MAGNETOMETER_NOISE_STD = [0.0, 0.0, 0.0]
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

# Keep consistent with your existing file
DIPOLE_QUANTIZATION_STEP = 0.0005


# ==========================================================
# EXPORT
# ==========================================================

__all__ = [
    "FSW_DT_S",
    "ACTIVE_CONTROLLER",
    "BDOT_GAIN",
    "OMEGA_DEADBAND_RADPS",
    "MAX_DIPOLE_AM2",
    "NS",
    "NA",
    "TOTAL_ACTUATION_TIME",
    "HORIZON_USE_NOISE",
    "HORIZON_NOISE_STD",
    "HORIZON_BIAS",
    "MAGNETOMETER_SCALE_FACTOR",
    "MAGNETOMETER_NOISE_STD",
    "NUM_MTB",
    "GT_MATRIX_B",
    "STEERING_MATRIX",
    "DIPOLE_QUANTIZATION_STEP",
    "get_controller_mode_int",
    "ESTIMATION_BUFFER_SIZE",
]