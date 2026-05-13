#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

# FINAL CONTRACT:
# - simulation_config.py owns physics, geometry, orbit, environment, and initial conditions
# - fsw_config.py owns FSW timing, gains, controller selection, and sensor tuning 
# - geometry_utils.py owns all facet construction
# - mass_properties.py owns mass, inertia, and CG derivation

# ==========================================================
# CONFIG IMPORTS
# ==========================================================
# simulation_config → ALL physical + environment + orbit + geometry
# fsw_config        → ALL FSW logic (timing, gains, controller selection)
# CHANGE:
# - Removed all get_*() config access functions
#
# NOW COMES FROM:
# - simulation_config.py → physics, geometry, orbit, environment
# - fsw_config.py → timing, gains, controller behavior
#
# WHY:
# - strict separation of simulation vs FSW configuration
# - eliminates duplicated configuration paths

from simulation_config import *
from fsw_config import *


# ==========================================================
# INTERNAL UTILITIES
# ==========================================================
# These ensure consistency between geometry and inertia

from mass_properties import compute_mass_properties

from geometry_utils import ALL_AERO_FACETS

import os
from pathlib import Path
# import csv 
import numpy as np
import matplotlib.pyplot as plt
#import general simulation support files
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    unitTestSupport,
    RigidBodyKinematics as rbk,
    # vizSupport,
)

#import simulation related support
from Basilisk.simulation import (
    spacecraft,
    extForceTorque,
    simpleNav,
    magneticFieldWMM,
    magnetometer,
    MtbEffector,
    GravityGradientEffector,
    exponentialAtmosphere,
)

#import FSW Algo related support, might need to be rewritten
from Basilisk.fswAlgorithms import(
    tamComm,
    dipoleMapping,
)

#import message declarations
from Basilisk.architecture import messaging

#get location of supporting data
fileName = os.path.basename(os.path.splitext(__file__)[0])
# =====================================================
# OUTPUT DIRECTORY
# =====================================================

OUTPUT_DIR = Path(OUTPUT_FOLDER_NAME)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
 
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile
from aero_torque_calculator import AeroTorqueCalculator
from magnetic_disturbance import ResidualMagneticDipoleTorque
from plot_orbits import plotOrbits
from horizon_sensor import HorizonSensor
from horizon_comm import HorizonComm
from mode_scheduler import ModeScheduler
from bdot_predictor import BdotPredictor
from bdot_controller import BdotController
from dipoleselector import DipoleSelector
from Dipole_quantizer import DipoleQuantizer
from DipoleConditioner import DipoleConditioner
from nadir_pointing_controller import NadirPointingController

def run():

    # =====================================================
    # SIMULATION + PROCESS SETUP
    # =====================================================

    dynTaskName = "dynTask"
    dynProcessName = "dynProcess"

    fswProcessName = "fswProcess"
    fswCoreTask = "fswCoreTask"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)

    dynProcess = scSim.CreateNewProcess(dynProcessName)
    fswProcess = scSim.CreateNewProcess(fswProcessName)

    # =====================================================
    # TIMING (FROM CONFIG)
    # =====================================================
    # CHANGE:
    # - simulation duration now from simulation_config
    # - dyn step time from simulation_config
    # - fsw step time from fsw_config
    #
    # WHY:
    # - single point of control for all timing
    # - allows changing dt without touching core code

    simulationTime = macros.sec2nano(SIMULATION_TIME_S)

    simStepTime = macros.sec2nano(DYN_DT_S)
    fswStepTime = macros.sec2nano(FSW_STEP_TIME_S)

    loggingSamplingTime = macros.sec2nano(LOGGING_DT_S)

    dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simStepTime))
    fswProcess.addTask(scSim.CreateNewTask(fswCoreTask, fswStepTime))
    # fswProcess.addTask(scSim.CreateNewTask(senseTaskName, fswStepTime))
    # fswProcess.addTask(scSim.CreateNewTask(fswCoreTask, fswStepTime))

    # =====================================================
    # SPACECRAFT PHYSICAL PARAMETERS (FROM CONFIG + DERIVED)
    # =====================================================
    # CHANGE:
    # - Removed ALL get_*() calls
    # - Mass + inertia now computed from geometry
    # - Geometry is single source of truth
    #
    # WHY:
    # - eliminates inconsistency between CAD and inertia
    # - ensures panels, dimensions, and inertia match
    # - removes duplicated definitions across files

    # --- Mass + inertia from geometry ---
    m, I_matrix, cg_B = compute_mass_properties()

    I_matrix = np.array(I_matrix, dtype=float)

    # --- Magnetic properties ---
    m_residual_B = np.array(RESIDUAL_DIPOLE_B_AM2).reshape(3, 1)

    # --- Actuator limits ---
    maxDipole = MAX_DIPOLE_AM2

    # =====================================================
    # SPACECRAFT DYNAMICS
    # =====================================================

    delfi = spacecraft.Spacecraft()
    scSim.AddModelToTask(dynTaskName, delfi)

    # =====================================================
    # ENVIRONMENT MODELS
    # =====================================================

    atmo = exponentialAtmosphere.ExponentialAtmosphere()
    atmo.addSpacecraftToModel(delfi.scStateOutMsg)
    scSim.AddModelToTask(dynTaskName, atmo)

    magModule = magneticFieldWMM.MagneticFieldWMM()
    magModule.addSpacecraftToModel(delfi.scStateOutMsg)
    scSim.AddModelToTask(dynTaskName, magModule)

    facets = ALL_AERO_FACETS

    # --- Register facets with Basilisk drag model ---
    # for facet in facets:
    #     dragEff.addFacet(
    #         facet["area"],
    #         facet["Cd"],
    #         facet["normal"],
    #         facet["location"]
    #     )

    # CHANGE:
    # - aero torque now uses shared facet list from geometry_utils
    #
    # NOW COMES FROM:
    # - geometry_utils.build_facets_from_config()
    #
    # WHY:
    # - guarantees identical geometry between drag and torque models


    aeroTorque = AeroTorqueCalculator(facets)
    scSim.AddModelToTask(dynTaskName, aeroTorque)

    aeroTorqueEff = extForceTorque.ExtForceTorque()
    delfi.addDynamicEffector(aeroTorqueEff)
    scSim.AddModelToTask(dynTaskName, aeroTorqueEff)

    mag_Dist = extForceTorque.ExtForceTorque()
    delfi.addDynamicEffector(mag_Dist)
    scSim.AddModelToTask(dynTaskName, mag_Dist)

    magDist = ResidualMagneticDipoleTorque(m_residual_B)
    scSim.AddModelToTask(dynTaskName, magDist)

    # =====================================================
    # ACTUATORS
    # =====================================================

    mtbEff = MtbEffector.MtbEffector()
    delfi.addDynamicEffector(mtbEff)
    scSim.AddModelToTask(dynTaskName, mtbEff)

    # =====================================================
    # GRAVITY
    # =====================================================

    ggEff = GravityGradientEffector.GravityGradientEffector()

    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    planet.isCentralBody = True

    gravFactory.addBodiesTo(delfi)

    ggEff.addPlanetName(planet.planetName)
    delfi.addDynamicEffector(ggEff)
    scSim.AddModelToTask(dynTaskName, ggEff)

    # =====================================================
    # NAVIGATION + SENSORS
    # =====================================================

    delfi_NavObj = simpleNav.SimpleNav()
    scSim.AddModelToTask(dynTaskName, delfi_NavObj)

    TAM = magnetometer.Magnetometer()
    scSim.AddModelToTask(dynTaskName, TAM)

    horizon = HorizonSensor()
    scSim.AddModelToTask(dynTaskName, horizon)

    # =====================================================
    # FSW — SENSOR INTERFACES
    # =====================================================

    tamCommObj = tamComm.tamComm()
    scSim.AddModelToTask(fswCoreTask, tamCommObj)

    horizonCommObj = HorizonComm()
    scSim.AddModelToTask(fswCoreTask, horizonCommObj)

    # =====================================================
    # MODE SCHEDULER
    # =====================================================

    modeScheduler = ModeScheduler()
    scSim.AddModelToTask(fswCoreTask, modeScheduler)

    # =====================================================
    # CONTROL PIPELINE
    # =====================================================

    bdotPredictor = BdotPredictor()
    scSim.AddModelToTask(fswCoreTask, bdotPredictor)


    # ---------------------------------
    # Mode-dependent B-dot gain
    # ---------------------------------

    controller_mode = ACTIVE_CONTROLLER.strip().upper()

    if controller_mode == "BDOT":
        bdot_gain = BDOT_GAIN

    elif controller_mode == "NADIR_POINTING":
        bdot_gain = BDOT_GAIN_NADIR

    else:
        raise ValueError(
            f"Unsupported ACTIVE_CONTROLLER: {controller_mode}"
        )

    bdotController = BdotController(
        I_matrix,
        bdot_gain
    )
    scSim.AddModelToTask(fswCoreTask, bdotController)

    nadirController = NadirPointingController(I_matrix, KP_NADIR)
    scSim.AddModelToTask(fswCoreTask, nadirController)

    dipoleSelector = DipoleSelector()
    scSim.AddModelToTask(fswCoreTask, dipoleSelector)

    dipoleConditioner = DipoleConditioner()
    scSim.AddModelToTask(fswCoreTask, dipoleConditioner)

    quantizerObj = DipoleQuantizer(
        step_percentage=DIPOLE_QUANTIZATION_STEP,
        max_dipole=maxDipole
    )
    scSim.AddModelToTask(fswCoreTask, quantizerObj)

    dipoleMappingObj = dipoleMapping.dipoleMapping()
    scSim.AddModelToTask(fswCoreTask, dipoleMappingObj)

    # =====================================================
    # LOGGING
    # =====================================================

    dataRec = delfi.scStateOutMsg.recorder(
        loggingSamplingTime
    )
    scSim.AddModelToTask(dynTaskName, dataRec)

    MagDistLog = magDist.disturbanceTorqueOutMsg.recorder(
        loggingSamplingTime
    )
    scSim.AddModelToTask(dynTaskName, MagDistLog)

    ggLog = ggEff.gravityGradientOutMsg.recorder(
        loggingSamplingTime
    )
    scSim.AddModelToTask(dynTaskName, ggLog)

    aeroTorqueLog = aeroTorque.torqueOutMsg.recorder(
        loggingSamplingTime
    )
    scSim.AddModelToTask(dynTaskName, aeroTorqueLog)

    # # =====================================================
    # # MAGNETIC FIELD LOGGING
    # # =====================================================

    # # True magnetic field from environment model
    # magFieldLog = magModule.envOutMsgs[0].recorder(
    #     loggingSamplingTime
    # )
    # scSim.AddModelToTask(dynTaskName, magFieldLog)

    # # Magnetometer sensor output
    # tamLog = TAM.tamDataOutMsg.recorder(
    #     loggingSamplingTime
    # )
    # scSim.AddModelToTask(dynTaskName, tamLog)

    # =====================================================
    # FSW LOGGING
    # =====================================================

    # mtbDipoleCmdsLog = dipoleMappingObj.dipoleRequestMtbOutMsg.recorder(fswStepTime)
    # scSim.AddModelToTask(fswCoreTask, mtbDipoleCmdsLog)


    # modePhaseLog = modeScheduler.modeOutMsg.recorder(
    #     fswStepTime
    # )
    # scSim.AddModelToTask(
    #     fswCoreTask,
    #     modePhaseLog
    # )

    # modeTypeLog = modeScheduler.modeTypeOutMsg.recorder(
    #     fswStepTime
    # )
    # scSim.AddModelToTask(
    #     fswCoreTask,
    #     modeTypeLog
    # )

    # actuateLog = modeScheduler.actuateOutMsg.recorder(
    #     fswStepTime
    # )
    # scSim.AddModelToTask(
    #     fswCoreTask,
    #     actuateLog
    # )

    # bdotLog = bdotPredictor.bdotOutMsg.recorder(
    # fswStepTime
    # )
    # scSim.AddModelToTask(
    #     fswCoreTask,
    #     bdotLog
    # )

    # BpredictorLog = bdotPredictor.bOutMsg.recorder(
    #     fswStepTime
    # )
    # scSim.AddModelToTask(
    #     fswCoreTask,
    #     BpredictorLog
    # )
    
    # =====================================================
    # ORBIT DEFINITION
    # =====================================================

    # CHANGE:
    # - removed get_orbit_elements()
    #
    # REMOVED FROM PREVIOUS VERSION:
    # - function-based orbit retrieval
    #
    # NOW COMES FROM:
    # - simulation_config.ORBIT_ELEMENTS
    #
    # WHY:
    # - centralize orbit definition in single config file

    cfg = ORBIT_ELEMENTS
    oe = orbitalMotion.ClassicElements()
    oe.a = cfg["a"]
    oe.e = cfg["e"]
    oe.i = cfg["i_deg"] * macros.D2R
    oe.Omega = cfg["Omega_deg"] * macros.D2R
    oe.omega = cfg["omega_deg"] * macros.D2R
    oe.f = cfg["f_deg"] * macros.D2R


    # =====================================================
    # GRAVITY MODEL
    # =====================================================

    ggm03s__j2_only_path = get_path(DataFile.LocalGravData.GGM03S_J2_only)

    planet.useSphericalHarmonicsGravityModel(
        str(ggm03s__j2_only_path),
        2
    )

    mu = planet.mu


    # =====================================================
    # INITIAL ORBIT STATE
    # =====================================================

    rN, vN = orbitalMotion.elem2rv(mu, oe)

    # regenerate elements for consistency
    oe = orbitalMotion.rv2elem(mu, rN, vN)


    # =====================================================
    # SPACECRAFT INITIAL CONDITIONS
    # =====================================================

    delfi.hub.r_CN_NInit = rN
    delfi.hub.v_CN_NInit = vN

    delfi.hub.mHub = m

    delfi.hub.r_BcB_B = np.array(cg_B, dtype=float).reshape(3, 1).tolist()

    delfi.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
        np.array(I_matrix, dtype=float).reshape(9).tolist()
    )

    delfi.hub.sigma_BNInit = np.array(INITIAL_SIGMA_BN, dtype=float).reshape(3, 1).tolist()

    delfi.hub.omega_BN_BInit = np.array(INITIAL_OMEGA_BN_B_RADPS, dtype=float).reshape(3, 1).tolist()


    # =====================================================
    # ORBITAL TIME PARAMETERS
    # =====================================================

    n = np.sqrt(mu / oe.a**3)
    P = 2.0 * np.pi / n


    # =====================================================
    # ATMOSPHERIC MODEL
    # =====================================================

    atmo.planetRadius = ATMOSPHERE_PLANET_RADIUS_M

    # atmospheric density model
    atmo.baseDensity = ATMOSPHERE_BASE_DENSITY_KG_M3
    atmo.scaleHeight = ATMOSPHERE_SCALE_HEIGHT_M

    # altitude bounds
    atmo.envMinReach = ATMOSPHERE_ENV_MIN_REACH_M
    atmo.envMaxReach = ATMOSPHERE_ENV_MAX_REACH_M


    # =====================================================
    # DRAG MODEL CONNECTIONS
    # =====================================================

    aeroTorque.atmoInMsg.subscribeTo(atmo.envOutMsgs[0])
    aeroTorque.stateInMsg.subscribeTo(delfi.scStateOutMsg)


    # =====================================================
    # SENSOR STATE INPUTS
    # =====================================================

    horizon.stateInMsg.subscribeTo(delfi.scStateOutMsg)
    TAM.stateInMsg.subscribeTo(delfi.scStateOutMsg)

    delfi_NavObj.scStateInMsg.subscribeTo(delfi.scStateOutMsg)

    aeroTorque.attInMsg.subscribeTo(delfi_NavObj.attOutMsg)
    horizon.attInMsg.subscribeTo(delfi_NavObj.attOutMsg)

    magDist.attNavInMsg.subscribeTo(delfi_NavObj.attOutMsg)


    # =====================================================
    # DISTURBANCE TORQUE CONNECTION
    # =====================================================

    aeroTorqueEff.cmdTorqueInMsg.subscribeTo(aeroTorque.torqueOutMsg)


    # =====================================================
    # MAGNETIC FIELD MODEL
    # =====================================================

    epochMsg = unitTestSupport.timeStringToGregorianUTCMsg(MAG_FIELD_EPOCH)

    wmm_path = get_path(DataFile.MagneticFieldData.WMM)

    magModule.configureWMMFile(str(wmm_path))
    magModule.epochInMsg.subscribeTo(epochMsg)

    mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])
    TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])
    magDist.magInMsg.subscribeTo(magModule.envOutMsgs[0])

    mag_Dist.cmdTorqueInMsg.subscribeTo(
        magDist.disturbanceTorqueOutMsg
    )


    # =====================================================
    # MAGNETORQUER CONFIGURATION
    # =====================================================

    mtbConfigParams = messaging.MTBArrayConfigMsgPayload()

    mtbConfigParams.numMTB = NUM_MTB
    mtbConfigParams.GtMatrix_B = GT_MATRIX_B

    maxDipole_scalar = float(np.array(maxDipole).flatten()[0])

    mtbConfigParams.maxMtbDipoles = [
        maxDipole_scalar
    ] * NUM_MTB

    mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(
        mtbConfigParams
    )

    mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)

 
    # =====================================================
    # SENSOR CONFIGURATION
    # =====================================================
    # CHANGE:
    # - removed inline sensor noise values
    #
    # NOW COMES FROM:
    # - fsw_config (sensor noise parameters)
    #
    # WHY:
    # - centralize all FSW tuning

    horizon.useNoise = HORIZON_USE_NOISE
    horizon.senNoiseStd = HORIZON_NOISE_STD
    horizon.bias = HORIZON_BIAS

    TAM.scaleFactor = MAGNETOMETER_SCALE_FACTOR
    TAM.senNoiseStd = MAGNETOMETER_NOISE_STD
    TAM.walkBounds = MAGNETOMETER_WALK_BOUNDS
    TAM.senBias = MAGNETOMETER_BIAS
    TAM.maxOutput = MAGNETOMETER_MAX_OUTPUT
    TAM.minOutput = MAGNETOMETER_MIN_OUTPUT

    # =====================================================
    # SENSOR → FSW INTERFACES
    # =====================================================

    # TAM communication
    tamCommObj.dcm_BS = [
        1., 0., 0.,
        0., 1., 0.,
        0., 0., 1.
    ]

    tamCommObj.tamInMsg.subscribeTo(
        TAM.tamDataOutMsg
    )

    # Horizon communication
    horizonCommObj.nadirInMsg.subscribeTo(
        horizon.nadirOutMsg
    )

    # =====================================================
    # BDOT PREDICTOR
    # =====================================================

    bdotPredictor.bInMsg.subscribeTo(
        tamCommObj.tamOutMsg
    )

    bdotPredictor.modeInMsg.subscribeTo(
        modeScheduler.modeOutMsg
    )


    # =====================================================
    # BDOT CONTROLLER
    # =====================================================

    bdotController.bInMsg.subscribeTo(
        bdotPredictor.bOutMsg
    )

    bdotController.bdotInMsg.subscribeTo(
        bdotPredictor.bdotOutMsg
    )

    # =====================================================
    # ACTUATION TRIGGER CONNECTION (CRITICAL)
    #=====================================================

    bdotController.actuateInMsg.subscribeTo(
        modeScheduler.actuateOutMsg
    )

    # =====================================================
    # NADIR CONTROLLER
    # =====================================================

    nadirController.actuateInMsg.subscribeTo(
        modeScheduler.actuateOutMsg
    )

    nadirController.nadirInMsg.subscribeTo(
        horizonCommObj.nadirOutMsg
    )

    nadirController.bInMsg.subscribeTo(
        bdotPredictor.bOutMsg
    )


    # =====================================================
    # CONTROLLER SELECTION
    # =====================================================

    dipoleSelector.modeTypeInMsg.subscribeTo(
        modeScheduler.modeTypeOutMsg
    )

    # ---------------------------------
    # SIMPLE CONTROLLER SWITCH
    # ---------------------------------

    valid_modes = ["BDOT", "NADIR_POINTING"]

    if controller_mode not in valid_modes:
        raise ValueError(
            f"Unsupported ACTIVE_CONTROLLER: {controller_mode}"
        )

    if controller_mode == "BDOT":

        dipoleSelector.bdotDipoleInMsg.subscribeTo(
            bdotController.cmdDipoleOutMsg
        )

    elif controller_mode == "NADIR_POINTING":

        dipoleSelector.bdotDipoleInMsg.subscribeTo(
            bdotController.cmdDipoleOutMsg
        )

        dipoleSelector.nadirDipoleInMsg.subscribeTo(
            nadirController.cmdDipoleOutMsg
        )

    dipoleConditioner.dipoleInMsg.subscribeTo(
    dipoleSelector.dipoleOutMsg
    )

    # ---------------------------------
    # Quantization BEFORE dipole mapping
    # ---------------------------------

    quantizerObj.dipoleInMsg.subscribeTo(
        dipoleConditioner.dipoleOutMsg
    )

    dipoleMappingObj.steeringMatrix = STEERING_MATRIX

    dipoleMappingObj.dipoleRequestBodyInMsg.subscribeTo(
        quantizerObj.dipoleOutMsg
    )

    dipoleMappingObj.mtbArrayConfigParamsInMsg.subscribeTo(
        mtbParamsInMsg
    )


    # =====================================================
    # MTB COMMAND CONNECTION
    # =====================================================

    mtbEff.mtbCmdInMsg.subscribeTo(
        dipoleMappingObj.dipoleRequestMtbOutMsg
    )


    # =====================================================
    # SIMULATION EXECUTION
    # =====================================================

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()


    # =====================================================
    # EXTRACT LOGGED DATA
    # =====================================================

    times_sec = dataRec.times() * macros.NANO2SEC

    times_min = times_sec / 60.0
    # mtb_times_sec = (mtbDipoleCmdsLog.times()* macros.NANO2SEC) 

    omega = np.array(dataRec.omega_BN_B)

    omega_mag = np.linalg.norm(
        omega,
        axis=1
    )

    magDistTorque = np.array(
        MagDistLog.torqueRequestBody
    )

    ggTorque = np.array(
        ggLog.gravityGradientTorque_B
    )

    aeroTorqueVec = np.array(
        aeroTorqueLog.torqueRequestBody
    )

    # trueMagField = np.array(
    #     magFieldLog.magField_N
    # )

    # tamMeasurement = np.array(
    #     tamLog.tam_S
    # )

    # dipole = np.array(
    #     mtbDipoleCmdsLog.mtbDipoleCmds
    # )
    # modePhase = np.array(
    #     modePhaseLog.dataValue
    # )

    # modeType = np.array(
    #     modeTypeLog.dataValue
    # )

    # actuateFlag = np.array(
    #     actuateLog.dataValue
    # )

    # Bdot_logged = np.array(
    #     bdotLog.rHat_XB_B
    # )

    # B_logged = np.array(
    #     BpredictorLog.tam_B
    # )

    posData = np.array(dataRec.r_BN_N)

    velData = np.array(dataRec.v_BN_N)

    sigmaData = np.array(dataRec.sigma_BN)
    # trueMagField_B = np.zeros_like(trueMagField)

    # for i in range(len(times_sec)):

    #     C_BN = rbk.MRP2C(sigmaData[i])

    #     trueMagField_B[i] = C_BN @ trueMagField[i]

    # =====================================================
    # POINTING ANGLES
    # =====================================================

    angle_z_vel_deg = np.zeros(len(times_sec))

    angle_x_nadir_deg = np.zeros(len(times_sec))

    for i in range(len(times_sec)):

        r_N = posData[i]

        v_N = velData[i]

        sigma_BN = sigmaData[i]

        r_norm = np.linalg.norm(r_N)

        v_norm = np.linalg.norm(v_N)

        if r_norm < 1e-12 or v_norm < 1e-12:
            continue

        nadir_hat_N = -r_N / r_norm

        v_hat_N = v_N / v_norm

        C_BN = rbk.MRP2C(sigma_BN)

        plus_x_N = C_BN.T @ np.array([
            1.0,
            0.0,
            0.0
        ])

        plus_z_N = C_BN.T @ np.array([
            0.0,
            0.0,
            1.0
        ])

        cos_x = np.clip(
            np.dot(plus_x_N, nadir_hat_N),
            -1.0,
            1.0
        )

        cos_z = np.clip(
            np.dot(plus_z_N, v_hat_N),
            -1.0,
            1.0
        )

        angle_x_nadir_deg[i] = np.degrees(
            np.arccos(cos_x)
        )

        angle_z_vel_deg[i] = np.degrees(
            np.arccos(cos_z)
        )

    if (
        not np.all(np.isfinite(omega))
        or not np.all(np.isfinite(magDistTorque))
        or not np.all(np.isfinite(ggTorque))
        or not np.all(np.isfinite(aeroTorqueVec))
        #or not np.all(np.isfinite(dipole))
        or not np.all(np.isfinite(angle_z_vel_deg))
        or not np.all(np.isfinite(angle_x_nadir_deg))
    ):
        raise RuntimeError(
            "Non-finite values detected in logged data"
        )


    # =====================================================
    # CSV EXPORT
    # =====================================================

    data = np.column_stack((

        times_sec,

        omega[:,0],
        omega[:,1],
        omega[:,2],
        omega_mag,

        magDistTorque[:,0],
        magDistTorque[:,1],
        magDistTorque[:,2],

        ggTorque[:,0],
        ggTorque[:,1],
        ggTorque[:,2],

        aeroTorqueVec[:,0],
        aeroTorqueVec[:,1],
        aeroTorqueVec[:,2],

        angle_z_vel_deg,
        angle_x_nadir_deg
    ))

    csv_metadata = f"""# =====================================================
    # ADCS Simulation Results
    # =====================================================
    #
    # Simulation File:
    # {fileName}
    #
    # Active Controller:
    # {ACTIVE_CONTROLLER}
    #
    # Dynamics Timestep [s]:
    # {DYN_DT_S}
    #
    # FSW Timestep [s]:
    # {FSW_STEP_TIME_S}
    #
    # Logging Interval [s]:
    # {LOGGING_DT_S}
    #
    # Number of Logged Samples:
    # {len(times_sec)}
    #
    # =====================================================
    # COLUMN DESCRIPTIONS
    # =====================================================
    #
    # time_sec
    #     Simulation time [sec]
    #
    # omega_x_rad_s
    # omega_y_rad_s
    # omega_z_rad_s
    #     Body angular velocity components [rad/s]
    #
    # omega_mag_rad_s
    #     Body angular velocity magnitude [rad/s]
    #
    # mag_dist_x_Nm
    # mag_dist_y_Nm
    # mag_dist_z_Nm
    #     Residual magnetic disturbance torque [N·m]
    #
    # gg_x_Nm
    # gg_y_Nm
    # gg_z_Nm
    #     Gravity-gradient torque [N·m]
    #
    # aero_x_Nm
    # aero_y_Nm
    # aero_z_Nm
    #     Aerodynamic disturbance torque [N·m]
    #
    # angle_z_vel_deg
    #     Angle between body +Z axis and velocity vector [deg]
    #
    # angle_x_nadir_deg
    #     Angle between body +X axis and nadir vector [deg]
    #
    # =====================================================
    """

    csv_columns = ",".join([

        "time_sec",

        "omega_x_rad_s",
        "omega_y_rad_s",
        "omega_z_rad_s",
        "omega_mag_rad_s",

        "mag_dist_x_Nm",
        "mag_dist_y_Nm",
        "mag_dist_z_Nm",

        "gg_x_Nm",
        "gg_y_Nm",
        "gg_z_Nm",

        "aero_x_Nm",
        "aero_y_Nm",
        "aero_z_Nm",

        "angle_z_vel_deg",
        "angle_x_nadir_deg"
    ])

    csv_path = OUTPUT_DIR / "adcs_log.csv"

    with open(csv_path, "w") as f:

        # Metadata block
        f.write(csv_metadata)

        # Column names
        f.write(csv_columns + "\n")

        # Numeric data
        np.savetxt(
            f,
            data,
            delimiter=",",
            fmt="%.6e"
        )

        print(
            f"Saved ADCS log with {len(times_sec)} samples"
        )

        print(
            f"CSV output: {csv_path}"
        )
    # # =====================================================
    # # MTB COMMAND CSV EXPORT
    # # =====================================================

    # mtb_data = np.column_stack((

    #     mtb_times_sec,

    #     modePhase,
    #     modeType,
    #     actuateFlag,

    #     B_logged[:,0],
    #     B_logged[:,1],
    #     B_logged[:,2],

    #     Bdot_logged[:,0],
    #     Bdot_logged[:,1],
    #     Bdot_logged[:,2],

    #     dipole[:,0],
    #     dipole[:,1],
    #     dipole[:,2]
    # ))

    # mtb_csv_metadata = f"""# =====================================================
    # # Magnetorquer Command Log
    # # =====================================================
    # #
    # # Simulation File:
    # # {fileName}
    # #
    # # Active Controller:
    # # {ACTIVE_CONTROLLER}
    # #
    # # FSW Timestep [s]:
    # # {FSW_STEP_TIME_S}
    # #
    # # Number of Logged Samples:
    # # {len(mtb_times_sec)}
    # #
    # # =====================================================
    # # COLUMN DESCRIPTIONS
    # # =====================================================
    # #
    # # time_sec
    # #     Simulation time [sec]
    # #
    # # mode_phase
    # #     Scheduler phase
    # #     1 = sensing
    # #     2 = actuation
    # #
    # # controller_type
    # #     Active controller type
    # #     0 = BDOT
    # #     1 = NADIR_POINTING
    # #
    # # actuate_flag
    # #     Actuation enable flag
    # #     0 = disabled
    # #     1 = enabled
    # #
    # # B_x_T
    # # B_y_T
    # # B_z_T
    # #     Magnetic field vector in body frame [T]
    # #
    # # Bdot_x_Tps
    # # Bdot_y_Tps
    # # Bdot_z_Tps
    # #     Estimated magnetic field derivative [T/s]
    # #
    # # mtb_x_Am2
    # # mtb_y_Am2
    # # mtb_z_Am2
    # #     Commanded magnetorquer dipole moments [A·m²]
    # #
    # # =====================================================
    # """

    # mtb_csv_columns = ",".join([

    #     "time_sec",

    #     "mode_phase",
    #     "controller_type",
    # "actuate_flag",

    # "B_x_T",
    # "B_y_T",
    # "B_z_T",

    # "Bdot_x_Tps",
    # "Bdot_y_Tps",
    # "Bdot_z_Tps",

    # "mtb_x_Am2",
    # "mtb_y_Am2",
    # "mtb_z_Am2"
    # ])

    # mtb_csv_path = OUTPUT_DIR / "mtb_log.csv"

    # with open(mtb_csv_path, "w") as f:

    #     # Metadata block
    #     f.write(mtb_csv_metadata)

    #     # Column names
    #     f.write(mtb_csv_columns + "\n")

    #     # Numeric data
    #     np.savetxt(
    #         f,
    #         mtb_data,
    #         delimiter=",",
    #         fmt="%.6e"
    #     )

    # print(
    #     f"Saved MTB log with {len(mtb_times_sec)} samples"
    # )

    # print(
    #     f"MTB CSV output: {mtb_csv_path}"
    # )

    # print(
    #     "Unique phase values:",
    #     np.unique(modePhase)
    # )

    # print(
    #     "Unique controller values:",
    #     np.unique(modeType)
    # )

    # print(
    #     "Unique actuate values:",
    #     np.unique(actuateFlag)
    # )

    # print(
    #     "Maximum MTB magnitude:",
    #     np.max(np.abs(dipole))
    # )

    # =====================================================
    # PLOTTING
    # =====================================================

    figureList = plotOrbits(

        times_min,
        # mtb_times_sec,

        omega,
        omega_mag,

        magDistTorque,
        ggTorque,
        aeroTorqueVec,

        # dipole,

        # trueMagField_B,
        # tamMeasurement,

        angle_z_vel_deg,
        angle_x_nadir_deg,

        fileName,

        OUTPUT_DIR
    )

    return figureList


    # =====================================================
    # SCRIPT ENTRY POINT
    # =====================================================

if __name__ == "__main__":
    run()