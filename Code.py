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
from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])
 
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
    fswStepTime = macros.sec2nano(FSW_DT_S)

    dynsamplingTime = simStepTime
    fswsamplingTime = fswStepTime

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


    bdotController = BdotController(I_matrix)
    scSim.AddModelToTask(fswCoreTask, bdotController)

    dipoleSelector = DipoleSelector()
    scSim.AddModelToTask(fswCoreTask, dipoleSelector)

    dipoleMappingObj = dipoleMapping.dipoleMapping()
    scSim.AddModelToTask(fswCoreTask, dipoleMappingObj)

    quantizerObj = DipoleQuantizer(
        step_percentage=DIPOLE_QUANTIZATION_STEP,
        max_dipole=maxDipole
    )
    scSim.AddModelToTask(fswCoreTask, quantizerObj)

    # =====================================================
    # LOGGING
    # =====================================================

    mtbTorqueLog = mtbEff.logger("torqueExternalPntB_B", dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, mtbTorqueLog)

    MagDistLog = magDist.disturbanceTorqueOutMsg.recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, MagDistLog)

    ggLog = ggEff.gravityGradientOutMsg.recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, ggLog)

    # dragForceLog = dragEff.logger("forceExternal_B", dynsamplingTime)
    # scSim.AddModelToTask(dynTaskName, dragForceLog)

    rhoLog = atmo.envOutMsgs[0].recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, rhoLog)

    dataRec = delfi.scStateOutMsg.recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, dataRec)

    magLog = magModule.envOutMsgs[0].recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, magLog)

    tamLog = TAM.tamDataOutMsg.recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, tamLog)

    aeroTorqueLog = aeroTorque.torqueOutMsg.recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, aeroTorqueLog)

    # =====================================================
    # FSW LOGGING
    # =====================================================

    tamCommLog = tamCommObj.tamOutMsg.recorder(fswsamplingTime)
    scSim.AddModelToTask(fswCoreTask, tamCommLog)

    horizonLog = horizonCommObj.nadirOutMsg.recorder(fswsamplingTime)
    scSim.AddModelToTask(fswCoreTask, horizonLog)

    mtbDipoleCmdsLog = dipoleMappingObj.dipoleRequestMtbOutMsg.recorder(fswsamplingTime)
    scSim.AddModelToTask(fswCoreTask, mtbDipoleCmdsLog)

    modeLog = modeScheduler.modeOutMsg.recorder(fswsamplingTime)
    scSim.AddModelToTask(fswCoreTask, modeLog)

    modeTypeLog = modeScheduler.modeTypeOutMsg.recorder(fswsamplingTime)
    scSim.AddModelToTask(fswCoreTask, modeTypeLog)

    trueOmegaLog = delfi.scStateOutMsg.recorder(dynsamplingTime)
    scSim.AddModelToTask(dynTaskName, trueOmegaLog)

    bdotLog = bdotPredictor.bdotOutMsg.recorder(fswsamplingTime)
    scSim.AddModelToTask(fswCoreTask, bdotLog)
    
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

    # delfi.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]

    # delfi.hub.omega_BN_BInit = [
    #     omega0,
    #     -omega0,
    #     omega0
    # ]
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

    # Earth radius [m]
    # atmo.planetRadius = orbitalMotion.REQ_EARTH * 1e3

    # # atmospheric density model
    # atmo.baseDensity = 3.2e-11
    # atmo.scaleHeight = 70000.0

    # # altitude bounds
    # atmo.envMinReach = -300e3
    # atmo.envMaxReach = 1000e3
    # Earth radius [m]
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

    # dragEff.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])

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

    # epochMsg = unitTestSupport.timeStringToGregorianUTCMsg(
    #     "2019 June 27, 10:23:0.0 (UTC)"
    # )
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
    #legacy code then replacement
    # mtbConfigParams = messaging.MTBArrayConfigMsgPayload()

    # mtbConfigParams.numMTB = 3

    # mtbConfigParams.GtMatrix_B = [
    #     1., 0., 0.,
    #     0., 1., 0.,
    #     0., 0., 1.
    # ]

    

    # maxDipole_scalar = float(maxDipole[0]) if isinstance(maxDipole, (list, np.ndarray)) else float(maxDipole)

    # mtbConfigParams.maxMtbDipoles = [
    #     maxDipole_scalar
    # ] * mtbConfigParams.numMTB
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
    # # Horizon sensor noise
    # horizon.useNoise = ENABLE_SENSOR_NOISE

    # # Magnetometer configuration
    # TAM.scaleFactor = 1.0
    # TAM.senNoiseStd = [0.0, 0.0, 0.0]


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
    # CONTROLLER SELECTION
    # =====================================================

    dipoleSelector.modeTypeInMsg.subscribeTo(
        modeScheduler.modeTypeOutMsg
    )

    # ---------------------------------
    # SIMPLE CONTROLLER SWITCH
    # ---------------------------------

    controller_mode = ACTIVE_CONTROLLER.strip().upper()

    if controller_mode == "BDOT":

        dipoleSelector.bdotDipoleInMsg.subscribeTo(
            bdotController.cmdDipoleOutMsg
        )

    dipoleMappingObj.steeringMatrix = STEERING_MATRIX

    dipoleMappingObj.dipoleRequestBodyInMsg.subscribeTo(
        dipoleSelector.dipoleOutMsg
    )

    dipoleMappingObj.mtbArrayConfigParamsInMsg.subscribeTo(
        mtbParamsInMsg
    )


    # =====================================================
    # MTB QUANTIZATION
    # =====================================================

    quantizerObj.mtbCmdInMsg.subscribeTo(
        dipoleMappingObj.dipoleRequestMtbOutMsg
    )

    mtbEff.mtbCmdInMsg.subscribeTo(
        quantizerObj.mtbCmdOutMsg
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

    # FSW time base
    times = bdotLog.times() * 1e-9

    mode = np.array(modeLog.dataValue)
    modeType = np.array(modeTypeLog.dataValue)

    dipole = np.array(mtbDipoleCmdsLog.mtbDipoleCmds)
    dipole_norm = np.linalg.norm(dipole, axis=1)

    # dynamics time base
    dynTimes = magLog.times() * 1e-9

    # magnetic field
    B_dyn = np.array(magLog.magField_N)
    B_norm_dyn = np.linalg.norm(B_dyn, axis=1)

    # BDOT estimate
    bdot = np.array(bdotLog.rHat_XB_B)
    bdot_norm = np.linalg.norm(bdot, axis=1)

    # actual MTB torque
    mtbTorque_dyn = np.array(mtbTorqueLog.torqueExternalPntB_B)

    trueOmega_dyn = np.array(trueOmegaLog.omega_BN_B)


    # =====================================================
    # RESAMPLING UTILITIES
    # =====================================================

    def resample_dyn(data_dyn):

        return np.vstack([
            np.interp(times, dynTimes, data_dyn[:,0]),
            np.interp(times, dynTimes, data_dyn[:,1]),
            np.interp(times, dynTimes, data_dyn[:,2])
        ]).T


    def resample_dyn_scalar(data_dyn):

        return np.interp(times, dynTimes, data_dyn)


    # =====================================================
    # RESAMPLE DYNAMICS DATA TO FSW TIMELINE
    # =====================================================

    B_norm = resample_dyn_scalar(B_norm_dyn)

    mtbTorque = resample_dyn(mtbTorque_dyn)

    trueOmega = resample_dyn(trueOmega_dyn)
    trueOmega_norm = np.linalg.norm(trueOmega, axis=1)


    # =====================================================
    # BUILD DATA MATRIX
    # =====================================================

    data = np.column_stack((

        times,
        mode,
        modeType,

        B_norm,

        bdot[:,0],
        bdot[:,1],
        bdot[:,2],
        bdot_norm,

        dipole[:,0],
        dipole[:,1],
        dipole[:,2],
        dipole_norm,

        mtbTorque[:,0],
        mtbTorque[:,1],
        mtbTorque[:,2],

        # -----------------------------
        # TRUE OMEGA (ADD THIS BLOCK)
        # -----------------------------
        trueOmega[:,0],
        trueOmega[:,1],
        trueOmega[:,2],
        trueOmega_norm

    ))


    # =====================================================
    # CSV EXPORT
    # =====================================================

    header = (
    "time_s,"
    "mode,"
    "modeType,"
    "B_norm_T,"
    "bdot_x_T_s,bdot_y_T_s,bdot_z_T_s,bdot_norm_T_s,"
    "dipole_x_A_m2,dipole_y_A_m2,dipole_z_A_m2,dipole_norm_A_m2,"
    "mtb_torque_x_Nm,mtb_torque_y_Nm,mtb_torque_z_Nm,"
    "true_omega_x_rad_s,true_omega_y_rad_s,true_omega_z_rad_s,true_omega_norm_rad_s"
    )

    np.savetxt(
        "adcs_log.csv",
        data,
        delimiter=",",
        header=header,
        comments=""
    )


    # =====================================================
    # QUICK DIAGNOSTICS
    # =====================================================

    print("Mean density:", np.mean(rhoLog.neutralDensity))

    # print(
    #     "Mean drag force:",
    #     np.mean(np.linalg.norm(dragForceLog.forceExternal_B, axis=1))
    # )

    print(
        "Mean drag torque:",
        np.mean(np.linalg.norm(aeroTorqueLog.torqueRequestBody, axis=1))
    )

    print("Mode sample:", modeLog.dataValue[:20])
    print("ModeType sample:", modeTypeLog.dataValue[:20])
    print("ModeType (first 20):", modeTypeLog.dataValue[:20])


    # =====================================================
    # ORBIT DATA FOR PLOTTING
    # =====================================================

    posData = dataRec.r_BN_N
    velData = dataRec.v_BN_N

    dynTime = dataRec.times()
    fswTime = tamCommLog.times()

    magData = magLog.magField_N


    print("NaN in r:", np.isnan(posData).any())
    print("NaN in v:", np.isnan(velData).any())
    print(
        "Mean magnetic field magnitude:",
        np.mean(np.linalg.norm(magData, axis=1))
    )


    # =====================================================
    # PLOTTING
    # =====================================================

    figureList, finalDiff = plotOrbits(
        fswTime,
        dynTime,
        posData,
        velData,
        dataRec.sigma_BN,
        oe,
        mu,
        P,
        dataRec,
        magData,
        mtbDipoleCmdsLog,
        MagDistLog,
        ggLog,
        # dragForceLog,
        aeroTorqueLog,
        fileName
    )

plt.figure()

labels = ['x', 'y', 'z']

for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(times, trueOmega[:,i], label=f"ω_{labels[i]}")
    plt.ylabel("rad/s")
    plt.legend()
    plt.grid()

plt.xlabel("Time [s]")
plt.suptitle("Body Angular Velocity")

    plt.show()
    plt.close("all")

    return finalDiff, figureList


    # =====================================================
    # SCRIPT ENTRY POINT
    # =====================================================

if __name__ == "__main__":
    run()