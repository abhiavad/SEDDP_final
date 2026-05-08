import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros, orbitalMotion, unitTestSupport


def plotOrbits(
    fswTime,
    dynTime,
    posData,
    velData,
    attData,    
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
    fileName,   # <-- explicit dependency injection
):
    plt.close("all")
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style="plain")
    finalDiff = 0.0

    for idx in range(3):
        plt.plot(
            dynTime * macros.NANO2SEC / P,
            posData[:, idx] / 1000.0,
            color=unitTestSupport.getLineColor(idx, 3),
            label="$r_{BN," + str(idx) + "}$"
        )

    plt.legend(loc="lower right")
    plt.xlabel("Time [orbits]")
    plt.ylabel("Inertial Position [km]")

    figureList = {}
    pltName = fileName + "1"
    figureList[pltName] = plt.figure(1)

    plt.figure(2)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style="plain")

    smaData = []
    for idx in range(len(posData)):
        oeData = orbitalMotion.rv2elem(mu, posData[idx], velData[idx])
        smaData.append(oeData.a / 1000.0)

    plt.plot(dynTime * macros.NANO2SEC / P, smaData)
    plt.xlabel("time [orbit]")
    plt.ylabel("SMA [km]")

    pltName = fileName + "2"
    figureList[pltName] = plt.figure(2)

    plt.figure(6)
    for idx in range(3):
        plt.plot(
            dynTime * macros.NANO2MIN,
            dataRec.omega_BN_B[:, idx],
            color=unitTestSupport.getLineColor(idx, 3),
            label=r'$\omega_{B/N,' + str(idx) + '}$'
        )
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Body Rate $\omega_{B/N}$ [rad/s]')

    pltName = fileName + "6"
    figureList[pltName] = plt.figure(6)

    plt.figure(7)
    fig1 = plt.gcf()
    ax = fig1.gca()
    ax.ticklabel_format(useOffset=False, style='sci')
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )

    for idx in range(3):
        plt.plot(
            dynTime * macros.NANO2SEC / P,
            magData[:, idx] * 1e9,
            color=unitTestSupport.getLineColor(idx, 3),
            label=r'$B\_N_{' + str(idx) + '}$'
        )

    plt.legend(loc='lower right')
    plt.xlabel('Time [orbits]')
    plt.ylabel('Magnetic Field [nT]')

    pltName = fileName + "7"
    figureList[pltName] = plt.figure(7)

    plt.figure(8)
    omegaMag = np.linalg.norm(dataRec.omega_BN_B, axis=1)
    plt.plot(dynTime * macros.NANO2MIN, omegaMag)
    plt.xlabel("Time [min]")
    plt.ylabel("|ω| [rad/s]")
    plt.title("Angular Velocity Magnitude")

    pltName = fileName + "8"
    figureList[pltName] = plt.figure(8)

    plt.figure(9)
    for idx in range(3):
        plt.plot(
            fswTime * macros.NANO2MIN,
            mtbDipoleCmdsLog.mtbDipoleCmds[:, idx],
            label=f"$m_{idx}$"
        )

    plt.xlabel("Time [min]")
    plt.ylabel("Magnetic Moment [A m²]")
    plt.title("Magnetorquer Magnetic Moment")
    plt.legend()

    pltName = fileName + "9"
    figureList[pltName] = plt.figure(9)

    plt.figure(10)
    for idx in range(3):
        plt.plot(
            dynTime * macros.NANO2MIN,
            MagDistLog.torqueRequestBody[:, idx],
            label=f"tau_d_{idx}"
        )
    plt.xlabel("Time [min]")
    plt.ylabel("Mag Disturbance Torque [Nm]")
    plt.legend()

    plt.figure(11)
    for idx in range(3):
        plt.plot(
            dynTime * macros.NANO2MIN,
            ggLog.gravityGradientTorque_B[:, idx],
            label=f"tau_gg_{idx}"
        )

    plt.xlabel("Time [min]")
    plt.ylabel("Gravity Gradient Torque [Nm]")
    plt.legend()

    pltName = fileName + "11"
    figureList[pltName] = plt.figure(11)

    # plt.figure(12)
    # for idx in range(3):
    #     plt.plot(
    #         dynTime * macros.NANO2MIN,
    #         dragForceLog.forceExternal_B[:, idx],
    #         label=f"F_drag_{idx}"
    #     )

    # plt.xlabel("Time [min]")
    # plt.ylabel("Aerodynamic Force [N]")
    # plt.legend()

    plt.figure(13)
    for idx in range(3):
        plt.plot(
            dynTime * macros.NANO2MIN,
            aeroTorqueLog.torqueRequestBody[:, idx],
            label=f"tau_drag_{idx}"
        )

    plt.xlabel("Time [min]")
    plt.ylabel("Aerodynamic Torque [Nm]")
    plt.legend()

    # =====================================================
    # +Z AXIS VS VELOCITY VECTOR ANGLE
    # =====================================================

    angle_z_vel_deg = np.zeros(len(dynTime))

    for i in range(len(dynTime)):

        v_N = velData[i]

        v_hat_N = v_N / np.linalg.norm(v_N)

        sigma_BN = attData[i]

        C_BN = rbk.MRP2C(sigma_BN)

        # body +Z axis expressed in inertial frame
        plus_z_N = C_BN.T @ np.array([0.0, 0.0, 1.0])

        cos_angle = np.dot(
            plus_z_N,
            v_hat_N
        )

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0
        )

        angle_z_vel_deg[i] = np.degrees(
            np.arccos(cos_angle)
        )

    plt.figure(14)

    plt.plot(
        dynTime * macros.NANO2MIN,
        angle_z_vel_deg
    )

    plt.xlabel("Time [min]")

    plt.ylabel("Angle [deg]")

    plt.title("+Z Axis vs Velocity Vector")

    plt.grid(True)

    pltName = fileName + "14"

    figureList[pltName] = plt.figure(14)

    

    return figureList, finalDiff