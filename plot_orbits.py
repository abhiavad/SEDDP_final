import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities import unitTestSupport
from pathlib import Path


def plotOrbits(

    time_min,
    # mtb_times_sec,

    omega,
    omega_mag,

    magDistTorque,
    ggTorque,
    aeroTorque,

    # dipole,

    # trueMagField_B,
    # tamMeasurement,

    angle_z_vel_deg,
    angle_x_nadir_deg,

    fileName,
    output_dir
):

    plt.close("all")

    plt.rcParams["figure.figsize"] = (10, 5)

    plt.rcParams["axes.grid"] = True

    figureList = {}

    labels = ["x", "y", "z"]

    # =====================================================
    # BODY ANGULAR VELOCITY COMPONENTS
    # =====================================================

    plt.figure(1)

    for i in range(3):

        plt.plot(
            time_min, 
            omega[:, i], 
            color=unitTestSupport.getLineColor(i, 3), 
            linewidth=2.0, 
            label=f"$\\omega_{{B/N,{labels[i]}}}$" 
        )
    
    plt.title(
        "Body Angular Velocity Components",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Angular Velocity [rad/s]",
        fontsize=12
    )

    plt.legend(loc="best")

    plt.tight_layout()

    figureList[fileName + "1"] = plt.figure(1)

    # =====================================================
    # ANGULAR VELOCITY MAGNITUDE
    # =====================================================

    plt.figure(2)

    plt.plot(
        time_min,
        omega_mag,
        linewidth=2.0,
        label=r"$|\omega|$"
    )

    plt.title(
        "Body Angular Velocity Magnitude",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Angular Velocity Magnitude [rad/s]",
        fontsize=12
    )

    plt.legend(loc="best")

    plt.tight_layout()

    figureList[fileName + "2"] = plt.figure(2)

    # =====================================================
    # RESIDUAL MAGNETIC DISTURBANCE TORQUE
    # =====================================================

    plt.figure(3)

    for i in range(3):

        plt.plot(
            time_min,
            magDistTorque[:, i],
            linewidth=2.0,
            label=f"$\\tau_{{mag,{labels[i]}}}$"
        )

    plt.title(
        "Residual Magnetic Disturbance Torque",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Residual Magnetic Disturbance Torque [N·m]",
        fontsize=12
    )

    plt.legend(loc="best")

    plt.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    plt.tight_layout()

    figureList[fileName + "3"] = plt.figure(3)

    # =====================================================
    # GRAVITY GRADIENT TORQUE
    # =====================================================

    plt.figure(4)

    for i in range(3):

        plt.plot(
            time_min,
            ggTorque[:, i],
            linewidth=2.0,
            label=f"$\\tau_{{gg,{labels[i]}}}$"
        )

    plt.title(
        "Gravity Gradient Torque",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Gravity Gradient Torque [N·m]",
        fontsize=12
    )

    plt.legend(loc="best")

    plt.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    plt.tight_layout()

    figureList[fileName + "4"] = plt.figure(4)

    # =====================================================
    # AERODYNAMIC DISTURBANCE TORQUE
    # =====================================================

    plt.figure(5)

    for i in range(3):

        plt.plot(
            time_min,
            aeroTorque[:, i],
            linewidth=2.0,
            label=f"$\\tau_{{aero,{labels[i]}}}$"
        )

    plt.title(
        "Aerodynamic Disturbance Torque",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Aerodynamic Disturbance Torque [N·m]",
        fontsize=12
    )

    plt.legend(loc="best")

    plt.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    plt.tight_layout()

    figureList[fileName + "5"] = plt.figure(5)

    # # =====================================================
    # # MAGNETORQUER MAGNETIC MOMENT
    # # =====================================================

    # plt.figure(6)

    # for i in range(3):

    #     plt.plot(
    #         mtb_times_sec,
    #         dipole[:, i],
    #         linewidth=2.0,
    #         label=f"$m_{{MTB,{labels[i]}}}$"
    #     )

    # plt.title(
    #     "Magnetorquer Commanded Magnetic Moment",
    #     fontsize=14
    # )

    # plt.xlabel(
    #     "Time [sec]",
    #     fontsize=12
    # )

    # plt.ylabel(
    #     "Magnetic Moment [A·m²]",
    #     fontsize=12
    # )

    # plt.legend(loc="best")
    # plt.tight_layout()

    # figureList[fileName + "6"] = plt.figure(6)

    # =====================================================
    # +Z AXIS VS VELOCITY VECTOR
    # =====================================================

    plt.figure(7)

    plt.plot(
        time_min,
        angle_z_vel_deg,
        linewidth=2.0,
        color="tab:blue",
        label="+Z Axis vs Velocity Vector"
    )

    plt.title(
        "Angle Between Body +Z Axis and Velocity Vector",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Angle [deg]",
        fontsize=12
    )

    plt.legend(loc="best")
    plt.tight_layout()

    figureList[fileName + "7"] = plt.figure(7)

    # =====================================================
    # +X AXIS VS NADIR VECTOR
    # =====================================================

    plt.figure(8)

    plt.plot(
        time_min,
        angle_x_nadir_deg,
        linewidth=2.0,
        color="tab:orange",
        label="+X Axis vs Nadir Vector"
    )

    plt.title(
        "Angle Between Body +X Axis and Nadir Vector",
        fontsize=14
    )

    plt.xlabel(
        "Time [min]",
        fontsize=12
    )

    plt.ylabel(
        "Angle [deg]",
        fontsize=12
    )

    plt.legend(loc="best")

    plt.tight_layout()

    figureList[fileName + "8"] = plt.figure(8)


    # # =====================================================
    # # TRUE MAGNETIC FIELD IN BODY FRAME
    # # =====================================================

    # plt.figure(9)

    # for i in range(3):

    #     plt.plot(
    #         time_min,
    #         trueMagField_B[:, i],
    #         linewidth=2.0,
    #         label=f"$B_{{true,{labels[i]}}}$"
    #     )

    # plt.title(
    #     "True Magnetic Field in Body Frame",
    #     fontsize=14
    # )

    # plt.xlabel(
    #     "Time [min]",
    #     fontsize=12
    # )

    # plt.ylabel(
    #     "Magnetic Field [T]",
    #     fontsize=12
    # )

    # plt.legend(loc="best")

    # plt.ticklabel_format(
    #     axis="y",
    #     style="sci",
    #     scilimits=(0, 0)
    # )

    # plt.tight_layout()

    # figureList[fileName + "9"] = plt.figure(9)


    # # =====================================================
    # # MAGNETOMETER MEASUREMENT
    # # =====================================================

    # plt.figure(10)

    # for i in range(3):

    #     plt.plot(
    #         time_min,
    #         tamMeasurement[:, i],
    #         linewidth=2.0,
    #         label=f"$B_{{meas,{labels[i]}}}$"
    #     )

    # plt.title(
    #     "Magnetometer Measurement",
    #     fontsize=14
    # )

    # plt.xlabel(
    #     "Time [min]",
    #     fontsize=12
    # )

    # plt.ylabel(
    #     "Magnetic Field [T]",
    #     fontsize=12
    # )

    # plt.legend(loc="best")

    # plt.ticklabel_format(
    #     axis="y",
    #     style="sci",
    #     scilimits=(0, 0)
    # )

    # plt.tight_layout()

    # figureList[fileName + "10"] = plt.figure(10)

    # =====================================================
    # SAVE FIGURES
    # =====================================================

    figure_names = {

        1: "angular_velocity_components",
        2: "angular_velocity_magnitude",
        3: "magnetic_disturbance_torque",
        4: "gravity_gradient_torque",
        5: "aerodynamic_disturbance_torque",
        # 6: "magnetorquer_magnetic_moment",
        7: "z_axis_vs_velocity",
        8: "x_axis_vs_nadir",
        # 9: "true_magnetic_field_body_frame",
        # 10: "magnetometer_measurement"
    }

    for fig_num, fig_name in figure_names.items():

        fig = plt.figure(fig_num)

        save_path = Path(output_dir) / f"{fig_name}.png"

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved figure: {save_path}")

    plt.close("all")

    return figureList