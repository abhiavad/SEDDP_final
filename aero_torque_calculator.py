"""
Aerodynamic drag torque calculator.

FINAL AERODYNAMIC PHILOSOPHY
----------------------------
- Uses explicit MATLAB-style spacecraft topology.
- Aerodynamic surfaces come from geometry_utils.py.
- Each aerodynamic facet explicitly contains:
    * area
    * drag coefficient
    * normal
    * center location

IMPORTANT:
- Torque arms are computed relative to TOTAL spacecraft CG.
- Total spacecraft CG comes from mass_properties.py.
- This automatically captures:
    * deployment-angle-dependent torque arms
    * panel deployment effects
    * shifted spacecraft CG

BASILISK ARCHITECTURE
---------------------
This preserves the original Basilisk disturbance architecture:
- SysModel disturbance module
- message readers/writers
- update-loop structure

Only the geometry source philosophy changed.
"""

import numpy as np

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel

from Basilisk.utilities import (
    RigidBodyKinematics as rbk,
)

from mass_properties import cg_B_m


class AeroTorqueCalculator(sysModel.SysModel):

    def __init__(self, facets):

        super().__init__()

        self.ModelTag = "AeroTorqueCalc"

        # ==================================================
        # INPUT MESSAGES
        # ==================================================

        self.atmoInMsg = messaging.AtmoPropsMsgReader()

        self.stateInMsg = messaging.SCStatesMsgReader()

        self.attInMsg = messaging.NavAttMsgReader()

        # ==================================================
        # GEOMETRY
        # ==================================================

        """
        facets:
        Explicit aerodynamic surfaces from geometry_utils.py

        Each facet contains:
        - area
        - Cd
        - normal
        - location
        """

        self.facets = facets

        # ==================================================
        # OUTPUT MESSAGE
        # ==================================================

        self.torqueOutMsg = messaging.CmdTorqueBodyMsg()

    # ======================================================
    # UPDATE
    # ======================================================

    def UpdateState(self, CurrentSimNanos):

        # ==================================================
        # READ INPUTS
        # ==================================================

        atmo = self.atmoInMsg()

        state = self.stateInMsg()

        att = self.attInMsg()

        # ==================================================
        # ATMOSPHERIC DENSITY
        # ==================================================

        rho = float(atmo.neutralDensity)

        # ==================================================
        # INERTIAL VELOCITY
        # ==================================================

        v_N = np.asarray(
            state.v_BN_N,
            dtype=float,
        ).reshape(3)

        # ==================================================
        # ATTITUDE
        # ==================================================

        sigma_BN = np.asarray(
            att.sigma_BN,
            dtype=float,
        )

        C_BN = rbk.MRP2C(sigma_BN)

        # ==================================================
        # BODY-FRAME VELOCITY
        # ==================================================

        v_B = C_BN @ v_N

        v_norm = np.linalg.norm(v_B)

        # ==================================================
        # ZERO-VELOCITY SAFETY
        # ==================================================

        if v_norm < 1e-12:

            payload = messaging.CmdTorqueBodyMsgPayload()

            payload.torqueRequestBody = [0.0, 0.0, 0.0]

            self.torqueOutMsg.write(
                payload,
                CurrentSimNanos,
            )

            return

        # ==================================================
        # FLOW DIRECTION
        # ==================================================

        v_hat_B = v_B / v_norm

        # ==================================================
        # TOTAL TORQUE
        # ==================================================

        tau_total_B = np.zeros(3)

        # ==================================================
        # FACET LOOP
        # ==================================================

        for facet in self.facets:

            # ----------------------------------------------
            # FACET PROPERTIES
            # ----------------------------------------------

            area_m2 = float(facet["area"])

            Cd = float(facet["Cd"])

            normal_B = np.asarray(
                facet["normal"],
                dtype=float,
            )

            center_B_m = np.asarray(
                facet["location"],
                dtype=float,
            )

            # ----------------------------------------------
            # TORQUE ARM
            # ----------------------------------------------

            """
            IMPORTANT:
            Torque arm is measured from TOTAL spacecraft CG,
            not body-frame origin.

            This automatically captures:
            - panel deployment effects
            - shifted CG effects
            """

            r_B_m = center_B_m - cg_B_m

            # ----------------------------------------------
            # FACET VISIBILITY
            # ----------------------------------------------

            cos_theta = np.dot(
                normal_B,
                v_hat_B,
            )


            # ----------------------------------------------
            # SKIP BACK-FACING SURFACES
            # ----------------------------------------------

            if cos_theta <= 0.0:
                continue

            # ----------------------------------------------
            # DYNAMIC PRESSURE
            # ----------------------------------------------

            q = 0.5 * rho * v_norm**2

            # ----------------------------------------------
            # DRAG FORCE MAGNITUDE
            # ----------------------------------------------

            """
            Simple flat-plate drag model.

            Effective projected area:
            A_proj = A * cos(theta)
            """

            F_drag_mag = (
                q
                *
                Cd
                *
                area_m2
                *
                cos_theta
            )

            # ----------------------------------------------
            # DRAG FORCE VECTOR
            # ----------------------------------------------

            F_drag_B = (
                F_drag_mag
                *
                (-v_hat_B)
            )

            # ----------------------------------------------
            # DRAG TORQUE
            # ----------------------------------------------

            tau_drag_B = np.cross(
                r_B_m,
                F_drag_B,
            )

            # ----------------------------------------------
            # TOTAL TORQUE
            # ----------------------------------------------

            tau_total_B += tau_drag_B

        # ==================================================
        # OUTPUT MESSAGE
        # ==================================================

        payload = messaging.CmdTorqueBodyMsgPayload()

        payload.torqueRequestBody = (
            tau_total_B.tolist()
        )

        self.torqueOutMsg.write(
            payload,
            CurrentSimNanos,
        )