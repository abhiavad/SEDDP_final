import numpy as np

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel

from magnetic_control_shared import (
    write_zero_dipole,
)

from fsw_config import (
    KP_NADIR,
    RECOVERY_ENTER_ANGLE_DEG,
    RECOVERY_EXIT_ANGLE_DEG,
    NADIR_POINTING_DEADBAND_DEG,
)


class NadirPointingController(sysModel.SysModel):

    def __init__(self, I_matrix, baseGain):

        super().__init__()

        self.ModelTag = "NadirPointingController"

        # Inputs
        self.nadirInMsg = messaging.BodyHeadingMsgReader()
        self.bInMsg = messaging.TAMSensorBodyMsgReader()
        self.actuateInMsg = messaging.SwDataMsgReader()

        # Output
        self.cmdDipoleOutMsg = messaging.DipoleRequestBodyMsg()

        # Controller parameters
        self.I = np.asarray(I_matrix, dtype=float)

        if self.I.shape != (3, 3):
            raise ValueError(
                "I_matrix must be 3x3"
            )

        if not np.all(np.isfinite(self.I)):
            raise ValueError(
                "I_matrix contains invalid values"
            )

        self.Iyy = float(self.I[1, 1])
        self.Izz = float(self.I[2, 2])

        self.gain = float(baseGain)

        self.recoveryEnterCos = np.cos(
            np.deg2rad(
                RECOVERY_ENTER_ANGLE_DEG
            )
        )
        self.deadbandAngleRad = np.deg2rad(
            NADIR_POINTING_DEADBAND_DEG
        )

        self.recoveryExitCos = np.cos(
            np.deg2rad(
                RECOVERY_EXIT_ANGLE_DEG
            )
        )

        self.inRecoveryMode = False

    def Reset(self, CurrentSimNanos):

        self.inRecoveryMode = False

    def UpdateState(self, CurrentSimNanos):

        # --------------------------
        # ACTUATION TRIGGER
        # --------------------------

        actuate = (
            int(self.actuateInMsg().dataValue)
            if self.actuateInMsg.isWritten()
            else 0
        )

        if actuate != 1:

            write_zero_dipole(
                self.cmdDipoleOutMsg,
                CurrentSimNanos
            )

            return

        # --------------------------
        # Validate inputs
        # --------------------------

        if (
            not self.bInMsg.isWritten()
            or not self.nadirInMsg.isWritten()
        ):

            write_zero_dipole(
                self.cmdDipoleOutMsg,
                CurrentSimNanos
            )

            return

        # --------------------------
        # Read messages
        # --------------------------

        bMsg = self.bInMsg()

        nadirMsg = self.nadirInMsg()

        B = np.asarray(
            bMsg.tam_B,
            dtype=float
        ).reshape(3)

        nadir_B = np.asarray(
            nadirMsg.rHat_XB_B,
            dtype=float
        ).reshape(3)

        B_norm_sq = np.dot(B, B)

        # --------------------------
        # Validate vectors
        # --------------------------

        if (
            B_norm_sq < 1e-12
        ):

            m = np.zeros(3)

        else:

            nadir_norm = np.linalg.norm(
                nadir_B
            )

            if nadir_norm < 1e-12:

                m = np.zeros(3)

            else:

                nadir_B = nadir_B / nadir_norm

                # ----------------------------------
                # Alignment metric
                #
                # +X should align with nadir
                #
                # dot([1,0,0], nadir_B)
                # = nadir_B[0]
                # ----------------------------------

                x_alignment = np.clip(
                    nadir_B[0],
                    -1.0,
                    1.0
                )

                total_angle_error = np.arctan2(
                    np.linalg.norm(nadir_B[1:]),
                    nadir_B[0]
                )

                # ----------------------------------
                # Recovery-mode hysteresis
                # ----------------------------------

                if self.inRecoveryMode:

                    if x_alignment > self.recoveryExitCos:
                        self.inRecoveryMode = False

                else:

                    if x_alignment < self.recoveryEnterCos:
                        self.inRecoveryMode = True

                # ----------------------------------
                # Desired control torque
                # ----------------------------------

                tau_cmd = np.zeros(3)
                # ----------------------------------
                # Angular deadband
                #
                # Disable nadir steering if +X
                # already sufficiently aligned
                # with nadir.
                # ----------------------------------

                if total_angle_error < self.deadbandAngleRad:

                    m = np.zeros(3)

                else:

                    # ----------------------------------
                    # RECOVERY MODE
                    #
                    # Only rotate about body Z
                    # until +X faces Earth again.
                    # ----------------------------------

                    if self.inRecoveryMode:

                        heading_error = np.arctan2(
                            nadir_B[1],
                            nadir_B[0]
                        )

                        tau_cmd[2] = (
                            self.gain
                            * self.Izz
                            * heading_error
                        )

                    else:

                        y_angle = -np.arctan2(
                            nadir_B[2],
                            nadir_B[0]
                        )

                        z_angle = np.arctan2(
                            nadir_B[1],
                            nadir_B[0]
                        )

                        tau_cmd[1] = (
                            self.gain
                            * self.Iyy
                            * y_angle
                        )

                        tau_cmd[2] = (
                            self.gain
                            * self.Izz
                            * z_angle
                        )
                    # ----------------------------------
                    # Magnetic inversion
                    #
                    # tau = m x B
                    #
                    # minimum-norm solution:
                    #
                    # m = (B x tau) / |B|²
                    # ----------------------------------

                    m = np.cross(
                        B,
                        tau_cmd
                    ) / B_norm_sq


        # --------------------------
        # Output
        # --------------------------

        payload = messaging.DipoleRequestBodyMsgPayload()

        payload.dipole_B = m.tolist()

        self.cmdDipoleOutMsg.write(
            payload,
            CurrentSimNanos
        )