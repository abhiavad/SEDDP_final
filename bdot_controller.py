import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from magnetic_control_shared import (
    write_zero_dipole,
)

from fsw_config import (
    OMEGA_DEADBAND_RADPS,
    TOTAL_ACTUATION_TIME
)

class BdotController(sysModel.SysModel):

    def __init__(self, I_matrix, baseGain):
        super().__init__()
        self.ModelTag = "BdotController"

        # Inputs
        self.bdotInMsg = messaging.BodyHeadingMsgReader()
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
        self.gain = float(baseGain)

    def UpdateState(self, CurrentSimNanos):

        # --------------------------
        # ACTUATION TRIGGER
        # --------------------------
        actuate = int(self.actuateInMsg().dataValue) if self.actuateInMsg.isWritten() else 0

        if actuate != 1:
            write_zero_dipole(
                self.cmdDipoleOutMsg,
                CurrentSimNanos
            )
            return

        # --------------------------
        # Read messages
        # --------------------------
        bMsg = self.bInMsg()
        bdotMsg = self.bdotInMsg()

        B = np.asarray(bMsg.tam_B, dtype=float).reshape(3)
        Bdot = np.asarray(bdotMsg.rHat_XB_B, dtype=float).reshape(3)


        B_norm_sq = np.dot(B, B)

        # --------------------------
        # Control law
        # --------------------------

        if B_norm_sq < 1e-12:
            m = np.zeros(3)

        else:

            B_norm = np.sqrt(B_norm_sq)

            B_hat = B / B_norm

            S_Bhat = np.array([
                [0.0,         -B_hat[2],  B_hat[1]],
                [B_hat[2],     0.0,      -B_hat[0]],
                [-B_hat[1],    B_hat[0],  0.0]
            ])

            tmp1 = S_Bhat @ Bdot

            tmp2 = self.I @ tmp1

            m = (
                self.gain
                * (S_Bhat @ tmp2)
                / B_norm_sq
            )
            m /= TOTAL_ACTUATION_TIME
                
        # --------------------------
        # Output
        # --------------------------
        payload = messaging.DipoleRequestBodyMsgPayload()
        payload.dipole_B = m.tolist()
        self.cmdDipoleOutMsg.write(payload, CurrentSimNanos)