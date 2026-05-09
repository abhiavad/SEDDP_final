import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from magnetic_control_shared import (
    write_zero_dipole,
)

from fsw_config import (
    OMEGA_DEADBAND_RADPS
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
        # Validate inputs
        # --------------------------
        if not self.bInMsg.isWritten() or not self.bdotInMsg.isWritten():
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
        if (
            B_norm_sq < 1e-12
            or not np.all(np.isfinite(B))
            or not np.all(np.isfinite(Bdot))
        ):
            m = np.zeros(3)
        else:
            omega_perp = -np.cross(B, Bdot) / B_norm_sq

            if not np.all(np.isfinite(omega_perp)):
                omega_perp = np.zeros(3)

            if np.all(np.abs(omega_perp) <= OMEGA_DEADBAND_RADPS):
                m = np.zeros(3)
            
            else:

                L_perp = self.I @ omega_perp

                m = -self.gain * np.cross(B, L_perp) / B_norm_sq
                

        if not np.all(np.isfinite(m)):
            m = np.zeros(3)
        # --------------------------
        # Output
        # --------------------------
        payload = messaging.DipoleRequestBodyMsgPayload()
        payload.dipole_B = m.tolist()
        self.cmdDipoleOutMsg.write(payload, CurrentSimNanos)