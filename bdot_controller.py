import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel

from fsw_config import (
    BDOT_GAIN,
    MAX_DIPOLE_AM2,
    TOTAL_ACTUATION_TIME,
    OMEGA_DEADBAND_RADPS
)

class BdotController(sysModel.SysModel):

    def __init__(self, I_matrix):
        super().__init__()
        self.ModelTag = "BdotController"

        # Inputs
        self.bdotInMsg = messaging.BodyHeadingMsgReader()
        self.bInMsg = messaging.TAMSensorBodyMsgReader()
        self.actuateInMsg = messaging.SwDataMsgReader()

        # Output
        self.cmdDipoleOutMsg = messaging.DipoleRequestBodyMsg()

        # Controller parameters
        self.I = np.array(I_matrix, dtype=float)
        
        self.max_dipole = np.array(MAX_DIPOLE_AM2, dtype=float)
        self.k = BDOT_GAIN   # keeps same sign behavior as before

    def UpdateState(self, CurrentSimNanos):

        # --------------------------
        # ACTUATION TRIGGER
        # --------------------------
        actuate = int(self.actuateInMsg().dataValue) if self.actuateInMsg.isWritten() else 0

        if actuate != 1:
            payload = messaging.DipoleRequestBodyMsgPayload()
            payload.dipole_B = [0.0, 0.0, 0.0]
            self.cmdDipoleOutMsg.write(payload, CurrentSimNanos)
            return

        # --------------------------
        # Validate inputs
        # --------------------------
        if not self.bInMsg.isWritten() or not self.bdotInMsg.isWritten():
            payload = messaging.DipoleRequestBodyMsgPayload()
            payload.dipole_B = [0.0, 0.0, 0.0]
            self.cmdDipoleOutMsg.write(payload, CurrentSimNanos)
            return

        # --------------------------
        # Read messages
        # --------------------------
        bMsg = self.bInMsg()
        bdotMsg = self.bdotInMsg()

        B = np.asarray(bMsg.tam_B, dtype=float)
        Bdot = np.asarray(bdotMsg.rHat_XB_B, dtype=float)
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
            omega_perp = - np.cross(B, Bdot)/ B_norm_sq

            if np.all(np.abs(omega_perp) <= OMEGA_DEADBAND_RADPS):
                m = np.zeros(3)
            
            else:

                L_perp = self.I @ omega_perp

                m = -self.k * np.cross(B, L_perp) / B_norm_sq

                # time scaling
                m /= TOTAL_ACTUATION_TIME

                # saturation
                ratios = np.abs(m) / self.max_dipole
                max_ratio = np.max(ratios)

                if max_ratio > 1.0:
                    m = m / max_ratio

        # --------------------------
        # Output
        # --------------------------
        payload = messaging.DipoleRequestBodyMsgPayload()
        payload.dipole_B = m.tolist()
        self.cmdDipoleOutMsg.write(payload, CurrentSimNanos)