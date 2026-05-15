import numpy as np

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel

from magnetic_control_shared import (
    write_zero_dipole,
    saturate_dipole_uniform,
)

from fsw_config import (
    MAX_DIPOLE_AM2,
    TOTAL_ACTUATION_TIME,
)


class DipoleConditioner(sysModel.SysModel):

    def __init__(self):
        super().__init__()

        self.ModelTag = "DipoleConditioner"

        # --------------------------------------------------
        # Inputs
        # --------------------------------------------------

        self.dipoleInMsg = messaging.DipoleRequestBodyMsgReader()

        # --------------------------------------------------
        # Outputs
        # --------------------------------------------------

        self.dipoleOutMsg = messaging.DipoleRequestBodyMsg()

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.max_dipole = np.asarray(MAX_DIPOLE_AM2, dtype=float)

        if self.max_dipole.shape != (3,):
            raise ValueError(
                "MAX_DIPOLE_AM2 must be length 3"
            )

        if not np.all(np.isfinite(self.max_dipole)):
            raise ValueError(
                "MAX_DIPOLE_AM2 contains invalid values"
            )

        if np.any(self.max_dipole <= 0.0):
            raise ValueError(
                "MAX_DIPOLE_AM2 must be positive"
            )

        if TOTAL_ACTUATION_TIME <= 0.0:
            raise ValueError(
                "TOTAL_ACTUATION_TIME must be positive"
            )

    def UpdateState(self, CurrentSimNanos):

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if not self.dipoleInMsg.isWritten():

            write_zero_dipole(
                self.dipoleOutMsg,
                CurrentSimNanos
            )

            return

        # --------------------------------------------------
        # Read raw dipole
        # --------------------------------------------------

        msg = self.dipoleInMsg()

        m = np.asarray(msg.dipole_B, dtype=float).reshape(3)

        # --------------------------------------------------
        # Validate dipole
        # --------------------------------------------------

        if not np.all(np.isfinite(m)):

            write_zero_dipole(
                self.dipoleOutMsg,
                CurrentSimNanos
            )

            return

        # --------------------------------------------------
        # Shared actuator conditioning
        # --------------------------------------------------
    
        # shared saturation
        m = saturate_dipole_uniform(m, self.max_dipole)
        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        payload = messaging.DipoleRequestBodyMsgPayload()

        payload.dipole_B = m.tolist()

        self.dipoleOutMsg.write(
            payload,
            CurrentSimNanos
        )