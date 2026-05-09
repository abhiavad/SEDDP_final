from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
import numpy as np


class DipoleSelector(sysModel.SysModel):

    def __init__(self):
        super().__init__()
        self.ModelTag = "DipoleSelector"

        # Inputs
        self.bdotDipoleInMsg = messaging.DipoleRequestBodyMsgReader()
        self.nadirDipoleInMsg = messaging.DipoleRequestBodyMsgReader()

        self.modeTypeInMsg = messaging.SwDataMsgReader()

        # Output
        self.dipoleOutMsg = messaging.DipoleRequestBodyMsg()

    def UpdateState(self, CurrentSimNanos):

        modeType = int(self.modeTypeInMsg().dataValue) if self.modeTypeInMsg.isWritten() else 0

        m_total = np.zeros(3, dtype=float)

        m_bdot = np.zeros(3, dtype=float)

        m_nadir = np.zeros(3, dtype=float)

        timeTag = CurrentSimNanos

        # --------------------------------------
        # Read B-dot dipole
        # --------------------------------------

        if self.bdotDipoleInMsg.isWritten():

            msg = self.bdotDipoleInMsg()

            m_bdot = np.asarray(msg.dipole_B, dtype=float ).reshape(3)

            if not np.all(np.isfinite(m_bdot)):
                m_bdot = np.zeros(3)

        # --------------------------------------
        # Mode-dependent dipole selection
        # --------------------------------------

        if modeType == 0:

            # Pure magnetic damping
            m_total = m_bdot

        elif modeType == 1:

            # Magnetic damping remains active
            # during nadir pointing.

            if (
                self.nadirDipoleInMsg.isLinked()
                and self.nadirDipoleInMsg.isWritten()
            ):

                msg = self.nadirDipoleInMsg()

                m_nadir = np.asarray(msg.dipole_B, dtype=float ).reshape(3)
                if not np.all(np.isfinite(m_nadir)):
                    m_nadir = np.zeros(3)
                    
            m_total = m_bdot + m_nadir

        if not np.all(np.isfinite(m_total)):
            m_total = np.zeros(3)

        payload = messaging.DipoleRequestBodyMsgPayload()
        payload.dipole_B = m_total.tolist()
        payload.timeTag = timeTag

        self.dipoleOutMsg.write(payload, CurrentSimNanos)