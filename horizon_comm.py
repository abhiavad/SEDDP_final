import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel

class HorizonComm(sysModel.SysModel):
    def __init__(self):
        super().__init__()
        self.ModelTag = "horizonComm"

        # FIX 1: Use BodyHeadingMsgReader
        self.nadirInMsg = messaging.BodyHeadingMsgReader()
        # FIX 2: Use BodyHeadingMsg
        self.nadirOutMsg = messaging.BodyHeadingMsg()

    def UpdateState(self, CurrentSimNanos):

        # If no valid input, do NOT output anything
        # This avoids fake data and invalid timestamps
        if not self.nadirInMsg.isWritten():
            return

        nadir = self.nadirInMsg()
        nadir_B = np.asarray(nadir.rHat_XB_B, dtype=float).reshape(3)
        if not np.all(np.isfinite(nadir_B)):
            return

        # Normalize safely
        norm = np.linalg.norm(nadir_B)
        if norm > 1e-10:
            nadir_B = nadir_B / norm
            if not np.all(np.isfinite(nadir_B)):
                return
        else:
            return   # invalid vector → skip

        payload = messaging.BodyHeadingMsgPayload()
        payload.rHat_XB_B = nadir_B.tolist()
        # --- TIMESTAMP HANDLING ---
        # Use sensor timestamp if available, else fallback to current time
        payload.timeTag = CurrentSimNanos

        self.nadirOutMsg.write(payload, CurrentSimNanos)