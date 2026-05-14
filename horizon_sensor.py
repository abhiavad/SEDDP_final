import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from Basilisk.utilities import RigidBodyKinematics as rbk

class HorizonSensor(sysModel.SysModel):
    """
    Horizon sensor (minimal)
    Output: nadir_B (unit vector in body frame)
    """

    def __init__(self):
        super().__init__()
        self.ModelTag = "HorizonSensor"

        # Inputs
        self.stateInMsg = messaging.SCStatesMsgReader()
        self.attInMsg   = messaging.NavAttMsgReader()

        # Output
        self.nadirOutMsg = messaging.BodyHeadingMsg()

        # Noise parameters
        self.useNoise = False
        self.bias = np.zeros(3)
        self.senNoiseStd = np.zeros(3) # Represents the MAX angular error in radians

    def UpdateState(self, CurrentSimNanos):
        if not self.stateInMsg.isWritten() or not self.attInMsg.isWritten():
            return

        state = self.stateInMsg()
        att   = self.attInMsg()

        # Get position and normalize for Nadir (N-frame)

        r_N = np.asarray(state.r_BN_N, dtype=float).reshape(3)

        sigma_BN = np.asarray(att.sigma_BN, dtype=float).reshape(3)

        if (
            not np.all(np.isfinite(r_N))
            or not np.all(np.isfinite(sigma_BN))
        ):
            return

        r_norm = np.linalg.norm(r_N)

        if r_norm < 1e-10:
            return

        nadir_N = -r_N / r_norm

        C_BN = rbk.MRP2C(sigma_BN)
        nadir_B = (C_BN @ nadir_N).flatten()

        # Apply angular noise
        if self.useNoise:
            # Sample from a uniform distribution for a hard "at most 0.5 deg" limit
            roll_err = np.random.uniform(-self.senNoiseStd[0], self.senNoiseStd[0]) + self.bias[0]
            pitch_err = np.random.uniform(-self.senNoiseStd[1], self.senNoiseStd[1]) + self.bias[1]
            yaw_err = np.random.uniform(-self.senNoiseStd[2], self.senNoiseStd[2]) + self.bias[2]

            # Construct true small-angle rotation

            C_err = rbk.euler1232C(np.array([roll_err, pitch_err, yaw_err]))
            # Rotate true vector into noisy measurement
            nadir_B = C_err @ nadir_B
            if not np.all(np.isfinite(nadir_B)):
                return

        # Final normalization to ensure it remains a valid unit direction vector
        norm = np.linalg.norm(nadir_B)
        if norm < 1e-10:
            return

        nadir_B = nadir_B / norm

        payload = messaging.BodyHeadingMsgPayload()
        payload.rHat_XB_B = nadir_B.tolist() 
        payload.timeTag = CurrentSimNanos
        
        self.nadirOutMsg.write(payload, CurrentSimNanos)