import numpy as np

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from Basilisk.utilities import RigidBodyKinematics as rbk


class ResidualMagneticDipoleTorque(sysModel.SysModel):

    def __init__(self, m_residual_B):
        super().__init__()
        self.ModelTag = "magDisturbance"

        self.magInMsg = messaging.MagneticFieldMsgReader()
        self.attNavInMsg = messaging.NavAttMsgReader()

        self.m_residual = np.asarray(m_residual_B).reshape(3,1)

        self.disturbanceTorqueOutMsg = messaging.CmdTorqueBodyMsg()

    def UpdateState(self, CurrentSimNanos):

        mag = self.magInMsg()
        nav = self.attNavInMsg()

        B_N = np.asarray(mag.magField_N).reshape(3,1)

        sigma_BN = np.asarray(nav.sigma_BN)
        C_BN = rbk.MRP2C(sigma_BN)

        B_B = C_BN @ B_N

        tau = np.cross(
            self.m_residual.flatten(),
            B_B.flatten()
        ).reshape(3,1)

        payload = messaging.CmdTorqueBodyMsgPayload()
        payload.torqueRequestBody = tau.flatten().tolist()
        # Timestamp for disturbance torque
        payload.timeTag = CurrentSimNanos

        self.disturbanceTorqueOutMsg.write(payload, CurrentSimNanos)