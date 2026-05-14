from Basilisk.architecture import sysModel
from Basilisk.architecture import messaging
import numpy as np

class DipoleQuantizer(sysModel.SysModel):
    """
    Quantizes body-frame dipole requests before dipole mapping.
    """

    def __init__(self, step_percentage=0.05, max_dipole=0.02):
        super().__init__()

        self.ModelTag = "DipoleQuantizer"

        # Inputs and Outputs
        self.dipoleInMsg = messaging.DipoleRequestBodyMsgReader()
        self.dipoleOutMsg = messaging.DipoleRequestBodyMsg()

        # Discretization parameters
        self.max_dipole = np.asarray(max_dipole, dtype=float)

        if self.max_dipole.shape != (3,):
            raise ValueError(
                "max_dipole must be length 3"
            )

        if not np.all(np.isfinite(self.max_dipole)):
            raise ValueError(
                "max_dipole contains invalid values"
            )

        if np.any(self.max_dipole <= 0.0):
            raise ValueError(
                "max_dipole must be positive"
            )

        if not np.isfinite(step_percentage):
            raise ValueError(
                "step_percentage must be finite"
            )

        if step_percentage <= 0.0:
            raise ValueError(
                "step_percentage must be positive"
            )

        self.step_size = self.max_dipole * step_percentage

        if not np.all(np.isfinite(self.step_size)):
            raise ValueError(
                "Invalid quantization step size"
            )

        if np.any(self.step_size <= 0.0):
            raise ValueError(
                "Quantization step size must be positive"
            )

    def UpdateState(self, CurrentSimNanos):

        # Pass zeros if input message unavailable
        if not self.dipoleInMsg.isWritten():

            out_msg = messaging.DipoleRequestBodyMsgPayload()
            out_msg.dipole_B = [0.0, 0.0, 0.0]

            self.dipoleOutMsg.write(
                out_msg,
                CurrentSimNanos
            )

            return

        # Read body dipole request
        in_msg = self.dipoleInMsg()

        dipole_cmds = np.asarray(
            in_msg.dipole_B,
            dtype=float
        ).reshape(3)

        # Quantization
        quantized_cmds = (
            np.sign(dipole_cmds)
            * np.floor(np.abs(dipole_cmds) / self.step_size)
            * self.step_size
        )

        # Saturation
        quantized_cmds = np.clip(
            quantized_cmds,
            -self.max_dipole,
            self.max_dipole
        )

        # Output quantized body dipole
        out_msg = messaging.DipoleRequestBodyMsgPayload()
        out_msg.dipole_B = quantized_cmds.tolist()

        self.dipoleOutMsg.write(
            out_msg,
            CurrentSimNanos
        )