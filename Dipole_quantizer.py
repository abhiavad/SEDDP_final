from Basilisk.architecture import sysModel
from Basilisk.architecture import messaging
import numpy as np

class DipoleQuantizer(sysModel.SysModel):
    """
    Intercepts continuous MTB commands and quantizes them into discrete steps.
    """
    def __init__(self, step_percentage=0.05, max_dipole=0.02):
        super().__init__()
        
        self.ModelTag = "DipoleQuantizer"
        # Inputs and Outputs
        self.mtbCmdInMsg = messaging.MTBCmdMsgReader()
        self.mtbCmdOutMsg = messaging.MTBCmdMsg()
        
        # Discretization parameters
        self.max_dipole = np.asarray(max_dipole, dtype = float)

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
        # Pass zeros if the input message hasn't been written yet
        if not self.mtbCmdInMsg.isWritten():
            return
            
        in_msg = self.mtbCmdInMsg()
        dipole_cmds = np.asarray(in_msg.mtbDipoleCmds, dtype=float)
        if not np.all(np.isfinite(dipole_cmds)):
            dipole_cmds = np.zeros_like(dipole_cmds)

        # Ensure shape is (N,3)
        dipole_cmds = dipole_cmds.reshape(-1, 3)

        quantized_cmds = np.round(dipole_cmds / self.step_size) * self.step_size
        quantized_cmds = np.clip(quantized_cmds, -self.max_dipole, self.max_dipole)

        # Flatten back to original Basilisk format
        quantized_cmds = quantized_cmds.reshape(-1)
        # Write the quantized commands to the output message
        out_msg = messaging.MTBCmdMsgPayload()
        out_msg.mtbDipoleCmds = quantized_cmds.tolist()
        
        self.mtbCmdOutMsg.write(out_msg, CurrentSimNanos)
