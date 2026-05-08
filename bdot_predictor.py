import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from fsw_config import FSW_STEP_TIME_S, ESTIMATION_BUFFER_SIZE


class BdotPredictor(sysModel.SysModel):

    def __init__(self):
        super().__init__()
        self.dt = float(FSW_STEP_TIME_S)
        self.ModelTag = "BdotPredictor"

        self.bInMsg = messaging.TAMSensorBodyMsgReader()
        self.modeInMsg = messaging.SwDataMsgReader()

        self.bOutMsg = messaging.TAMSensorBodyMsg()
        self.bdotOutMsg = messaging.BodyHeadingMsg()

        self.prevMode = None

        # Buffers sized by sensing window
        self.B_buffer = []
        self.t_buffer = []

        self.Bdot_est = np.zeros(3)

    def Reset(self, CurrentSimNanos):
        self.dt = float(FSW_STEP_TIME_S)
        
    def UpdateState(self, CurrentSimNanos):

        if not self.bInMsg.isWritten() or not self.modeInMsg.isWritten():
            return

        mode = int(self.modeInMsg().dataValue)

        bMsg = self.bInMsg()
        B = np.asarray(bMsg.tam_B)
        # Use simulation time only (consistent with scheduler)
        t = CurrentSimNanos * 1e-9

        # --------------------------------------
        # START OF SENSING → RESET BUFFERS
        # --------------------------------------
        if mode == 1 and self.prevMode != 1:
            self.B_buffer = []
            self.t_buffer = []
            self.Bdot_est = np.zeros(3)

        # --------------------------------------
        # SENSING PHASE → COLLECT DATA
        # --------------------------------------
        if mode == 1:
            self.B_buffer.append(B.copy())
            self.t_buffer.append(t)

            if len(self.B_buffer) > ESTIMATION_BUFFER_SIZE:
                self.B_buffer.pop(0)
                self.t_buffer.pop(0)

            # Compute Bdot once sensing window is filled
            if len(self.B_buffer) == ESTIMATION_BUFFER_SIZE:

                bdot_samples = []

                for i in range(1, ESTIMATION_BUFFER_SIZE):

                    dB = self.B_buffer[i] - self.B_buffer[i - 1]
                    dt = self.t_buffer[i] - self.t_buffer[i - 1]

                    # Fallback protection
                    if dt <= 1e-9:
                        dt = self.dt

                    bdot_samples.append(dB / dt)

                self.Bdot_est = np.mean(bdot_samples, axis=0)

        # --------------------------------------
        # OUTPUT (HOLD LAST ESTIMATED VALUES)
        # --------------------------------------
        if len(self.B_buffer) > 0:
            B_out = self.B_buffer[-1]
        else:
            B_out = B

        payload = messaging.TAMSensorBodyMsgPayload()
        payload.tam_B = B_out.tolist()
        payload.timeTag = CurrentSimNanos
        self.bOutMsg.write(payload, CurrentSimNanos)

        bdot_payload = messaging.BodyHeadingMsgPayload()
        bdot_payload.rHat_XB_B = self.Bdot_est.tolist()
        self.bdotOutMsg.write(bdot_payload, CurrentSimNanos)

        self.prevMode = mode