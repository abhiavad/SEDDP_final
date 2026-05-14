import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from fsw_config import FSW_STEP_TIME_S, ESTIMATION_BUFFER_SIZE, TOTAL_ACTUATION_TIME

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

        self.Bdot_est = np.zeros(3,dtype=np.float64)

    def Reset(self, CurrentSimNanos):

        self.dt = np.float64(FSW_STEP_TIME_S)
        self.prevMode = None
        self.B_buffer = []
        self.t_buffer = []
        self.Bdot_est = np.zeros(3,dtype=np.float64)
        
    def UpdateState(self, CurrentSimNanos):

        if not self.bInMsg.isWritten() or not self.modeInMsg.isWritten():
            return

        mode = int(self.modeInMsg().dataValue)

        bMsg = self.bInMsg()

        B = np.asarray(bMsg.tam_B, dtype=np.float64).reshape(3)

        if not np.all(np.isfinite(B)):
            return
        # Use simulation time only (consistent with scheduler)
        t = np.float64(CurrentSimNanos) * np.float64(1e-9)

        # --------------------------------------
        # START OF SENSING → RESET BUFFERS
        # --------------------------------------
        if mode == 1 and self.prevMode != 1:
            self.B_buffer = []
            self.t_buffer = []
            self.Bdot_est = np.zeros(3,dtype=np.float64)

        # --------------------------------------
        # SENSING PHASE → COLLECT DATA
        # --------------------------------------
        if mode == 1:
            self.B_buffer.append(np.asarray(B.copy(), dtype=np.float64))
            self.t_buffer.append(t)

            if len(self.B_buffer) > ESTIMATION_BUFFER_SIZE:
                self.B_buffer.pop(0)
                self.t_buffer.pop(0)

            # Compute Bdot once sensing window is filled
            if len(self.B_buffer) == ESTIMATION_BUFFER_SIZE:

                # --------------------------------------
                # Linear least-squares Bdot estimate
                # --------------------------------------

                t_arr = np.asarray(
                    self.t_buffer,
                    dtype=np.float64
                )

                B_arr = np.asarray(
                    self.B_buffer,
                    dtype=np.float64
                )

                # Center time axis for numerical conditioning
                t_mean = np.mean(t_arr)

                t_centered = t_arr - t_mean

                den = np.dot(
                    t_centered,
                    t_centered
                )

                # Least-squares slope estimate
                self.Bdot_est = (
                    t_centered @ B_arr
                ) / den

                # Final finite check
                if not np.all(np.isfinite(self.Bdot_est)):

                    self.Bdot_est = np.zeros(
                        3,
                        dtype=np.float64
                    )

        # --------------------------------------
        # OUTPUT PREDICTED FIELD
        # --------------------------------------

        if len(self.B_buffer) == ESTIMATION_BUFFER_SIZE:

            # Average sensed field
            B_avg = np.mean(
                np.asarray(
                    self.B_buffer,
                    dtype=np.float64
                ),
                axis=0,
                dtype=np.float64
            )

            # ----------------------------------
            # Predict toward midpoint of
            # actuation interval
            # ----------------------------------

            prediction_time = (
                0.5 * TOTAL_ACTUATION_TIME
                + 0.5 * ESTIMATION_BUFFER_SIZE * self.dt
            )

            B_out = (
                B_avg
                + self.Bdot_est * prediction_time
            )

            # ----------------------------------
            # Preserve physical field magnitude
            # ----------------------------------

            B_mag = np.linalg.norm(B_avg)

            B_out = (
                B_mag
                * B_out
                / np.linalg.norm(B_out)
            )

            # Safety fallback
            if not np.all(np.isfinite(B_out)):

                B_out = np.asarray(
                    self.B_buffer[-1],
                    dtype=np.float64
                )

        elif len(self.B_buffer) > 0:

            B_out = np.asarray(
                self.B_buffer[-1],
                dtype=np.float64
            )

        else:

            B_out = B

        payload = messaging.TAMSensorBodyMsgPayload()
        payload.tam_B = (np.asarray(B_out,dtype=np.float64).tolist())
        payload.timeTag = CurrentSimNanos
        self.bOutMsg.write(payload, CurrentSimNanos)

        bdot_payload = messaging.BodyHeadingMsgPayload()
        bdot_payload.rHat_XB_B = (np.asarray(self.Bdot_est,dtype=np.float64).tolist())
        self.bdotOutMsg.write(bdot_payload, CurrentSimNanos)

        self.prevMode = mode