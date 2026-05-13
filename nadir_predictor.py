import numpy as np

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel

from fsw_config import (
    FSW_STEP_TIME_S,
    ESTIMATION_BUFFER_SIZE,
    TOTAL_ACTUATION_TIME
)


class NadirPredictor(sysModel.SysModel):

    def __init__(self):

        super().__init__()

        self.dt = float(FSW_STEP_TIME_S)

        self.ModelTag = "NadirPredictor"

        # -------------------------------------------------
        # Inputs
        # -------------------------------------------------

        self.nadirInMsg = messaging.BodyHeadingMsgReader()

        self.modeInMsg = messaging.SwDataMsgReader()

        # -------------------------------------------------
        # Outputs
        # -------------------------------------------------

        self.nadirOutMsg = messaging.BodyHeadingMsg()

        # -------------------------------------------------
        # Internal state
        # -------------------------------------------------

        self.prevMode = None

        self.nadir_buffer = []

        self.t_buffer = []

        self.nadirRate_est = np.zeros(
            3,
            dtype=np.float64
        )

    # -----------------------------------------------------

    def Reset(self, CurrentSimNanos):

        self.dt = np.float64(FSW_STEP_TIME_S)

        self.prevMode = None

        self.nadir_buffer = []

        self.t_buffer = []

        self.nadirRate_est = np.zeros(
            3,
            dtype=np.float64
        )

    # -----------------------------------------------------

    def UpdateState(self, CurrentSimNanos):

        # -------------------------------------------------
        # Validate message availability
        # -------------------------------------------------

        if (
            not self.nadirInMsg.isWritten()
            or not self.modeInMsg.isWritten()
        ):
            return

        # -------------------------------------------------
        # Read inputs
        # -------------------------------------------------

        mode = int(
            self.modeInMsg().dataValue
        )

        nadirMsg = self.nadirInMsg()

        nadir_B = np.asarray(
            nadirMsg.rHat_XB_B,
            dtype=np.float64
        ).reshape(3)

        # -------------------------------------------------
        # Validate vector
        # -------------------------------------------------

        if not np.all(np.isfinite(nadir_B)):
            return

        norm = np.linalg.norm(nadir_B)

        if norm < 1e-12:
            return

        # -------------------------------------------------
        # Enforce unit-vector geometry
        # BEFORE buffering
        # -------------------------------------------------

        nadir_B = nadir_B / norm

        # -------------------------------------------------
        # Simulation time
        # -------------------------------------------------

        t = (
            np.float64(CurrentSimNanos)
            * np.float64(1e-9)
        )

        # -------------------------------------------------
        # START OF SENSING → RESET BUFFERS
        # -------------------------------------------------

        if mode == 1 and self.prevMode != 1:

            self.nadir_buffer = []

            self.t_buffer = []

            self.nadirRate_est = np.zeros(
                3,
                dtype=np.float64
            )

        # -------------------------------------------------
        # SENSING PHASE
        # -------------------------------------------------

        if mode == 1:

            self.nadir_buffer.append(
                np.asarray(
                    nadir_B.copy(),
                    dtype=np.float64
                )
            )

            self.t_buffer.append(t)

            # ---------------------------------------------
            # Maintain fixed buffer size
            # ---------------------------------------------

            if len(self.nadir_buffer) > ESTIMATION_BUFFER_SIZE:

                self.nadir_buffer.pop(0)

                self.t_buffer.pop(0)

            # ---------------------------------------------
            # Compute least-squares direction-rate estimate
            # ---------------------------------------------

            if len(self.nadir_buffer) == ESTIMATION_BUFFER_SIZE:

                t_arr = np.asarray(
                    self.t_buffer,
                    dtype=np.float64
                )

                nadir_arr = np.asarray(
                    self.nadir_buffer,
                    dtype=np.float64
                )

                # -----------------------------------------
                # Center time axis
                # -----------------------------------------

                t_mean = np.mean(t_arr)

                t_centered = (
                    t_arr - t_mean
                )

                den = np.dot(
                    t_centered,
                    t_centered
                )

                # -----------------------------------------
                # Least-squares slope estimate
                # -----------------------------------------

                self.nadirRate_est = (
                    t_centered @ nadir_arr
                ) / den

                # -----------------------------------------
                # Final finite check
                # -----------------------------------------

                if not np.all(
                    np.isfinite(
                        self.nadirRate_est
                    )
                ):

                    self.nadirRate_est = np.zeros(
                        3,
                        dtype=np.float64
                    )

        # -------------------------------------------------
        # OUTPUT PREDICTED NADIR VECTOR
        # -------------------------------------------------

        if len(self.nadir_buffer) == ESTIMATION_BUFFER_SIZE:

            # ---------------------------------------------
            # Average sensed nadir vector
            # ---------------------------------------------

            nadir_avg = np.mean(
                np.asarray(
                    self.nadir_buffer,
                    dtype=np.float64
                ),
                axis=0,
                dtype=np.float64
            )

            # ---------------------------------------------
            # Predict toward midpoint of
            # actuation interval
            # ---------------------------------------------

            prediction_time = (
                0.5 * TOTAL_ACTUATION_TIME
                + 0.5 * ESTIMATION_BUFFER_SIZE * self.dt
            )

            nadir_out = (
                nadir_avg
                + self.nadirRate_est
                * prediction_time
            )

            # ---------------------------------------------
            # Re-normalize to preserve valid geometry
            # ---------------------------------------------

            norm_out = np.linalg.norm(
                nadir_out
            )

            if norm_out > 1e-12:

                nadir_out = (
                    nadir_out / norm_out
                )

            else:

                nadir_out = np.asarray(
                    self.nadir_buffer[-1],
                    dtype=np.float64
                )

            # ---------------------------------------------
            # Final safety fallback
            # ---------------------------------------------

            if not np.all(
                np.isfinite(nadir_out)
            ):

                nadir_out = np.asarray(
                    self.nadir_buffer[-1],
                    dtype=np.float64
                )

        elif len(self.nadir_buffer) > 0:

            nadir_out = np.asarray(
                self.nadir_buffer[-1],
                dtype=np.float64
            )

        else:

            nadir_out = nadir_B

        # -------------------------------------------------
        # Output message
        # -------------------------------------------------

        payload = messaging.BodyHeadingMsgPayload()

        payload.rHat_XB_B = (
            np.asarray(
                nadir_out,
                dtype=np.float64
            ).tolist()
        )

        payload.timeTag = CurrentSimNanos

        self.nadirOutMsg.write(
            payload,
            CurrentSimNanos
        )

        self.prevMode = mode