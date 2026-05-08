from Basilisk.architecture import sysModel
from Basilisk.architecture import messaging

from fsw_config import (
    NS,
    NA,
    get_controller_mode_int
)


class ModeScheduler(sysModel.SysModel):

    def __init__(self):
        super().__init__()
        self.ModelTag = "ModeScheduler"

        # -----------------------------
        # CONFIG
        # -----------------------------
        self.Ns = NS
        self.Na = NA
        self.cycleLength = self.Ns + self.Na

        self.counter = 0

        # -----------------------------
        # OUTPUT MESSAGES
        # -----------------------------
        self.modeOutMsg = messaging.SwDataMsg()
        self.modeTypeOutMsg = messaging.SwDataMsg()
        self.actuateOutMsg = messaging.SwDataMsg()

    # -------------------------------------------------

    def Reset(self, CurrentSimNanos):
        self.counter = 0

        # runtime validation
        assert self.Na > 0, "Na must be > 0"
        assert self.Ns > 0, "Ns must be > 0"

        self.cycleLength = self.Ns + self.Na

    # -------------------------------------------------

    def UpdateState(self, CurrentSimNanos):

        idx = self.counter % self.cycleLength

        # -----------------------------
        # PHASE LOGIC
        # -----------------------------
        if idx < self.Ns:
            phase = 1  # sensing
            actuate = 0

        else:
            phase = 2  # actuation
            actuate = 1   # ← CRITICAL FIX: active for ALL Na steps

        # -----------------------------
        # OUTPUT: MODE
        # -----------------------------
        payload_mode = messaging.SwDataMsgPayload()
        payload_mode.dataValue = phase
        self.modeOutMsg.write(payload_mode, CurrentSimNanos)

        # -----------------------------
        # OUTPUT: CONTROLLER TYPE
        # -----------------------------
        payload_type = messaging.SwDataMsgPayload()
        payload_type.dataValue = get_controller_mode_int()
        self.modeTypeOutMsg.write(payload_type, CurrentSimNanos)

        # -----------------------------
        # OUTPUT: ACTUATION FLAG
        # -----------------------------
        payload_act = messaging.SwDataMsgPayload()
        payload_act.dataValue = actuate
        self.actuateOutMsg.write(payload_act, CurrentSimNanos)

        # -----------------------------
        # INCREMENT COUNTER
        # -----------------------------
        self.counter += 1