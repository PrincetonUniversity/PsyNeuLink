from enum import Enum

# Time scale modes
class TimeScale(Enum):
        REAL_TIME = 0
        TRIAL = 1
        BOUNDS = 2 # Used for type checking where TimeScale value is passed

# Central clock
class CentralClock:

    time_step = 0
    trial = 0
    block = 0
    task = 0

class CurrentTime:
    def __init__(self):
        self.time_step = CentralClock.time_step
        self.trial = CentralClock.trial
        self.block = CentralClock.block
        self.task = CentralClock.task




