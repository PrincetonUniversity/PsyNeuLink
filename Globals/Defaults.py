#
# ********************************************  System Defaults ********************************************************
#

from enum import Enum
from Globals.TimeScale import TimeScale

# MechanismState values:
inputValueSystemDefault = [0]
outputValueSystemDefault = [0]

# TimeScale:
timeScaleSystemDefault = TimeScale.TRIAL

# Default input:
SystemDefaultInputValue = 0.0

# Default control allocation mode values:
class DefaultControlAllocationMode(Enum):
    GUMBY_MODE = 0.0
    BADGER_MODE = 1.0
# defaultControlAllocation = DefaultControlAllocationMode.BADGER_MODE.value
defaultControlAllocation = DefaultControlAllocationMode.BADGER_MODE.value

# IMPLEMENTATION NOTE:  WOULD REQUIRE A DEFAULT MECHANISM AS WELL
DEFAULT_ALLOCATION_SAMPLES = [0.0, 1.0, 0.1] # min, max, step size
