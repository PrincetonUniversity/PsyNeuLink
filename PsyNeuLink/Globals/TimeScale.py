# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
from enum import Enum
from PsyNeuLink.Globals.Keywords import CENTRAL_CLOCK

# Time scale modes
class TimeScale(Enum):
    """Values used to specify ``time_scale`` argument for mechanisms, processes, and systems.
    """
    TIME_STEP = 0
    PASS = 1
    TRIAL = 2
    RUN = 3
    LIFE = 4
    BOUNDS = 5 # Used for type checking where TimeScale value is passed


class Clock:
    """Clock object used by all systems, processes, mechanisms, and projections
    """
    def __init__(self, name):
        self.name = name
        self.time_step = 0
        self.trial = 0
        self.block = 0
        self.task = 0

CentralClock = Clock(name=CENTRAL_CLOCK)


class CurrentTime:
    def __init__(self):
        self.time_step = CentralClock.time_step
        self.trial = CentralClock.trial
        self.block = CentralClock.block
        self.task = CentralClock.task




