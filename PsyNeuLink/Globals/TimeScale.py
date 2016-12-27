# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
from enum import Enum

# Time scale modes
class TimeScale(Enum):
    """Values used to specify ``time_scale`` argument for mechanisms, processes, and systems.
    """
    TIME_STEP = 0
    TRIAL = 1
    BOUNDS = 2 # Used for type checking where TimeScale value is passed


# Central clock
class CentralClock:
    """Central clock used by all mechanisms, processes and systems.
    """
    time_step = 0
    trial = 0
    block = 0
    task = 0

    def __init__(self):
        self.time_step = 0
        self.trial = 0
        self.block = 0
        self.task = 0


class CurrentTime:
    def __init__(self):
        self.time_step = CentralClock.time_step
        self.trial = CentralClock.trial
        self.block = CentralClock.block
        self.task = CentralClock.task




