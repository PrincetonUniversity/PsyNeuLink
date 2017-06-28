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

# ********************************************* TimeScale ***************************************************************

"""
TimeScale (Enum) represents the various divisions of time used elsewhere in PsyNeuLink, and are defined as follows
(in order of increasingly coarse granularity):

    - TIME_STEP
        The nuclear unit of time, consisting of a set of mechanisms that are considered to run simultaneously
    - PASS
        A PASS consists of an entire iteration through the `Scheduler`<Scheduler>'s consideration queue (i.e. its
        toposort ordering), during which zero or more TIME_STEPs will occur and mechanisms set to execute
    - TRIAL
        An open-ended unit of time consisting of all activity occurring within the scope of a single input to a
        `System`<System> (or similar composition)
    - RUN
        A loosely-defined unit of time consisting of zero or more TRIALs
    - LIFE
        LIFE consists of all time since the creation of an object

"""

# Time scale modes
class TimeScale(Enum):
    """Values used to specify ``time_scale`` argument for mechanisms, processes, and systems.
    """
    TIME_STEP = 0
    PASS = 1
    TRIAL = 2
    RUN = 3
    LIFE = 4


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




