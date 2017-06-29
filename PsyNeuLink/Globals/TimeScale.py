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


# Time scale modes
class TimeScale(Enum):
    """Represents divisions of time used by the `Scheduler`, `Conditions <Condition>`, and the **time_scale**
    argument of execute methods.

    The values of TimeScale are defined as follows (in order of increasingly coarse granularity):

    Attributes
    ----------

    TIME_STEP
        the nuclear unit of time, consisting of a set of mechanisms that are eligible to execute simultaneously;

    PASS
        an entire iteration through the `Scheduler's <Scheduler>` `consideration_queue` (i.e. its
        toposort ordering), during which one or more `TIME_STEPs <TIME_STEP>` can occur and mechanisms may execute;

    TRIAL
        an open-ended unit of time consisting of all activity occurring within the scope of a single input to a
        `Composition`;

    RUN
        a loosely-defined unit of time consisting of one or more `TRIALs <TRIAL>`;

    LIFE
        all time since the creation of an object
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




