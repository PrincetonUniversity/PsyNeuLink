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
    argument of a Composition's `run <Composition.run>` method.

    The values of TimeScale are defined as follows (in order of increasingly coarse granularity):

    Attributes
    ----------

    TIME_STEP
        the nuclear unit of time, corresponding to the execution of all `Mechanism <Mechanism>`\\ s allowed to execute
        from a single `consideration_set` of a `Scheduler`, and which are considered to have executed simultaneously.

    PASS
        a full iteration through all of the `consideration_sets <consideration_set>` in a `Scheduler's <Scheduler>`
        `consideration_queue`, consisting of one or more `TIME_STEPs <TIME_STEP>`, over which every `Component
        <Component>` `specified to a Scheduler <Scheduler_Creation>` is considered for execution at least once.

    TRIAL
        an open-ended unit of time consisting of all action that occurs within the scope of a single input to a
        `Composition <Composition>`.

    RUN
        the scope of a call to the `run <Composition.run>` method of a `Composition <Composition>`,
        consisting of one or more `TRIALs <TRIAL>`.

    LIFE
        the scope of time since the creation of an object.
    """
    TIME_STEP = 0
    PASS = 1
    TRIAL = 2
    RUN = 3
    LIFE = 4


class Clock:
    """Clock object used by a `Components <Component>` and `Compositions <Composition>` to reference the current
    state of their assigned `Scheduler`.
    """

    def __init__(self, name):
        self.name = name
        self.time_step = 0
        # FIX: WHY ISN'T THE FOLLOWING ASSIGNMENT OK?
        # self.pass = 0
        self.trial = 0
        self.run = 0
        self.life = 0
        # FIX: THE FOLLOWING TWO PRESUMABLY DEFUNCT ITEMS STILL SEEM TO BE GETTING REFERENCED SOMEWHERE:
        self.block = 0
        self.task = 0


CentralClock = Clock(name=CENTRAL_CLOCK)


class CurrentTime:
    def __init__(self):
        self.time_step = CentralClock.time_step
        self.trial = CentralClock.trial
        self.block = CentralClock.block
        self.task = CentralClock.task
