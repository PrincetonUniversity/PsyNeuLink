# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ********************************************  System Defaults ********************************************************


from uuid import UUID
from enum import IntEnum
from collections import namedtuple
import warnings


from psyneulink.globals.keywords import INITIALIZING, VALIDATE, EXECUTING, CONTROL, LEARNING
# from psyneulink.composition import Composition


__all__ = [
    'Context',
    'ContextStatus',
    '_get_context'
]

STATUS = 'status'

time = namedtuple('time', 'run trial pass_ time_step')

class ContextError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class Context():
    __name__ = 'Context'
    def __init__(self, owner, status, composition=None, execution_id:UUID=None, string:str='', time=None):

        self.owner = owner
        self.status = status
        self.composition = composition
        self.execution_id = execution_id
        self.execution_time = None
        self.string = string

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        # if isinstance(status, ContextStatus):
        #     self._status = status
        # elif isinstance(status, int):
        #     self._status = ContextStatus(status)
        # else:
        #     raise ContextError("{} argument in call to {} must be a {} or an int".
        #                        format(STATUS, self.__name__, ContextStatus.__name__))
        if isinstance(status, (ContextStatus, int)):
            self._status = status
        else:
            raise ContextError("{} argument in call to {} must be a {} or an int".
                               format(STATUS, self.__name__, ContextStatus.__name__))

    @property
    def composition(self):
        try:
            return self._composition
        except AttributeError:
            self._composition = None

    @composition.setter
    def composition(self, composition):
        # from psyneulink.composition import Composition
        # if isinstance(composition, Composition):
        if composition is None or composition.__class__.__name__ in {'Composition', 'System'}:
            self._composition = composition
        else:
            raise ContextError("Assignment to context.composition for {} ({}) "
                               "must be a Composition (or \'None\').".format(self.owner.name, composition))

    @property
    def execution_time(self):
        try:
            return self._execution_time
        except:
            return None

    @execution_time.setter
    def execution_time(self, time):
        self._execution_time = time

    def update_execution_time(self):
        if self.status & ContextStatus.EXECUTION:
            self.execution_time = _get_time(self.owner, self.context.status)
        else:
            raise ContextError("PROGRAM ERROR: attempt to call update_execution_time for {} "
                               "when 'EXECUTION' was not in its context".format(self.owner.name))


# FIX: REPLACE IntEnum WITH Flags and auto IF/WHEN MOVE TO Python 3.6
class ContextStatus(IntEnum):
    """Used to identify the status of a `Component` when its value or one of its attributes is being accessed.
    Also used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`.
    """
    OFF = 0
    # """No recording."""
    INITIALIZATION = 1<<1  # 2
    """Set during execution of the Component's constructor."""
    VALIDATION =     1<<2  # 4
    """Set during validation of the value of a Component or its attribute."""
    EXECUTION =      1<<3  # 8
    """Set during any execution of the Component."""
    PROCESSING =     1<<4  # 16
    """Set during the `processing phase <System_Execution_Processing>` of execution of a Composition."""
    LEARNING =       1<<5  # 32
    """Set during the `learning phase <System_Execution_Learning>` of execution of a Composition."""
    CONTROL =        1<<6  # 64
    """Set during the `control phase System_Execution_Control>` of execution of a Composition."""
    TRIAL =          1<<7  # 128
    """Set at the end of a `TRIAL`."""
    RUN =            1<<8  # 256
    """Set at the end of a `RUN`."""
    SIMULATION =     1<<9  # 512
    # Set during simulation by Composition.controller
    COMMAND_LINE =   1<<10 # 1024
    # Component accessed by user
    CONSTRUCTOR =    1<<11 # 2048
    # Component being constructor (used in call to super.__init__)
    ALL_ASSIGNMENTS = \
        INITIALIZATION | VALIDATION | EXECUTION | PROCESSING | LEARNING | CONTROL
    """Specifies all contexts."""

    @classmethod
    def _get_context_string(cls, condition, string=None):
        """Return string with the names of all flags that are set in **condition**, prepended by **string**"""
        if string:
            string += ": "
        else:
            string = ""
        flagged_items = []
        # If OFF or ALL_ASSIGNMENTS, just return that
        if condition in (ContextStatus.ALL_ASSIGNMENTS, ContextStatus.OFF):
            return condition.name
        # Otherwise, append each flag's name to the string
        for c in list(cls.__members__):
            # Skip ALL_ASSIGNMENTS (handled above)
            if c is ContextStatus.ALL_ASSIGNMENTS.name:
                continue
            if ContextStatus[c] & condition:
               flagged_items.append(c)
        string += ", ".join(flagged_items)
        return string


def _get_context(context):

    if isinstance(context, ContextStatus):
        return context
    context_flag = ContextStatus.OFF
    if INITIALIZING in context:
        context_flag |= ContextStatus.INITIALIZATION
    if VALIDATE in context:
        context_flag |= ContextStatus.VALIDATION
    if EXECUTING in context:
        context_flag |= ContextStatus.EXECUTION
    if CONTROL in context:
        context_flag |= ContextStatus.CONTROL
    if LEARNING in context:
        context_flag |= ContextStatus.LEARNING
    if context == ContextStatus.TRIAL.name: # cxt-test
        context_flag |= ContextStatus.TRIAL
    if context == ContextStatus.RUN.name:
        context_flag |= ContextStatus.RUN
    if context == ContextStatus.COMMAND_LINE.name:
        context_flag |= ContextStatus.COMMAND_LINE
    return context_flag

def _get_time(component, context_flags):

    """Get time from Scheduler of System in which Component is being executed.

    Returns tuple with (run, trial, time_step) if being executed during Processing or Learning
    Otherwise, returns (None, None, None)

    """

    from psyneulink.globals.context import time
    from psyneulink.components.mechanisms.mechanism import Mechanism
    from psyneulink.components.states.state import State
    from psyneulink.components.projections.projection import Projection

    no_time = time(None, None, None, None)

    # Get mechanism to which Component being logged belongs
    if isinstance(component, Mechanism):
        ref_mech = component
    elif isinstance(component, State):
        if isinstance(component.owner, Mechanism):
            ref_mech = component.owner
        elif isinstance(component.owner, Projection):
            ref_mech = component.owner.receiver.owner
        else:
            raise ContextError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                           format(component.__class__.__name__,
                                  Mechanism.__name__, State.__name__, Projection.__name__))
    elif isinstance(component, Projection):
        ref_mech = component.receiver.owner
    else:
        raise ContextError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                       format(component.__class__.__name__,
                              Mechanism.__name__, State.__name__, Projection.__name__))

    # FIX: Modify to use component.owner.context.composition once that is implemented
    # Get System in which it is being (or was last) executed (if any):

    # If called from COMMAND_LINE, get context for last time value was assigned:
    # if context_flags & ContextStatus.COMMAND_LINE:
    if context_flags & (ContextStatus.COMMAND_LINE | ContextStatus.RUN | ContextStatus.TRIAL):
        context_flags = component.prev_context.status
        execution_context = component.prev_context.string
    else:
        execution_context = component.context.string

    system = ref_mech.context.composition

    if system:
        # FIX: Add ContextStatus.VALIDATE?
        if context_flags == ContextStatus.EXECUTION:
            t = system.scheduler_processing.clock.time
            t = time(t.run, t.trial, t.pass_, t.time_step)
        elif context_flags == ContextStatus.CONTROL:
            t = system.scheduler_processing.clock.time
            t = time(t.run, t.trial, t.pass_, t.time_step)
        elif context_flags == ContextStatus.LEARNING:
            t = system.scheduler_learning.clock.time
            t = time(t.run, t.trial, t.pass_, t.time_step)
        else:
            t = None

    else:
        if component.verbosePref:
            offender = "\'{}\'".format(component.name)
            if ref_mech is not component:
                offender += " [{} of {}]".format(component.__class__.__name__, ref_mech.name)
            warnings.warn("Attempt to log {} which is not in a System (logging is currently supported only "
                          "when running Components within a System".format(offender))
        t = None

    return t or no_time
