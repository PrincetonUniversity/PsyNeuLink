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
    'ContextFlags',
    '_get_context'
]

STATUS = 'status'

time = namedtuple('time', 'run trial pass_ time_step')

class ContextError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ContextFlags(IntEnum):
    """Used to identify the status of a `Component` when its value or one of its attributes is being accessed.
    Also used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`.
    """
    UNINITIALIZED = 0
    """Not Initialized."""
    DEFERRED_INIT = 1
    """Set if flagged for deferred initialization."""
    INITIALIZING =  2
    """Set during initialization of the Component."""
    VALIDATING =    3
    """Set during validation of the value of a Component or its attribute."""
    INITIALIZED =   4
    """Set after completion of initialization of the Component."""


# FIX: REPLACE IntEnum WITH Flags and auto IF/WHEN MOVE TO Python 3.6
class Status(IntEnum):
    """Used to identify the status of a `Component` when its value or one of its attributes is being accessed.
    Also used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`.
    """
    UNINITIALIZED = 0
    """Not Initialized."""
    DEFERRED_INIT = 1
    """Set if flagged for deferred initialization."""
    INITIALIZING =  2
    """Set during initialization of the Component."""
    VALIDATING =    3
    """Set during validation of the value of a Component or its attribute."""
    INITIALIZED =   4
    """Set after completion of initialization of the Component."""

class Source(IntEnum):
    CONSTRUCTOR =  0
    """Call to method from Component's constructor."""
    COMMAND_LINE = 1
    """Direct call to method by user (either interactively from the command line, or in a script)."""
    COMPONENT =    2
    """Call to method by the Component."""
    COMPOSITION =  3
    """Call to method by a/the Composition to which the Component belongs."""

    # @classmethod
    # def _get_context_string(cls, condition, string=None):
    #     """Return string with the names of all flags that are set in **condition**, prepended by **string**"""
    #     if string:
    #         string += ": "
    #     else:
    #         string = ""
    #     flagged_items = []
    #     # If OFF or ALL_ASSIGNMENTS, just return that
    #     if condition in (ContextFlags.ALL_ASSIGNMENTS, ContextFlags.OFF):
    #         return condition.name
    #     # Otherwise, append each flag's name to the string
    #     for c in list(cls.__members__):
    #         # Skip ALL_ASSIGNMENTS (handled above)
    #         if c is ContextFlags.ALL_ASSIGNMENTS.name:
    #             continue
    #         if ContextFlags[c] & condition:
    #            flagged_items.append(c)
    #     string += ", ".join(flagged_items)
    #     return string

class ExecutionPhase(IntEnum):
    """Used to identify the status of a `Component` when its value or one of its attributes is being accessed.
    Also used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`.
    """
    IDLE =         0
    """Not currently executin."""
    PROCESSING =   1
    """Set during the `processing phase <System_Execution_Processing>` of execution of a Composition."""
    LEARNING =     2
    """Set during the `learning phase <System_Execution_Learning>` of execution of a Composition."""
    CONTROL =      3
    """Set during the `control phase System_Execution_Control>` of execution of a Composition."""
    SIMULATION =   4
    """Set during simulation by Composition.controller"""


class Context():
    __name__ = 'Context'
    def __init__(self,
                 owner,
                 status=Status.UNINITIALIZED,
                 execution_phase=Status.IDLE,
                 source=Source.COMPONENT,
                 composition=None,
                 execution_id:UUID=None,
                 string:str='', time=None):

        self.owner = owner
        self.status = status
        self.execution_phase = execution_phase
        self.source = source
        self.composition = composition
        self.execution_id = execution_id
        self.execution_time = None
        self.string = string

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        # if isinstance(status, ContextFlags):
        #     self._status = status
        # elif isinstance(status, int):
        #     self._status = ContextFlags(status)
        # else:
        #     raise ContextError("{} argument in call to {} must be a {} or an int".
        #                        format(STATUS, self.__name__, ContextFlags.__name__))
        if isinstance(status, (ContextFlags, int)):
            self._status = status
        else:
            raise ContextError("{} argument in call to {} must be a {} or an int".
                               format(STATUS, self.__name__, ContextFlags.__name__))

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
        if self.status & ContextFlags.EXECUTION:
            self.execution_time = _get_time(self.owner, self.context.status)
        else:
            raise ContextError("PROGRAM ERROR: attempt to call update_execution_time for {} "
                               "when 'EXECUTION' was not in its context".format(self.owner.name))


def _get_context(context):

    if isinstance(context, ContextFlags):
        return context
    context_flag = ContextFlags.OFF
    if INITIALIZING in context:
        context_flag |= ContextFlags.INITIALIZATION
    if VALIDATE in context:
        context_flag |= ContextFlags.VALIDATION
    if EXECUTING in context:
        context_flag |= ContextFlags.EXECUTION
    if CONTROL in context:
        context_flag |= ContextFlags.CONTROL
    if LEARNING in context:
        context_flag |= ContextFlags.LEARNING
    if context == ContextFlags.TRIAL.name: # cxt-test
        context_flag |= ContextFlags.TRIAL
    if context == ContextFlags.RUN.name:
        context_flag |= ContextFlags.RUN
    if context == ContextFlags.COMMAND_LINE.name:
        context_flag |= ContextFlags.COMMAND_LINE
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
    # if context_flags & ContextFlags.COMMAND_LINE:
    if context_flags & (ContextFlags.COMMAND_LINE | ContextFlags.RUN | ContextFlags.TRIAL):
        context_flags = component.prev_context.status
        execution_context = component.prev_context.string
    else:
        execution_context = component.context.string

    system = ref_mech.context.composition

    if system:
        # FIX: Add ContextFlags.VALIDATE?
        if context_flags == ContextFlags.EXECUTION:
            t = system.scheduler_processing.clock.time
            t = time(t.run, t.trial, t.pass_, t.time_step)
        elif context_flags == ContextFlags.CONTROL:
            t = system.scheduler_processing.clock.time
            t = time(t.run, t.trial, t.pass_, t.time_step)
        elif context_flags == ContextFlags.LEARNING:
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
