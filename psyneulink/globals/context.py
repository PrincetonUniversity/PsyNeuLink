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


from psyneulink.globals.keywords import INITIALIZING, VALIDATE, EXECUTING, CONTROL, LEARNING
# from psyneulink.composition import Composition


__all__ = [
    'Context',
    'ContextStatus',
    '_get_context'
]

STATUS = 'status'


class ContextError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class Context():
    def __init__(self, status, composition, execution_id:UUID, string:str=''):

        self.status = status

        from psyneulink.composition import Composition
        if isinstance(composition, Composition):
            self.composition = composition
        else:
            raise ContextError("\'composition\' argument in call to {} must be a {}".
                               format(self.__name__, Composition.__name__))

        self.execution_id = execution_id
        self.string = string

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        if isinstance(status, ContextStatus):
            self._status = status
        else:
            raise ContextError("{} argument in call to {} must be a {}".
                               format(STATUS, self.__name__, ContextStatus.__name__))


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
    COMMAND_LINE =   1<<9  # 512
    # Component accessed by user
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
    if context == ContextStatus.TRIAL.name:
        context_flag |= ContextStatus.TRIAL
    if context == ContextStatus.RUN.name:
        context_flag |= ContextStatus.RUN
    if context == ContextStatus.COMMAND_LINE.name:
        context_flag |= ContextStatus.COMMAND_LINE
    return context_flag
