# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  EpisodicMemoryMechanism ****************************************************

"""

.. _EpisodicMemoryMechanism_Overview:

Overview
--------

An EpisodicMemoryMechanism is an `IntegratorMechanism` that can store and retrieve content-associate pairs.
Only the content is used for determining which pairs are retrieved

.. _EpisodicMemoryMechanism_Creation:

Creating a TransferMechanism
-----------------------------

An EpisodicMemoryMechanism is created by calling its constructor with **content_size** and, optionally, **assoc_size**,
that define the shapes of the items stored in its memory.

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

An EpisodicMemoryMechanism has at least one `InputStates <InputState>`, its *CONTENT_INPUT* and,
optionally, an *ASSOC_INPUT* InputState (if its *assoc_size* is specified and is not 0) that represent
an item to store;  a `function <EpisodicMemoryMechanism.function>` that stores and retrieves content-assoc pairs from its
memory; and at least one `OutputStates <OutputState>`, *CONTENT_OUTPUT*, as well as a 2nd, *CONTENT_OUTPUT* if it has
an *ASSOC_INPUT* InputState, that represent a retrieved item. The default function is a `ContentAddressableMemory` that
implements a simple form of content-addressable memory, but a custom function can be specified, so long as it meets the
following requirements:

    * It must accept a 2d array as its first argument, the first item of which is the content and the second the associate.
    ..
    * It must return a 2d array, the first item of which is the retrieved content and the second of which is the
    assoc with which it is associated in the `function <EpisodicMemoryMechanism.function>`\\'s `memory
    <EpisodicMemoryMechanism.memory>`.
    ..
    * It may also implement a memory attribute;  if it does, it can be accessed by the EpisodicMemoryMechanism's
      `memory <EpisodicMemoryMechanism.memory>` attribute.

.. _EpisodicMemoryMechanism_Execution:

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieves an item from its memory based on the `value <InputState.value>` of its *CONTENT_INPUT* `InputState`;
      if no retrieval is made, appropriately shaped zero-valued arrays are assigned to the `value
      <OutputState.value>` of the *CONTENT_OUTPUT* and, if specified, it *ASSOC_OUTPUT* OutputStates.
    ..
    * stores the `value <InputState.value>` of its *CONTENT_INPUT* and, if specified, *ASSOC_INPUT* `InputStates
    <InputState>` in its memory.
    ..
    * assigns the value of the retrieved item's content in the EpisodicMemoryMechanism's  *CONTENT_OUTPUT*
    `OutputState`, and the value of the assoc of the retrieved item in the *ASSOC_OUTPUT* OutputState.

    .. note::
         The order of storage and retieval is determined by the function.

         The value of the content of the item retrieved from memory (and stored in *CONTENT_OUTPUT*) may be different than the
         `value <InputState.value>` of *CONTENT* used to retrieve the item.

         If no retrieval is made, appropriately shaped zero-valued arrays are assigned as the `value
         <OutputState.value>` of the *CONTENT_OUTPUT* and, if specified, *ASSOC_OUTPUT* OutputStates.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""
import warnings

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import ContentAddressableMemory
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.inputstate import OutputState
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import CONTEXT, NAME, OWNER_VALUE, SIZE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set

__all__ = ['EpisodicMemoryMechanism', 'CONTENT_INPUT', 'ASSOC_INPUT', 'CONTENT_OUTPUT', 'ASSOC_OUTPUT']

CONTENT_INPUT = 'CONTENT_INPUT'
ASSOC_INPUT = 'ASSOC_INPUT'
CONTENT_OUTPUT = 'CONTENT_OUTPUT'
ASSOC_OUTPUT = 'ASSOC_OUTPUT'


class EpisodicMemoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism(                \
        content_size=1,                     \
        assoc_size=1,                       \
        function=ContentAddressableMemory,  \
        params=None,                        \
        name=None,                          \
        prefs=None                          \
    )

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary
    (ContentAddressableMemory)<HTML>`_

    Arguments
    ---------

    content_size : int : default 1
        specifies length of the content stored in the `function <EpisodicMemoryMechanism.function>`\\s memory.

    assoc_size : int : default 0
        specifies length of the assoc stored in the `function <EpisodicMemoryMechanism.function>`\\s memory;
        if it is 0 (the default) then no *ASSOC_INPUT* InputState or *ASSOC_OUTPUT* OutputState are created.

    function : function : default ContentAddressableMemory
        specifies the function that implements a memory store and methods to store to and retrieve from it.  It
        must take as its `variable <Function.variable>` a 2d array, the first item of which is the content and the second
        the associate to be stored in its memory, and must return a 2d array that is the value of the
        content and assoc retrieved from its memory.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its `function <Mechanism_Base.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <EpisodicMemoryMechanism.name>`
        specifies the name of the EpisodicMemoryMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the EpisodicMemoryMechanism; see `prefs <TransferMechanism.prefs>` for details.

    Attributes
    ----------

    function : function
        function that implements storage and retrieval from a memory.

    memory : 3d array
        contains key-value pairs stored in the `function <EpisodicMemoryMechanism.function>'\\s `memory` attribute
        (if it has one).

    name : str
        the name of the EpisodicMemoryMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the EpisodicMemoryMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <EpisodicMemoryMechanism.variable>`

                    :default value: [[0]]
                    :type: list

        """
        variable = Parameter([[0]])

    def __init__(self,
                 content_size:int=1,
                 assoc_size:int=0,
                 function:Function=ContentAddressableMemory,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        # Template for memory_store entries
        default_variable = [np.zeros(content_size)]

        input_states = [{NAME:CONTENT_INPUT, SIZE:content_size}]
        output_states = [{NAME: CONTENT_OUTPUT, VARIABLE: (OWNER_VALUE, 0)}]

        if assoc_size:
            input_states.append({NAME:ASSOC_INPUT, SIZE:assoc_size})
            output_states.append({NAME: ASSOC_OUTPUT, VARIABLE: (OWNER_VALUE, 1)})
            default_variable.append(np.zeros(assoc_size))

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs
                         )

    def _execute(self, variable=None, context=None, runtime_params=None):

        value =  super()._execute(variable=variable,
                                           context=context,
                                           runtime_params=runtime_params,
                                           )
        # Only return content if assoc has not been specified (in which case second element of value should be empty)
        if len(value[1]) == 0:
            return np.delete(value,1)
        else:
            return value


    def _instantiate_output_states(self, context=None):
        if len(self.input_states) != len(self.output_states):
            assert False, \
                f'PROGRAM ERROR: Number of {InputState.__class__.__name__}s and ' \
                f'{OutputState.__class__.__name__}s do not match in {self.name}'
        for i, input_state_spec, output_state_spec in zip(range(len(self.input_states)-1),
                                                          self.input_states,
                                                          self.output_states):
            if input_state_spec.value is []:
                del self.output_states[i]

        return super()._instantiate_output_states(context=context)

    def _parse_function_variable(self, variable, context=None):

        # If assoc has not been specified, add empty list to call to function (which expects two items in its variable)
        if len(variable) != 2:
            return np.array([variable[0],[]])
        else:
            return variable

    @property
    def memory(self):
        """Return function's memory attribute"""
        try:
            return self.function.memory
        except:
            warnings.warning(f'Function of {self.name} (self.function.name) has no memory attribute')
