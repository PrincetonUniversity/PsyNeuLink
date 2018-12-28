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

A EpisodicMemoryMechanism is an `IntegratorFunction` Function that can store and retrieve cue-associate pairs.

.. _EpisodicMemoryMechanism_Creation:

Creating a TransferMechanism
-----------------------------

An EpisodicMemoryMechanism is created by calling its constructor with **cue_size** and **assoc_size** that define
the shapes of the items stored in its memory.

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

A EpisodicMemoryMechanism has two `InputStates <InputState>`, *CUE_INPUT* and *ASSOC_INPUT*, that represent
an item to store;  a `function <EpisodicMemoryMechanism.function>` that stores and retrieves cue-assoc pairs from its
memory; and two `OutputStates <OutputState>`, *ASSOC_OUTPUT* and *CUE_OUTPUT* that represent a retrieved item.
The default function is a `DND` that implements a simple form of differentiable neural dictionary, but a custom
function can be specified, so long as it meets the following requirements:

    * It must accept a 2d array as its first argument, the first item of which is the cue and the second the associate.
    ..
    * It must retur a 2d array, the first item of which is the retrieved associate and the cue with which it is
      associated in the `function <EpisodicMemoryMechanism.function>`\\'s memory.
    ..
    * It may also implement `storage_prob` and `retrieval_prob` attributes;  if it does, they are assigned the values
      specified in the corresponding arguments of the EpisodicMemoryMechanism's constructor, otherwise those are
      ignored.

.. _EpisodicMemoryMechanism_Execution:

Execution
---------

When an EpisodicMemoryMechanism is executed, its `function <EpisodicMemoryMechanism.function>` carries out
the following operations:

    * retrieve an item from its memory based on the `value <InputState.value>` of its *CUE_INPUT* `InputState`
      and `retrieval_prob <EpisodicMemory.storage_prob>`;  if no retrieval is made, appropriately shaped zero-valued
      arrays are assigned to the `value <OutputState.value>` of the *ASSOC_OUTPUT* and *CUE_OUTPUT* OutputStates.
    ..
    * store the `value <InputState.value>` of its *CUE_INPUT* and *ASSOC_INPUT* `InputStates <InputState>` in
      its memory, based on its `storage_prob <EpisodicMemoryMechanism.storage_prob>`.
    ..
    * assign the value of the retrieved item's assoc in the EpisodicMemoryMechanism's  *ASSOC_OUTPUT* `OutputState`,
      and the value of the cue of the retrieved item in the *CUE_OUTPUT* OutputState.

    .. note::
         The order of storage and retieval is determined by the function.

         The value of the cue of the item retrieved from memory (and stored in *CUE_OUTPUT*) may be different than the
         `value <InputState.value>` of *CUE* used to retrieve the item.

         If no retrieval is made, appropriately shaped zero-valued arrays are assigned as the `value
         <OutputState.value>` of the *ASSOC_OUTPUT* and *CUE_OUTPUT* OutputStates.

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import RETRIEVAL_PROB, STORAGE_PROB, DND
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import NAME, OWNER_VALUE, SIZE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.utilities import all_within_range

__all__ = ['EpisodicMemoryMechanism', 'CUE_INPUT', 'ASSOC_INPUT', 'CUE_OUTPUT', 'ASSOC_OUTPUT']

CUE_INPUT = 'CUE_INPUT'
ASSOC_INPUT = 'ASSOC_INPUT'
CUE_OUTPUT = 'CUE_OUTPUT'
ASSOC_OUTPUT = 'ASSOC_OUTPUT'


class EpisodicMemoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism( \
        cue_size=1,          \
        assoc_size=1,        \
        storage_prob=1.0     \
        retrieval_prob=1.0   \
        function=DND,        \
        params=None,         \
        name=None,           \
        prefs=None           \
    )

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary (DND)
    <HTML>`_

    Arguments
    ---------

    cue_size : int : default 1
        specifies length of the cue stored in the `function <EpisodicMemoryMechanism.function>`\s memory.

    assoc_size : int : default 1
        specifies length of the assoc stored in the `function <EpisodicMemoryMechanism.function>`\s memory.

    storage_prob : float : default 1.0
        specifies probability that the cue and assoc are stored in the `function
        <EpisodicMemoryMechanism.function>`\\'s memory.

    retrieval_prob : float : default 1.0
        specifies probability that the cue and assoc are retrieved from the `function
        <EpisodicMemoryMechanism.function>`\\'s memory.

    function : function : default DND
        specifies the function that implements a memory store and methods to store to and retrieve from it.  It
        must take as its `variable <Function.variable>` a 2d array, the first item of which is the cue and the second
        the associate to be stored in its memory, and must return a 2d array that is the value of the
        retriefved associate and the actual cue associated with it in memory.

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

    storage_prob : float : default 1.0
        probability that cue and assoc are stored in the `function <EpisodicMemoryMechanism.function>`\s memory.

    retrieval_prob : float : default 1.0
        probability that cue and assoc are retrieved from the `function <EpisodicMemoryMechanism.function>`\s memory;
        if no retrieval is made, appropriately-shaped zero-valued arrays are assigned to the the `value
        <OutputState.value>` of the *ASSOC_OUTPUT* and *CUE_OUTPUT* OutputStates (see <Structure
        <EpisodicMemoryMechanism_Structure>`.

    function : function : default DND
        function that implements storage and retrieval from a memory.

    name : str
        the name of the EpisodicMemoryMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the EpisodicMemoryMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    class Parameters(ProcessingMechanism_Base.Parameters):
        variable = Parameter([[0],[0]])

    def __init__(self,
                 cue_size:int=1,
                 assoc_size:int=1,
                 storage_prob:float=1.0,
                 retrieval_prob:float=1.0,
                 function:Function=DND,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # Template for memory_store entries
        default_variable = [np.zeros(cue_size), np.zeros(assoc_size)]

        input_states = [{NAME:CUE_INPUT, SIZE:cue_size},
                        {NAME:ASSOC_INPUT, SIZE:assoc_size}]

        output_states = [{NAME: ASSOC_OUTPUT, VARIABLE: (OWNER_VALUE, 0)},
                         {NAME: CUE_OUTPUT, VARIABLE: (OWNER_VALUE, 1)}]

        self._storage_prob = storage_prob
        self._retrieval_prob = retrieval_prob

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR
                         )

    def _instantiate_attributes_after_function(self, context=None):
        super()._instantiate_attributes_after_function(context=context)

        if not all_within_range(self._storage_prob, 0, 1):
            raise EpisodicMemoryMechanismError("{} arg of {} ({}) must be a float in the interval [0,1]".
                                format(repr(STORAGE_PROB), self.__class___.__name__, self._storage_prob))
        if hasattr(self.function, STORAGE_PROB):
            self.function.parameters.storage_prob.set(self._storage_prob)

        if not all_within_range(self._retrieval_prob, 0, 1):
            raise EpisodicMemoryMechanismError("{} arg of {} ({}) must be a float in the interval [0,1]".
                                format(repr(RETRIEVAL_PROB), self.__class___.__name__, self._retrieval_prob))
        if hasattr(self.function, RETRIEVAL_PROB):
            self.function.parameters.retrieval_prob.set(self._retrieval_prob)

    def _execute(self, variable=None, execution_id=None, runtime_params=None, context=None):
        return super()._execute(variable=variable,
                                  execution_id=execution_id,
                                  runtime_params=runtime_params,
                                  context=context)
