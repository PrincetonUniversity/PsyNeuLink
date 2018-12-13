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

A EpisodicMemoryMechanism is an `IntegratorFunction` that can store and retrieve cue-associate pairs.

.. _EpisodicMemoryMechanism_Creation:

Creating a TransferMechanism
-----------------------------

An EpisodicMemoryMechanism is created by calling its constructor with **cue_size** and **assoc_size** that define
the shapes of the items stored in its memory.

.. _EpisodicMemoryMechanism_Structure:

Structure
---------

A EpisodicMemoryMechanism has two `InputStates`, *CUE_INPUT* and *ASSOC_INPUT*.  Its `function
<EpisodicMemoryMechanism.function>` takes the `value <InputState.value>` of these as a 2d array ([CUE, ASSOC]),
stores it in its memory, and uses the `value <InputState.value>` of its *CUE_INPUT* to retrieve an item from memory
that it assigns as the `value <OutputState>` of its `primary OutputState <OutputState_Primary>` (named *ASSOC_OUTPUT*);
the value of the cue associated with the assoc that is retrieved (which may be different than the `value
<InputState.value>` of *CUE*) is assigned to the *CUE_OUTPUT* `OutputState`.

.. _EpisodicMemoryMechanism_Execution:

Execution
---------

Function stores CUE_INPUT and ASSOC_INPUT as entry in `memory_store <EpisodicMemoryMechanism.memory_store>`, retrieves an item that
matches CUE_INPUT
THe oreder of storage  and rretieval is determined by the function

.. _EpisodicMemoryMechanism_Class_Reference:

Class Reference
---------------


"""

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.integratorfunctions import DND
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.globals.keywords import NAME, SIZE, VARIABLE, OWNER_VALUE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.context import ContextFlags

__all__ = ['EpisodicMemoryMechanism', 'CUE_INPUT', 'ASSOC_INPUT', 'CUE_OUTPUT', 'ASSOC_OUTPUT']

CUE_INPUT = 'CUE_INPUT'
ASSOC_INPUT = 'ASSOC_INPUT'
CUE_OUTPUT = 'CUE_OUTPUT'
ASSOC_OUTPUT = 'ASSOC_OUTPUT'

class EpisodicMemoryMechanism(ProcessingMechanism_Base):
    """
    EpisodicMemoryMechanism(          \
        cue_size=1,        \
        assoc_size=1,      \
        function=DND,      \
        params=None,       \
        name=None,         \
        prefs=None         \
    )

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary (DND)
    <HTML>`_

    Arguments
    ---------

    cue_size : int : default 1
        specifies length of the cue stored in  `memory_store <EpisodicMemoryMechanism.memory_store>`

    assoc_size : int : default 1
        specifies length of the value of entries in `memory_store <EpisodicMemoryMechanism.memory_store>`

    function : function : default DND
        specifies the function that implements storage and retrieval from `memory_store <EpisodicMemoryMechanism.memory_store>`.  It must take
        as its `variable <Function.variable>` a 2d array, the first item of which is the cue and the second the value
        to be stored in its `memory_store` attribute, and must return a 1d array that is the value of the entry retrieved.

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

    function : function : default DND
        function that implements storage and retrieval from memory <EpisodicMemory_Memory>`.

    name : str
        the name of the EpisodicMemoryMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the EpisodicMemoryMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    class Params(ProcessingMechanism_Base.Params):
        variable = [[0],[0]],


    def __init__(self,
                 cue_size=1,
                 assoc_size=1,
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

    def _execute(self, variable=None, execution_id=None, runtime_params=None, context=None):
        return super()._execute(variable=variable,
                                 execution_id=execution_id,
                                 runtime_params=runtime_params,
                                 context=context)
