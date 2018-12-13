# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  DNDMechanism ***************************************************************

"""

.. _DNDMechanism_Overview:

Overview
--------

A DNDMechanism is an `IntegratorFunction` that implements a `differential neural dictionary <HTML_REF>`_.

.. _DNDMechanism_Creation:

Creating a TransferMechanism
-----------------------------

A DNDMechanism is created by calling its constructor with **key_size** and **value_size** for the entries of its
`dict <DNDMechanism.dict>`, and a function used to implement the `dict` including methods for storing and retrieving
from it.

.. _DNDMechanism_Structure:

Structure
---------

A DNDMechanism has two `InputState`, *KEY_INPUT* and *VALUE_INPUT*.  Its `function <DNDMechanism.function>` takes the
`value <InputState.value>` of these as a 2d array ([key, value]) and stores it as an entry in its `dict
<DNDMechanism.dict>`, and uses the `value <InputState.value>` of its *KEY_INPUT* to retrieve an item from the
`dict <DNDMechanism.dict>` that is assigned as the `value <OutputState>` of its `primary OutputState
<OutputState_Primary>`.

.. _DNDMechanism_Execution:

Execution
---------

Function stores KEY_INPUT and VALUE_INPUT as entry in `dict <DNDMechanism.dict>`, retrieves an item that
matches KEY_INPUT
THe oreder of storage  and rretieval is determined by the function

.. _DNDMechanism_Class_Reference:

Class Reference
---------------


"""

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.integratorfunctions import DND
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.globals.keywords import NAME, SIZE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.context import ContextFlags

__all__ = ['DNDMechanism']


class DNDMechanism(ProcessingMechanism_Base):
    """
    DNDMechanism(          \
        key_size=1,        \
        value_size=1,      \
        function=DND,      \
        params=None,       \
        name=None,         \
        prefs=None         \
    )

    Subclass of `IntegratorMechanism <IntegratorMechanism>` that implements a `differentiable neural dictionary (DND)
    <HTML>`_

    Arguments
    ---------

    key_size : int : default 1
        specifies length of the key of entries in `dict <DNDMechanism.dict>`

    value_size : int : default 1
        specifies length of the value of entries in `dict <DNDMechanism.dict>`

    function : function : default DND
        specifies the function that implements storage and retrieval from `dict <DNDMechanism.dict>`.  It must take
        as its `variable <Function.variable>` a 2d array, the first item of which is the key and the second the value
        to be stored in its `dict` attribute, and must return a 1d array that is the value of the entry retrieved.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its `function <Mechanism_Base.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <DNDMechanism.name>`
        specifies the name of the DNDMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the DNDMechanism; see `prefs <TransferMechanism.prefs>` for details.

    Attributes
    ----------

    dict : OrderedDict
        member of `function <DNDMechanism.function>` that stores entries of DNDMechanism.

    function : function : default DND
        function that implements storage and retrieval from it maintains in its `dict` attribute.

    name : str
        the name of the DNDMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the DNDMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    class Params(ProcessingMechanism_Base.Params):
        variable = [[0],[0]]

    def __init__(self,
                 key_size=1,
                 value_size=1,
                 function:Function=DND,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # Template for dict entries
        default_variable = [np.zeros(key_size), np.zeros(value_size)]

        input_states = [{NAME:'KEY INPUT', SIZE:key_size},
                        {NAME:'VALUE INPUT', SIZE:value_size}]

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_states=input_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR
                         )

    def _instantiate_attributes_after_function(self, context=None):
        super()._instantiate_attributes_after_function(context=context)
        self.dict = self.function_object.dict
