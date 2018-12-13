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

A DNDMechanism
independent ones.

The function used to carry out the transformation can be selected from the following PsyNeuLink
`Functions <Function>`: `Linear`, `Exponential`, `Logistic`, or `SoftMax`.

The **integrator_mode** argument can switch the transformation from an "instantaneous"  to a "time averaged"
(integrated) manner of execution. When `integrator_mode <TransferMechanism.integrator_mode>` is set to True, the
mechanism's input is first transformed by its `integrator_function <TransferMechanism.integrator_function>` (the
`AdaptiveIntegrator`). That result is then transformed by the mechanism's `function <TransferMechanism.function>`.

.. _DNDMechanism_Creation:

Creating a TransferMechanism
-----------------------------

A DNDMechanism is created by calling its constructor.

.. _DNDMechanism_Structure:

Structure
---------

.. _DNDMechanism_Execution:

Execution
---------

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
