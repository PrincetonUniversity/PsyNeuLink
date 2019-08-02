# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  CompositionInterfaceMechanism *************************************************


"""
Overview
--------

A ParameterInterfaceMechanism is a specialized subclass of CompositionInterfaceMechanism. The ParameterInterfaceMechanism
stores inputs from outside the Composition so that those can be delivered to ParameterStates of its nested nodes.

.. _ParameterInterfaceMechanism_Creation:

Creating a ParameterInterfaceMechanism
-----------------------------------------

A ParameterInterfaceMechanism is created automatically when a controller is added to a Composition. When created, the
CompositionInterfaceMechanism's OutputState is set directly by the Composition. This Mechanism should never be executed,
and should never be created by a user.

.. _ParameterInterfaceMechanism_Structure

Structure
---------

[TBD]

.. _ParameterInterfaceMechanism_Execution

Execution
---------

[TBD]

.. _ParameterInterfaceMechanism_Class_Reference:

Class Reference
---------------

"""

import typecheck as tc

from collections.abc import Iterable

from psyneulink.core.components.functions.transferfunctions import Identity
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import PARAMETER_INTERFACE_MECHANISM, kwPreferenceSetName
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = ['ParameterInterfaceMechanism']


class ParameterInterfaceMechanism(CompositionInterfaceMechanism):
    """
    ParameterInterfaceMechanism(                            \
    default_variable=None,                               \
    size=None,                                              \
    function=Identity() \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Implements the ParameterInterfaceMechanism subclass of CompositionInterfaceMechanism.

    Arguments
    ---------

    default_variable : number, list or np.ndarray
        the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <ParameterInterfaceMechanism.variable>` for
        `function <ParameterInterfaceMechanism.function>`, and the `primary outputState <OutputState_Primary>` of the
        Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.

    function : InterfaceFunction : default Identity
        specifies the function used to transform the variable before assigning it to the Mechanism's OutputState(s)

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the `Mechanism <Mechanism>`, parameters for its `function <ParameterInterfaceMechanism.function>`, and/or a
        custom function and its parameters.  Values specified for parameters in the dictionary override any assigned
        to those parameters in arguments of the constructor.

    name : str : default ParameterInterfaceMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------
    variable : value: default
        the input to Mechanism's ``function``.

    name : str : default ParameterInterfaceMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Mechanism;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentType = PARAMETER_INTERFACE_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ParameterInterfaceMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({})
    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states: tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState)) = None,
                 function=Identity(),
                 composition=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None):
        super(ParameterInterfaceMechanism, self).__init__(default_variable=default_variable,
                                                        size=size,
                                                        input_states=input_states,
                                                        function=function,
                                                        params=params,
                                                        name=name,
                                                        prefs=prefs,
                                                        context=ContextFlags.CONSTRUCTOR)