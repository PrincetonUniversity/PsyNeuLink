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

A CompositionInterfaceMechanism stores inputs from outside the Composition so that those can be delivered to the
Composition's `INPUT <NodeRole.INPUT>` Mechanism(s).

.. _CompositionInterfaceMechanism_Creation:

Creating an CompositionInterfaceMechanism
-----------------------------------------

A CompositionInterfaceMechanism is created automatically when an `INPUT <NodeRole.INPUT>` Mechanism is identified in a
Composition. When created, the CompositionInterfaceMechanism's OutputPort is set directly by the Composition. This
Mechanism should never be executed, and should never be created by a user.

.. _CompositionInterfaceMechanism_Structure

Structure
---------

[TBD]

.. _CompositionInterfaceMechanism_Execution

Execution
---------

[TBD]

.. _CompositionInterfaceMechanism_Class_Reference:

Class Reference
---------------

"""

import typecheck as tc

from collections.abc import Iterable

from psyneulink.core.components.functions.transferfunctions import Identity
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import COMPOSITION_INTERFACE_MECHANISM, PREFERENCE_SET_NAME
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = ['CompositionInterfaceMechanism']


class CompositionInterfaceMechanism(ProcessingMechanism_Base):
    """
    CompositionInterfaceMechanism(                            \
    default_variable=None,                               \
    size=None,                                              \
    function=Identity() \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Implements the CompositionInterfaceMechanism subclass of Mechanism.

    Arguments
    ---------

    default_variable : number, list or np.ndarray
        the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <CompositionInterfaceMechanism.variable>` for
        `function <CompositionInterfaceMechanism.function>`, and the `primary outputPort <OutputPort_Primary>` of the
        Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.

    function : InterfaceFunction : default Identity
        specifies the function used to transform the variable before assigning it to the Mechanism's OutputPort(s)

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterPort_Specifying_Parameters>` that can be used to specify the parameters for
        the `Mechanism <Mechanism>`, parameters for its `function <CompositionInterfaceMechanism.function>`, and/or a
        custom function and its parameters.  Values specified for parameters in the dictionary override any assigned
        to those parameters in arguments of the constructor.

    name : str : default CompositionInterfaceMechanism-<index>
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

    name : str : default CompositionInterfaceMechanism-<index>
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

    componentType = COMPOSITION_INTERFACE_MECHANISM
    outputPortTypes = [OutputPort, ControlSignal]

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'CompositionInterfaceMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({})
    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports: tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort)) = None,
                 function=Identity(),
                 composition=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        if default_variable is None and size is None:
            default_variable = self.class_defaults.variable
        self.composition = composition
        self.connected_to_composition = False

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(function=function,
                                                  input_ports=input_ports,
                                                  params=params)

        super(CompositionInterfaceMechanism, self).__init__(default_variable=default_variable,
                                                            size=size,
                                                            input_ports=input_ports,
                                                            function=function,
                                                            params=params,
                                                            name=name,
                                                            prefs=prefs,
                                                            )
