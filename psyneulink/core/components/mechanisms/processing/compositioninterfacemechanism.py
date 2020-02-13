# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  CompositionInterfaceMechanism *************************************************

"""

Contents
--------

  * `CompositionInterfaceMechanism_Overview`
  * `CompositionInterfaceMechanism_Creation`
  * `CompositionInterfaceMechanism_Structure`
  * `CompositionInterfaceMechanism_Execution`
  * `CompositionInterfaceMechanism_Class_Reference`


.. _CompositionInterfaceMechanism_Overview:

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
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = ['CompositionInterfaceMechanism']


class CompositionInterfaceMechanism(ProcessingMechanism_Base):
    """
    CompositionInterfaceMechanism(  \
        function=Identity())

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that acts as interface between a Composition and its
    inputs from and outputs to the environment or other Mechanisms (if it is a nested Composition).

    See `Mechanism <Mechanism_Class_Reference>` for arguments and additonal attributes.

    Attributes
    ----------

    function : InterfaceFunction : default Identity
        the function used to transform the variable before assigning it to the Mechanism's OutputPort(s)

    """

    componentType = COMPOSITION_INTERFACE_MECHANISM
    outputPortTypes = [OutputPort, ControlSignal]

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'CompositionInterfaceMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <CompositionInterfaceMechanism.function>`

                    :default value: `Identity`
                    :type: `Function`
        """
        function = Parameter(Identity, stateful=False, loggable=False)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports: tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort)) = None,
                 function=None,
                 composition=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        if default_variable is None and size is None:
            default_variable = self.class_defaults.variable
        self.composition = composition
        self.connected_to_composition = False

        super(CompositionInterfaceMechanism, self).__init__(default_variable=default_variable,
                                                            size=size,
                                                            input_ports=input_ports,
                                                            function=function,
                                                            params=params,
                                                            name=name,
                                                            prefs=prefs,
                                                            )
