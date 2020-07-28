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

import warnings
import typecheck as tc

from collections.abc import Iterable

from psyneulink.core.components.functions.transferfunctions import Identity
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import COMPOSITION_INTERFACE_MECHANISM, INPUT_PORTS, OUTPUT_PORTS, PREFERENCE_SET_NAME
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
                 input_ports: tc.optional(tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort))) = None,
                 function=None,
                 composition=None,
                 port_map=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        if default_variable is None and size is None:
            default_variable = self.class_defaults.variable
        self.composition = composition
        self.port_map = port_map
        self.connected_to_composition = False
        self.user_added_ports = {
            INPUT_PORTS: set(),
            OUTPUT_PORTS: set()
        }
        super(CompositionInterfaceMechanism, self).__init__(default_variable=default_variable,
                                                            size=size,
                                                            input_ports=input_ports,
                                                            function=function,
                                                            params=params,
                                                            name=name,
                                                            prefs=prefs,
                                                            )

    @handle_external_context()
    def add_ports(self, ports, context=None):
        ports = super(CompositionInterfaceMechanism, self).add_ports(ports, context=context)
        if context.source == ContextFlags.COMMAND_LINE:
            warnings.warn(
                'You are attempting to add custom ports to a CIM, which can result in unpredictable behavior and '
                'is therefore recommended against. If suitable, you should instead add ports to the mechanism(s) '
                'that project to or are projected to from the CIM.')
            if ports[INPUT_PORTS]:
                self.user_added_ports[INPUT_PORTS].update([port for port in ports[INPUT_PORTS].data])
            if ports[OUTPUT_PORTS]:
                self.user_added_ports[OUTPUT_PORTS].update([port for port in ports[OUTPUT_PORTS].data])
        return ports

    @handle_external_context()
    def remove_ports(self, ports, context=None):
        super(CompositionInterfaceMechanism, self).remove_ports(ports, context)
        input_ports_marked_for_deletion = set()
        for port in self.user_added_ports[INPUT_PORTS]:
            if port not in self.input_ports:
                input_ports_marked_for_deletion.add(port)
        self.user_added_ports[INPUT_PORTS] = self.user_added_ports[INPUT_PORTS] - input_ports_marked_for_deletion
        output_ports_marked_for_deletion = set()
        for port in self.user_added_ports[OUTPUT_PORTS]:
            if port not in self.output_ports:
                output_ports_marked_for_deletion.add(port)
        self.user_added_ports[OUTPUT_PORTS] = self.user_added_ports[OUTPUT_PORTS] - output_ports_marked_for_deletion
