# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  DefaultControlMechanism ************************************************

"""

The DefaultControlMechanism is created for a `System` if no other controller type is specified. The
DefaultControlMechanism creates an `ControlSignal` for each `ControlProjection` it is assigned, and uses
`defaultControlAllocation` as the `value <ControlSignal.value>` for the ControlSignal.  By default,
`defaultControlAllocation` =  1, so that ControlProjections from the DefaultControlMechanism have no effect on their
parameters.  However, it can be used to uniformly control the parameters that receive ControlProjections from it,
by manually changing the value of `defaultControlAllocation`.  See `ControlMechanism <ControlMechanism>` for additional
details of how ControlMechanism are created, executed and their attributes.

COMMENT:
   ADD LINK FOR defaultControlAllocation

    TEST FOR defaultControlAllocation:  |defaultControlAllocation|

    ANOTHER TEST FOR defaultControlAllocation:  :py:print:`defaultControlAllocation`

    AND YET ANOTHER TEST FOR defaultControlAllocation:  :py:print:|defaultControlAllocation|

    LINK TO DEFAULTS: :doc:`Defaults`
COMMENT


"""

import numpy as np
import typecheck as tc

from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import CONTROL, INPUT_PORTS, NAME
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import ContentAddressableList

__all__ = [
    'DefaultControlMechanism', 'DefaultControlMechanismError'
]


class DefaultControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class DefaultControlMechanism(ControlMechanism):
    """Subclass of `ControlMechanism <ControlMechanism>` that implements a DefaultControlMechanism.

    COMMENT:
        Description:
            Implements default source of control signals, with one inputPort and outputPort for each.
            Uses defaultControlAllocation as input(s) and pass value(s) unchanged to outputPort(s) and
            ControlProjection(s)

            Every ControlProjection is assigned this Mechanism as its sender by default (i.e., unless a sender is
                explicitly specified in its constructor).

            An inputPort and outputPort is created for each ControlProjection assigned:
                the inputPort is assigned the
                :py:constant:`defaultControlAllocation <Defaults.defaultControlAllocation>` value;
                when the DefaultControlMechanism executes, it simply assigns the same value to the ControlProjection.

            Class attributes:
                + componentType (str): System Default Mechanism
    COMMENT
    """

    componentType = "DefaultControlMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE

    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'DefaultControlMechanismCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    from psyneulink.core.components.functions.transferfunctions import Linear

    @tc.typecheck
    def __init__(self,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 control_signals:tc.optional(list)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 function=None,
                 **kwargs
                 ):

        super(DefaultControlMechanism, self).__init__(
                objective_mechanism=objective_mechanism,
                control_signals=control_signals,
                function=function,
                params=params,
                name=name,
                prefs=prefs,

                **kwargs)

    def _instantiate_input_ports(self, context=None):
        """Instantiate input_value attribute

        Instantiate input_ports and monitored_output_ports attributes (in case they are referenced)
            and assign any OutputPorts that project to the input_ports to monitored_output_ports

        IMPLEMENTATION NOTE:  At present, these are dummy assignments, simply to satisfy the requirements for
                              subclasses of ControlMechanism;  in the future, an _instantiate_objective_mechanism()
                              method should be implemented that also implements an _instantiate_monitored_output_ports
                              method, and that can be used to add OutputPorts/Mechanisms to be monitored.
        """

        if not hasattr(self, INPUT_PORTS):
            self._input_ports = None
        elif self.input_ports:
            for input_port in self.input_ports:
                for projection in input_port.path_afferents:
                    self.monitored_output_ports.append(projection.sender)

    def _instantiate_control_signal(self, control_signal, context=None):
        """Instantiate requested ControlSignal, ControlProjection and associated InputPort
        """
        from psyneulink.core.components.ports.parameterport import ParameterPort

        if isinstance(control_signal, dict):
            if CONTROL in control_signal:
                projection = control_signal[CONTROL][0]
                input_name = 'DefaultControlAllocation for ' + projection.receiver.name + '_ControlSignal'
            elif NAME in control_signal:
                input_name = 'DefaultControlAllocation for ' + control_signal[NAME] + '_ControlSignal'

        elif isinstance(control_signal, tuple):
            input_name = 'DefaultControlAllocation for ' + control_signal[0] + '_ControlSignal'

        elif isinstance(control_signal, ParameterPort):
            input_name = 'DefaultControlAllocation for ' + control_signal.name + '_ControlSignal'

        else:
            raise DefaultControlMechanismError("control signal ({}) was not a dict, tuple, or ParameterPort".
                                               format(control_signal))

        # Instantiate input_ports and control_allocation attribute for control_signal allocations
        self._instantiate_default_input_port(input_name, [defaultControlAllocation], context=context)
        self.control_allocation = self.input_values

        # Call super to instantiate ControlSignal
        # Note: any params specified with ControlProjection for the control_signal
        #           should be in PARAMS entry of dict passed in control_signal arg
        control_signal = super()._instantiate_control_signal(control_signal=control_signal, context=context)

    def _instantiate_default_input_port(self, input_port_name, input_port_value, context=None):
        """Instantiate inputPort for ControlMechanism

        NOTE: This parallels ObjectMechanism._instantiate_input_port_for_monitored_port()
              It is implemented here to spare having to instantiate a "dummy" (and superfluous) ObjectiveMechanism
              for the sole purpose of creating input_ports for each value of defaultControlAllocation to assign
              to the ControlProjections.

        Extend self.defaults.variable by one item to accommodate new inputPort
        Instantiate the inputPort using input_port_name and input_port_value
        Update self.input_port and self.input_ports

        Args:
            input_port_name (str):
            input_port_value (2D np.array):
            context:

        Returns:
            input_port (InputPort):

        """

        # First, test for initialization conditions:

        # This is for generality (in case, for any subclass in the future, variable is assigned to None on init)
        if self.defaults.variable is None:
            self.defaults.variable = np.atleast_2d(input_port_value)

        # If there is a single item in self.defaults.variable, it could be the one assigned on initialization
        #     (in order to validate ``function`` and get its return value as a template for value);
        #     in that case, there should be no input_ports yet, so pass
        #     (i.e., don't bother to extend self.defaults.variable): it will be used for the new inputPort
        elif len(self.defaults.variable) == 1:
            if self.input_ports:
                self.defaults.variable = np.append(self.defaults.variable, np.atleast_2d(input_port_value), 0)
            else:
                # If there are no input_ports, this is the usual initialization condition;
                # Pass to create a new inputPort that will be assigned to existing the first item of self.defaults.variable
                pass
        # Other than on initialization (handled above), it is a PROGRAM ERROR if
        #    the number of input_ports is not equal to the number of items in self.defaults.variable
        elif len(self.defaults.variable) != len(self.input_ports):
            raise DefaultControlMechanismError(
                "PROGRAM ERROR:  The number of input_ports ({}) does not match "
                "the number of items found for the variable attribute ({}) of {}"
                "when creating {}".format(
                    len(self.input_ports),
                    len(self.defaults.variable),
                    self.name,
                    input_port_name,
                )
            )

        # Extend self.defaults.variable to accommodate new inputPort
        else:
            self.defaults.variable = np.append(self.defaults.variable, np.atleast_2d(input_port_value), 0)

        variable_item_index = self.defaults.variable.size - 1

        # Instantiate inputPort
        from psyneulink.core.components.ports.port import _instantiate_port
        from psyneulink.core.components.ports.inputport import InputPort
        input_port = _instantiate_port(owner=self,
                                         port_type=InputPort,
                                         name=input_port_name,
                                         reference_value=np.array(self.defaults.variable[variable_item_index]),
                                         reference_value_name='Default control allocation',
                                         params=None,
                                         context=context)

        #  Update inputPort and input_ports
        if self.input_ports:
            self._input_ports[input_port.name] = input_port
        else:
            from psyneulink.core.components.ports.port import Port_Base
            self._input_ports = ContentAddressableList(component_type=Port_Base,
                                                        list=[input_port],
                                                        name=self.name + '.input_ports')

        # self.input_value = [port.value for port in self.input_ports]

        return input_port
