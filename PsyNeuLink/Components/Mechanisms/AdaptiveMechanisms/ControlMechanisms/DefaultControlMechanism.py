# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  DefaultControlMechanism ************************************************

"""

The DefaultControlMechanism is created for a `System` if no other controller type is specified. The 
DefaultControlMechanism createsan inputState for each ControlProjection it is assigned, and uses 
`defaultControlAllocation` as the value for the control signal.  By default,  :py:data:`defaultControlAllocation` =  1, 
so that ControlProjections from the DefaultControlMechanism have no effect on their parameters.  However, it can be 
used to uniformly control the parameters that receive ControlProjections from it, by manually changing the value of
`defaultControlAllocation`.  See :doc:`ControlMechanism` for additional details of how ControlMechanisms are
created, executed and their attributes.

COMMENT:
   ADD LINK FOR defaultControlAllocation

    TEST FOR defaultControlAllocation:  |defaultControlAllocation|

    ANOTHER TEST FOR defaultControlAllocation:  :py:print:`defaultControlAllocation`

    AND YET ANOTHER TEST FOR defaultControlAllocation:  :py:print:|defaultControlAllocation|

    LINK TO DEFAULTS: :doc:`Defaults`
COMMENT


"""

from collections import OrderedDict

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Components.ShellClasses import *


class DefaultControlMechanism(ControlMechanism_Base):
    """Implements the DefaultControlMechanism

    COMMENT:
        Description:
            Implements default source of control signals, with one inputState and outputState for each.
            Uses defaultControlAllocation as input(s) and pass value(s) unchanged to outputState(s) and ControlProjection(s)

            Every ControlProjection is assigned this mechanism as its sender by default (i.e., unless a sender is
                explicitly specified in its constructor).

            An inputState and outputState is created for each ControlProjection assigned:
                the inputState is assigned the
                :py:constant:`defaultControlAllocation <Defaults.defaultControlAllocation>` value;
                when the DefaultControlMechanism executes, it simply assigns the same value to the ControlProjection.

            Class attributes:
                + componentType (str): System Default Mechanism
                + paramClassDefaults (dict):
                    + FUNCTION: Linear
    COMMENT
    """

    componentType = "DefaultControlMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE

    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}


    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = ControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({# MAKE_DEFAULT_CONTROLLER:True  <- No need, it is the default by default
                               FUNCTION:Linear,
                               FUNCTION_PARAMS:{SLOPE:1, INTERCEPT:0},
                               MONITOR_FOR_CONTROL:None
                               })

    from PsyNeuLink.Components.Functions.Function import Linear
    @tc.typecheck
    def __init__(self,
                 # default_input_value=None,
                 system=None,
                 monitor_for_control:tc.optional(list)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        super(DefaultControlMechanism, self).__init__(#default_input_value =default_input_value,
                                                    monitor_for_control=monitor_for_control,
                                                    params=params,
                                                    name=name,
                                                    prefs=prefs,
                                                    context=self)

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):

        return self.input_values or [defaultControlAllocation]

    def _instantiate_input_states(self, context=None):
        """Instantiate input_value attribute

        Instantiate input_value, inputState and input_states attributes (in case they are referenced).
        Otherwise, no need to do anything, as DefaultControllerMechanism only adds input_states
        when a ControlProjection is instantiated, and uses _instantiate_control_mechanism_input_state to do so.

        """

        if not hasattr(self, INPUT_STATES):
            self._input_states = None
        # if self.input_states is None:
        #     self.input_value = None

    def _instantiate_control_projection(self, projection, params=None, context=None):
        """Instantiate requested controlProjection and associated inputState
        """

        # Instantiate input_states and allocation_policy attribute for control_signal allocations
        input_name = 'DefaultControlAllocation for ' + projection.receiver.name + '_ControlSignal'
        self._instantiate_default_input_state(input_name, defaultControlAllocation, context=context)
        self.allocation_policy = self.input_values

        # Call super to instantiate outputStates
        # Note: params carries any specified with ControlProjection for the control_signal
        super()._instantiate_control_projection(projection=projection,
                                                params=params,
                                                context=context)

    def _instantiate_default_input_state(self, input_state_name, input_state_value, context=None):
        """Instantiate inputState for ControlMechanism

        NOTE: This parallels ObjectMechanism._instantiate_input_state_for_monitored_state()
              It is implemented here to spare having to instantiate a "dummy" (and superfluous) ObjectiveMechanism
              for the sole purpose of creating input_states for each value of defaultControlAllocation to assign
              to the ControlProjections.

        Extend self.variable by one item to accommodate new inputState
        Instantiate the inputState using input_state_name and input_state_value
        Update self.input_state and self.input_states

        Args:
            input_state_name (str):
            input_state_value (2D np.array):
            context:

        Returns:
            input_state (InputState):

        """

        # First, test for initialization conditions:

        # This is for generality (in case, for any subclass in the future, variable is assigned to None on init)
        if self.variable is None:
            self.variable = np.atleast_2d(input_state_value)

        # If there is a single item in self.variable, it could be the one assigned on initialization
        #     (in order to validate ``function`` and get its return value as a template for self.value);
        #     in that case, there should be no input_states yet, so pass
        #     (i.e., don't bother to extend self.variable): it will be used for the new inputState
        elif len(self.variable) == 1:
            if self.input_states:
                self.variable = np.append(self.variable, np.atleast_2d(input_state_value), 0)
            else:
                # If there are no input_states, this is the usual initialization condition;
                # Pass to create a new inputState that will be assigned to existing the first item of self.variable
                pass
        # Other than on initialization (handled above), it is a PROGRAM ERROR if
        #    the number of input_states is not equal to the number of items in self.variable
        elif len(self.variable) != len(self.input_states):
            raise ControlMechanismError("PROGRAM ERROR:  The number of input_states ({}) does not match "
                                        "the number of items found for the variable attribute ({}) of {}"
                                        "when creating {}".
                                        format(len(self.input_states),
                                               len(self.variable),
                                               self.name,input_state_name))

        # Extend self.variable to accommodate new inputState
        else:
            self.variable = np.append(self.variable, np.atleast_2d(input_state_value), 0)

        variable_item_index = self.variable.size-1

        # Instantiate inputState
        from PsyNeuLink.Components.States.State import _instantiate_state
        from PsyNeuLink.Components.States.InputState import InputState
        input_state = _instantiate_state(owner=self,
                                         state_type=InputState,
                                         state_name=input_state_name,
                                         state_spec=defaultControlAllocation,
                                         state_params=None,
                                         constraint_value=np.array(self.variable[variable_item_index]),
                                         constraint_value_name='Default control allocation',
                                         context=context)

        #  Update inputState and input_states
        if self.input_states:
            self._input_states[input_state.name] = input_state
        else:
            from PsyNeuLink.Components.States.State import State_Base
            self._input_states = ContentAddressableList(component_type=State_Base, list=[input_state])

        # self.input_value = [state.value for state in self.input_states]

        return input_state
