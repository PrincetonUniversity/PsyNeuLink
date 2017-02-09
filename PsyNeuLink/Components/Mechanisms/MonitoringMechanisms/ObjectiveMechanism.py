# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ObjectiveMechanism *******************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

"""

from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.Functions.Function import LinearCombination

OBJECT = 0
WEIGHT = 1
EXPONENT = 2

OBJECTIVE_RESULT = "ObjectiveResult"

class ObjectiveError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ObjectiveMechanism(MonitoringMechanism_Base):
    """Implement ObjectiveMechanism subclass
    """

    componentType = OBJECTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ObjectiveCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = [[0],[0]]  # By default, ObjectiveMechanism compares two 1D np.array inputStates

    # ObjectiveMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        TIME_SCALE: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        OUTPUT_STATES:[{NAME:OBJECTIVE_RESULT}]})
        # MODIFIED 12/7/16 NEW:

    paramNames = paramClassDefaults.keys()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 monitor=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """
        """

        if default_input_value is None:
            default_input_value = self.variableClassDefault

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitor=monitor,
                                                  params=params)

        super().__init__(variable=default_input_value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)


    def _validate_monitored_state_spec(self, state_spec, context=None):
        """Validate specified outputstate is for a Mechanism in the System

        Called by both self._validate_params() and self.add_monitored_state() (in ControlMechanism)
        """
        super(ControlMechanism_Base, self)._validate_monitored_state(state_spec=state_spec, context=context)

        # Get outputState's owner
        from PsyNeuLink.Components.States.OutputState import OutputState
        if isinstance(state_spec, OutputState):
            state_spec = state_spec.owner

        # Confirm it is a mechanism in the system
        if not state_spec in self.system.mechanisms:
            raise ObjectiveError("Request for controller in {0} to monitor the outputState(s) of "
                                              "a mechanism ({1}) that is not in {2}".
                                              format(self.system.name, state_spec.name, self.system.name))

        # Warn if it is not a terminalMechanism
        if not state_spec in self.system.terminalMechanisms.mechanisms:
            if self.prefs.verbosePref:
                print("Request for controller in {0} to monitor the outputState(s) of a mechanism ({1}) that is not"
                      " a terminal mechanism in {2}".format(self.system.name, state_spec.name, self.system.name))



    def _instantiate_input_states(self, context=None):

        # Clear self.variable, as items will be assigned in call(s) to _instantiate_input_state_for_monitored_state()
        self.variable = None

        from PsyNeuLink.Components.States.OutputState import OutputState

        # Instantiate inputState for each monitored state in the list
        # from Components.States.OutputState import OutputState

        # FIX: PARSE LIST HERE, STANDARDINZING FORMAT INTO (ITEM, (WEIGHT, EXPONENT)) TUPLES
        # FIX: SHOULD THIS RESPECT SPECIFICATIONS ON THE MECHANISM (OR, IF NOT SPECIFIED THERE, THE SYSTEM)
        # FIX:    FOR WHICH OUTPUT STATES TO INCLUDE?  -- SEE _get_monitored_states IN EVCMechanism
        # FIX:    IF OUTPUTSTATES MONITOR_FOR_CONTROL = NONE, THEN WARN AND IGNORE (DON'T CREATE INPUTSTATE)
        #     if isinstance(item, OutputState):
        #         self._instantiate_input_state_for_monitored_state(item, context=context)
        #     elif isinstance(item, Mechanism):
        #         for output_state in item.outputStates:
        #             self._instantiate_input_state_for_monitored_state(output_state, context=context)
        #     else:
        #         raise ObjectiveError("PROGRAM ERROR: outputState specification ({0}) slipped through that is "
        #                              "neither an OutputState nor Mechanism".format(item))

        monitored_items = list(zip(*self.monitor))[0]
        weights_and_exponents = list(zip(*self.monitor))[1]

        for item in monitored_items:
            self._instantiate_input_state_for_monitored_state(item, context=context)

        if self.prefs.verbosePref:
            print ("{0} monitoring:".format(self.name))
            for state in self.monitoredOutputStates:
                weight = self.monitor_for_control_weights_and_exponents[self.monitoredOutputStates.index(state)][0]
                exponent = self.monitor_for_control_weights_and_exponents[self.monitoredOutputStates.index(state)][1]

                print ("\t{0} (exp: {1}; wt: {2})".format(state.name, weight, exponent))

        self.inputValue = self.variable.copy() * 0.0


    # def _instantiate_control_mechanism_input_state(self, input_state_name, input_state_value, context=None):
    def _instantiate_input_state_for_monitored_state(self,monitored_state, context=None):
        """Instantiate inputState with projection from monitoredOutputState

        Validate specification for state to be monitored
        Instantiate the inputState (assign name, and value of monitored_state)
        Extend self.variable by one item to accommodate new inputState
        Update self.inputState and self.inputStates
        Instantiate MappingProjection to inputState from monitored_state

        Args:
            input_state_name (str):
            input_state_value (2D np.array):
            context:

        Returns:
            input_state (InputState):

        """

        self._validate_monitored_state_spec(monitored_state, context=context)

        input_state_name = monitored_state.owner.name + '_' + monitored_state.name + '_Monitor'
        input_state_value = monitored_state.value

        # First, test for initialization conditions:

        # This is for generality (in case, for any subclass in the future, variable is assigned to None on init)
        if self.variable is None:
            self.variable = np.atleast_2d(input_state_value)

        # If there is a single item in self.variable, it could be the one assigned on initialization
        #     (in order to validate ``function`` and get its return value as a template for self.value);
        #     in that case, there should be no inputStates yet, so pass
        #     (i.e., don't bother to extend self.variable): it will be used for the new inputState
        elif len(self.variable) == 1:
            try:
                self.inputStates
            except AttributeError:
                # If there are no inputStates, this is the usual initialization condition;
                # Pass to create a new inputState that will be assigned to existing the first item of self.variable
                pass
            else:
                self.variable = np.append(self.variable, np.atleast_2d(input_state_value), 0)
        # Other than on initialization (handled above), it is a PROGRAM ERROR if
        #    the number of inputStates is not equal to the number of items in self.variable
        elif len(self.variable) != len(self.inputStates):
            raise ObjectiveError("PROGRAM ERROR:  The number of inputStates ({}) does not match "
                                 "the number of items found for the variable attribute ({}) of {}"
                                 "when creating {}".
                                 format(len(self.inputStates),
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

        #  Update inputState and inputStates
        try:
            self.inputStates[input_state.name] = input_state
        except AttributeError:
            self.inputStates = OrderedDict({input_state_name:input_state})
            self.inputState = list(self.inputStates.values())[0]

        self.inputValue = list(state.value for state in self.inputStates.values())

        # Instantiate MappingProjection from monitored_state to new input_state
        from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
        MappingProjection(sender=monitored_state, receiver=input_state, matrix=IDENTITY_MATRIX)


    def _add_monitored_states(self, states_spec, context=None):
        """Validate and then instantiate outputStates to be monitored by EVC

        Use by other objects to add a state or list of states to be monitored by EVC
        states_spec can be a Mechanism, OutputState or list of either or both
        If item is a Mechanism, each of its outputStates will be used
        All of the outputStates specified must be for a Mechanism that is in self.System

        Args:
            states_spec (Mechanism, MechanimsOutputState or list of either or both:
            context:
        """
        states_spec = list(states_spec)
        self._validate_monitored_state_spec(states_spec, context=context)
        self._instantiate_monitored_output_states(states_spec, context=context)

