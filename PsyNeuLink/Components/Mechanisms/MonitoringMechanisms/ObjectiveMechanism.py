# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ObjectiveMechanism *******************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**
One inputState is assigned to each of the
`outputStates <OutputState>` that have been specified to be evaluated. The EVCMechanism's
`MONITOR_FOR_CONTROL <monitor_for_control>` parameter is used to specify which outputStates are evaluated, and how.
The contribution of each outputState to the overall evaluation can be specified by an exponent and/or a weight
(see `ControlMechanism_Monitored_OutputStates` for specifying monitored outputStates; and
`below <EVCMechanism_Examples>` for examples). By default, the value of the EVCMechanism's `MONITOR_FOR_CONTROL`
parameter is `MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES`, which specifies monitoring the
`primary outputState <OutputState_Primary>` of every `TERMINAL` mechanism in the system, each of which is assigned an
exponent and weight of 1.  When an EVCMechanism is `created automatically <EVCMechanism_Creation>`, an inputState is
created for each outputState specified in its `MONITOR_FOR_CONTROL` parameter,  and a `MappingProjection` is created
that projects to that inputState from the outputState to be monitored.  The outputStates of a system being monitored
by an EVCMechanism are listed in its `monitored_output_states` attribute.

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
                 function=LinearCombination,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """
        """

        if default_input_value is None:
            default_input_value = self.variableClassDefault

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  monitor=monitor,
                                                  params=params)

        super().__init__(variable=default_input_value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate monitor argument
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        #region VALIDATE MONITORED STATES (for use by ControlMechanism)
        # Note: this must be validated after OUTPUT_STATES (and therefore call to super._validate_params)
        #       as it can reference entries in that param
        try:
            if not target_set[MONITOR_FOR_CONTROL] or target_set[MONITOR_FOR_CONTROL] is NotImplemented:
                pass
            # It is a MonitoredOutputStatesOption specification
            elif isinstance(target_set[MONITOR_FOR_CONTROL], MonitoredOutputStatesOption):
                # Put in a list (standard format for processing by _instantiate_monitored_output_states)
                target_set[MONITOR_FOR_CONTROL] = [target_set[MONITOR_FOR_CONTROL]]
            # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or States
            else:
                # Validate each item of MONITOR_FOR_CONTROL
                for item in target_set[MONITOR_FOR_CONTROL]:
                    validate_monitored_state(self, item, context=context)
                # FIX: PRINT WARNING (IF VERBOSE) IF WEIGHTS or EXPONENTS IS SPECIFIED,
                # FIX:     INDICATING THAT IT WILL BE IGNORED;
                # FIX:     weights AND exponents ARE SPECIFIED IN TUPLES
                # FIX:     WEIGHTS and EXPONENTS ARE VALIDATED IN SystemContro.Mechanism_instantiate_monitored_output_states
                # # Validate WEIGHTS if it is specified
                # try:
                #     num_weights = len(target_set[FUNCTION_PARAMS][WEIGHTS])
                # except KeyError:
                #     # WEIGHTS not specified, so ignore
                #     pass
                # else:
                #     # Insure that number of weights specified in WEIGHTS
                #     #    equals the number of states instantiated from MONITOR_FOR_CONTROL
                #     num_monitored_states = len(target_set[MONITOR_FOR_CONTROL])
                #     if not num_weights != num_monitored_states:
                #         raise MechanismError("Number of entries ({0}) in WEIGHTS of kwFunctionParam for EVC "
                #                        "does not match the number of monitored states ({1})".
                #                        format(num_weights, num_monitored_states))
        except KeyError:
            pass
        #endregion

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

        for item in self.monitor:
            self._instantiate_input_state_for_monitored_state(item, context=context)

        # self.inputValue = self.variableClassDefault = self.variable.copy() * 0.0
        self.inputValue = self.variableClassDefault = self.variable.copy() * 0.0

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

        # variable_item_index = self.variable.size-1
        variable_item_index = self.variable.shape[0]-1

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


    def add_monitored_states(self, states_spec, context=None):
        """Validate specification and then add inputState to ObjectiveFunction + MappingProjection to it from state

        Use by other objects to add a state or list of states to be monitored by EVC
        states_spec can be a Mechanism, OutputState or list of either or both
        If item is a Mechanism, each of its outputStates will be used

        Args:
            states_spec (Mechanism, MechanimsOutputState or list of either or both:
            context:
        """
        states_spec = list(states_spec)
        validate_monitored_state(self, states_spec, context=context)
        self._instantiate_monitored_output_states(states_spec, context=context)

def validate_monitored_state(self, state_spec, context=None):
    """Validate specification is a Mechanism or OutputState, the name of one, or a MonitoredOutpuStatesOption value

    Called by both self._validate_params() and self.add_monitored_state()
    """
    state_spec_is_OK = False

    if isinstance(state_spec, MonitoredOutputStatesOption):
        state_spec_is_OK = True

    if isinstance(state_spec, tuple):
        if len(state_spec) != 3:
            raise MechanismError("Specification of tuple ({0}) in MONITOR_FOR_CONTROL for {1} "
                                 "has {2} items;  it should be 3".
                                 format(state_spec, self.name, len(state_spec)))

        if not isinstance(state_spec[1], numbers.Number):
            raise MechanismError("Specification of the exponent ({0}) for MONITOR_FOR_CONTROL of {1} "
                                 "must be a number".
                                 format(state_spec, self.name, state_spec[0]))

        if not isinstance(state_spec[2], numbers.Number):
            raise MechanismError("Specification of the weight ({0}) for MONITOR_FOR_CONTROL of {1} "
                                 "must be a number".
                                 format(state_spec, self.name, state_spec[0]))

        # Set state_spec to the output_state item for validation below
        state_spec = state_spec[0]

    from PsyNeuLink.Components.States.OutputState import OutputState
    if isinstance(state_spec, (Mechanism, OutputState)):
        state_spec_is_OK = True

    if isinstance(state_spec, str):
        # FIX: TEST THAT STR IS THE NAME OF AN OUTPUTSTATER OF A MECHANISM IN THE SELF.SYSTEM
        # if state_spec in self.paramInstanceDefaults[OUTPUT_STATES]:
        if any(state_spec in m.outputStates for m in self.system.mechanisms):
            state_spec_is_OK = True
    try:
        self.outputStates[state_spec]
    except (KeyError, AttributeError):
        pass
    else:
        state_spec_is_OK = True

    if not state_spec_is_OK:
        raise ObjectiveError("Specification of state to be monitored ({0}) by {1} is not "
                             "a Mechanism, OutputState, the name of one, or a MonitoredOutputStatesOption value".
                             format(state_spec, self.name))
