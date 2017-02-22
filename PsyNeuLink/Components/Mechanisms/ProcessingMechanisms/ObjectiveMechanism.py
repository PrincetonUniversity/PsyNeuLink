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



Overview
--------

An ObjectiveMechanism monitors the `outputStates <OutputState>` of one or more `ProcessingMechanism` specified in its
`monitor <ObjectiveMechanism.monitor>` attribute, and evaluates them using its `function`.  The contribution of each
outputState to the evaluation can be specified by an exponent and/or a weight
(see `ControlMechanism_Monitored_OutputStates` for specifying monitored outputStates; and
`below <EVCMechanism_Examples>` for examples). By default, the value of the EVCMechanism's `MONITOR_FOR_CONTROL`
parameter is `MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES`, which specifies monitoring the
`primary outputState <OutputState_Primary>` of every `TERMINAL` mechanism in the system, each of which is assigned an
exponent and weight of 1.  When an EVCMechanism is `created automatically <EVCMechanism_Creation>`, an inputState is
created for each outputState specified in its `MONITOR_FOR_CONTROL` parameter,  and a `MappingProjection` is created
that projects to that inputState from the outputState to be monitored.  The outputStates of a system being monitored
by an EVCMechanism are listed in its `monitored_output_states` attribute.

.. _Comparator_Creation:

Creating a ComparatorMechanism
------------------------------

An ObjectiveMechanism can be created directly by calling its constructor.  ObjectiveMechanisms are also created
automatically by other types of mechanisms (such as an `EVCMechanism`
COMMENT:
   or a `LearningMechanism`).
COMMENT
).
The mechanisms and/or outputStates monitored by an ObjectiveMechanism are specified by the
:keyword:`monitor` argument in its constructor.  These can be specified in a variety of ways, and assigned
weights and/or exponents to specify the contribution of each to the evaluation (as described
`below <Monitored OutputStates>`); however all are converted to references to outputStates, that are listed in the
ObjectiveMechanism's `monitor <ObjectiveMechanism.monitor>` attribute.

.. _ObjectiveMechanism_Monitored_OutputStates:

Monitored OutputStates
~~~~~~~~~~~~~~~~~~~~~~

COMMENT:
     RE-WRITE TO INDICATE:  (SEE ATTRIBUTE DESCRIPTION FOR monitored_values)
     VALUES AND INPUTSTATE CAN BE SPECIFIED
     AUTOMATIC IMPLEMETNATION BY PROCESS AND/OR SYSTEM
     SPECIFCATION OF WEIGHTS AND EXPONENTS IN LinearCombination FUNCTION, AS SPECIAL CASE / EXAMPLE
COMMENT

The outputState(s) monitored by an ObjectiveMechanism can be specified in any of the places listed below.  The
list also describes the order of precedence when more than one specification pertains to the same
outputState(s). In all cases, specifications can be a references to an outputState object, or a string that is the
name of one (see :ref:ControlMechanism_Examples' below). The specification of whether an outputState is monitored by
a ControlMechanism can be done in the following places:

* **OutputState**: an outputState can be *excluded* from being monitored by assigning `None` as the value of the
  :keyword:`MONITOR_FOR_CONTROL` entry of a parameter specification dictionary in the outputState's ``params``
  argument.  This specification takes precedence over any others;  that is, specifying `None` will suppress
  monitoring of that outputState, irrespective of any other specifications that might otherwise apply to that
  outputState;  thus, it can be used to exclude the outputState for cases in which it would otherwise be monitored
  based on one of the other specification methods below.
..
* **Mechanism**: the outputState of a particular mechanism can be designated to be monitored, by specifying it in the
  `MONITOR_FOR_CONTROL` entry of a parameter specification dictionary in the mechanism's `params` argument.  The value
  of the entry must be either a list containing the outputState(s) and/or their name(s),
  a `monitoredOutputState tuple <ControlMechanism_OutputState_Tuple>`, a `MonitoredOutputStatesOption` value, or `None`.
  The values of `MonitoredOutputStatesOption` are treated as follows:

    * `PRIMARY_OUTPUT_STATES`: only the primary (first) outputState of the mechanism is monitored;
    |
    * `ALL_OUTPUT_STATES`:  all of the mechanism's outputStates are monitored.

  This specification takes precedence over any of the other types listed below:  if it is `None`, then none of
  that mechanism's outputStates will be monitored;   if it specifies outputStates to be monitored, those will be
  monitored even if the mechanism is not a `TERMINAL` mechanism (see below).
..
* **ControlMechanism** or **System**: outputStates to be monitored by a `ControlMechanism` can be specified in the
  ControlMechanism itself , or in the system for which that ControlMechanism is the `controller`.  The specification can
  be in the :keyword:`monitor_for_control` argument of the ControlMechanism or System's constructor, or in the
  `MONITOR_FOR_CONTROL` entry of a parameter specification dictionary in the `params` argument of the constructor.
  In either case, the value must be a list, each item of which must be one of the following:

  * an existing **outputState** or the name of one.
  |
  * a **mechanism** or the name of one -- only the mechanism's primary (first) outputState will be monitored,
    unless a `MonitoredOutputStatesOption` value is also in the list (see below) or the specification is
    overridden in a params dictionary for the mechanism (see above);
  |
  * a `monitoredOutputState tuple <ControlMechanism_OutputState_Tuple>`;
  |
  * a value of `MonitoredOutputStatesOption` --  this applies to any mechanisms that appear in the list
    (except those that override it with their own :keyword:`monitor_for_control` specification); if the value of
    `MonitoredOutputStatesOption` appears alone in the list, it is treated as follows:

    * `PRIMARY_OUTPUT_STATES` -- only the primary (first) outputState of the `TERMINAL` mechanism(s)
      in the system for which the ControlMechanism is the `controller` is monitored;
    |
    * `ALL_OUTPUT_STATES` -- all of the outputStates of the `TERMINAL` mechanism(s)
      in the system for which the ControlMechanism is the `controller` are monitored;
  * `None`.

  Specifications in a ControlMechanism take precedence over any in the system; both are superceded by specifications
  in the constructor or params dictionary for an outputState or mechanism.

.. _ObjectiveMechanism_OutputState_Tuple:

**MonitoredOutputState Tuple**

A tuple can be used wherever an outputState can be specified, to determine how its value is combined with others by
the ObjectiveMechanism's `function <ObjectiveMechanism.function>`. Each tuple must have the three following items in
the order listed:

  * an outputState or mechanism, the name of one, or a specification dictionary for one;
  ..
  * a weight (int) - multiplies the value of the outputState.
  ..
  * an exponent (int) - exponentiates the value of the outputState;

COMMENT:
    The set of weights and exponents assigned to each outputState is listed in the ObjectiveMechanism's
    `monitor_for_control_weights_and_exponents` attribute, in the same order as the outputStates are listed in its
    `monitored_output_states` attribute.  Each item in the list is a tuple with the weight and exponent for a given
    outputState.
COMMENT


.. _ObjectiveMechanism_Structure:

Structure
---------

An ObjectiveMechanism has an `inputState <InputState>` for each of the outputStates listed in its
`monitored_values` attribute. Each inputState receives a projection from the corresponding
outputState.

.. _ObjectiveMechanism_Execution:

Execution
---------

Each time an ObjectiveMechanism is executed, it updates its inputStates with the values of outputStates listed in
its `monitored_values` attribute, and then uses its `function <ObjectiveMechanism.function>` to
evaluate these.  The result is assigned as the value of its outputState.

.. _ObjectiveMechanism_Class_Reference:

Class Reference
---------------

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

# from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.Functions.Function import LinearCombination

OBJECT = 0
WEIGHT = 1
EXPONENT = 2
ROLE = 'role'
NAMES = 'names'
MONITORED_VALUES = 'monitored_values'

OBJECTIVE_RESULT = "ObjectiveResult"

class ObjectiveError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# class ObjectiveMechanism(MonitoringMechanism_Base):
class ObjectiveMechanism(ProcessingMechanism_Base):
    """Implement ObjectiveMechanism subclass
    """

    componentType = OBJECTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ObjectiveCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

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
                 monitored_values=None,
                 names:tc.optional(list)=None,
                 function=LinearCombination,
                 role:tc.optional(str)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """
        ObjectiveMechanism(           \
        monitored_values=None,        \
        names=None,                   \
        function=LinearCombination,   \
        role=None                     \
        params=None,                  \
        name=None,                    \
        prefs=None)

    Implements the ObjectiveMechanism subclass of `ProcessingMechanism`.

    COMMENT:
        Description:
            ObjectiveMechanism is a subtype of the ProcessingMechanism Type of the Mechanism Category of the
                Component class
            It's function uses the LinearCombination Function to compare two input variables
            COMPARISON_OPERATION (functionParams) determines whether the comparison is subtractive or divisive
            The function returns an array with the Hadamard (element-wise) differece/quotient of target vs. sample,
                as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

        Class attributes:
            + componentType (str): ComparatorMechanism
            + classPreference (PreferenceSet): Comparator_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (value):  Comparator_DEFAULT_STARTING_POINT // QUESTION: What to change here
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL,
                                          FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}
            + paramNames (dict): names as above

        Class methods:
            None

        MechanismRegistry:
            All instances of ComparatorMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    default_sample_and_target : Optional[List[array, array] or 2d np.array]
        the input to the ComparatorMechanism to use if none is provided in a call to its
        `execute <Mechanism.Mechanism_Base.execute>` or `run <Mechanism.Mechanism_Base.run>` methods.
        The first item is the `COMPARATOR_SAMPLE` item of the input and the second is the `COMPARATOR_TARGET`
        item of the input, which must be the same length.  This also serves as a template to specify the length of
        inputs to the `function <ComparatorMechanism.function>`.

    monitored_values : [List[value, InputState, OutputState, Mechanism, or MonitoredOutputStateOption]
        specifies the values that will will be monitored, and evaluated by `function <ObjectiveMechanism>`
        (see `monitored_values` for details of specification).

    names: List[str]
        specifies names for the outputStates listed in `monitored_values <ObjectiveMechanism.monitor>`.  If specified,
        the number of items in the list must equal the number of items in `monitored_values`.

    function: Function, function or method
        specifies the function used to evaluate the value of the outputStates listed in
        `monitored_values`.

    role: Optional[LEARNING, CONTROL]
        specifies if the ObjectiveMechanism is being used for learning or control.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  The following entries can be
        included:

        * `COMPARATOR_SAMPLE`:  Mechanism, InputState, or the name of or specification dictionary for one;
        ..
        * `COMPARATOR_TARGET`:  Mechanism, InputState, or the name of or specification dictionary for one;
        ..
        * `FUNCTION`: Function, function or method;  default is `LinearCombination`.

        Values specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    COMMENT:
        [TBI]
        time_scale :  TimeScale : TimeScale.TRIAL
            specifies whether the mechanism is executed on the :keyword:`TIME_STEP` or :keyword:`TRIAL` time scale.
            This must be set to :keyword:`TimeScale.TIME_STEP` for the ``rate`` parameter to have an effect.
    COMMENT

    name : str : default ComparatorMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    monitored_values : [List[value, InputState, OutputState, Mechanism, or MonitoredOutputStateOption]
        serves as the ObjectiveMechahnism's variable;  determines  the values monitored, and evaluated by
        `function <ObjectiveMechanism>`.  An `inputState <InputState>` is created in the ObjectiveMechanism for each
        item in the list.  It is assumed, for any item that is a value or an inputState that has no projections to
        it, these will be generated later (possibly automatically, as part of a `Process` or `System`).  For any item
        that is an `OutputState`, `Mechanism`, or `MonitoredOutputStateOption`, projections will be created
        automatically by any `systems <System>` and/or `processes <Process>` to which the ObjectiveMechanism belongs.

    function : CombinationFunction : default LinearCombination
        the function used to compare `COMPARATOR_SAMPLE` with `COMPARATOR_TARGET`.

    value : 2d np.array
        holds the output of the `comparison_operation` carried out by the ComparatorMechanism's
        `function <ComparatorMechanism.function>`; its value is  also assigned to the `COMPARISON_RESULT` outputState
        and the the first item of `outputValue <ComparatorMechanism.outputValue>`.

    outputValue : List[1d np.array, float, float, float, float]
        a list with the following items:

        * **result** of the `function <ComparatorMechanism.function>` calculation
          and the :keyword:`value` of the `COMPARISON_RESULT` outputState;
        * **mean** of the result's elements and the :keyword:`value` of the `COMPARISON_MEAN` outputState;
        * **sum** of the result's elements and the :keyword:`value` of the `COMPARISON_SUM` outputState;
        * **sum of squares** of the result's elements and the :keyword:`value` of the `COMPARISON_SSE` outputState;
        * **mean of squares** of the result's elements and the :keyword:`value` of the `COMPARISON_MSE` outputState.

    name : str : default ComparatorMechanism-<index>
        the name of the mechanism.
        Specified in the `name` argument of the constructor for the mechanism;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for mechanism.
        Specified in the `prefs` argument of the constructor for the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


        """

        if monitored_values is None:
            default_input_value = self.variableClassDefault

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(names=names,
                                                  function=function,
                                                  role=role,
                                                  params=params)

        super().__init__(variable=monitored_values,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

        # IMPLEMENATION NOTE: THIS IS HERE UNTIL Composition IS IMPLEMENTED,
        # SO THAT SYSTEMS AND PROCESSES CAN FIND THE OBJECTIVE MECHANISSMS SERVING AS TARGETS
        self.learning_role = None


    def _validate_variable(self, variable, context=None):
        pass


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate monitored_values argument
        """
        # FIX: NEED TO VALIDATE FOR VALUES AND/OR INPUT STATES AS WELL AS OUTPUTSTATESS, MECHAINSMS AND OPTIONS
        # FIX: AND TEST THAT IT ALLOWS THESE

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if target_set[ROLE] and not target_set[ROLE] in {LEARNING, CONTROL}:
            raise ObjectiveError("\'role\'arg ({}) of {} must be either \'LEARNING\' or \'CONTROL\'".
                                 format(target_set[ROLE], self.name))

        if target_set[NAMES]:
            if len(target_set[NAMES]) != len(target_set[MONITORED_VALUES]):
                raise ObjectiveError("The number of items in \'names\'arg ({}) must equal of the number in the "
                                     "\`monitored_values\` arg for {}".
                                     format(len(target_set[NAMES]), len(target_set[MONITORED_VALUES]), self.name))

            for name in target_set[NAMES]:
                if not isinstance(name, str):
                    raise ObjectiveError("it in \'names\'arg ({}) of {} is not a string".
                                         format(target_set[NAMES], self.name))

        #region VALIDATE MONITORED STATES
        # MOVE THIS TO _validate_variable() ABOVE
        # FIX: IS THE FOLLOWING STILL TRUE:
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
        """Instantiate input state for each outputState specified in `monitored_values` arg

          Note:  assumes that `monitored_values` has been parse into a list of outputStates
                 in _validate_monitored_state()
        """

        # Clear self.variable, as items will be assigned in call(s) to _instantiate_input_state_for_monitored_state()
        self.variable = None

        from PsyNeuLink.Components.States.OutputState import OutputState

        # Instantiate inputState for each monitored state in the list
        # from Components.States.OutputState import OutputState

        # FIX: PARSE LIST HERE, STANDARDIZING FORMAT INTO (ITEM, (WEIGHT, EXPONENT)) TUPLES
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

        names = self.names or [None] * len(self.monitored_values)
        for monitored_state, name in zip(self.monitored_values, names):
            self._instantiate_input_state_for_monitored_state(monitored_state, name, context=context)

        # self.inputValue = self.variableClassDefault = self.variable.copy() * 0.0
        self.inputValue = self.variableClassDefault = self.variable.copy() * 0.0

    def _instantiate_input_state_for_monitored_state(self,monitored_state, name=None, context=None):
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
        input_state_value = monitored_state.value
        input_state_name = name or monitored_state.owner.name + '_' + monitored_state.name + '_Monitor'

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

        # IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
        _instantiate_monitoring_projection(sender=monitored_state, receiver=input_state, matrix=AUTO_ASSIGN_MATRIX)


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

def objective_mechanism_role(mech, role):
    if isinstance(mech, ObjectiveMechanism):
        if mech.role is role:
            return True
        else:
            return False
    else:
        return False

# IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
def _instantiate_monitoring_projection(sender, receiver, matrix=DEFAULT_MATRIX):
    from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
    MappingProjection(sender=sender, receiver=receiver, matrix=matrix)