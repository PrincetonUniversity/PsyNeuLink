# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NEW DOCUMENTATION:
#  input_states is used to name and/or specify the value of the input_states; it can be:
#      just a list of strings (used as names)
#      just a list of values (used as values of inputState, overrides use of monitored_values such as for RL)
#      if both, use dict with NAME and VARIABLE entries
#      # if input_states must == monitored_values
#      if none are specified, they will autojatically be created based on the monitored_values spec
#  monitored_values no longer takes an inputState specification (that must be in input_states)
#      if it is a name or value, not projection is called for (assumes that it is a TARGET, and that the projection
#                    will be implemented by Composition

# *********************************************  ObjectiveMechanism *******************************************************

"""

Overview
--------

An ObjectiveMechanism is a `ProcessingMechanism` that monitors the `outputStates <OutputState>` of one or more other
ProcessingMechanisms specified in its `monitor <ObjectiveMechanism.monitor>` attribute, and evaluates them using its
`function <ObjectiveMechanism.function>`. The result of the evaluation is placed in the ObjectiveMechanism's
`primary outputState <OutputState_Primary>`.  ObjectiveMechanisms are typically used closely (and often created
automatically) with `AdaptiveMechanisms <AdpativeMechanism>`.

.. _Comparator_Creation:

Creating a ComparatorMechanism
------------------------------

An ObjectiveMechanism can be created directly by calling its constructor.  ObjectiveMechanisms are also created
automatically when other PsyNeuLink components are created (such as `LearningMechanisms <LearningMechanism_Creation>`
and `ControlMechanisms <ControlMechanism_Creation>`.

.. _ObjectiveMechanism_Structure:

Structure
---------

An ObjectiveMechanism has one `inputState <InputState>` for each of the values that are specified
to be monitored in its `monitored_values` attribute.  When an ObjectiveMechanism is created, an inputState is created
for each of those values, and assigned a `MappingProjection` from the outputState to which the value belongs.  The
ObjectiveMechanism's `function  <ObjectiveMechanism.function>` uses these values to compute an `objective (or "loss")
function <https://en.wikipedia.org/wiki/Loss_function>`_, that is assigned as the value of its outputState.


COMMENT:
Input States
~~~~~~~~~~~~~~~~
ADD DOCUMENTATION HERE (SEE NEW DOCUMENTATION ABOVE)
COMMENT

.. _ObjectiveMechanism_Monitored_States:

Monitored Values
~~~~~~~~~~~~~~~~

The values to be monitored by an ObjectiveMechanism are specified in the :keyword:`monitored_values` argument of its
constructor.  These can be specified in a variety of ways, each of which must eventually resolve to an outputState, the
value of which is to be monitored.  Those outputStates are listed in the ObjectiveMechanism's `monitored_values`
attribute.

The number of items in `monitored_values` must match the length of the number of items in the 
**input_states** argument if it is specified
COMMENT:
, or the `default_input_value
<ObjectiveMechanism.Additional_Attributes>` if it is specified
COMMENT
.  Note that some forms of
specification may depend on specifications made for the outputState referenced, the mechanism to which it belongs,
and/or the process or system to which that mechanism belongs. These interactions (and the precedence afforded to
each) are described below.

If an outputState is specified at the time the ObjectiveMechanism is created, or the specification can be resolved
to an outputState, a MappingProjection is automatically created from it to the corresponding inputState
using `AUTO_ASSIGN_MATRIX` as its `matrix <MapppingProjection.matrix>` parameter.  If the outputState can't be
determined, no MappingProjection is assigned, and this must be done by some other means;  any values in
`monitored_values` that are not associated with an outputState at the time the ObjectiveMechanism is executed are
ignored.

The specification of item in `monitored_values` can take any of the following forms:

* **OutputState**:  a reference to the `outputState <OutputState>` of a mechanism.  This will create a
  `MappingProjection` from it to the corresponding inputState in `input_states <ObjectiveMechanism.input_states>`.
COMMENT: TBI
    Note that an outputState can be *excluded* from being monitored by assigning `None` as the value of its
    `monitoring_status` attribute.  This specification takes precedence over any others;  that is, it will suppress
    monitoring of that outputState, irrespective of any other specifications that might otherwise apply to that
    outputState, including those described below.
COMMENT
..
* **Mechanism**: by default, the mechanism's `primary outputState <OutputState_Primary>` will be used.  However,
  if the mechanism has any outputStates specified in its `monitored_states` attribute, those will be used (except for
  any that specify `None` as their `monitoring_status`). This specification takes precedence over any of the other
  types listed below:  if it is `None`, then none of that mechanism's outputStates will be monitored; if it
  specifies outputStates to be monitored, those will be monitored even if they do not satisfy any of the conditions
  described in the specifications below.
..
COMMENT: [OLD - REPLACED BY input_states ARG]
    * **InputState**:  this creates a "placemarker" inputState, that will later be assigned to an outputState to be
      monitored and a projection from it.  It can be any of the following:
    
      * **existing inputState**:  its name, value, and parameters will be used to create an identical
        inputState for the ObjectiveMechanism;
      |
      * `specification dictionary <InputState_Creation>` **for an inputState**:  the specifications will be used to
        create an inputState for the ObjectiveMechanism;
      |
      * **value**: a default inputState will be created using that value;
      |
      * **string**: a default inputState will be created using the string as its name, and a scalar as its value.
COMMENT

COMMENT: TBI
    .. _ObjectiveMechanism_OutputState_Tuple:

    * **MonitoredOutputState Tuple**  tuple can be used wherever an outputState can be specified, to determine how
      its value is combined with others by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`. Each
      tuple must have the three following items in the order listed:

          * an outputState or mechanism, the name of one, or a specification dictionary for one;
          ..
          * a weight (int) - multiplies the value of the outputState.
          ..
          * an exponent (int) - exponentiates the value of the outputState;
COMMENT
* **string**, **value** or **dict**: These can be used as placemarkers for a monitored_state that will be instantiated
  later (for example, for the TARGET input of a Composition).  If a string is specified, it is used as the
  default name of the corresponding inputState (specified in the `input_states <ObjectiveMechanism.input_states>`
  attribute of the ObjectiveMechanism) If a value is specified, it is used as the default value for the corresponding
  inputState.  If a dict is specified, it must have a single entry, the key of which will be used a string
  specification and the value as a value specification. 

Additional Attributes
~~~~~~~~~~~~~~~~~~~~~

* `default_input_value`
   This specifies the format of each value monitored by the ObjectiveMechanism and the variable for the corresponding
   inputState.  These values take precedence over the specification of values in `monitored_values`, and can be used
   to override the defaults assumed there. If `default_input_value` is specified, it must have the same number of items
   as `monitored_values`. If  `default_input_value` is `None` (the default), then the specifications in
   `monitored_values` are used.  The use of `default_input_value` to override defaults used by `monitored_values`
   can be helpful in some situations (see `example <ObjectiveMechanism_Default_Input_Value_Example>` below).


.. _ObjectiveMechanism_Function:

Function
~~~~~~~~

The ObjectiveMechanism's `function` can be customized to implement a wide variety of
`objective (or "loss") functions <https://en.wikipedia.org/wiki/Loss_function>`_.  The default is the
`LinearCombination` function, which simply sums the values of the outputStates listed in `monitored_values`.
However, this can easily be configured to calculate differnces, ratios, etc. (see
`example <ObjectiveMechanism_Weights_and_Exponents_Example>` below).  It can also be replaced with any
`CombinationFunction`, or any python function that takes a 2d array with an arbitrary number of
items or a number equal to the number of items in the ObjectiveMechanism's variable (and its number of
input_states), and returns a 1d array.

.. _ObjectiveMechanism_Execution:

Execution
---------

When an ObjectiveMechanism is executed, it updates its input_states with the values of the outputStates listed in
its `monitored_values` attribute, and then uses its `function <ObjectiveMechanism.function>` to
evaluate these.  The result is assigned as to its `value <ObjectiveMechanism.value>` attribute as the value of its
`primary outputState <OutputState_Primary>`.

.. _ObjectiveMechanism_Class_Reference:

Examples
--------

.. _ObjectiveMechanism_Default_Input_Value_Example:

*Formatting inputState values*

The use of default_input_value to override a specification in `monitored_values` can be useful in some situations.
For example, for `Reinforcement Learning <Reinforcement>`, an ObjectiveMechanism is used to monitor an action
selection mechanism.  In the example below, the latter uses a `TransferMechanism` with the `SoftMax` function (and the
`PROB <Softmax.PROB>` as its output format) to select the action.  This generates a vector with a single non-zero
value, which designates the predicted reward for the selected action.  Because the output is a vector,
by default the inputState of the ObjectiveMechanism created to monitor it will also be a vector.  However, the
ObjectiveMechanism requires that this be a single value, that it can compare with the value of the reward mechanism.
This can be dealt with by using `default_input_value` in the construction of the ObjectiveMechanism, to force
the inputState for the ObjectiveMechanism to have a single value, as in the example below::

    my_action_select_mech = TransferMechanism(default_input_value = [0,0,0],
                                function=SoftMax(output=PROB))

    my_reward_mech = TransferMechanism(default_input_value = [0])

    my_objective_mech = ObjectiveMechanism(default_input_value = [[0],[0]],
                                          monitored_values = [my_action_select_mech, my_reward_mech])

Note that the outputState for the `my_action_selection` and `my_reward_mech` are specified
in `monitored_values`.  If that were the only specification, the inputState created for `my_action_select_mech`
would be a vector of length 3.  This is overridden by specifying `default_input_value` as an array with two
single-value arrays (one corresponding to `my_action_select_mech` and the other to `my_reward_mech`).  This forces
the inputState for `my_action_select_mech` to have only a single element which, in turn, will cause a
MappingProjection to be created from  `my_action_select_mech` to the ObjectiveMechanism's inputState using a
`FULL_CONNECTIVITY_MATRIX` (the one used for `AUTO_ASSIGN_MATRIX` when the sender and receiver have values of
different lengths).  This produces the desired effect, since the action selected is the only non-zero value in the
output of `my_action_select_mech`, and so the `FULL_CONNECTIVITY_MATRIX` will combine it with zeros (the other values
in the vector), and so its value will be assigned as the value of the corresponding inputState in the
ObjectiveMechanism.  Another option would have been to customize the ObjectiveMechanism's
`function <ObjectiveMechanism.function>` to convert the output of `my_action_select_mech` to a length 1 vector, though
this would have been more involved.  The next example describes a simple case of customizing the ObjectiveMechanism's
`function <ObjectiveMechanism.function>`, however more sophisticated ones are possible, just as the one just suggested.

.. _ObjectiveMechanism_Weights_and_Exponents_Example:

*Customizing the ObjectiveMechanism's function*

The simplest way to customize the `function <ObjectiveMechanism.function>` of an ObjectiveMechanism is to
parameterize its default function (`LinearCombination`).  In the example below, the ObjectiveMechanism used in the
`previous example <ObjectiveMechanism_Default_Input_Value_Example>` is further customized to subtract the value
of the action selected from the value of the reward::

    my_objective_mech = ObjectiveMechanism(default_input_value = [[0],[0]],
                                          monitored_values = [my_action_select_mech, my_reward_mech],
                                          function=LinearCombination(weights=[[-1], [1]]))

This is done by specifying the `weights <LinearCombination.weights>` parameter of the `LinearCombination` function,
with two values [-1] and [1] corresponding to the two items in `monitored_values` (and `default_input_value`).  This
will multiply the value from `my_action_select_mech` by -1 before adding it to (and thus
subtracting it from) the value of `my_reward_mech`.  Similarly, the `operation <LinearCombination.operation>`
and `exponents <LinearCombination.exponents>` parameters of `LinearCombination` can be used together to multiply and
divide quantities.


Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import LinearCombination
# from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.States.InputState import InputState

OBJECT = 0
WEIGHT = 1
EXPONENT = 2
ROLE = 'role'
NAMES = 'names'
MONITORED_VALUES = 'monitored_values'
MONITORED_VALUE_NAME_SUFFIX = '_Monitor'
DEFAULT_MONITORED_VALUE = [0]


class ObjectiveMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ObjectiveMechanism(ProcessingMechanism_Base):
    """
    ObjectiveMechanism(           \
    monitored_values,             \
    input_states=None,            \
    names=None,                   \
    function=LinearCombination,   \
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

    COMMENT:
    default_input_value : Optional[List[array] or 2d np.array]
        specifies the format of the values monitored by the ObjectiveMechanism;  each item corresponds to the value
        of an outputState monitored, and to the value of the corresponding inputState of the ObjectiveMechanism.  It
        must have the same length as the number items in monitored_values.  The values specified here take precedence
        over those in :keyword:`monitored_values`;  if none are provided, the ones in :keyword:`monitored_values`
        will be used.
    COMMENT

    monitored_values : [List[OutputState, Mechanism, string, value, dict or MonitoredOutputStateOption]
        specifies the values that will will be monitored, and evaluated by the `function <ObjectiveMechanism>`
        (see `monitored_values` for details of specification).  The number of items must equal the length
        of `default_input_value` if that is specified.

    names: List[str]
        specifies the names to use for the input_states created for the list in
        `monitored_values <ObjectiveMechanism.monitor>`.  If specified,
        the number of items in the list must equal the number of items in `monitored_values`, and takes precedence
        over any names specified there.

    function: Function, function or method
        specifies the function used to evaluate the values listed in :keyword:`monitored_values`
        (see `function <LearningMechanism.function>` for details.

    role: Optional[LEARNING, CONTROL]
        specifies if the ObjectiveMechanism is being used for learning or control (see `role` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the
        constructor.

    COMMENT:
        [TBI]
        time_scale :  TimeScale : TimeScale.TRIAL
            specifies whether the mechanism is executed on the TIME_STEP or TRIAL time scale.
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

    COMMENT:
    default_input_value : Optional[List[array] or 2d np.array]
    COMMENT

    monitored_values : [List[OutputState]
        determines  the values monitored, and evaluated by `function <ObjectiveMechanism>`.  Once the
        ObjectiveMechanism is fully instantiated, each item in the list refers to an outputState containing the
        value to be monitored, with a `MappingProjection` from it to the corresponding inputState.

    function : CombinationFunction or function : default LinearCombination
        the function used to compare evaluate the values monitored by the ObjectiveMechanism.  The function can be
        any PsyNeuLink `CombinationFunction` or a python function that takes a 2d array with an arbitrary number of
        items or a number equal to the number of items in the ObjectiveMechanism's variable (and its number of
        input_states), and returns a 1d array.

    role : None, LEARNING or CONTROL
        specifies whether the ObjectiveMechanism is being used for learning in a process or system (in conjunction
        with a `LearningMechanism`), or for control in a system (in conjunction with a `ControlMechanism`).

    value : 1d np.array
        the output of the evaluation carried out by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.

    output_values : 2d np.array
        1st and only item is same as `value <ObjectiveMechanisms.value>`.

    name : str : default ComparatorMechanism-<index>
        the name of the mechanism.
        Specified in the **name** argument of the constructor for the mechanism;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for mechanism.
        Specified in the **prefs** argument of the constructor for the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentType = OBJECTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ObjectiveCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # variableClassDefault = [[0],[0]]  # By default, ObjectiveMechanism compares two 1D np.array input_states
    variableClassDefault = None

    # ObjectiveMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        MONITORED_VALUES: None,
        TIME_SCALE: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        OUTPUT_STATES:[{NAME:RESULT}]})
        # MODIFIED 12/7/16 NEW:

    paramNames = paramClassDefaults.keys()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 # MODIFIED 5/8/17 OLD:
                 # default_input_value=None,
                 # MODIFIED 5/8/17 END
                 monitored_values,
                 input_states=None,
                 # names:tc.optional(list)=None,
                 function=LinearCombination,
                 # role:tc.optional(str)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitored_values=monitored_values,
                                                  input_states=input_states,
                                                  function=function,
                                                  params=params)

        self._learning_role = None

        super().__init__(
                         # MODIFIED 5/8/17 OLD:
                         # variable=default_input_value,
                         # MODIFIED 5/8/17 NEW:
                         variable=None,
                         # MODIFIED 5/8/17 END
                         input_states=input_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

        # IMPLEMENATION NOTE: THIS IS HERE UNTIL Composition IS IMPLEMENTED,
        # SO THAT SYSTEMS AND PROCESSES CAN FIND THE OBJECTIVE MECHANISSMS SERVING AS TARGETS

    def _validate_variable(self, variable, context=None):
        """Validate that if default_input_value is specified the number of values matches the number of monitored_values

        """
        # IMPLEMENTATION NOTE:  use self.user_params (i.e., values specified in constructor)
        #                       since params have not yet been validated and so self.params is not yet available
        if variable is not None and len(variable) != len(self.user_params[MONITORED_VALUES]):
                raise ObjectiveMechanismError("The number of items specified for the default_input_value arg ({}) of {} "
                                     "must match the number of items specified for its monitored_values arg ({})".
                                     format(len(variable), self.name, len(self.user_params[MONITORED_VALUES])))

        super()._validate_variable(variable=variable, context=context)


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate `role`, `monitored_values`, amd `input_states <ObjectiveMechanism.input_states>` arguments

        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if ROLE in target_set and target_set[ROLE] and not target_set[ROLE] in {LEARNING, CONTROL}:
            raise ObjectiveMechanismError("\'role\'arg ({}) of {} must be either \'LEARNING\' or \'CONTROL\'".
                                          format(target_set[ROLE], self.name))

        if INPUT_STATES in target_set and target_set[INPUT_STATES] is not None:
            if MONITORED_VALUES in target_set:
                monitored_values = target_set[MONITORED_VALUES]
            elif hasattr(self, 'monitored_values'):
                monitored_values = self.monitored_values
            else:
                raise ObjectiveMechanismError("PROGRAM ERROR: monitored_values not instantiated as param or attirb")

            if len(target_set[INPUT_STATES]) != len(monitored_values):
                raise ObjectiveMechanismError("The number of items in the \'{}\'arg for {} ({}) "
                                              "must equal of the number in the \`{}\` arg ({})".
                                     format(INPUT_STATES,
                                            self.name,
                                            len(target_set[INPUT_STATES]),
                                            MONITORED_VALUES,
                                        len(target_set[MONITORED_VALUES])))

            #FIX: IS THIS HANDLED BY _instantiate_input_states??
            for state_spec in target_set[INPUT_STATES]:
                if not isinstance(state_spec, (str, InputState, Mechanism, dict)):
                    raise ObjectiveMechanismError("Item in \'{}\'arg for {} is not a "
                                                  "valid specificaton for an InputState".
                                                  format(INPUT_STATES,
                                                         self.name,
                                                         target_set[INPUT_STATES]))

        #region VALIDATE MONITORED VALUES
        # FIX: IS THE FOLLOWING STILL TRUE:
        # Note: this must be validated after OUTPUT_STATES (and therefore call to super._validate_params)
        #       as it can reference entries in that param
        if MONITORED_VALUES in target_set:
            try:
                if not target_set[MONITORED_VALUES] or target_set[MONITORED_VALUES] is NotImplemented:
                    pass
                # It is a MonitoredOutputStatesOption specification
                elif isinstance(target_set[MONITORED_VALUES], MonitoredOutputStatesOption):
                    # Put in a list (standard format for processing by _instantiate_monitored_output_states)
                    target_set[MONITORED_VALUES] = [target_set[MONITORED_VALUES]]
                # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or States
                else:
                    # Validate each item of MONITORED_VALUES
                    for item in target_set[MONITORED_VALUES]:
                        _validate_monitored_value(self, item, context=context)

                    # FIX: PRINT WARNING (IF VERBOSE) IF WEIGHTS or EXPONENTS IS SPECIFIED,
                    # FIX:  INDICATING THAT IT WILL BE IGNORED;
                    # FIX:  weights AND exponents ARE SPECIFIED IN TUPLES
                    # FIX:  WEIGHTS and EXPONENTS ARE VALIDATED IN
                    # FIX:           SystemContro.Mechanism_instantiate_monitored_output_states
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


    def _instantiate_input_states(self, context=None):
        """Instantiate input state for each value specified in `monitored_values` arg and instantiate self.variable
        
        Parse specifications for input_states, using monitored_values where relevant, and instantiate input_states. 
        Re-specify corresponding items of variable to match the values of the inputStates in input_states.
        Update self.input_state and self.input_states.
        Call _instantiate_monitoring_projection() to instantiate MappingProjection to inputState
            if an outputState has been specified.

        """

        # # FIX: THIS SHOULD DETERMINE WHETHER ARG IS A LIST OR A DICT (OF THE FORM: STATE_NAME:STATE_VALUE:
        # super()._instantiate_input_states(context=context)

        from PsyNeuLink.Components.States.State import _parse_state_spec

        # If input_states were not specified, instantiate empty list (to invoke default assignments)
        if self.input_states is None:
            self._input_states = [None] * len(self.monitored_values)

        # Parse monitored_values
        monitored_values = _parse_monitored_values(owner=self, monitored_values=self.monitored_values)

        # Parse input_states to determine whether its specifications need to be derived from monitored_value
        for i, input_state, monitored_value in zip(range(len(self.input_states)),
                                                   self.input_states,
                                                   monitored_values):

            # if monitored_values[i][PROJECTION]:
            #     projections = {SENDER: monitored_values[OUTPUT_STATE],
            #                    RECEIVER: inputState}


            # Parse input_state to determine its specifications and assign values from monitored_values
            #    to any missing projections, including any projections requested.
            self._input_states[i] = _parse_state_spec(self,
                                                     state_spec=input_state,
                                                     default_name=monitored_values[i][NAME],
                                                     default_value=monitored_values[i][VALUE],
                                                     projections=monitored_values[i][PROJECTION])

        constraint_value = []
        for input_state in self.input_states:
            constraint_value.append(input_state[VARIABLE])

        from PsyNeuLink.Components.States.State import _instantiate_state_list
        self._input_states = _instantiate_state_list(owner=self,
                                                     state_list=self.input_states,
                                                     state_type=InputState,
                                                     state_param_identifier=INPUT_STATE,
                                                     constraint_value=constraint_value,
                                                     constraint_value_name=self.__class__.__name__ + ' variable',
                                                     context=context)



        # ??ARE THESE DONE SOMEWHERE ELSE (E.G. IN super()._instantiate_input_states????
        self.variable = self.input_states.values.copy()
        self.variableClassDefault = self.variable.copy()

        # IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
        #                      SHOULD PROBABLY BE INTEGRATED INTO State MODULE (IN _instantate_state)
        # Instantiate inputState with projection from outputState specified by monitored_value
        for monitored_value, input_state in zip(self.monitored_values, self.input_states):
            if monitored_value[PROJECTION]:
                _instantiate_monitoring_projection(sender=monitored_value[OUTPUT_STATE],
                                                   receiver=input_state,
                                                   matrix=AUTO_ASSIGN_MATRIX)


@tc.typecheck
def _parse_monitored_values(owner, monitored_values:tc.any(list, dict)):
    """Parse specifications contained in monitored_values list or dict, 
    
    Can take either a list or dict of specifications.
    If it is a list, each item must be one of the following:
        - OuptutState
        - Mechanism
        - string
        - value
        - dict

    If it is a dict, each item must be an entry, the key of which must be a string that is used as a name
        specification, and the value of which can be any of the above. 
        
    Return a list of specification dicts, one for each item of monitored_values
    """

    from PsyNeuLink.Components.States.OutputState import OutputState

    def parse_spec(spec):

        # OutputState:
        if isinstance(spec, OutputState):
            name = spec.owner.name + MONITORED_VALUE_NAME_SUFFIX
            output_state = spec
            value = spec.value
            projections = True

        # Mechanism:
        elif isinstance(spec, Mechanism_Base):
            name = spec.name + MONITORED_VALUE_NAME_SUFFIX
            output_state = spec.output_state
            value = spec.output_state.value
            projections = True

        # # If spec is a MonitoredOutputStatesOption:
        # # FIX: NOT SURE WHAT TO DO HERE YET
        # elif isinstance(montiored_value, MonitoredOutputStateOption):
        #     value = ???
        #     projections = True

        # If spec is a string:
        # - use as name of inputState
        # - instantiate InputState with defalut value (1d array with single scalar item??)

        # str:
        elif isinstance(spec, str):
            name = spec
            output_state = DEFERRED_ASSIGNMENT
            value = DEFAULT_MONITORED_VALUE
            projections = False

        # value:
        elif is_value_spec(spec):
            name = owner.name + MONITORED_VALUE_NAME_SUFFIX
            output_state = DEFERRED_ASSIGNMENT
            value = spec
            projections = False

        elif isinstance(spec, tuple):
            # FIX: REPLACE CALL TO parse_spec WITH CALL TO _parse_state_spec
            name = owner.name + MONITORED_VALUE_NAME_SUFFIX
            output_state = DEFERRED_ASSIGNMENT
            value = spec[0]
            projections = spec[1]

        # dict:
        elif isinstance(spec, dict):

            name = None
            for k, v in spec.items():
                # Key is not a spec keyword, so dict must be of the following form: STATE_NAME_ASSIGNMENT:STATE_SPEC
                #
                if not k in {NAME, VALUE, STATE_PROJECTIONS}:
                    name = k
                    value = v

            if NAME in spec:
                name = spec[NAME]

            projections = False
            if STATE_PROJECTIONS in spec:
                projections = spec[STATE_PROJECTIONS]

            output_state = DEFERRED_ASSIGNMENT
            if OUTPUT_STATE in spec:
                output_state = spec[OUTPUT_STATE]

            if isinstance(spec[VALUE], (dict, tuple)):
                # FIX: REPLACE CALL TO parse_spec WITH CALL TO _parse_state_spec
                entry_name, value, projections = parse_spec(spec[VALUE])

            else:
                value = spec[VALUE]

        else:
            raise ObjectiveMechanismError("Specification for {} arg of {} ({}) must be an "
                                          "OutputState, Mechanism, value or string".
                                          format(MONITORED_VALUES, owner.name, spec))

        return name, output_state, value, projections

    # If it is a dict, convert to list by:
    #    - assigning the key of each entry to a NAME entry of the dict
    #    - placing the value in a VALUE entry of the dict
    if isinstance(monitored_values, dict):
        monitored_values_list = []
        for name, spec in monitored_values.items():
            monitored_values_list.append({NAME: name, VALUE: spec})
        monitored_values = monitored_values_list

    if isinstance(monitored_values, list):

        for i, monitored_value in enumerate(monitored_values):
            name, output_state, value, projections = parse_spec(monitored_value)
            monitored_values[i] = {NAME: name,
                                   OUTPUT_STATE: output_state,
                                   VALUE: value,
                                   PROJECTION: projections}

    else:
        raise ObjectiveMechanismError("{} arg for {} ({} )must be a list or dict".
                                      format(MONITORED_VALUES, owner.name, monitored_values))

    return monitored_values


    # def add_monitored_values(self, states_spec, context=None):
    #     """Validate specification and then add inputState to ObjectiveFunction + MappingProjection to it from state
    #
    #     Use by other objects to add a state or list of states to be monitored by EVC
    #     states_spec can be a Mechanism, OutputState or list of either or both
    #     If item is a Mechanism, each of its outputStates will be used
    #
    #     Args:
    #         states_spec (Mechanism, MechanimsOutputState or list of either or both:
    #         context:
    #     """
    #     states_spec = list(states_spec)
    #     validate_monitored_value(self, states_spec, context=context)
    #     self._instantiate_monitored_output_states(states_spec, context=context)


def _validate_monitored_value(objective_mech, state_spec, context=None):
    """Validate specification for monitored_value arg

    Validate that each item of monitored_value arg is: 
        * OutputState
        * Mechanism, 
        * string, or 
        * MonitoredOutpuStatesOption value.
    
    Called by both self._validate_variable(), self.add_monitored_value(), and EVCMechanism._get_monitored_states()
    """
    # state_spec_is_OK = False

    # if _is_value_spec(state_spec):
    #     state_spec_is_OK = True

    from PsyNeuLink.Components.States.OutputState import OutputState
    if not isinstance(state_spec, (str, OutputState, Mechanism, MonitoredOutputStatesOption)):
        state_spec_is_OK = True

    # if isinstance(state_spec, dict):
    #     state_spec_is_OK = True
    #
    # if isinstance(state_spec, str):
    #     # Will be used as the name of the inputState
    #     state_spec_is_OK = True

    # if isinstance(state_spec, MonitoredOutputStatesOption):
    #     state_spec_is_OK = True
    #
    # if not state_spec_is_OK:
        raise ObjectiveMechanismError("Specification of {} arg for {} ({}) must be"
                             "an OutputState, Mechanism, or a MonitoredOutputStatesOption value".
                             format(MONITORED_VALUES, objective_mech.name, state_spec))


def _objective_mechanism_role(mech, role):
    if isinstance(mech, ObjectiveMechanism):
        if mech._role is role:
            return True
        else:
            return False
    else:
        return False


# IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
def _instantiate_monitoring_projection(sender, receiver, matrix=DEFAULT_MATRIX):
    from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
    MappingProjection(sender=sender,
                      receiver=receiver,
                      matrix=matrix,
                      name = sender.name + ' monitor')