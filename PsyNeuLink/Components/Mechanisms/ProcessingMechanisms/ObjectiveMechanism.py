# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

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

.. _ObjectiveMechanism_Monitored_OutputStates:


Monitored Values
~~~~~~~~~~~~~~~~

The values to be monitored by an ObjectiveMechanism are specified in the :keyword:`monitored_values` argument of its
constructor.  These can be specified in a variety of ways, each of which must eventually resolve to an outputState, the
value of which is to be monitored.  Those outputStates are listed in the ObjectiveMechanism's `monitored_values`
attribute.

The number of items in `monotired_values` must match the length of the `default_input_value
<ObjectiveMechanism.Additional_Attributes>` if it is specified.  Note that some forms of
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

* **OutputState**:  a reference to the `outputState <OutputState>` of a mechanism.  This will create an inputState
  using the outputState's value as the template for its variable.
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
  specifies outputStates to be monitored, those will be monitored even if they do not satisify any of the conditions
  described in the specifications below.
..
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

OBJECTIVE_RESULT = "ObjectiveResult"


class ObjectiveMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ObjectiveMechanism(ProcessingMechanism_Base):
    """
    ObjectiveMechanism(           \
    default_input_value=None,     \
    monitored_values=None,        \
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

    default_input_value : Optional[List[array] or 2d np.array]
        specifies the format of the values monitored by the ObjectiveMechanism;  each item corresponds to the value
        of an outputState monitored, and to the value of the corresponding inputState of the ObjectiveMechanism.  It
        must have the same length as the number items in monitored_values.  The values specified here take precedence
        over those in :keyword:`monitored_values`;  if none are provided, the ones in :keyword:`monitored_values`
        will be used.

    monitored_values : [List[value, InputState, OutputState, Mechanism, string, or MonitoredOutputStateOption]
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

    monitored_values : [List[OutputState, Mechanism, InputState, dict, value, or str]
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

    outputValue : 2d np.array
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
        TIME_SCALE: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        OUTPUT_STATES:[{NAME:OBJECTIVE_RESULT}]})
        # MODIFIED 12/7/16 NEW:

    paramNames = paramClassDefaults.keys()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 monitored_values=None,
                 names:tc.optional(list)=None,
                 function=LinearCombination,
                 # role:tc.optional(str)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # if default_input_value is None:
        #     default_input_value = self.variableClassDefault

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitored_values=monitored_values,
                                                  names=names,
                                                  function=function,
                                                  params=params)

        self._learning_role = None

        super().__init__(variable=default_input_value,
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
        """Validate `monitored_values`, `role` and `names <ObjectiveMechanism.names>` arguments

        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if ROLE in target_set and target_set[ROLE] and not target_set[ROLE] in {LEARNING, CONTROL}:
            raise ObjectiveMechanismError("\'role\'arg ({}) of {} must be either \'LEARNING\' or \'CONTROL\'".
                                 format(target_set[ROLE], self.name))

        if NAMES in target_set and target_set[NAMES]:
            if len(target_set[NAMES]) != len(target_set[MONITORED_VALUES]):
                raise ObjectiveMechanismError("The number of items in \'names\'arg ({}) must equal of the number in the "
                                     "\`monitored_values\` arg for {}".
                                     format(len(target_set[NAMES]), len(target_set[MONITORED_VALUES]), self.name))

            for name in target_set[NAMES]:
                if not isinstance(name, str):
                    raise ObjectiveMechanismError("it in \'names\'arg ({}) of {} is not a string".
                                         format(target_set[NAMES], self.name))

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
                        validate_monitored_value(self, item, context=context)
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

        """
        num_values = len(self.monitored_values)
        values = [None] * num_values
        names = self.names or [None] * num_values

        # If default_input_value arg (assigned to variable in __init__) was used to specify the size of input_states,
        #   pass those values for use in instantiating input_states
        if self.variable is not None:
            input_state_sizes = self.variable
        else:
            input_state_sizes = values
        for i, monitored_value, name in zip(range(num_values), self.monitored_values, names):
            values[i] = self._instantiate_input_state_for_monitored_value(input_state_sizes[i],
                                                                          monitored_value,
                                                                          name,
                                                                          context=context)

        # If self.variable was not specified, construct from values of input_states
        if self.variable is None:
            # If all items of self.variable are numeric and of the same length, convert to ndarray
            dim_axis_0 = len(values)
            dim_axis_1 = len(values[0])
            if all((is_numeric(values[i]) and len(values[i])==dim_axis_1) for i in range(dim_axis_0)):
                self.variable = np.zeros((dim_axis_0,dim_axis_1), dtype=float)
            # Otherwise, just use list of values returned from instantiation above
            else:
                self.variable = values.copy()

        self.variableClassDefault = self.variable.copy()
        self.inputValue = list(self.variable)

    def _instantiate_input_state_for_monitored_value(self, variable, monitored_value, name=None, context=None):
        """Instantiate inputState with projection from monitoredOutputState

        Validate specification for value to be monitored (using call to validate_monitored_value)
        Instantiate the inputState (assign name if specified, and value of monitored_state)
        Re-specify corresponding item of variable to match the value of the new inputState
        Update self.input_state and self.input_states
        Call _instantiate_monitoring_projection() to instantiate MappingProjection to inputState
            if an outputState has been specified.

        Parameters
        ----------
        monitored_value (value, InputState, OutputState, Mechanisms, str, dict, or MonitoredOutputStateOption)
        name (str)
        context (str)

        Returns
        -------

        """
        from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
        from PsyNeuLink.Components.States.InputState import InputState
        from PsyNeuLink.Components.States.OutputState import OutputState

        call_for_projection = False
        value = DEFAULT_MONITORED_VALUE
        input_state_name = name
        input_state_params = None

        # If monitored_value is a value:
        # - create default InputState using monitored_value as its variable specification
        if _is_value_spec(monitored_value):
            value = monitored_value
            input_state_name = name or self.name + MONITORED_VALUE_NAME_SUFFIX

        # If monitored_value is an InputState:
        # - use InputState's variable, params, and name
        elif isinstance(monitored_value, InputState):
            value = monitored_value.variable
            # Note: name specified in names argument of constructor takes precedence over existing name of inputState
            input_state_name = name or monitored_value.name
            input_state_params = monitored_value.params

        # If monitored_value is a specification dictionary for an InputState:
        elif isinstance(monitored_value, InputState):
            try:
                value = monitored_value[VARIABLE]
                # Note: name specified in a specification dictionary takes precedence over names argument in constructor
                input_state_name = monitored_value[NAME]
                input_state_params = monitored_value[INPUT_STATE_PARAMS]
            except KeyError:
                if (value == DEFAULT_MONITORED_VALUE and
                            input_state_name is name and
                            input_state_params is None):
                    raise ObjectiveMechanismError("Specification dictionary in monitored_values arg for {}"
                                         "did not contain any entries relevant to an inputState".format(self.name))
                else:
                    pass

        # elif isinstance(monitored_value, tuple):
        #     monitored_value = monitored_value[0]
        #     # FIX:
        #     # IF IT IS A STRING, LOOK FOR OUTPUTSTATE OR MECHANISM WITH THAT NAME AND REASSIGN??

        # If monitored_value is an OutputState:
        # - match inputState to the value of the outputState's value
        # - and specify the need for a projection from it
        elif isinstance(monitored_value, OutputState):
            value = monitored_value.value
            input_state_name = name or monitored_value.owner.name + MONITORED_VALUE_NAME_SUFFIX
            call_for_projection = True
            
        # If monitored_value is a Mechanism:
        # - match inputState to the value of its primary outputState
        # - and specify the need for a projection from it
        elif isinstance(monitored_value, Mechanism):
            value = monitored_value.outputState.value
            input_state_name = name or monitored_value.name + MONITORED_VALUE_NAME_SUFFIX
            call_for_projection = True
            
        # # If monitored_value is a MonitoredOutputStatesOption:
        # # FIX: NOT SURE WHAT TO DO HERE YET
        # elif isinstance(montiored_value, MonitoredOutputStateOption):
        #     value = ???
        #     call_for_projection = True

        # If monitored_value is a string:
        # - use as name of inputState
        # - instantiate InputState with defalut value (1d array with single scalar item??)

        elif isinstance(monitored_value, str):
            input_state_name = monitored_value
            value = DEFAULT_MONITORED_VALUE

        # Format the item of self.variable that corresponds to the inputState
        # Give precedence to item specified in self.variable for inputState's variable
        if variable is not None:
            input_state_variable = variable
        # Otherwise, set to value derived from monitored_value above
        else:
            input_state_variable = value

        from PsyNeuLink.Components.States.State import _instantiate_state
        input_state = _instantiate_state(owner=self,
                                         state_type=InputState,
                                         state_name=input_state_name,
                                         state_spec=input_state_variable,
                                         state_params=input_state_params,
                                         constraint_value=input_state_variable,
                                         constraint_value_name='ObjectiveMechanism inputState value',
                                         context=context)

        #  Update inputState and input_states
        try:
            self.input_states[input_state.name] = input_state
        except (AttributeError, TypeError):
            self.input_states = OrderedDict({input_state_name:input_state})
            self.input_state = list(self.input_states.values())[0]

        self.inputValue = list(state.value for state in self.input_states.values())

        # IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
        if call_for_projection:
            _instantiate_monitoring_projection(sender=monitored_value, receiver=input_state, matrix=AUTO_ASSIGN_MATRIX)

        return input_state.value

    def add_monitored_values(self, states_spec, context=None):
        """Validate specification and then add inputState to ObjectiveFunction + MappingProjection to it from state

        Use by other objects to add a state or list of states to be monitored by EVC
        states_spec can be a Mechanism, OutputState or list of either or both
        If item is a Mechanism, each of its outputStates will be used

        Args:
            states_spec (Mechanism, MechanimsOutputState or list of either or both:
            context:
        """
        states_spec = list(states_spec)
        validate_monitored_value(self, states_spec, context=context)
        self._instantiate_monitored_output_states(states_spec, context=context)


def validate_monitored_value(objective_mech, state_spec, context=None):
    """Validate specification for monitored_value arg

    Validate that each item of monitored_value arg is an inputState, OutputState, mechanism, string,
    or a MonitoredOutpuStatesOption value.
    
    Called by both self._validate_variable(), self.add_monitored_value(), and EVCMechanism._get_monitored_states()
    """
    state_spec_is_OK = False

    if _is_value_spec(state_spec):
        state_spec_is_OK = True

    from PsyNeuLink.Components.States.OutputState import OutputState
    if isinstance(state_spec, (InputState, OutputState, Mechanism)):
        state_spec_is_OK = True

    if isinstance(state_spec, dict):
        state_spec_is_OK = True

    if isinstance(state_spec, str):
        # Will be used as the name of the inputState
        state_spec_is_OK = True

    if isinstance(state_spec, MonitoredOutputStatesOption):
        state_spec_is_OK = True

    if not state_spec_is_OK:
        raise ObjectiveMechanismError("Specification of state to be monitored ({0}) by {1} is not "
                             "a value, Mechanism, OutputState, string, dict, or a value of MonitoredOutputStatesOption".
                             format(state_spec, self.name))


def _objective_mechanism_role(mech, role):
    if isinstance(mech, ObjectiveMechanism):
        if mech._role is role:
            return True
        else:
            return False
    else:
        return False


def _is_value_spec(spec):
    if isinstance(spec, (int, float, list, np.ndarray)):
        return True
    else:
        return False


# IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
def _instantiate_monitoring_projection(sender, receiver, matrix=DEFAULT_MATRIX):
    from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
    MappingProjection(sender=sender,
                      receiver=receiver,
                      matrix=matrix,
                      name = sender.name + ' monitor')