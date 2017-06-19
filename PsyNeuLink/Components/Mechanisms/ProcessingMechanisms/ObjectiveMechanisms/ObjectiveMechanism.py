# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ADD TO DOCUMENTATION: [per ComparatorMechanism]
#  input_states is used to name and/or specify the value of the input_states; it can be:
#      just a list of strings (used as names)
#      just a list of values (used as values of inputState, overrides use of monitored_values such as for RL)
#      if both, use dict with NAME and VARIABLE entries
#      # if input_states must == monitored_values
#      if none are specified, they will autojatically be created based on the monitored_values spec
#  monitored_values no longer takes an inputState specification (that must be in input_states)
#      if it is a name or value, no projection is called for (assumes that it is a TARGET, and that the projection
#                    will be implemented by Composition

# *********************************************  ObjectiveMechanism ****************************************************

"""

Overview
--------

An ObjectiveMechanism is a `ProcessingMechanism` that monitors the `outputStates <OutputState>` of one or more other
ProcessingMechanisms specified in its `monitor <ObjectiveMechanism.monitor>` attribute, and evaluates them using its
`function <ObjectiveMechanism.function>`. The result of the evaluation is placed in the ObjectiveMechanism's
`primary outputState <OutputState_Primary>`.  ObjectiveMechanisms are typically used closely with (and often created
automatically by) `AdaptiveMechanisms <AdaptiveMechanism>`.

.. _ObjectiveMechanism_Creation:

Creating an ObjectiveMechanism
------------------------------

ObjectiveMechanisms are often created automatically when other PsyNeuLink components are created (in particular, 
AdaptiveMechanisms such as `LearningMechanisms <LearningMechanism_Creation>` and 
`ControlMechanisms <ControlMechanism_Creation>`.  An ObjectiveMechanism can also be created directly by calling its 
constructor.  Its **monitored_values** argument is used to specify the outputStates to be monitored.  Any of the forms 
used for `specifying outputStates <OutputState_Specification>` can be used, as well as a value of 
MonitoredOutputStateOption.  When an ObjectiveMechanism is created, an inputState is created for each of the 
outputStates specified in its **monitored_values** argument, and a `MappingProjection` is assigned from each of those 
to the corresponding inputState.  By default, the value of each inputState uses the format of its corresponding 
outputState, and the MappingProjection between them uses an `IDENTITY_MATRIX`.  However, the **input_states** argument 
can be used to customize the inputStates, and/or to specify their names. Any of the forms used for 
`specifying inputStates <InputState_Specification>` can be used in the **input_states** argument, however the number
of inputStates specified must equal the number of outputStates specified in the **monitored_values** argument (see the 
`examples <ObjectiveMechanism_Examples>` below). The value of each must also be of the same type as the value of the 
corresponding outputState, however their lengths can differ;  in that case, by default, the MappingProjection created 
uses a  `FULL_CONNECTIVITY` matrix, although this too can be customized using the *PROJECTION* entry of a 
`state specification dictionary <InputState_Specification>` for the inputState in the **input_states** argument.


.. _ObjectiveMechanism_Structure:

Structure
---------

An ObjectiveMechanism has one `inputState <InputState>` for each of the outputStates specified in its
**monitored_values** argument, each pair of which is connected by a `MappingProjection`.  The monitored outputStates
are listed in the ObjectiveMechanism's `monitored_values <ObjectiveMechanism.monitored_values>` attribute, and  
the inputStates to which they project are listed in the ObjectiveMechanism's 
`input_states <ObjectiveMechanism.input_states>` attribute.  The ObjectiveMechanism's 
`function <ObjectiveMechanism.function>` uses these values to compute an 
`objective (or "loss") function <https://en.wikipedia.org/wiki/Loss_function>`_, that is assigned as the value of its 
*ERROR_SIGNAL* (`primary <OutputState_Primary>`) outputState.  By default, it uses a `LinearCombination` function to 
sum the values of its inputStates.  However, the `function <ComparatorMechanism.function>` can be customized to 
calculate other quantities (differences, ratios, etc. -- see 
`example <ObjectiveMechanism_Weights_and_Exponents_Example>` below). It can also be replaced with any Python function
or method, so long as it takes a 2d array as its input, and generates a 1d array as its result.  

.. _ObjectiveMechanism_Monitored_Values:

Monitored Values
~~~~~~~~~~~~~~~~

These can be specified in a variety of ways, each of which must eventually resolve to an outputState, the
value of which is to be monitored.

Note that some forms of
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
  inputState.  If a dict is specified, it must have a single entry, the key of which will be used as a string
  specification -- i.e., as the name of the inputState -- and the value as its value specification. 

.. _ObjectiveMechanism_Function:

Function
~~~~~~~~

The ObjectiveMechanism's `function` can be customized to implement a wide variety of
`objective (or "loss") functions <https://en.wikipedia.org/wiki/Loss_function>`_.  The default is the
`LinearCombination` function, which simply sums the values of the outputStates listed in `monitored_values`.
However, this can easily be configured to calculate differences, ratios, etc. (see
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


.. _ObjectiveMechanism_Examples:

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
This can be dealt with by using `default_input_value` in the constructor of the ObjectiveMechanism, to force
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

.. _ObjectiveMechanism_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import LinearCombination
# from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.OutputState import standard_output_states, PRIMARY_OUTPUT_STATE

ROLE = 'role'
MONITORED_VALUES = 'monitored_values'
MONITORED_VALUE_NAME_SUFFIX = '_Monitor'
ERROR_SIGNAL = 'error_signal'

# This is a convenience class that provides list of standard_output_state names in IDE
class OBJECTIVE_OUTPUT():
        ERROR_SIGNAL=ERROR_SIGNAL


class ObjectiveMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ObjectiveMechanism(ProcessingMechanism_Base):
    """
    ObjectiveMechanism(               \
        monitored_values,             \
        input_states=None,            \
        function=LinearCombination,   \
        output_states=[ERROR_SIGNAL], \
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
            + componentType (str): ObjectiveMechanism
            + classPreference (PreferenceSet): Comparator_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (value):  Comparator_DEFAULT_STARTING_POINT // QUESTION: What to change here
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL,
                                          FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}
            + paramNames (dict): names as above

        Class methods:
            None

        MechanismRegistry:
            All instances of ObjectiveMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    monitored_values : List[OutputState, Mechanism, string, value, dict, MonitoredOutputStateOption] or Dict[]
        specifies the values that will will be monitored, and evaluated by the `function <ObjectiveMechanism>`
        (see `monitored_values` for details of specification).

    input_states :  List[InputState, value, str or dict] or Dict[] : default None
        specifies the names and/or formats to use for the values of the inputStates that receive the input from the
        outputStates specified in the **monitored_values** argument; if specified, there must be one for each item 
        specified in the **monitored_values** argument.

    COMMENT:
    names: List[str]
        specifies the names to use for the input_states created for the list in
        `monitored_values <ObjectiveMechanism.monitor>`.  If specified,
        the number of items in the list must equal the number of items in `monitored_values`, and takes precedence
        over any names specified there.
    COMMENT

    function: CombinationFunction, ObjectiveFunction, function or method : default LinearCombination
        specifies the function used to evaluate the values listed in :keyword:`monitored_values`
        (see `function <LearningMechanism.function>` for details.

    output_states :  List[OutputState, value, str or dict] or Dict[] : default [ERROR_SIGNAL]  
        specifies the outputStates for the mechanism;

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

    name : str : default ObjectiveMechanism-<index>
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

    monitored_values : ContentAddressableList[OutputState]
        determines  the values monitored, and evaluated by `function <ObjectiveMechanism>`.  Each item in the list 
        refers to an outputState containing the value to be monitored, with a `MappingProjection` from it to the 
        corresponding inputState listed in the `input_states <ComparatorMechanism.input_states>` attribute.

    input_states : ContentAddressableList[InputState]
        contains the inputStates of the ObjectiveMechanism, each of which receives a `MappingProjection` from the 
        outputStates specified in its `monitored_values <ObjectiveMechanism.monitored_values>` attribute.

    function : CombinationFunction, ObjectiveFunction, function, or method
        the function used to compare evaluate the values monitored by the ObjectiveMechanism.  The function can be
        any PsyNeuLink `CombinationFunction` or a python function that takes a 2d array with an arbitrary number of
        items or a number equal to the number of items in the ObjectiveMechanism's variable (and its number of
        input_states), and returns a 1d array.

    role : None, LEARNING or CONTROL
        specifies whether the ObjectiveMechanism is being used for learning in a process or system (in conjunction
        with a `LearningMechanism`), or for control in a system (in conjunction with a `ControlMechanism`).

    value : 1d np.array
        the output of the evaluation carried out by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.

    output_state : OutputState
        contains the 'primary <OutputState_Primary>` outputState of the ObjectiveMechanism; the default is  
        its *ERROR_SIGNAL* outputState, the value of which is equal to the `value <ObjectiveMechanism.value>` 
        attribute of the ObjectiveMechanism.

    output_states : ContentAddressableList[OutputState]
        contains, by default, only the *ERROR_SIGNAL* (primary) outputState of the ObjectiveMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *ERROR_SIGNAL* outputState.

    name : str : default ObjectiveMechanism-<index>
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
        })

    paramNames = paramClassDefaults.keys()

    standard_output_states = standard_output_states.copy()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 monitored_values:tc.any(list, dict),
                 input_states=None,
                 function=LinearCombination,
                 output_states:tc.optional(tc.any(list, dict))=[ERROR_SIGNAL],
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitored_values=monitored_values,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  function=function,
                                                  params=params)
        self._learning_role = None

        from PsyNeuLink.Components.States.OutputState import StandardOutputStates
        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)

        super().__init__(
                         variable=None,
                         input_states=input_states,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

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

        if (INPUT_STATES in target_set and
                    target_set[INPUT_STATES] is not None and
                not all(input_state is None for input_state in target_set[INPUT_STATES])):
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
                    # FIX:           System.ControlMechanism_instantiate_monitored_output_states
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

    # IMPLEMENTATION NOTE: FOR Composition, BREAK THIS UP SO THAT monitored_values CAN BE IMPLEMENTED
    #                      ONE AT TIME (IN A CALL TO _instantiate_monitored_value THAT INSTANTATEs
    #                      ADDS an input_state FOR EACH ITEM IN monitored_values
    #                      (AKIN _instantiate_control_signal)
    def _instantiate_input_states(self, context=None):
        """Instantiate input state for each value specified in `monitored_values` arg and instantiate self.variable
        
        Parse specifications for input_states, using monitored_values where relevant, and instantiate input_states. 
        Re-specify corresponding items of variable to match the values of the inputStates in input_states.
        Update self.input_state and self.input_states.
        Call _instantiate_monitoring_projection() to instantiate MappingProjection to inputState
            if an outputState has been specified.
        """

        from PsyNeuLink.Components.States.State import _parse_state_spec
        from PsyNeuLink.Components.States.OutputState import OutputState

        # Parse monitored_values
        monitored_values = []
        for value in self.monitored_values:
            monitored_value_dict = {}
            monitored_value = _parse_state_spec(owner=self,
                                                state_type=OutputState,
                                                state_spec=value)
            if isinstance(monitored_value, dict):
                monitored_value_dict = monitored_value
            elif isinstance(monitored_value, State):
                monitored_value_dict[NAME] = monitored_value.name
                monitored_value_dict[VARIABLE] = monitored_value.variable
                monitored_value_dict[VALUE] = monitored_value.variable
                # monitored_value_dict[PARAMS] = monitored_value.params
            else:
                raise ObjectiveMechanismError("PROGRAM ERROR: call to State._parse_state_spec() for {} of {} "
                                              "should have returned dict or State, but returned {} instead".
                                              format(OUTPUT_STATE, self.name, type(monitored_value)))
            monitored_value_dict[OUTPUT_STATE]=value
            monitored_value_dict[NAME] = monitored_value_dict[NAME] + MONITORED_VALUE_NAME_SUFFIX
            monitored_values.append(monitored_value_dict)

        # If input_states were not specified, assign value of monitored_valued for each
        #    (to invoke a default assignment for each input_state)
        if self.input_states is None:
            self._input_states = [m[VALUE] for m in monitored_values]

        # Parse input_states into a state specification dict, passing monitored_values as defaults from monitored_value
        for i, input_state, monitored_value in zip(range(len(self.input_states)),
                                                   self.input_states,
                                                   monitored_values):

            # Parse input_state to determine its specifications and assign values from monitored_values
            #    to any missing specifications, including any projections requested.
            self._input_states[i] = _parse_state_spec(self,
                                                      state_type=InputState,
                                                      state_spec=input_state,
                                                      name=monitored_values[i][NAME],
                                                      value=monitored_values[i][VALUE])

        constraint_value = []
        for input_state in self.input_states:
            constraint_value.append(input_state[VARIABLE])
        self.variable = constraint_value

        super()._instantiate_input_states(context=context)

        # self.variableClassDefault = self.variable.copy()

        # Get any projections specified in input_states arg, else set to default (AUTO_ASSIGN_MATRIX)
        input_state_projection_specs = []
        for i, state in enumerate(self._input_states):
            input_state_projection_specs.append(state.params[STATE_PROJECTIONS] or [AUTO_ASSIGN_MATRIX])

        # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
        _instantiate_monitoring_projections(owner=self,
                                            sender_list=[mon_val[OUTPUT_STATE] for mon_val in monitored_values],
                                            receiver_list=self.input_states,
                                            receiver_projection_specs=input_state_projection_specs,
                                            context=context)

def _validate_monitored_value(objective_mech, state_spec, context=None):
    """Validate specification for monitored_value arg

    Validate that each item of monitored_value arg is: 
        * OutputState
        * Mechanism, 
        * string, or 
        * MonitoredOutpuStatesOption value.
    
    Called by both self._validate_variable(), self.add_monitored_value(), and EVCMechanism._get_monitored_states()
    """
    from PsyNeuLink.Components.States.OutputState import OutputState
    if not isinstance(state_spec, (str, OutputState, Mechanism, MonitoredOutputStatesOption, dict)):
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
#                      ??MAYBE INTEGRATE INTO State MODULE (IN _instantate_state)
@tc.typecheck
def _instantiate_monitoring_projections(owner,
                                        sender_list:tc.any(list, ContentAddressableList),
                                        receiver_list:tc.any(list, ContentAddressableList),
                                        receiver_projection_specs:tc.optional(list)=None,
                                        context=None):

    from PsyNeuLink.Components.States.OutputState import OutputState
    from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

    receiver_projection_specs = receiver_projection_specs or [DEFAULT_MATRIX] * len(sender_list)

    if len(sender_list) != len(receiver_list):
        raise ObjectiveMechanismError("PROGRAM ERROR: Number of senders ({}) does not equal number of receivers ({}) "
                                     "in call to instantiate monitoring projections for {}".
                                     format(len(sender_list), len(receiver_list), owner.name))

    if len(receiver_projection_specs) != len(receiver_list):
        raise ObjectiveMechanismError("PROGRAM ERROR: Number of projection specs ({}) "
                                     "does not equal number of receivers ({}) "
                                     "in call to instantiate monitoring projections for {}".
                                     format(len(receiver_projection_specs), len(receiver_list), owner.name))

    # Instantiate inputState with projection from outputState specified by sender
    for sender, receiver, recvr_projs in zip(sender_list, receiver_list, receiver_projection_specs):

        # IMPLEMENTATION NOTE:  If there is more than one projection specified for a receiver, only the 1st is used;
        #                           (there should only be one if a 2-item tuple was used to specify the inputState,
        #                            however other forms of specifications could produce more)
        if len(recvr_projs) > 1 and owner.verbosePref:
            warnings.warn("{} projections were specified for inputState ({}) of {} ;"
                          "only the first ({}) will be used".
                          format(len(recvr_projs), receiver.name, owner.name))
        projection_spec = recvr_projs[0]

        # IMPLEMENTATION NOTE:  This may not handle situations properly in which the outputState is specified
        #                           by a 2-item tuple (i.e., with a projection specification as its second item)
        if isinstance(sender, OutputState):
            # Projection has been specified for receiver and initialization begun, so call deferred_init()
            if receiver.path_afferents:
                if not receiver.path_afferents[0].value is DEFERRED_INITIALIZATION:
                    raise ObjectiveMechanismError("PROGRAM ERROR: {} of {} already has an afferent projection "
                                                  "implemented and initialized ({})".
                                                  format(receiver.name, owner.name, receiver.aferents[0].name))
                if not receiver.path_afferents[0].function_params[MATRIX] is projection_spec:
                    raise ObjectiveMechanismError("PROGRAM ERROR: Projection specification for {} of {} ({}) "
                                                  "does not match matrix already assigned ({})".
                                                  format(receiver.name,
                                                         owner.name,
                                                         projection_spec,
                                                         receiver.path_afferents[0].function_params[MATRIX]))
                receiver.path_afferents[0].init_args[SENDER] = sender
                receiver.path_afferents[0]._deferred_init()
            else:
                MappingProjection(sender=sender,
                                  receiver=receiver,
                                  matrix=projection_spec,
                                  name = sender.name + ' monitor')
