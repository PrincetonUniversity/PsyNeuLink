# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************  ComparatorMechanism ***************************************************

"""

Overview
--------

A ComparatorMechanism is a subclass of `ComparatorMechanism` that receives two inputs (a sample and a target), compares 
them using its `function <ComparatorMechanism.function>`, and generates and an `error_signal <Comparator.error_signal>`
as its output.

.. _ComparatorMechanism_Creation:

Creating a ComparatorMechanism
------------------------------

A CompartorMechanism can be created directly by calling its constructor.  ComparatorMechanisms are also created
automatically when other PsyNeuLink components are created (such as `LearningMechanisms <LearningMechanism_Creation>`
and `ControlMechanisms <ControlMechanism_Creation>`.

.. _ComparatorMechanism_Structure:

Structure
---------

An ComparatorMechanism has one `inputState <InputState>` for each of the values that are specified
to be monitored in its `monitored_values` attribute.  When an ComparatorMechanism is created, an inputState is created
for each of those values, and assigned a `MappingProjection` from the outputState to which the value belongs.  The
ComparatorMechanism's `function  <ComparatorMechanism.function>` uses these values to compute an `objective (or "loss")
function <https://en.wikipedia.org/wiki/Loss_function>`_, that is assigned as the value of its outputState.


COMMENT:
Input States
~~~~~~~~~~~~~~~~
ADD DOCUMENTATION HERE (SEE NEW DOCUMENTATION ABOVE)
COMMENT

.. _ComparatorMechanism_Monitored_States:

Monitored Values
~~~~~~~~~~~~~~~~

The values to be monitored by an ComparatorMechanism are specified in the :keyword:`monitored_values` argument of its
constructor.  These can be specified in a variety of ways, each of which must eventually resolve to an outputState, the
value of which is to be monitored.  Those outputStates are listed in the ComparatorMechanism's `monitored_values`
attribute.

The number of items in `monitored_values` must match the length of the number of items in the 
**input_states** argument if it is specified
COMMENT:
, or the `default_input_value
<ComparatorMechanism.Additional_Attributes>` if it is specified
COMMENT
.  Note that some forms of
specification may depend on specifications made for the outputState referenced, the mechanism to which it belongs,
and/or the process or system to which that mechanism belongs. These interactions (and the precedence afforded to
each) are described below.

If an outputState is specified at the time the ComparatorMechanism is created, or the specification can be resolved
to an outputState, a MappingProjection is automatically created from it to the corresponding inputState
using `AUTO_ASSIGN_MATRIX` as its `matrix <MapppingProjection.matrix>` parameter.  If the outputState can't be
determined, no MappingProjection is assigned, and this must be done by some other means;  any values in
`monitored_values` that are not associated with an outputState at the time the ComparatorMechanism is executed are
ignored.

The specification of item in `monitored_values` can take any of the following forms:

* **OutputState**:  a reference to the `outputState <OutputState>` of a mechanism.  This will create a
  `MappingProjection` from it to the corresponding inputState in `input_states <ComparatorMechanism.input_states>`.
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
        inputState for the ComparatorMechanism;
      |
      * `specification dictionary <InputState_Creation>` **for an inputState**:  the specifications will be used to
        create an inputState for the ComparatorMechanism;
      |
      * **value**: a default inputState will be created using that value;
      |
      * **string**: a default inputState will be created using the string as its name, and a scalar as its value.
COMMENT

COMMENT: TBI
    .. _ComparatorMechanism_OutputState_Tuple:

    * **MonitoredOutputState Tuple**  tuple can be used wherever an outputState can be specified, to determine how
      its value is combined with others by the ComparatorMechanism's `function <ComparatorMechanism.function>`. Each
      tuple must have the three following items in the order listed:

          * an outputState or mechanism, the name of one, or a specification dictionary for one;
          ..
          * a weight (int) - multiplies the value of the outputState.
          ..
          * an exponent (int) - exponentiates the value of the outputState;
COMMENT
* **string**, **value** or **dict**: These can be used as placemarkers for a monitored_state that will be instantiated
  later (for example, for the TARGET input of a Composition).  If a string is specified, it is used as the
  default name of the corresponding inputState (specified in the `input_states <ComparatorMechanism.input_states>`
  attribute of the ComparatorMechanism) If a value is specified, it is used as the default value for the corresponding
  inputState.  If a dict is specified, it must have a single entry, the key of which will be used a string
  specification and the value as a value specification. 

Additional Attributes
~~~~~~~~~~~~~~~~~~~~~

* `default_input_value`
   This specifies the format of each value monitored by the ComparatorMechanism and the variable for the corresponding
   inputState.  These values take precedence over the specification of values in `monitored_values`, and can be used
   to override the defaults assumed there. If `default_input_value` is specified, it must have the same number of items
   as `monitored_values`. If  `default_input_value` is `None` (the default), then the specifications in
   `monitored_values` are used.  The use of `default_input_value` to override defaults used by `monitored_values`
   can be helpful in some situations (see `example <ComparatorMechanism_Default_Input_Value_Example>` below).


.. _ComparatorMechanism_Function:

Function
~~~~~~~~

The ComparatorMechanism's `function` can be customized to implement a wide variety of
`objective (or "loss") functions <https://en.wikipedia.org/wiki/Loss_function>`_.  The default is the
`LinearCombination` function, which simply sums the values of the outputStates listed in `monitored_values`.
However, this can easily be configured to calculate differnces, ratios, etc. (see
`example <ComparatorMechanism_Weights_and_Exponents_Example>` below).  It can also be replaced with any
`CombinationFunction`, or any python function that takes a 2d array with an arbitrary number of
items or a number equal to the number of items in the ComparatorMechanism's variable (and its number of
input_states), and returns a 1d array.

.. _ComparatorMechanism_Execution:

Execution
---------

When an ComparatorMechanism is executed, it updates its input_states with the values of the outputStates listed in
its `monitored_values` attribute, and then uses its `function <ComparatorMechanism.function>` to
evaluate these.  The result is assigned as to its `value <ComparatorMechanism.value>` attribute as the value of its
`primary outputState <OutputState_Primary>`.

.. _ComparatorMechanism_Class_Reference:

Examples
--------

.. _ComparatorMechanism_Default_Input_Value_Example:

*Formatting inputState values*

The use of default_input_value to override a specification in `monitored_values` can be useful in some situations.
For example, for `Reinforcement Learning <Reinforcement>`, an ComparatorMechanism is used to monitor an action
selection mechanism.  In the example below, the latter uses a `TransferMechanism` with the `SoftMax` function (and the
`PROB <Softmax.PROB>` as its output format) to select the action.  This generates a vector with a single non-zero
value, which designates the predicted reward for the selected action.  Because the output is a vector,
by default the inputState of the ComparatorMechanism created to monitor it will also be a vector.  However, the
ComparatorMechanism requires that this be a single value, that it can compare with the value of the reward mechanism.
This can be dealt with by using `default_input_value` in the construction of the ComparatorMechanism, to force
the inputState for the ComparatorMechanism to have a single value, as in the example below::

    my_action_select_mech = TransferMechanism(default_input_value = [0,0,0],
                                function=SoftMax(output=PROB))

    my_reward_mech = TransferMechanism(default_input_value = [0])

    my_objective_mech = ComparatorMechanism(default_input_value = [[0],[0]],
                                          monitored_values = [my_action_select_mech, my_reward_mech])

Note that the outputState for the `my_action_selection` and `my_reward_mech` are specified
in `monitored_values`.  If that were the only specification, the inputState created for `my_action_select_mech`
would be a vector of length 3.  This is overridden by specifying `default_input_value` as an array with two
single-value arrays (one corresponding to `my_action_select_mech` and the other to `my_reward_mech`).  This forces
the inputState for `my_action_select_mech` to have only a single element which, in turn, will cause a
MappingProjection to be created from  `my_action_select_mech` to the ComparatorMechanism's inputState using a
`FULL_CONNECTIVITY_MATRIX` (the one used for `AUTO_ASSIGN_MATRIX` when the sender and receiver have values of
different lengths).  This produces the desired effect, since the action selected is the only non-zero value in the
output of `my_action_select_mech`, and so the `FULL_CONNECTIVITY_MATRIX` will combine it with zeros (the other values
in the vector), and so its value will be assigned as the value of the corresponding inputState in the
ComparatorMechanism.  Another option would have been to customize the ComparatorMechanism's
`function <ComparatorMechanism.function>` to convert the output of `my_action_select_mech` to a length 1 vector, though
this would have been more involved.  The next example describes a simple case of customizing the ComparatorMechanism's
`function <ComparatorMechanism.function>`, however more sophisticated ones are possible, just as the one just suggested.

.. _ComparatorMechanism_Weights_and_Exponents_Example:

*Customizing the ComparatorMechanism's function*

The simplest way to customize the `function <ComparatorMechanism.function>` of an ComparatorMechanism is to
parameterize its default function (`LinearCombination`).  In the example below, the ComparatorMechanism used in the
`previous example <ComparatorMechanism_Default_Input_Value_Example>` is further customized to subtract the value
of the action selected from the value of the reward::

    my_objective_mech = ComparatorMechanism(default_input_value = [[0],[0]],
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
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism \
    import ObjectiveMechanism
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.Functions.Function import Distance, DISTANCE_METRICS


class ComparatorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ComparatorMechanism(ObjectiveMechanism):
    """
    ComparatorMechanism(                    \
    sample=None,                            \
    target=None,                            \
    function=Distance(metric=DIFFERENCE),   \
    params=None,                            \
    name=None,                              \
    prefs=None)

    Implements the ComparatorMechanism subclass of `ObjectiveMechanism`.

    COMMENT:
        Description:
            ComparatorMechanism is a subtype of the ObjectiveMechanism Subtype of the ProcssingMechanism Type 
            of the Mechanism Category of the Component class.
            By default, it's function uses the LinearCombination Function to compare two input variables.
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
        specifies the format of the values monitored by the ComparatorMechanism;  each item corresponds to the value
        of an outputState monitored, and to the value of the corresponding inputState of the ComparatorMechanism.  It
        must have the same length as the number items in monitored_values.  The values specified here take precedence
        over those in :keyword:`monitored_values`;  if none are provided, the ones in :keyword:`monitored_values`
        will be used.
    COMMENT

    sample : OutputState, Mechanism, value, or string
        specifies the value to compare with the `target` by the `function <ComparatorMechanism>`.

    target : OutputState, Mechanism, value, or string
        specifies the value with which to compare the `sample` by the `function <ComparatorMechanism>`.

    function: Function, function or method : default Distance(metric=DIFFERENCE)
        specifies the function used to compare the `sample` with the `target` 
        (see `function <LearningMechanism.function>` for details.

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

    sample : OutputState
        determines the value to compare with the `target` by the `function <ComparatorMechanism>`.

    target : OutputState
        determines the value with which to compare the `sample by the `function <ComparatorMechanism>`.

    function : Distance, function or method
        the function used to compare the sample with the target.  It can be any PsyNeuLink `CombinationFunction`,
        `Objecior a 
        python function that takes a 2d array with an arbitrary number of
        items or a number equal to the number of items in the ComparatorMechanism's variable (and its number of
        input_states), and returns a 1d array.

\    value : 1d np.array
        the output of the evaluation carried out by the ComparatorMechanism's `function <ComparatorMechanism.function>`.

    output_values : 2d np.array
        1st and only item is same as `value <ComparatorMechanisms.value>`.

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

    componentType = COMPARATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ComparatorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # variableClassDefault = [[0],[0]]  # By default, ComparatorMechanism compares two 1D np.array input_states
    variableClassDefault = None

    # ComparatorMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        TIME_SCALE: TimeScale.TRIAL,
        OUTPUT_STATES:[{NAME:RESULT}]})
        # MODIFIED 12/7/16 NEW:

    paramNames = paramClassDefaults.keys()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 sample:tc.any(OutputState, Mechanism_Base, dict, is_numeric, str),
                 target:tc.any(OutputState, Mechanism_Base, dict, is_numeric, str),
                 input_states=None,
                 function=Distance(metric=DIFFERENCE),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(sample=sample,
                                                  target=target,
                                                  input_states=input_states,
                                                  function=function,
                                                  params=params)

        super().__init__(
                         monitored_values=[sample, target],
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

        # IMPLEMENATION NOTE: THIS IS HERE UNTIL Composition IS IMPLEMENTED,
        # SO THAT SYSTEMS AND PROCESSES CAN FIND THE OBJECTIVE MECHANISSMS SERVING AS TARGETS

    def _validate_params(self, request_set, target_set=None, context=None):
        """If sample and target values are specified, validate that they are compatible 

        """

        if INPUT_STATES in request_set:
            input_states = request_set[INPUT_STATES]
            num_input_states = len(input_states)
            if num_input_states > 2:
                raise ComparatorMechanismError("Number of items in {} arg for {} ({}) must be 2 or less".
                                               format(INPUT_STATES, self.__class__.__name__, len(input_states)))
            if len(input_states[0]) != len(input_states[1]):
                raise ComparatorMechanismError("The length of the value specified for the {} inputState of {} ({})"
                                               "must be the same as the length of the value specified for the {} ({})".
                                               format(SAMPLE,
                                                      self.__class__.__name__,
                                                      len(input_states[0]),
                                                      TARGET,
                                                      len(input_states[1])))

        elif SAMPLE in request_set and TARGET in request_set:

            sample = request_set[SAMPLE]
            if isinstance(sample, InputState):
                sample_value = sample.value
            elif isinstance(sample, Mechanism):
                sample_value = sample.input_value[0]
            elif is_value_spec(sample):
                sample_value = sample
            else:
                sample_value = None

            target = request_set[TARGET]
            if isinstance(target, InputState):
                target_value = target.value
            elif isinstance(target, Mechanism):
                target_value = target.input_value[0]
            elif is_value_spec(target):
                target_value = target
            else:
                target_value = None

            if sample is not None and target is not None:
                if not iscompatible(sample, target, **{kwCompatibilityLength: True,
                                                       kwCompatibilityNumeric: True}):
                    raise ComparatorMechanismError("The length of the sample ({}) must be the same as for the target ({})"
                                                   "for {} {}".
                                                   format(len(sample),
                                                          len(target),
                                                          self.__class__.__name__,
                                                          self.name))


        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)
