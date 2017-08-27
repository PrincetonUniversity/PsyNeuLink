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

A ComparatorMechanism is a subclass of `ObjectiveMechanism` that receives two inputs (a sample and a target), compares
them using its `function <ComparatorMechanism.function>`, and places the calculted discrepancy between the two in its
*ERROR_SIGNAL* `output_state <ComparatorMechanism.output_state>`.

.. _ComparatorMechanism_Creation:

Creating a ComparatorMechanism
------------------------------

ComparatorMechanisms are generally created automatically when other PsyNeuLink components are created (such as
`LearningMechanisms <LearningMechanism_Creation>`.  A ComparatorMechanism can also be created directly by calling
its constructor.  Its **sample** and **target** arguments are used to specify the OutputStates that provide the
sample and target inputs, respectively (see `ObjectiveMechanism_Monitored_States` for details concerning their
specification, which are special versions of an ObjectiveMechanism's **monitored_values** argument).  When the
ComparatorMechanism is created, two InputStates are created, one each for its sample and target inputs (and named,
by default, *SAMPLE* and *TARGET*). Each is assigned a MappingProjection from the corresponding OutputState specified
in the **sample** and **target** arguments.

It is important to recognize that the value of the *SAMPLE* and *TARGET* InputStates must have the same length and type,
so that they can be compared using the ComparatorMechanism's `function <ComparatorMechanism.function>`.  By default,
they use the format of the OutputStates specified in the **sample** and **target** arguments, respectively,
and the `MappingProjection` to each uses an `IDENTITY_MATRIX`.  Therefore, for the default configuration, the
OutputStates specified in the **sample** and **target** arguments must have values of the same length and type.
If these differ, the **input_states** argument can be used to explicitly specify the format of the ComparatorMechanism's
*SAMPLE* and *TARGET* InputStates, to insure they are compatible with one another (as well as to customize their
names, if desired).  If the **input_states** argument is used, *both* the sample and target InputStates must be
specified.  Any of the formats for `specifying InputStates <InputState_Specification>` can be used in the argument.
If values are assigned for the InputStates, they must be of equal length and type.  Their types must be
also be compatible with the value of the OutputStates specified in the **sample** and **target** arguments.  However,
the length specified for an InputState can differ from its corresponding OutputState;  in that case, by default, the
MappingProjection created uses a `FULL_CONNECTIVITY` matrix.  Thus, OutputStates of differing lengths can be mapped
to the sample and target InputStates of a ComparatorMechanism (see the `example <ComparatorMechanism_Example>` below),
so long as the latter are of the same length.  If a projection other than a `FULL_CONNECTIVITY` matrix is needed, this
can be specified using the *PROJECTION* entry of a `State specification dictionary <State_Specification>` for the
InputState in the **input_states** argument.

.. _ComparatorMechanism_Structure:

Structure
---------

A ComparatorMechanism has two `input_states <ComparatorMechanism.input_states>`, each of which receives a
`MappingProjection` from a corresponding OutputState specified in the **sample** and **target** arguments of its
constructor.  The InputStates are listed in the Mechanism's `input_states <ComparatorMechanism.input_States>` attribute
and named, respectively, *SAMPLE* and *TARGET*.  The OutputStates from which they receive their projections (specified
in the **sample** and **target** arguments) are listed in the Mechanism's `sample <ComparatorMechanism.sample>` and
`target <ComparatorMechanism.target>` attributes as well as in its `monitored_values <Comparator.monitored_values>`
attribute. The ComparatorMechanism's `function <ComparatorMechanism.function>` compares the value of the sample and
target InputStates.  By default, it uses a `LinearCombination` function, assigning the sample InputState a `weight
<LinearCombination.weight>` of *-1* and the target a `weight <LinearCombination.weight>` of *1*, so that the sample is
subtracted from the target.  However, the `function <ComparatorMechanism.function>` can be customized, so long as it is
replaced with one that takes two arrays with the same format as its inputs, and generates a similar array as its result.
The result is assigned as the value of the Comparator Mechanism's *ERROR_SIGNAL* (`primary <OutputState_Primary>`)
OutputState.

.. _ComparatorMechanism_Function:

Execution
---------

When an ComparatorMechanism is executed, it updates its input_states with the values of the OutputStates specified
in its **sample** and **target** arguments, and then uses its `function <ComparatorMechanism.function>` to
compare these.  By default, the result is assigned as to the `value <ComparatorMechanism.value>` of its *ERROR_SIGNAL*
`output_state <ComparatorMechanism.output_state>`, and as the first item of the Mechanism's
`output_values <ComparatorMechanism.output_values>` attribute.

.. _ComparatorMechanism_Example:

Example
-------

.. _ComparatorMechanism_Default_Input_Value_Example:

*Formatting InputState values*

The **input_states** argument can be used to specify a particular format for the SAMPLE and/or TARGET InputStates
of a ComparatorMechanism.  This can be useful when one or both of these must be different than the format of the
OutputState(s) specified in the **sample** and **target** arguments. For example, for `Reinforcement Learning
<Reinforcement>`, a ComparatorMechanism is used to monitor an action selection Mechanism (the sample), and compare
this with a reinforcement signal (the target).  In the example below, the action selection Mechanism is a
`TransferMechanism` that uses the `SoftMax` function (and the `PROB <Softmax.PROB>` as its output format) to select
an action.  This generates a vector with a single non-zero value (the selected action). Because the output is a vector,
specifying it as the ComparatorMechanism's **sample** argument will generate a corresponding InputState with a vector
as its value.  This will not match the reward signal specified in the ComparatorMechanism's **target** argument, the
value of which is a single scalar.  This can be dealt with by explicitly specifying the format for the SAMPLE and
TARGET InputStates in the **input_states** argument of the ComparatorMechanism's constructor, as follows::

    my_action_selection_mech = TransferMechanism(size=5,
                                                 function=SoftMax(output=PROB))

    my_reward_mech = TransferMechanism(default_variable = [0])

    my_comparator_mech = ComparatorMechanism(sample=my_action_selection_mech,
                                             target=my_reward_mech,
                                             input_states = [[0],[0]])

Note that ``my_action_selection_mechanism`` is specified to take an array of length 5 as its input, and therefore
generate one of the same length as its `primary output <OutputState_Primary>`.  Since it is assigned as the **sample**
of the ComparatorMechanism, by default this will create a *SAMPLE* InputState of length 5, that will not match the
length of the *TARGET* InputState (which is 1).  This is taken care of, by specifying the **input_states** argument
as an array with two single-value arrays (corresponding to the *SAMPLE* and *TARGET* InputStates). (In this
example, the **sample** and **target** arguments are specified as Mechanisms since, by default, each has only a single
(`primary <OutputState_Primary>`) OutputState, that will be used;  if either had more than one OutputState, and
one of those was desired, it would have had to be specified explicitly in the **sample** or **target** argument).

.. _ComparatorMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Functions.Function import LinearCombination
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import ERROR_SIGNAL, MONITORED_VALUES, ObjectiveMechanism
from PsyNeuLink.Components.ShellClasses import Mechanism
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.OutputState import OutputState, PRIMARY_OUTPUT_STATE, StandardOutputStates
from PsyNeuLink.Globals.Keywords import CALCULATE, COMPARATOR_MECHANISM, INPUT_STATES, NAME, SAMPLE, TARGET, TIME_SCALE, VARIABLE, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Globals.Utilities import is_numeric, is_value_spec, iscompatible, kwCompatibilityLength, kwCompatibilityNumeric
from PsyNeuLink.Scheduling.TimeScale import TimeScale

SSE = 'SSE'
MSE = 'MSE'

class COMPARATOR_OUTPUT():
    """
    .. _ComparatorMechanism_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `ComparatorMechanism`

    .. _COMPARATOR_MECHANISM_SSE

    *SSE*
        the value of the sum squared error of the Mechanism's function

    .. _COMPARATOR_MECHANISM_MSE

    *MSE*
        the value of the mean squared error of the Mechanism's function

    """
    SSE = SSE
    MSE = MSE


class ComparatorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ComparatorMechanism(ObjectiveMechanism):
    """
    ComparatorMechanism(                                \
        sample,                                         \
        target,                                         \
        input_states=[SAMPLE,TARGET]                    \
        function=LinearCombination(weights=[[-1],[1]],  \
        input_states=[ERROR_SIGNAL]                     \
        params=None,                                    \
        name=None,                                      \
        prefs=None)

    Subclass of `ObjectiveMechanism` that compares the values of two `OutputStates <OutputState>`.

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
            + ClassDefaults.variable (value):  Comparator_DEFAULT_STARTING_POINT // QUESTION: What to change here
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL,
                                          FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}

        Class methods:
            None

        MechanismRegistry:
            All instances of ComparatorMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    sample : OutputState, Mechanism, value, or string
        specifies the value to compare with the `target` by the `function <ComparatorMechanism.function>`.

    target :  OutputState, Mechanism, value, or string
        specifies the value with which the `sample` is compared by the `function <ComparatorMechanism.function>`.

    input_states :  List[InputState, value, str or dict] or Dict[] : default [SAMPLE, TARGET]
        specifies the names and/or formats to use for the values of the sample and target InputStates;
        by default they are named *SAMPLE* and *TARGET*, and their formats are match the value of the OutputStates
        specified in the **sample** and **target** arguments, respectively (see `ComparatorMechanism_Structure`
        for additional details).

    function :  Function, function or method : default Distance(metric=DIFFERENCE)
        specifies the `function <Comparator.function>` used to compare the `sample` with the `target`.

    output_states :  List[OutputState, value, str or dict] or Dict[] : default [ERROR_SIGNAL]
        specifies the OutputStates for the Mechanism;

    params :  Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the
        constructor.

    COMMENT:
        [TBI]
        time_scale :  TimeScale : TimeScale.TRIAL
            specifies whether the Mechanism is executed on the TIME_STEP or TRIAL time scale.
            This must be set to :keyword:`TimeScale.TIME_STEP` for the ``rate`` parameter to have an effect.
    COMMENT

    name :  str : default ComparatorMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs :  Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    COMMENT:
    default_variable : Optional[List[array] or 2d np.array]
    COMMENT

    sample : OutputState
        determines the value to compare with the `target` by the `function <ComparatorMechanism.function>`.

    target : OutputState
        determines the value with which `sample` is compared by the `function <ComparatorMechanism.function>`.

    input_states : ContentAddressableList[InputState, InputState]
        contains the two InputStates named, by default, *SAMPLE* and *TARGET*, each of which receives a
        `MappingProjection` from the OutputStates referenced by the `sample` and `target` attributes
        (see `ComparatorMechanism_Structure` for additional details).

    function : CombinationFunction, function or method
        used to compare the `sample` with the `target`.  It can be any PsyNeuLink `CombinationFunction`,
        or a python function that takes a 2d array with two items and returns a 1d array of the same length
        as the two input items.

    value : 1d np.array
        the result of the comparison carried out by the `function <ComparatorMechanism.function>`.

    output_state : OutputState
        contains the `primary <OutputState_Primary>` OutputState of the ComparatorMechanism; the default is
        its *ERROR_SIGNAL* OutputState, the value of which is equal to the `value <ComparatorMechanism.value>`
        attribute of the ComparatorMechanism.

    output_states : ContentAddressableList[OutputState]
        contains, by default, only the *ERROR_SIGNAL* (primary) OutputState of the ComparatorMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *ERROR_SIGNAL* OutputState.

    name : str : default ComparatorMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Mechanism;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    """
    componentType = COMPARATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ComparatorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # ClassDefaults.variable = [[0],[0]]  # By default, ComparatorMechanism compares two 1D np.array input_states
    class ClassDefaults(ObjectiveMechanism.ClassDefaults):
        variable = None

    # ComparatorMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        TIME_SCALE: TimeScale.TRIAL,
        MONITORED_VALUES: None})

    standard_output_states = ObjectiveMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:SSE,
                                    CALCULATE:lambda x: np.sum(x*x)},
                                   {NAME:MSE,
                                    CALCULATE:lambda x: np.sum(x*x)/len(x)}])

    @tc.typecheck
    def __init__(self,
                 sample:tc.optional(tc.any(OutputState, Mechanism_Base, dict, is_numeric, str))=None,
                 target:tc.optional(tc.any(OutputState, Mechanism_Base, dict, is_numeric, str))=None,
                 input_states=[SAMPLE, TARGET],
                 function=LinearCombination(weights=[[-1], [1]]),
                 output_states:tc.optional(tc.any(list, dict))=[ERROR_SIGNAL, MSE],
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Parse items of input_states arg for validation (in _validate_params)
        input_states = input_states or [None] * 2
        from PsyNeuLink.Components.States.State import _parse_state_spec
        sample_input = _parse_state_spec(owner=self,
                                         state_type=InputState,
                                         state_spec=input_states[0],
                                         name=SAMPLE,
                                         value=None)
        target_input = _parse_state_spec(owner=self,
                                         state_type=InputState,
                                         state_spec=input_states[1],
                                         name=TARGET,
                                         value=None)

        # IMPLEMENTATION NOTE: The following prevents the default from being updated by subsequent assignment
        #                     (in this case, to [ERROR_SIGNAL, {NAME= MSE}]), but fails to expose default in IDE
        # output_states = output_states or [ERROR_SIGNAL, MSE]

        # Create a StandardOutputStates object from the list of stand_output_states specified for the class
        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)

        super().__init__(monitored_values=[sample, target],
                         input_states = [sample_input, target_input],
                         function=function,
                         output_states=output_states.copy(), # prevent default from getting overwritten by later assign
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """If sample and target values are specified, validate that they are compatible
        """

        if INPUT_STATES in request_set:
            input_states = request_set[INPUT_STATES]

            # Validate that there are exactly two input_states (for sample and target)
            num_input_states = len(input_states)
            if num_input_states != 2:
                raise ComparatorMechanismError("{} arg is specified for {} ({}), so it must have exactly 2 items, "
                                               "one each for {} and {}".
                                               format(INPUT_STATES,
                                                      self.__class__.__name__,
                                                      len(input_states),
                                                      SAMPLE,
                                                      TARGET))

            # Validate that input_states are specified as dicts
            if not all(isinstance(input_state,dict) for input_state in input_states):
                raise ComparatorMechanismError("PROGRAM ERROR: all items in input_state args must be converted to dicts"
                                               " by calling State._parse_state_spec() before calling super().__init__")

            # Validate length of variable for sample = target
            if VARIABLE in input_states[0]:
                # input_states arg specified in standard state specification dict format
                lengths = [len(input_state[VARIABLE]) for input_state in input_states]
            else:
                # input_states arg specified in {<STATE_NAME>:<STATE SPECIFICATION DICT>} format
                lengths = [len(list(input_state_dict.values())[0][VARIABLE]) for input_state_dict in input_states]

            if lengths[0] != lengths[1]:
                raise ComparatorMechanismError("Length of value specified for {} InputState of {} ({}) must be "
                                               "same as length of value specified for {} ({})".
                                               format(SAMPLE,
                                                      self.__class__.__name__,
                                                      lengths[0],
                                                      TARGET,
                                                      lengths[1]))

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
