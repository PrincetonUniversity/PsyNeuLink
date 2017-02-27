# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  ComparatorMechanism *****************************************************

"""
Overview
--------

A ComparatorMechanism monitors the `outputState <OutputState>` of a `ProcessingMechanism <ProcessingMechanism>` in a
`process <Process>`, and compares this to a `target <Run_Targets>` provided as input to the :keyword:`run` method of
the process or system to which it belongs. The comparison can be done using subtraction or division.

.. _Comparator_Creation:

Creating a ComparatorMechanism
------------------------------

A ComparatorMechanism can be created directly by calling its constructor
COMMENT:
    , or using the
    `mechanism` function and specifying keyword:`ComparatorMechanism` as its :keyword:`mech_spec` argument.
COMMENT
. The type of comparison is specified in the `comparison_operation` argument, which can be `SUBTRACTION` or
`DIVISION`.  It can also be created by `in-context specification <Projection_Creation>` of a LearningProjection for a
projection to the `TERMINAL` mechanism of a process.  One or more ComparatorMechanisms are also created automatically
when learning is specified for a `process <Process_Learning>` or `system <System_Execution_Learning>`. Each
ComparatorMechanism is assigned a projection from a `TERMINAL` mechanism that receives a MappingProjection being
learned. A LearningProjection to that MappingProjection is also created (see `learning in a process <Process_Learning>`,
and `automatic creation of LearningSignals  <LearningProjection_Automatic_Creation>` for details).

.. _Comparator_Structure:

Structure
---------

A ComparatorMechanism has two `inputStates <InputState>`:

    * :keyword:`COMPARATOR_SAMPLE` inputState receives a MappingProjection
      from the `primary outputState <OutputState_Primary>` of a `TERMINAL` mechanism in a process;
    ..
    * `COMPARATOR_TARGET` inputState is assigned its value from the :keyword:`target` argument of a call to the
      `run <Run>` method of a process or system.  It has five outputStates, described under
      :ref:`Execution <Comparator_Execution>` below.


.. _Comparator_Execution:

Execution
---------

A ComparatorMechanism always executes after the mechanism it is monitoring.  The :keyword:`value` of the
`primary outputState <OutputState_Primary>` of the mechanism being monitored is assigned as the :keyword:`value` of the
ComparatorMechanism's :keyword:`COMPARATOR_SAMPLE` inputState;  the value of the :keyword:`COMPARATOR_TARGET`
inputState is received from the process (or system to which it belongs) when it is run (i.e., the input provided
 in the process' or system's :keyword:`execute` method or :keyword:`run` method). When the ComparatorMechanism
is executed, if `comparison_operation` is:

    * `SUBTRACTION`, its `function <ComparatorMechanism.function>` subtracts the  `COMPARATOR_SAMPLE` from the
      `COMPARATOR_TARGET`;
    ..
    * `DIVISION`, its `function <ComparatorMechanism.function>` divides the `COMPARATOR_TARGET`by the
      `COMPARATOR_SAMPLE`.

After each execution of the mechanism:

.. _Comparator_Results:

    * the **result** of the `function <ComparatorMechanism.function>` calculation is assigned to the mechanism's
      `value <ComparatorMechanism.value>` attribute, the value of its `COMPARISON_RESULT`
      outputState, and to the 1st item of its `outputValue <ComparatorMechanism.outputValue>` attribute;
    ..
    * the **mean** of the result is assigned to the :keyword:`value` of the mechanism's `COMPARISON_MEAN` outputState,
      and to the 2nd item of its `outputValue <ComparatorMechanism.outputValue>` attribute.
    ..

    * the **sum** of the result is assigned to the :keyword:`value` of the mechanism's `COMPARISON_SUM` outputState,
      and to the 3rd item of its `outputValue <ComparatorMechanism.outputValue>` attribute.
    ..

    * the **sum of squares** of the result is assigned to the :keyword:`value` of the mechanism's `COMPARISON_SSE`
      outputState, and to the 4th item of its `outputValue <ComparatorMechanism.outputValue>` attribute.
    ..

    * the **mean of the squares** of the result is assigned to the :keyword:`value` of the mechanism's
      :keyword:`COMPARISON_MSE` outputState, and to the 5th item of its `outputValue <ComparatorMechanism.outputValue>`
      attribute.

.. _Comparator_Class_Reference:

Class Reference
---------------
"""

import typecheck as tc
import numpy as np
# from numpy import sqrt, random, abs, tanh, exp
from numpy import sqrt, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.Functions.Function import LinearCombination


ComparatorMechanism = 'ComparatorMechanism'

# ComparatorMechanism parameter keywords:
COMPARATOR_SAMPLE = "comparatorSampleSource"
COMPARATOR_TARGET = "comparatorTargetSource"
COMPARISON_OPERATION = "comparison_operation"

# ComparatorMechanism outputs (used to create and name outputStates):
COMPARISON_RESULT = 'ComparisonArray'
COMPARISON_MEAN = 'ComparisonMean'
COMPARISON_SUM = 'ComparisonSum'
COMPARISON_SSE = 'ComparisonSumSquares'
COMPARISON_MSE = 'ComparisonMSE'

# ComparatorMechanism output indices (used to index output values):
class ComparatorOutput(AutoNumber):
    """Indices of the `outputValue <Comparator.outputValue>` attribute of the ComparatorMechanism containing the
    values described below."""
    COMPARISON_RESULT = ()
    """Result of the ComparatorMechanism's `function <ComparatorMechanism.function>`."""
    COMPARISON_MEAN = ()
    """Mean of the elements in the :keyword`value` of the COMPARISON_RESULT outputState."""
    COMPARISON_SUM = ()
    """Sum of the elements in :keyword`value` of the COMPARISON_RESULT outputState."""
    COMPARISON_SSE = ()
    """Sum squares of the elements in :keyword`value` of the COMPARISON_RESULT outputState."""
    COMPARISON_MSE = ()
    """Mean of the squares of the elements in :keyword`value` of the COMPARISON_RESULT outputState."""


class ComparatorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ComparatorMechanism(MonitoringMechanism_Base):
    """
    ComparatorMechanism(              \
    default_sample_and_target=None,   \
    comparison_operation=SUBTRACTION, \
    params=None,                      \
    name=None,                        \
    prefs=None)

    Implements the ComparatorMechanism subclass of `MonitoringMechanism`.

    COMMENT:
        Description:
            ComparatorMechanism is a subtype of the MonitoringMechanism Type of the Mechanism Category of the
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

    comparison_operation : keyword[SUBTRACTION or DIVISION] : default SUBTRACTION
        specifies how the `COMPARATOR_SAMPLE` and `COMPARATOR_TARGET` will be compared:

        * `SUBTRACTION`: `COMPARATOR_TARGET` - `COMPASAMPLE`

        * `DIVISION`: `COMPARATOR_TARGET` ÷ `SAMPLE`

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

    variable : 2d np.array
        the input to `function <ComparatorMechanism.function>`.  The first item is the :keyword:`value` of the
        `COMPARATOR_SAMPLE` inputState, and the second is the :keyword:`value` of the `COMPARATOR_TARGET` inputState.

    sample : 1d np.array
        the first item of the `variable <ComparatorMechanism.variable>` and the :keyword:`value` of the
        `COMPARATOR_SAMPLE` inputState.

    target : 1d np.array
        the second item of the `variable <ComparatorMechanism.variable>` and the :keyword:`value` of the
        `COMPARATOR_TARGET` inputState.

    function : CombinationFunction : default LinearCombination
        the function used to compare `COMPARATOR_SAMPLE` with `COMPARATOR_TARGET`.

    comparison_operation : SUBTRACTION or DIVISION : default SUBTRACTION
        determines the operation used by `function <ComparatorMechanism.function>` to compare the `COMPARATOR_SAMPLE`
        with `_COMPARATOR_TARGET`.

        * `SUBTRACTION`: `COMPARATOR_TARGET` - `COMPARATOR_SAMPLE`;

        * `DIVISION`: `COMPARATOR_TARGET` ÷ `COMPARATOR_SAMPLE`.

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

    componentType = "ComparatorMechanism"
    # onlyFunctionOnInit = True

    initMethod = INIT__EXECUTE__METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ComparatorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    variableClassDefault = [[0],[0]]  # ComparatorMechanism compares two 1D np.array inputStates

    # ComparatorMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        TIME_SCALE: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        INPUT_STATES:[COMPARATOR_SAMPLE,   # Instantiate two inputStates, one for sample and target each
                      COMPARATOR_TARGET],  #    and name them using keyword names
        PARAMETER_STATES: None,             # This suppresses parameterStates
        OUTPUT_STATES:[
            {NAME:COMPARISON_RESULT},

            {NAME:COMPARISON_MEAN,
             CALCULATE:lambda x: np.mean(x)},

            {NAME:COMPARISON_SUM,
             CALCULATE:lambda x: np.sum(x)},

            {NAME:COMPARISON_SSE,
             CALCULATE:lambda x: np.sum(x*x)},

            {NAME:COMPARISON_MSE,
             CALCULATE:lambda x: np.sum(x*x)/len(x)}
        ]})

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_sample_and_target=None,
                 comparison_operation:tc.enum(SUBTRACTION, DIVISION)=SUBTRACTION,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(comparison_operation=comparison_operation,
                                                 params=params)

        # Assign componentType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if not name:
            self.name = self.componentType
        else:
            self.name = name
        self.componentName = self.componentType

        if default_sample_and_target is None:
            default_sample_and_target = self.variableClassDefault

        super().__init__(variable=default_sample_and_target,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):

        if len(variable) != 2:
            if INITIALIZING in context:
                raise ComparatorError("Variable argument in initializaton of {} must be a two item list or array".
                                            format(self.name))
            else:
                raise ComparatorError("Variable argument for function of {} "
                                            "must be a two item list or array".format(self.name))

        if len(variable[0]) != len(variable[1]):
            if INITIALIZING in context:
                raise ComparatorError("The two items in variable argument used to initialize {} "
                                            "must have the same length ({},{})".
                                            format(self.name, len(variable[0]), len(variable[1])))
            else:
                raise ComparatorError("The two items in variable argument for function of {} "
                                            "must have the same length ({},{})".
                                            format(self.name, len(variable[0]), len(variable[1])))

        super()._validate_variable(variable=variable, context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Get (and validate) [TBI: COMPARATOR_SAMPLE, COMPARATOR_TARGET and/or] FUNCTION if specified

        # TBI:
        # Validate COMPARATOR_SAMPLE and/or COMPARATOR_TARGET, if specified, are valid references to an inputState
        #     and, if so, use to replace default (name) specifications in paramClassDefault[INPUT_STATES]
        # Note: this is because COMPARATOR_SAMPLE and COMPARATOR_TARGET are
        #       declared but not defined in paramClassDefaults (above)

        Validate that FUNCTION, if specified, is a valid reference to a Function Function and, if so,
            assign to self.combinationFunction and delete FUNCTION param
        Note: this leaves definition of self.execute (below) intact, which will call combinationFunction

        Args:
            request_set:
            target_set:
            context:

        """

        # Validate SAMPLE (will be further parsed and instantiated in _instantiate_input_states())
        try:
            sample = request_set[COMPARATOR_SAMPLE]
        except KeyError:
            pass
        else:
            if not (isinstance(sample, (str, InputState, dict))):
                raise ComparatorError("Specification of {} for {} must be a InputState, "
                                            "or the name (string) or specification dict for one".
                                            format(sample, self.name))
            self.paramClassDefaults[INPUT_STATES][0] = sample

        try:
            target = request_set[COMPARATOR_TARGET]
        except KeyError:
            pass
        else:
            if not (isinstance(target, (str, InputState, dict))):
                raise ComparatorError("Specification of {} for {} must be a InputState, "
                                            "or the name (string) or specification dict for one".
                                            format(target, self.name))
            self.paramClassDefaults[INPUT_STATES][0] = target

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)


    def _instantiate_input_states(self, context=None):
        """Assign self.sample and self.target to value of corresponding inputStates

        Args:
            context:

        Returns:

        """
        super()._instantiate_input_states(context=context)
        self.sample = self.inputStates[COMPARATOR_SAMPLE].value
        self.target = self.inputStates[COMPARATOR_TARGET].value

    def _instantiate_attributes_before_function(self, context=None):
        """Assign sample and target specs to INPUT_STATES, use COMPARISON_OPERATION to re-assign FUNCTION_PARAMS

        Override super method to:
            check if combinationFunction is default (LinearCombination):
                assign combinationFunction params based on COMPARISON_OPERATION (in FUNCTION_PARAMS[])
                    + WEIGHTS: [-1,1] if COMPARISON_OPERATION is SUBTRACTION
                    + EXPONENTS: [-1,1] if COMPARISON_OPERATION is DIVISION
            instantiate self.combinationFunction

        """

        # FIX: USE _ASSIGN_DEFAULTS HERE (TO BE SURE INSTANCE DEFAULTS ARE UPDATED AS WELL AS PARAMS_CURRENT

        comparison_function_params = {}
        comparison_operation = self.paramsCurrent[COMPARISON_OPERATION]

        self.paramsCurrent[FUNCTION_PARAMS] = {}
        # For WEIGHTS and EXPONENTS: [<coefficient for COMPARATOR_SAMPLE>,<coefficient for COMPARATOR_TARGET>]
        # If the comparison operation is subtraction, set WEIGHTS
        if comparison_operation is SUBTRACTION:
            self.paramsCurrent[FUNCTION_PARAMS][OPERATION] = SUM
            self.paramsCurrent[FUNCTION_PARAMS][WEIGHTS] = np.array([-1,1])
        # If the comparison operation is division, set EXPONENTS
        elif comparison_operation is DIVISION:
            self.paramsCurrent[FUNCTION_PARAMS][OPERATION] = PRODUCT
            self.paramsCurrent[FUNCTION_PARAMS][EXPONENTS] = np.array([-1,1])
        else:
            raise ComparatorError("PROGRAM ERROR: specification of COMPARISON_OPERATION {} for {} "
                                        "not recognized; should have been detected in Function._validate_params".
                                        format(comparison_operation, self.name))

        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_function(self, context=None):
        super()._instantiate_function(context=context)

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        """Compare sample inputState.value with target inputState.value using comparison function

        Executes COMPARISON_OPERATION and returns outcome values (in self.value and values of self.outputStates):

        Computes:
            comparison of two inputStates of equal length and generates array of same length,
            as well as summary statistics (mean, sum, sum of squares, and MSE of comparison array values)

        Return:
            value of item-wise comparison of sample vs. target in outputState[ComparatorOutput.COMPARISON_RESULT].value
            mean of item-wise comparisons in outputState[ComparatorOutput.COMPARISON_MEAN].value
            sum of item-wise comparisons in outputState[ComparatorOutput.COMPARISON_SUM].value
            sum of squares of item-wise comparisions in outputState[ComparatorOutput.COMPARISON_SSE].value
            mean of sum of squares of item-wise comparisions in outputState[ComparatorOutput.COMPARISON_MSE].value
        """

        if not context:
            context = EXECUTING + self.name

        self._check_args(variable=variable, params=runtime_params, context=context)

        # Assign sample and target attributes
        #    which also checks (by way of target property) that target is within range of sample
        #    if the sample's source mechanism specifies a range parameter

        self.sample = self.inputStates[COMPARATOR_SAMPLE].value
        self.target = self.inputStates[COMPARATOR_TARGET].value

        # EXECUTE COMPARISON FUNCTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.TIME_STEP:
            raise MechanismError("TIME_STEP mode not yet implemented for ComparatorMechanism")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Implement terminate() below

        #region EXECUTE COMPARISON FUNCTION (TRIAL TIME SCALE) ---------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Calculate comparision and stats
            # FIX: MAKE SURE VARIABLE HAS BEEN SET TO self.inputValue SOMEWHERE
            comparison_array = self.function(variable=self.variable, params=runtime_params, context=context)

            self.summed_error_signal = sum

            return comparison_array

        else:
            raise MechanismError("time_scale not specified for ComparatorMechanism")


    def terminate_function(self, context=None):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for ComparatorMechanism

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        """Check that target is within range if specified for sample

        Check that target is compatible with the value of all projections to sample
           from source mechanisms that specify a range parameter;
        Note:  range must be in the form of a list or 1d np.array;
            first item: lower bound of target value (inclusive)
            second item: upper bound of target value (inclusive)
        """
        try:
            for projection in self.inputStates[COMPARATOR_SAMPLE].receivesFromProjections:
                sample_source = projection.sender.owner
                try:
                    sample_range = sample_source.range
                    if list(sample_range):
                        for target_item in value:
                            if not sample_range[0] <= target_item <= sample_range[1]:
                                raise ComparatorError("Item of target ({}) is out of range for {}: {}) ".
                                                      format(target_item, sample_source.name, sample_range))
                except AttributeError:
                    pass
        except (AttributeError):
            pass
        self._target = value


