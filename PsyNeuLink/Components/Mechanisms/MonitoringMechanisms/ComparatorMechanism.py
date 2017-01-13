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

A ComparatorMechanism monitors the outputState of a ProcessingMechanism in a :doc:`Process`, and compares this to a
target provided as input to the ``run`` method of the process (or system to which it belongs) when it is executed.
The comparison can be done using subtraction or division.

.. _Comparator_Creation:

Creating a ComparatorMechanism
------------------------------

A ComparatorMechanism can be created either directly, by calling its constructor, or using the
:class:`mechanism` function and specifying "ComparatorMechanism" as its ``mech_spec`` argument. The type of comparison
is specified in the ``comparison_operation`` argument, which can be the keyword :keyword:`SUBTRACTION` or
:keyword:`DIVISION`.  It can also be created by
:ref:`in context specification of a LearningProjection <Projection_Creation>` for a projection to the
`TERMINAL` mechanism of a process.  One or more ComparatorMechanisms are also
created automatically when a process or system is created for which learning is specified; each is assigned a
projection from the outputState of a `TERMINAL` mechanism that receives a MappingProjection being learned,
and a LearningProjection to that MappingProjection  (see :ref:`learning in a process <Process_Learning>`,
and :ref:`automatic creation of LearningSignals  <LearningProjection_Automatic_Creation> for details).


.. _Comparator_Structure

Structure
---------

A ComparatorMechanism has two inputStates:  the :keyword:`SAMPLE inputState receives a MappingProjection from
the primary outputState of a `TERMINAL` mechanism in a process;  the :keyword:`COMPARATOR_TARGET` inputState
is assigned its value from the ``target`` argument of a call to the :doc:`run <Run>` method of a process or system.
It has five outputStates, described under Execution below.


.. _Comparator_Execution

Execution
---------

A ComparatorMechanism always executes after the mechanism it is monitoring.  The ``value`` of the primary outputState of
the mechanism being monitored is assigned as the value of the ComparatorMechanism's :keyword:`COMPARATOR_SAMPLE`
inputState;  the value of the :keyword:`COMPARATOR_TARGET` inputState is received from the process (or system to
which it belongs) when it is run. When the ComparatorMechanism is executed, if ``comparison_operation`` is
:keyword:`SUBTRACTION`, then its ``function`` subtracts the  :keyword:`COMPARATOR_SAMPLE` from the
:keyword:`COMPARATOR_TARGET`; if ``comparison_operation`` is :keyword:`DIVISION`, the :keyword:`COMPARATOR_TARGET`
is divided by the :keyword:`COMPARATOR_SAMPLE`.  After each execution of the mechanism:

.. _Comparator_Results:

    * the **result** of the ``function`` calculation is assigned to the mechanism's ``value`` attribute, the value of
      its :keyword:`COMPARISON_RESULT` outputState, and to the 1st item of its
      :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attribute.

    * the **mean** of the result is assigned to the value of the mechanism's :keyword:`COMPARISON_MEAN` outputState,
      and to the 2nd item of its :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attribute.

    * the **sum** of the result is assigned to the value of the mechanism's :keyword:`COMPARISON_SUM` outputState,
      and to the 3rd item of its :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attribute.

    * the **sum of squares** of the result's elements ("sum squared error") is assigned to value of the mechanism's
      :keyword:`COMPARISON_SSE` outputState, and to the 4th item of its
      :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attribute.

    * the **mean of the squares** of the result's elements ("mean squared error") is assigned to value of the
      mechanism's :keyword:`COMPARISON_MSE` outputState, and to the 5th item of its
      :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attribute.

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


class ComparatorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ComparatorMechanism(MonitoringMechanism_Base):
    """
    ComparatorMechanism(                       \
    default_sample_and_target=None,   \
    comparison_operation=SUBTRACTION, \
    params=None,                      \
    name=None,                        \
    prefs=None)

    Implements ComparatorMechanism subclass of MonitoringMechanism

    COMMENT:

        Description:
            ComparatorMechanism is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the
                Function class
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
        :py:meth:`execute <Mechanism.Mechanism_Base.execute>` or :py:meth:`run <Mechanism.Mechanism_Base.run` methods.
        The first item is the :keyword:`COMPARATOR_SAMPLE` input and the second is the :keyword:`COMPARATOR_TARGET`
        input, which must be the same length.  This also serves as a template to specify the length of inputs to the
        ``function``.

    comparison_operation : keyword[SUBTRACTION or DIVISION] : default SUBTRACTION
        specifies how the :keyword:`COMPARATOR_SAMPLE` and :keyword:`COMPARATOR_TARGET` will be compared:
        * :keyword:`SUBTRACTION`: :keyword:`COMPARATOR_TARGET` - :keyword:`SAMPLE`;
        * :keyword:`DIVISION`: :keyword:`COMPARATOR_TARGET` ÷ :keyword:`SAMPLE`.

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the mechanism, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a params dict).
        The following entries can be included:
        * :keyword:`COMPARATOR_SAMPLE`:  Mechanism, InputState, or the name of or specification dictionary for one;
        * :keyword:`COMPARATOR_TARGET`:  Mechanism, InputState, or the name of or specification dictionary for one;
        * :kewyord:`FUNCTION`: Function or method;  default is :class:`LinearCombination`.

    COMMENT:
        [TBI]
        time_scale :  TimeScale : TimeScale.TRIAL
            specifies whether the mechanism is executed on the :keyword:`TIME_STEP` or :keyword:`TRIAL` time scale.
            This must be set to :keyword:`TimeScale.TIME_STEP` for the ``rate`` parameter to have an effect.
    COMMENT

    name : str : default ComparatorMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the PreferenceSet for mechanism.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : 2d np.array
        the input to ``function``.  The first item is the ``value`` of the :keyword:`COMPARATOR_SAMPLE` inputState,
        and is the second ``value`` of the :keyword:`COMPARATOR_TARGET` inputState.

    sample : 1d np.array
        the first item of the ``variable`` and the ``value`` of the :keyword:`COMPARATOR_SAMPLE` inputState.

    target : 1d np.array
        the second item of ``variable``, and the ``value`` of the :keyword:`COMPARATOR_TARGET` inputState.

    function : CombinationFunction : default LinearCombination
        the function used to compare :keyword:`COMPARATOR_SAMPLE` with :keyword:`COMPARATOR_TARGET`.

    comparison_operation : SUBTRACTION or DIVISION : default SUBTRACTION
        determines the operation used by ``function`` to compare the :keyword:`COMPARATOR_SAMPLE` with
        :keyword:`_COMPARATOR_TARGET`.
        * :keyword:`SUBTRACTION`: :keyword:`COMPARATOR_TARGET` - :keyword:`COMPARATOR_SAMPLE`;
        * :keyword:`DIVISION`: :keyword:`COMPARATOR_TARGET` ÷ :keyword:`COMPARATOR_SAMPLE`.

    value : 2d np.array
        holds the output of the comparison operation carried out by the ComparatorMechanism's ``function``.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to the ``value`` of the :keyword:`COMPARTOR_RESULT` outputState
            and the first item of ``outputValue``.
    COMMENT

    outputValue : List[1d np.array, float, float, float, float]
        a list with the following items:
        * **result** of the ``function`` calculation (value of :keyword:`COMPARISON_RESULT` outputState);
        * **mean** of the result's elements (``value`` of :keyword:`COMPARISON_MEAN` outputState)
        * **sum** of the result's elements (``value`` of :keyword:`COMPARISON_SUM` outputState)
        * **sum of squares** of the result's elements (``value`` of :keyword:`COMPARISON_SSE` outputState)
        * **mean of squares** of the result's elements (``value`` of :keyword:`COMPARISON_MSE` outputState)

    name : str : default ComparatorMechanism-<index>
        the name of the mechanism.
        Specified in the name argument of the call to create the mechanism;
        if not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the PreferenceSet for mechanism.
        Specified in the prefs argument of the call to create the mechanism;
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).

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
            comparison_array = self.function(variable=self.variable, params=runtime_params)

            self.summedErrorSignal = sum

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


