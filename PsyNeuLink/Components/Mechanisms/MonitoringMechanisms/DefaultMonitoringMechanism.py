# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  Comparator *******************************************************
#

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.Functions.Function import LinearCombination


class ComparatorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class Comparator(MonitoringMechanism_Base):
    """Implement Comparator subclass

    Description:
        Comparator is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the Function class
        It's function uses the LinearCombination Function Function to compare two input variables
        COMPARISON_OPERATION (functionParams) determines whether the comparison is subtractive or divisive
        The function returns an array with the Hadamard (element-wise) differece/quotient of target vs. sample,
            as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

    Instantiation:
        - A Comparator Mechanism can be instantiated in several ways:
            - directly, by calling Comparator()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), Comparator also implements the following params:
        - variable (2D np.array): [[comparatorSample], [comparatorTarget]]
        - params (dict):
            + COMPARATOR_SAMPLE (MechanismsInputState, dict or str): (default: automatic local instantiation)
                specifies inputState to be used for comparator sample
            + COMPARATOR_TARGET (MechanismsInputState, dict or str):  (default: automatic local instantiation)
                specifies inputState to be used for comparator target
            + FUNCTION (Function of method):  (default: LinearCombination)
            + FUNCTION_PARAMS (dict):
                + COMPARISON_OPERATION (str): (default: SUBTRACTION)
                    specifies operation used to compare COMPARATOR_SAMPLE with COMPARATOR_TARGET;
                    SUBTRACTION:  output = target-sample
                    DIVISION:  output = target/sample
        Notes:
        *  params can be set in the standard way for any Function subclass:
            - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
                 and used for paramsCurrent unless and until the latter are changed in a function call
            - paramInstanceDefaults can be later modified using assign_defaults
            - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of Comparator are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of Comparator can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "Comparator" with a hyphenated, indexed suffix ('Comparator-n')

    Execution:
        - Computes comparison of two inputStates of equal length and generates array of same length,
            as well as summary statistics (sum, sum of squares, and variance of comparison array values) 
        - self.execute returns self.value
        Notes:
        * Comparator handles "runtime" parameters (specified in call to execute method) differently than std Components:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): Comparator
        + classPreference (PreferenceSet): Comparator_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
        + variableClassDefault (value):  Comparator_DEFAULT_STARTING_POINT // QUESTION: What to change here
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value): input to mechanism's execute method (default:  Comparator_DEFAULT_STARTING_POINT)
        + value (value): output of execute method
        + sample (1D np.array): reference to inputState[COMPARATOR_SAMPLE].value
        + target (1D np.array): reference to inputState[COMPARATOR_TARGET].value
        + comparisonFunction (Function): Function Function used to compare sample and test
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, default set is created by copying Comparator_PreferenceSet

    Instance methods:
        - _instantiate_function(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        - execute(variable, time_scale, params, context)
            executes COMPARISON_OPERATION and returns outcome values (in self.value and values of self.outputStates)

    """

    functionType = "Comparator"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ComparatorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = [[0],[0]]  # Comparator compares two 1D np.array inputStates

    # Comparator parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        FUNCTION_PARAMS:{COMPARISON_OPERATION: DIFFERENCE},
        INPUT_STATES:[COMPARATOR_SAMPLE,   # Automatically instantiate local InputStates
                                COMPARATOR_TARGET],  # for sample and target, and name them using kw constants
        OUTPUT_STATES:[COMPARISON_ARRAY,
                                 COMPARISON_MEAN,
                                 COMPARISON_SUM,
                                 COMPARISON_SUM_SQUARES,
                                 COMPARISON_MSE]
    })

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_input_value=NotImplemented,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences, default input value (Comparator_DEFAULT_NET_INPUT) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        if default_input_value is NotImplemented:
            # default_input_value = Comparator_DEFAULT_INPUT
            # FIX: ??CORRECT:
            default_input_value = self.variableClassDefault

        super().__init__(variable=default_input_value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):

        if len(variable) != 2:
            if kwInit in context:
                raise ComparatorError("Variable argument in initializaton of {} must be a two item list or array".
                                            format(self.name))
            else:
                raise ComparatorError("Variable argument for execute method of {} "
                                            "must be a two item list or array".format(self.name))

        if len(variable[0]) != len(variable[1]):
            if kwInit in context:
                raise ComparatorError("The two items in variable argument used to initialize {} "
                                            "must have the same length ({},{})".
                                            format(self.name, len(variable[0]), len(variable[1])))
            else:
                raise ComparatorError("The two items in variable argument for execute method of {} "
                                            "must have the same length ({},{})".
                                            format(self.name, len(variable[0]), len(variable[1])))

        super()._validate_variable(variable=variable, context=context)

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Get (and validate) [TBI: COMPARATOR_SAMPLE, COMPARATOR_TARGET and/or] FUNCTION if specified

        # TBI:
        # Validate that COMPARATOR_SAMPLE and/or COMPARATOR_TARGET, if specified, are each a valid reference to an inputState and, if so,
        #     use to replace default (name) specifications in paramClassDefault[INPUT_STATES]
        # Note: this is because COMPARATOR_SAMPLE and COMPARATOR_TARGET are declared but not defined in paramClassDefaults (above)

        Validate that FUNCTION, if specified, is a valid reference to a Function Function and, if so,
            assign to self.combinationFunction and delete FUNCTION param
        Note: this leaves definition of self.execute (below) intact, which will call combinationFunction

        Args:
            request_set:
            target_set:
            context:
        """

        try:
            self.comparisonFunction = request_set[FUNCTION]
        except KeyError:
            self.comparisonFunction = LinearCombination
        else:
            # Delete FUNCTION so that it does not supercede self.execute
            del request_set[FUNCTION]
            comparison_function = self.comparisonFunction
            if isclass(comparison_function):
                comparison_function = comparison_function.__name__

            # Validate FUNCTION
            # IMPLEMENTATION NOTE: Currently, only LinearCombination is supported
            # IMPLEMENTATION:  TEST INSTEAD FOR FUNCTION CATEGORY == COMBINATION
            if not (comparison_function is kwLinearCombination):
                raise ComparatorError("Unrecognized function {} specified for FUNCTION".
                                            format(comparison_function))

        # CONFIRM THAT THESE WORK:

        # Validate COMPARATOR_SAMPLE (will be further parsed and instantiated in _instantiate_input_states())
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
        self.target = self.inputStates[COMPARATOR_SAMPLE].value

    def _instantiate_attributes_before_function(self, context=None):
        """Assign sample and target specs to INPUT_STATES, use COMPARISON_OPERATION to re-assign FUNCTION_PARAMS

        Override super method to:
            check if combinationFunction is default (LinearCombination):
                assign combinationFunction params based on COMPARISON_OPERATION (in FUNCTION_PARAMS[])
                    + WEIGHTS: [-1,1] if COMPARISON_OPERATION is SUBTRACTION
                    + EXPONENTS: [-1,1] if COMPARISON_OPERATION is DIVISION
            instantiate self.combinationFunction
        """

        # FIX: USE ASSIGN_DEFAULTS HERE (TO BE SURE INSTANCE DEFAULTS ARE UPDATED AS WELL AS PARAMS_CURRENT

        comparison_function_params = {}

        # Get comparisonFunction params from FUNCTION_PARAMS
        comparison_operation = self.paramsCurrent[FUNCTION_PARAMS][COMPARISON_OPERATION]
        del self.paramsCurrent[FUNCTION_PARAMS][COMPARISON_OPERATION]

        # For WEIGHTS and EXPONENTS: [<coefficient for COMPARATOR_SAMPLE>,<coefficient for COMPARATOR_TARGET>]
        # If the comparison operation is subtraction, set WEIGHTS
        if comparison_operation is DIFFERENCE:
            comparison_function_params[OPERATION] = SUM
            comparison_function_params[WEIGHTS] = np.array([-1,1])
        # If the comparison operation is division, set EXPONENTS
        elif comparison_operation is QUOTIENT:
            comparison_function_params[OPERATION] = PRODUCT
            comparison_function_params[EXPONENTS] = np.array([-1,1])
        else:
            raise ComparatorError("PROGRAM ERROR: specification of COMPARISON_OPERATION {} for {} "
                                        "not recognized; should have been detected in Function._validate_params".
                                        format(comparison_operation, self.name))

        # Instantiate comparisonFunction
        self.comparisonFunction = LinearCombination(variable_default=self.variable,
                                                    param_defaults=comparison_function_params)

        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_attributes_before_function(self, context=None):

        # Map indices of output to outputState(s)
        self._outputStateValueMapping = {}
        self._outputStateValueMapping[COMPARISON_ARRAY] = ComparatorOutput.COMPARISON_ARRAY.value
        self._outputStateValueMapping[COMPARISON_MEAN] = ComparatorOutput.COMPARISON_MEAN.value
        self._outputStateValueMapping[COMPARISON_SUM] = ComparatorOutput.COMPARISON_SUM.value
        self._outputStateValueMapping[COMPARISON_SUM_SQUARES] = ComparatorOutput.COMPARISON_SUM_SQUARES.value
        self._outputStateValueMapping[COMPARISON_MSE] = ComparatorOutput.COMPARISON_MSE.value

        super()._instantiate_attributes_before_function(context=context)

    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=None):

        # DOCUMENTATION:
        # variable (float): set to self.value (= self.inputValue)
        # params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
        # time_scale (TimeScale): determines "temporal granularity" with which mechanism is executed
        # context (str)
        #
        # :param self:
        # :param variable (float)
        # :param params: (dict)
        # :param time_scale: (TimeScale)
        # :param context: (str)
        # :rtype self.outputState.value: (number)
        """Compare sample inputState.value with target inputState.value using comparison function

        Return:
            value of item-wise comparison of sample vs. target in outputState[ComparatorOutput.COMPARISON_ARRAY].value
            mean of item-wise comparisons in outputState[ComparatorOutput.COMPARISON_MEAN].value
            sum of item-wise comparisons in outputState[ComparatorOutput.COMPARISON_SUM].value
            sum of squqres of item-wise comparisions in outputState[ComparatorOutput.COMPARISON_SUM_SQUARES].value
        """

        # #region ASSIGN SAMPLE AND TARGET ARRAYS
        # # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
        # # - assign convenience names to each param
        # sample = self.paramsCurrent[COMPARATOR_SAMPLE].value
        # target = self.paramsCurrent[COMPARATOR_TARGET].value
        #
        # #endregion

        if not context:
            context = kwExecuting + self.name

        self._check_args(variable=variable, params=params, context=context)


        # EXECUTE COMPARISON FUNCTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.TIME_STEP:
            raise MechanismError("TIME_STEP mode not yet implemented for Comparator")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Implement terminate() below

        # EXECUTE COMPARISON FUNCTION (TRIAL TIME SCALE) ------------------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Calculate comparision and stats
            # FIX: MAKE SURE VARIABLE HAS BEEN SET TO self.inputValue SOMEWHERE
            comparison_array = self.comparisonFunction.function(variable=self.variable, params=params)
            mean = np.mean(comparison_array)
            sum = np.sum(comparison_array)
            SSE = np.sum(comparison_array * comparison_array)
            MSE = SSE/len(comparison_array)

            self.outputValue[ComparatorOutput.COMPARISON_ARRAY.value] = comparison_array
            self.outputValue[ComparatorOutput.COMPARISON_MEAN.value] = mean
            self.outputValue[ComparatorOutput.COMPARISON_SUM.value] = sum
            self.outputValue[ComparatorOutput.COMPARISON_SUM_SQUARES.value] = SSE
            self.outputValue[ComparatorOutput.COMPARISON_MSE.value] = MSE

            return self.outputValue

        else:
            raise MechanismError("time_scale not specified for Comparator")


    def terminate_function(self, context=None):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for Comparator


