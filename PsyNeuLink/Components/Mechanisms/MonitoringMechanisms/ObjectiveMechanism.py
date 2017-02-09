# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ObjectiveMechanism *******************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

"""

from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.Functions.Function import LinearCombination

OBJECT = 0
WEIGHT = 1
EXPONENT = 2

class ObjectiveError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ObjectiveMechanism(MonitoringMechanism_Base):
    """Implement ObjectiveMechanism subclass

    Description:
        ObjectiveMechanism is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the Function class
        It's function uses the LinearCombination Function Function to compare two input variables
        COMPARISON_OPERATION (functionParams) determines whether the comparison is subtractive or divisive
        The function returns an array with the Hadamard (element-wise) differece/quotient of target vs. sample,
            as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

    Instantiation:
        - A ObjectiveMechanism can be instantiated in several ways:
            - directly, by calling ObjectiveMechanism()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), ObjectiveMechanism also implements the following params:
        - variable (2D np.array): [[objectiveSample], [objectiveTarget]]
        - params (dict):
            + OBJECTIVE_SAMPLE (MechanismsInputState, dict or str): (default: automatic local instantiation)
                specifies inputState to be used for objectiveMechanism sample
            + OBJECTIVE_TARGET (MechanismsInputState, dict or str):  (default: automatic local instantiation)
                specifies inputState to be used for objectiveMechanism target
            + FUNCTION (Function of method):  (default: LinearCombination)
            + FUNCTION_PARAMS (dict):
                + COMPARISON_OPERATION (str): (default: SUBTRACTION)
                    specifies operation used to compare OBJECTIVE_SAMPLE with OBJECTIVE_TARGET;
                    SUBTRACTION:  output = target-sample
                    DIVISION:  output = target/sample
        Notes:
        *  params can be set in the standard way for any Function subclass:
            - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
                 and used for paramsCurrent unless and until the latter are changed in a function call
            - paramInstanceDefaults can be later modified using _assign_defaults
            - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of ObjectiveMechanism are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of ObjectiveMechanism can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "ObjectiveMechanism" with a hyphenated, indexed suffix ('ObjectiveMechanism-n')

    Execution:
        - Computes comparison of two inputStates of equal length and generates array of same length,
            as well as summary statistics (sum, sum of squares, and variance of comparison array values) 
        - self.execute returns self.value
        Notes:
        * ObjectiveMechanism handles "runtime" parameters (specified in call to execute method) differently than std Components:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with ControlProjection output

    Class attributes:
        + componentType (str): ObjectiveMechanism
        + classPreference (PreferenceSet): Objective_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
        + variableClassDefault (value):  Objective_DEFAULT_STARTING_POINT // QUESTION: What to change here
        + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL,
                                      FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value): input to mechanism's execute method (default:  Objective_DEFAULT_STARTING_POINT)
        + value (value): output of execute method
        + sample (1D np.array): reference to inputState[OBJECTIVE_SAMPLE].value
        + target (1D np.array): reference to inputState[OBJECTIVE_TARGET].value
        + comparisonFunction (Function): Function Function used to compare sample and test
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, default set is created by copying Objective_PreferenceSet

    Instance methods:
        - _instantiate_function(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        - execute(variable, time_scale, params, context)
            executes COMPARISON_OPERATION and returns outcome values (in self.value and values of self.outputStates)

    """

    componentType = OBJECTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ObjectiveCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = [[0],[0]]  # ObjectiveMechanism compares two 1D np.array inputStates

    # ObjectiveMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        TIME_SCALE: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        FUNCTION_PARAMS:{COMPARISON_OPERATION: DIFFERENCE},
        INPUT_STATES:[OBJECTIVE_SAMPLE,   # Automatically instantiate local InputStates
                      OBJECTIVE_TARGET],  # for sample and target, and name them using kw constants
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
        # MODIFIED 12/7/16 NEW:

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences, default input value (Objective_DEFAULT_NET_INPUT) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        if default_input_value is None:
            # default_input_value = Objective_DEFAULT_INPUT
            # FIX: ??CORRECT:
            default_input_value = self.variableClassDefault

        super().__init__(variable=default_input_value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):

        if len(variable) != 2:
            if INITIALIZING in context:
                raise ObjectiveError("Variable argument in initializaton of {} must be a two item list or array".
                                            format(self.name))
            else:
                raise ObjectiveError("Variable argument for execute method of {} "
                                            "must be a two item list or array".format(self.name))

        if len(variable[0]) != len(variable[1]):
            if INITIALIZING in context:
                raise ObjectiveError("The two items in variable argument used to initialize {} "
                                            "must have the same length ({},{})".
                                            format(self.name, len(variable[0]), len(variable[1])))
            else:
                raise ObjectiveError("The two items in variable argument for execute method of {} "
                                            "must have the same length ({},{})".
                                            format(self.name, len(variable[0]), len(variable[1])))

        super()._validate_variable(variable=variable, context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Get (and validate) [TBI: OBJECTIVE_SAMPLE, OBJECTIVE_TARGET and/or] FUNCTION if specified

        # TBI:
        # Validate that OBJECTIVE_SAMPLE and/or OBJECTIVE_TARGET, if specified, are each a valid reference to an
        #     inputState and, if so, use to replace default (name) specifications in paramClassDefault[INPUT_STATES]
        # Note: this is because OBJECTIVE_SAMPLE and OBJECTIVE_TARGET are declared but not defined in
        #       paramClassDefaults (above)

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
            if not (comparison_function is LINEAR_COMBINATION_FUNCTION):
                raise ObjectiveError("Unrecognized function {} specified for FUNCTION".
                                            format(comparison_function))

        # CONFIRM THAT THESE WORK:

        # Validate SAMPLE (will be further parsed and instantiated in _instantiate_input_states())
        try:
            sample = request_set[OBJECTIVE_SAMPLE]
        except KeyError:
            pass
        else:
            if not (isinstance(sample, (str, InputState, dict))):
                raise ObjectiveError("Specification of {} for {} must be a InputState, "
                                            "or the name (string) or specification dict for one".
                                            format(sample, self.name))
            self.paramClassDefaults[INPUT_STATES][0] = sample

        try:
            target = request_set[OBJECTIVE_TARGET]
        except KeyError:
            pass
        else:
            if not (isinstance(target, (str, InputState, dict))):
                raise ObjectiveError("Specification of {} for {} must be a InputState, "
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
        self.sample = self.inputStates[OBJECTIVE_SAMPLE].value
        self.target = self.inputStates[OBJECTIVE_SAMPLE].value

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

        # Get comparisonFunction params from FUNCTION_PARAMS
        comparison_operation = self.paramsCurrent[FUNCTION_PARAMS][COMPARISON_OPERATION]
        del self.paramsCurrent[FUNCTION_PARAMS][COMPARISON_OPERATION]

        # For WEIGHTS and EXPONENTS: [<coefficient for OBJECTIVE_SAMPLE>, <coefficient for OBJECTIVE_TARGET>]
        # If the comparison operation is subtraction, set WEIGHTS
        if comparison_operation is DIFFERENCE:
            comparison_function_params[OPERATION] = SUM
            comparison_function_params[WEIGHTS] = np.array([-1,1])
        # If the comparison operation is division, set EXPONENTS
        elif comparison_operation is QUOTIENT:
            comparison_function_params[OPERATION] = PRODUCT
            comparison_function_params[EXPONENTS] = np.array([-1,1])
        else:
            raise ObjectiveError("PROGRAM ERROR: specification of COMPARISON_OPERATION {} for {} "
                                        "not recognized; should have been detected in Function._validate_params".
                                        format(comparison_operation, self.name))

        # Instantiate comparisonFunction
        self.comparisonFunction = LinearCombination(variable_default=self.variable,
                                                    param_defaults=comparison_function_params)

        super()._instantiate_attributes_before_function(context=context)

    # MODIFIED 12/7/16 OLD:
    # def _instantiate_attributes_before_function(self, context=None):
    #
    #     # Map indices of output to outputState(s)
    #     self._outputStateValueMapping = {}
    #     self._outputStateValueMapping[COMPARISON_RESULT] = ObjectiveOutput.COMPARISON_RESULT.value
    #     self._outputStateValueMapping[COMPARISON_MEAN] = ObjectiveOutput.COMPARISON_MEAN.value
    #     self._outputStateValueMapping[COMPARISON_SUM] = ObjectiveOutput.COMPARISON_SUM.value
    #     self._outputStateValueMapping[COMPARISON_SSE] = ObjectiveOutput.COMPARISON_SSE.value
    #     self._outputStateValueMapping[COMPARISON_MSE] = ObjectiveOutput.COMPARISON_MSE.value
    #
    #     super()._instantiate_attributes_before_function(context=context)
    # MODIFIED 12/7/16 END

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        # DOCUMENTATION:
        # variable (float): set to self.value (= self.inputValue)
        # runtime_params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
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
            value of item-wise comparison of sample vs. target in outputState[ObjectiveOutput.COMPARISON_RESULT].value
            mean of item-wise comparisons in outputState[ObjectiveOutput.COMPARISON_MEAN].value
            sum of item-wise comparisons in outputState[ObjectiveOutput.COMPARISON_SUM].value
            sum of squqres of item-wise comparisions in outputState[ObjectiveOutput.COMPARISON_SSE].value
        """

        # #region ASSIGN OBJECTIVE_SAMPLE AND OBJECTIVE_TARGET ARRAYS
        # # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
        # # - assign convenience names to each param
        # sample = self.paramsCurrent[OBJECTIVE_SAMPLE].value
        # target = self.paramsCurrent[OBJECTIVE_TARGET].value
        #
        # #endregion

        if not context:
            context = EXECUTING + self.name

        self._check_args(variable=variable, params=runtime_params, context=context)


        # EXECUTE COMPARISON FUNCTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.TIME_STEP:
            raise MechanismError("TIME_STEP mode not yet implemented for ObjectiveMechanism")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Implement terminate() below

        # EXECUTE COMPARISON FUNCTION (TRIAL TIME SCALE) ------------------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Calculate comparision and stats
            # FIX: MAKE SURE VARIABLE HAS BEEN SET TO self.inputValue SOMEWHERE
            comparison_array = self.comparisonFunction.function(variable=self.variable, params=runtime_params)

            # # MODIFIED 12/7/16 OLD:
            # mean = np.mean(comparison_array)
            # sum = np.sum(comparison_array)
            # SSE = np.sum(comparison_array * comparison_array)
            # MSE = SSE/len(comparison_array)
            #
            # self.outputValue[ObjectiveOutput.COMPARISON_RESULT.value] = comparison_array
            # self.outputValue[ObjectiveOutput.COMPARISON_MEAN.value] = mean
            # self.outputValue[ObjectiveOutput.COMPARISON_SUM.value] = sum
            # self.outputValue[ObjectiveOutput.COMPARISON_SSE.value] = SSE
            # self.outputValue[ObjectiveOutput.COMPARISON_MSE.value] = MSE
            #
            # return self.outputValue
            # MODIFIED 12/7/16 NEW:
            return comparison_array
            # MODIFIED 12/7/16 END

        else:
            raise MechanismError("time_scale not specified for ObjectiveMechanism")
