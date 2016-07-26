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

import numpy as np
# from numpy import sqrt, random, abs, tanh, exp
from numpy import sqrt, abs, tanh, exp
from Functions.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
from Functions.MechanismStates.MechanismInputState import MechanismInputState
from Functions.Utility import LinearCombination

# Comparator parameter keywords:
kwComparatorSample = "ComparatorSample"
kwComparatorTarget = "ComparatorTarget"
kwComparisonOperation = "ComparisonOperation"

# Comparator outputs (used to create and name outputStates):
kwComparisonArray = 'ComparisonArray'
kwComparisonMean = 'ComparisonMean'
kwComparisonSum = 'ComparisonSum'
kwComparisonSumSquares = 'ComparisonSumSquares'
kwComparisonMSE = 'ComparisonMSE'

# Comparator output indices (used to index output values):
class ComparatorOutput(AutoNumber):
    COMPARISON_ARRAY = ()
    COMPARISON_MEAN = ()
    COMPARISON_SUM = ()
    COMPARISON_SUM_SQUARES = ()
    COMPARISON_MSE = ()


class ComparisonOperation(IntEnum):
        SUBTRACTION = 0
        DIVISION = 1
        MUTUAL_ENTROPY = 2


class LinearComparatorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LinearComparator(MonitoringMechanism_Base):
    """Implement Comparator subclass

    Description:
        Comparator is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the Function class
        It's executeMethod uses the LinearCombination Utility Function to compare two input variables
        kwComparisonOperation (executeMethodParams) determines whether the comparison is subtractive or divisive
        The executeMethod returns an array with the Hadamard (element-wise) differece/quotient of target vs. sample,
            as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

    Instantiation:
        - A Comparator Mechanism can be instantiated in several ways:
            - directly, by calling Comparator()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), Comparator also implements the following params:
        - variable (2D np.array): [[comparatorSample], [comparatorTarget]]
        - params (dict):
            # + kwComparatorSample (MechanismsInputState, dict or str): (default: ???XXX)
            #     specifies array to be compared with kwComparatorTarget
            # + kwComparatorTarget (MechanismsInputState, dict or str):  (default: ???XXX)
            #     specifies array against which kwComparatorSample is compared
            + kwExecuteMethod (Utility of method):  (default: LinearCombination)
            + kwExecuteMethodParams (dict):
                + kwComparisonOperation (ComparisonOperation): (default: ComparisonOperation.SUBTRACTION)
                    specifies operation used to compare kwComparatorSample with kwComparatorTarget;
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
        * Comparator handles "runtime" parameters (specified in call to execute method) differently than std Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the EXECUTE_METHOD_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding MechanismParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): Comparator
        + classPreference (PreferenceSet): Comparator_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
        + variableClassDefault (value):  Comparator_DEFAULT_STARTING_POINT // QUESTION: What to change here
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      kwExecuteMethodParams:{kwComparisonOperation: ComparisonOperation.SUBTRACTION}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value): input to mechanism's execute method (default:  Comparator_DEFAULT_STARTING_POINT)
        + value (value): output of execute method
        + comparatorSample (MechanismInputSTate): reference to inputState[0]
        + comparatorTarget (MechanismInputSTate): reference to inputState[1]
        + comparisonFunction (Utility): Utility Function used to compare sample and test
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, default set is created by copying Comparator_PreferenceSet

    Instance methods:
        • instantiate_execute_method(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        • execute(variable, time_scale, params, context)
            executes kwComparisonOperation and returns outcome values (in self.value and values of self.outputStates)

    """

    functionType = "LinearComparator"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ComparatorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = [[0],[0]]  # Comparator compares two 1D np.array inputStates

    # requiredParamClassDefaultTypes = Function.requiredParamClassDefaultTypes.copy()
    # requiredParamClassDefaultTypes.update({kwComparatorSample : [str, dict, MechanismInputState],
    #                                        kwComparatorTarget: [str, dict, MechanismInputState]})

    # Comparator parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwExecuteMethod: LinearCombination,
        kwExecuteMethodParams:{kwComparisonOperation: ComparisonOperation.SUBTRACTION},
        kwMechanismInputStates:[kwComparatorSample,
                                kwComparatorTarget],
        kwMechanismOutputStates:[kwComparisonArray,
                                 kwComparisonMean,
                                 kwComparisonSum,
                                 kwComparisonSumSquares,
                                 kwComparisonMSE]
    })

    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign type-level preferences, default input value (Comparator_DEFAULT_NET_INPUT) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name
        self.functionName = self.functionType

        if default_input_value is NotImplemented:
            # default_input_value = Comparator_DEFAULT_INPUT
            # FIX: ??CORRECT:
            default_input_value = self.variableClassDefault

        super().__init__(variable=default_input_value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Get (and validate) self.comparisonFunction from kwExecuteMethod if specified

        Intercept definition of kwExecuteMethod and assign to self.combinationFunction;
            leave defintion of self.execute below intact;  it will call combinationFunction

        Args:
            request_set:
            target_set:
            context:

        """

        try:
            self.comparisonFunction = request_set[kwExecuteMethod]
        except KeyError:
            self.comparisonFunction = LinearCombination
        else:
            # Delete kwExecuteMethod so that it does not supercede self.execute
            del request_set[kwExecuteMethod]
            comparison_function = self.comparisonFunction
            if isclass(comparison_function):
                comparison_function = comparison_function.__name__

            # Validate kwExecuteMethod
            # IMPLEMENTATION NOTE: Currently, only LinearCombination is supported
            # IMPLEMENTATION:  TEST INSTEAD FOR FUNCTION CATEGORY == COMBINATION
            if not (comparison_function is kwLinearCombination):
                raise LinearComparatorError("Unrecognized function {} specified for kwExecuteMethod".
                                            format(comparison_function))

        super().validate_params(request_set=request_set, target_set=target_set, context=context)


    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Assign sample and target specs to kwInputStates, use kwComparisonOperation to re-assign kwExecuteMethodParams

        Override super method to:
            # assign kwComparatorSample and kwComparatorTarget in appropriate order to list in kwInputStates
            check if combinationFunction is default (LinearCombination):
                assign combinationFunction params based on kwComparisonOperation (in kwExecuteMethodParams[])
                    + kwWeights: [-1,1] if kwComparisonOperation is SUBTRACTION
                    + kwExponents: [-1,1] if kwComparisonOperation is DIVISION
            instantiate self.combinationFunction

        """

        # sample = self.paramsCurrent[kwComparatorSample]
        # target = self.paramsCurrent[kwComparatorTarget]
        # self.paramsCurrent[kwMechanismInputStates] = [sample, target]

        # FIX: USE ASSIGN_DEFAULTS HERE (TO BE SURE INSTANCE DEFAULTS ARE UPDATED AS WELL AS PARAMS_CURRENT

        comparison_function_params = {}

        # Get comparisonFunction params from kwExecuteMethodParams
        comparison_operation = self.paramsCurrent[kwExecuteMethodParams][kwComparisonOperation]
        del self.paramsCurrent[kwExecuteMethodParams][kwComparisonOperation]


        # For kwWeights and kwExponents: [<coefficient for kwSample>,<coefficient for kwTarget>]
        # If the comparison operation is subtraction, set kwWeights
        if comparison_operation is ComparisonOperation.SUBTRACTION:
            comparison_function_params[kwOperation] = LinearCombination.Operation.SUM
            comparison_function_params[kwWeights] = np.array([-1,1])
        # If the comparison operation is division, set kwExponents
        elif comparison_operation is ComparisonOperation.DIVISION:
            comparison_function_params[kwOperation] = LinearCombination.Operation.PRODUCT
            comparison_function_params[kwExponents] = np.array([-1,1])
        else:
            raise LinearComparatorError("PROGRAM ERROR: specification of kwComparisonOperation {} for {} "
                                        "not recognized; should have been detected in Function.validate_params".
                                        format(comparison_operation, self.name))

        # Instantiate comparisonFunction
        self.comparisonFunction = LinearCombination(variable_default=self.variable,
                                                    param_defaults=comparison_function_params)

        super().instantiate_attributes_before_execute_method(context=context)

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=NotImplemented):

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
        # sample = self.paramsCurrent[kwComparatorSample].value
        # target = self.paramsCurrent[kwComparatorTarget].value
        #
        # #endregion

        if context is NotImplemented:
            context = kwExecuting + self.name

        self.check_args(variable=variable, params=params, context=context)


        #region EXECUTE COMPARISON FUNCTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for Comparator")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.executeMethodParameterStates[kwBias]
            # Implement terminate() below
        #endregion

        #region EXECUTE COMPARISON FUNCTION (TRIAL TIME SCALE) ------------------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            #region Calculate comparision and stats
            # FIX: MAKE SURE VARIABLE HAS BEEN SET TO self.inputValue SOMEWHERE
            comparison_array = self.comparisonFunction.execute(variable=self.variable, params=params)
            deltas = comparison_array
            mean = np.mean(comparison_array)
            sum = np.sum(comparison_array)
            SSE = np.sum(comparison_array * comparison_array)
            MSE = SSE/len(comparison_array)

            # Map indices of output to outputState(s)
            self.outputStateValueMapping = {}
            self.outputStateValueMapping[kwComparisonArray] = ComparatorOutput.COMPARISON_ARRAY.value
            self.outputStateValueMapping[kwComparisonMean] = ComparatorOutput.COMPARISON_MEAN.value
            self.outputStateValueMapping[kwComparisonSum] = ComparatorOutput.COMPARISON_SUM.value
            self.outputStateValueMapping[kwComparisonSumSquares] = ComparatorOutput.COMPARISON_SUM_SQUARES.value
            self.outputStateValueMapping[kwComparisonMSE] = ComparatorOutput.COMPARISON_MSE.value

            # Assign output values
            # Get length of output from kwMechansimOutputState
            # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            #       is run (to evaluate output) before outputStates have been instantiated
            output = [None] * len(self.paramsCurrent[kwMechanismOutputStates])
            # FIX: USE NP ARRAY
            #     output = np.array([[None]]*len(self.paramsCurrent[kwMechanismOutputStates]))
            output[ComparatorOutput.COMPARISON_ARRAY.value] = deltas
            output[ComparatorOutput.COMPARISON_MEAN.value] = mean
            output[ComparatorOutput.COMPARISON_SUM.value] = sum
            output[ComparatorOutput.COMPARISON_SUM_SQUARES.value] = SSE
            output[ComparatorOutput.COMPARISON_MSE.value] = MSE
            #endregion

            #region Print results
            # FIX: MAKE SENSTIVE TO WHETHER CALLED FROM MECHANISM SUPER OR JUST FREE-STANDING (USE CONTEXT)
            # if (self.prefs.reportOutputPref and kwFunctionInit not in context):
            import re
            if (self.prefs.reportOutputPref and kwExecuting in context):
                print ("\n{} execute method:\n- sample: {}\n- target: {}".
                       format(self.name,
                              self.inputStates[kwComparatorSample].value.__str__().strip("[]"),
                              self.inputStates[kwComparatorTarget].value.__str__().strip("[]")))
                # print ("Output: ", re.sub('[\[,\],\n]','',str(output[ComparatorOutput.ACTIVATION.value])))
                print ("\nOutput:\n- Error: {}\n- MSE: {}".
                       format(self.outputStates[kwComparisonArray].value.__str__().strip("[]"),
                              self.outputStates[kwComparisonMSE].value.__str__().strip("[]")))
            #endregion

            return output
        #endregion

        else:
            raise MechanismError("time_scale not specified for Comparator")


    def terminate_function(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for Comparator


