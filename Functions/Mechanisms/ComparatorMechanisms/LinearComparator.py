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
from Functions.Mechanisms.ComparatorMechanisms.ComparatorMechanism import *
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
kwComparisonSumSquares = 'kwComparisonSumSquares'

# Comparator output indices (used to index output values):
class ComparatorOutput(AutoNumber):
    COMPARISON_ARRAY = ()
    COMPARISON_MEAN = ()
    COMPARISON_SUM = ()
    COMPARISON_SUM_SQUARES = ()
    

class ComparisonOperation(IntEnum):
        SUBTRACTION = 0
        DIVISION = 1
        MUTUAL_ENTROPY = 2


class LinearComparatorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LinearComparator(ComparatorMechanism_Base):
    """Implement Comparator subclass

    Description:
        Comparator is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the Function class
        It implements a Mechanism that compares two input variables and generates output based on kwComparisonOperation

    Instantiation:
        - A Comparator Mechanism can be instantiated in several ways:
            - directly, by calling Comparator()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), Comparator also implements the following params:
        - params (dict):
            + kwComparatorSample (MechanismsInputState, dict or str): (default: ???XXX)
                specifies array to be compared with kwComparatorTarget
            + kwComparatorTarget (MechanismsInputState, dict or str):  (default: ???XXX)
                specifies array against which kwComparatorSample is compared
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
        * Comparator handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
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
        + variable (value): input to mechanism's execute method (default:  Comparator_DEFAULT_STARTING_POINT) // QUESTION: What to change here
        + value (value): output of execute method
        + comparatorSample (MechanismInputSTate): reference to inputState[0]
        + comparatorTarget (MechanismInputSTate): reference to inputState[1]
        + comparisonOperation (Utility): Utility Function used to transform the input
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, a default set is created by copying Comparator_PreferenceSet

    Instance methods:
        • instantiate_execute_method(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        • comparisonFunction(variable, params, context): LinearCombination Utility Function
            variable (2D np.array): [[comparatorSample], [comparatorTarget]]
            params:
                + kwExecuteMethodParams:
                    + kwWeights: [-1,1] if comparisonOperation is SUBTRACTION
                    + kwExponents: [-1,1] if comparisonOperation is DIVISION
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

    requiredParamClassDefaultTypes = Function.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({kwComparatorSample : [str, dict, MechanismInputState],
                                           kwComparatorTarget: [str, dict, MechanismInputState]})

    # Comparator parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwExecuteMethod: LinearCombination,
        kwExecuteMethodParams:{kwComparisonOperation: ComparisonOperation.SUBTRACTION},
        kwMechanismOutputStates:[kwComparisonArray,
                                 kwComparisonMean,
                                 kwComparisonSum,
                                 kwComparisonSumSquares]
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

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Assign sample and target specs to kwInputStates, use kwComparisonOperation to re-assign kwExecuteMethodParams

        Override super method to:
            assign kwComparatorSample and kwComparatorTarget in appropriate order to list in kwInputStates
            intercept definition of kwExecuteMethod to assign to self.transferFunction
                (and leave self.execute intact, that will call transferFunction)
            instantiate self.transferFunction

        """

        sample = self.paramsCurrent[kwComparatorSample]
        target = self.paramsCurrent[kwComparatorTarget]
        self.paramsCurrent[kwMechanismInputStates] = [sample, target]

        comparison_operation = self.paramsCurrent[kwExecuteMethodParams][kwComparisonOperation]
        # If the comparison operation is subtraction, set the weights param to -1 (applied to kwComparisonSample.value)
        if comparison_operation is ComparisonOperation.SUBTRACTION:
            self.paramClassDefaults[kwExecuteMethodParams][kwWeights] = np.array([-1,1])
        # If the comparison operation is division, set the weights param to -1 (applied to kwComparisonSample.value)
        elif comparison_operation is ComparisonOperation.DIVISION:
            self.paramClassDefaults[kwExecuteMethodParams][kwExponents] = np.array([-1,1])
        else:
            raise LinearComparatorError("PROGRAM ERROR: specification of kwComparisonOperation {} for {} not recognized; "
                                        "should have been detected in Function.validate_params".
                                        format(comparison_operation, self.name))

        self.comparisonFunction = LinearCombination(variable_default=self.variable,
                                                  param_defaults=self.paramsCurrent[kwExecuteMethodParams])

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

        #region ASSIGN SAMPLE AND TARGET ARRAYS
        # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
        # - assign convenience names to each param
        sample = self.paramsCurrent[kwComparatorSample].value
        target = self.paramsCurrent[kwComparatorTarget].value

        #endregion

        #region EXECUTE INTEGRATOR FUNCTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for Comparator")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.executeMethodParameterStates[kwBias]
            # Implement terminate() below
        #endregion

        #region EXECUTE COMPARISON (TRIAL TIME SCALE) ------------------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:
            # FIX: MAKE SURE VARIABLE HAS BEEN SET TO self.inputValue SOMEWHERE
            comparison_output = self.comparisonFunction(variable=variable, params=params)


            # FIX: STILL NEEDS WORK:
            #region Print results
            # if (self.prefs.reportOutputPref and kwFunctionInit not in context):
            import re
            if (self.prefs.reportOutputPref and kwExecuting in context):
                print ("\n{0} execute method:\n- input: {1}\n- params:".
                       format(self.name, self.inputState.value.__str__().strip("[]")))
                print ("    nunits:", str(nunits).__str__().strip("[]"),
                       "\n    net_input:", re.sub('[\[,\],\n]','',str(net_input)),
                       "\n    gain:", gain,
                       "\n    bias:", bias,
                       "\n    activation range:", re.sub('[\[,\],\n]','',str(range)),
                       "\n- output:",
                       "\n    mean activation: {0}".format(output[ComparatorOutput.ACTIVATION_MEAN.value]),
                       "\n    activation variance: {0}".format(output[ComparatorOutput.ACTIVATION_VARIANCE.value]))
                print ("Output: ", re.sub('[\[,\],\n]','',str(output[ComparatorOutput.ACTIVATION.value])))
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


