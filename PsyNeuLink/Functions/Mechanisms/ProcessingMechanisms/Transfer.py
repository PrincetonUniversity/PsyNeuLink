# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  Transfer *******************************************************
#

import numpy as np
# from numpy import sqrt, random, abs, tanh, exp
from numpy import sqrt, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Functions.Utility import Linear, Exponential, Logistic

# Transfer parameter keywords:
kwTransfer_Length = "Transfer_Number_Of_Units"
kwTransfer_Gain = "gain"
kwTransfer_Offset = "bias"
kwTransfer_Range = "range"
kwTransfer_Bias = "Transfer_Net_Input"

# Transfer outputs (used to create and name outputStates):
kwTransfer_Output = "Transfer_Activation"
kwTransfer_Output_Mean = "Transfer_Activation_Mean "
kwTransfer_Output_Variance = "kwTransfer_Output_Variance"

# Transfer output indices (used to index output values):
class Transfer_Output(AutoNumber):
    ACTIVATION = ()
    ACTIVATION_MEAN = ()
    ACTIVATION_VARIANCE = ()

# Transfer default parameter values:
Transfer_DEFAULT_LENGTH= 1
Transfer_DEFAULT_GAIN = 1
Transfer_DEFAULT_BIAS = 0
Transfer_DEFAULT_OFFSET = 0
Transfer_DEFAULT_RANGE = np.array([])


class TransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# IMPLEMENTATION NOTE:  IMPLEMENTS kwOffset PARAM BUT IT IS NOT CURRENTLY BEING USED
class Transfer(ProcessingMechanism_Base):
    """Implement Transfer subclass

    Description:
        Transfer is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the Function class
        It implements a Mechanism that transforms its input variable based on kwFunction (default: Linear)

    Instantiation:
        - A Transfer Mechanism can be instantiated in several ways:
            - directly, by calling Transfer()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), Transfer also implements the following params:
        - params (dict):
            + kwFunctionParams (dict):
                + kwFunction (Utility class or str):   (default: Linear)
                    specifies the function used to transform the input;  can be one of the following:
                    + kwLinear or Linear
                    + kwExponential or Exponential
                    + kwLogistic or Logistic
                + kwTransfer_Gain (float): (default: Transfer_DEFAULT_GAIN)
                    specifies gain of the transfer function:
                        slope for Linear, rate for Exponential, gain for Logistic
                + kwTransfer_Bias (float): (default: Transfer_DEFAULT_BIAS)
                    convolved with input prior to applying transfer function
                        additive for Linear, multiplicative for Exponential (scale) and Logistic (bias)
                + kwTransfer_Offset (float): (default: Transfer_DEFAULT_OFFSET)
                    added to output of the transfer function 
                        intercept for Linear; added posthoc for Exponential and Logistic
                + kwTransfer_Range ([float, float]): (default: Transfer_DEFAULT_RANGE)
                    specifies the range of the input values:
                       the first item indicates the minimum value
                       the second item indicates the maximum value
                + kwTransfer_Length (int):   (default: Transfer_DEFAULT_LENGTH)
                # FIX: HOW IS THIS DIFFERENT THAN LENGTH OF self.variable
                + kwTransfer_Length (float): (default: Transfer_DEFAULT_LENGTH
                    specifies number of items (length of input array)
        Notes:
        *  params can be set in the standard way for any Function subclass:
            - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
                 and used for paramsCurrent unless and until the latter are changed in a function call
            - paramInstanceDefaults can be later modified using assign_defaults
            - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of Transfer are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of Transfer can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "Transfer" with a hyphenated, indexed suffix ('Transfer-n')

    Execution:
        - Multiplies input by gain then applies function and bias; the result is capped by the kwTransfer_Range
        - self.value (and values of outputStates) contain each outcome value
            (e.g., Activation, Activation_Mean, Activation_Variance)
        - self.execute returns self.value
        Notes:
        * Transfer handles "runtime" parameters (specified in call to execute method) differently than std Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): Transfer
        + classPreference (PreferenceSet): Transfer_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
        + variableClassDefault (value):  Transfer_DEFAULT_BIAS
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value): input to mechanism's execute method (default:  Transfer_DEFAULT_BIAS)
        + value (value): output of execute method
        + function (Utility): Utility Function used to transform the input
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, a default set is created by copying Transfer_PreferenceSet

    Instance methods:
        - instantiate_function(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        - execute(variable, time_scale, params, context)
            executes function and returns outcome values (in self.value and values of self.outputStates)

    """

    functionType = "Transfer"

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = Transfer_DEFAULT_BIAS # Sets template for variable (input)
                                                 #  to be compatible with Transfer_DEFAULT_BIAS

    # Transfer parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwOutputStates:[kwTransfer_Output,
                                 kwTransfer_Output_Mean,
                                 kwTransfer_Output_Variance]
    })

    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 function=Linear(),
                 range=np.array([]),
                 params=None,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=functionType+kwInit):
        """Assign type-level preferences, default input value (Transfer_DEFAULT_BIAS) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(function=function,
                                                 range=range,
                                                 params=params)

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name
        self.functionName = self.functionType

        if default_input_value is NotImplemented:
            default_input_value = Transfer_DEFAULT_BIAS

        super(Transfer, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context,
                                  context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Get (and validate) self.function from kwFunction if specified

        Intercept definition of kwFunction and assign to self.combinationFunction;
            leave defintion of self.execute below intact;  it will call combinationFunction

        Args:
            request_set:
            target_set:
            context:
        """

# FIX: EITHER CREATE LOCAL COPY OF functionParams AND DELETE FROM REQUEST_SET
#      OR MOVE ALL OF THIS TO AFTER VALIDATION

        # # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 OLD:
        # try:
        #     self.function = request_set[kwFunction]
        # except KeyError:
        #     self.function = Linear
        # else:
        #     # Delete kwFunction so that it does not supercede self.execute
        #     del request_set[kwFunction]
        #     transfer_function = self.function
        #     if isclass(transfer_function):
        #         transfer_function = transfer_function.__name__
        #
        #     # Validate kwFunction
        #     # IMPLEMENTATION:  TEST INSTEAD FOR FUNCTION CATEGORY == TRANSFER
        #     if not (transfer_function is kwLinear or
        #                     transfer_function is kwExponential or
        #                     transfer_function is kwLogistic):
        #         raise TransferError("Unrecognized function {} specified for kwFunction of {}".
        #                             format(transfer_function, self.name))
        #
        #     try:
        #         self.functionParams = request_set[kwFunctionParams]
        #     except KeyError:
        #         pass
        #     else:
        #         del request_set[kwFunctionParams]
        #         # FIX: VALIDATE self.functionParams HERE

        # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 NEW:
        # FIX: USE CATEGORY RATHER THAN EXPLICIT NAMES OF FUNCTIONS FOR VALIDATION
        transfer_function = request_set[kwFunction]
        if isinstance(transfer_function, Function):
            transfer_function = transfer_function.__class__.__name__
        elif isclass(transfer_function):
            transfer_function = transfer_function.__name__

        # Validate kwFunction
        # IMPLEMENTATION:  TEST INSTEAD FOR FUNCTION CATEGORY == TRANSFER
        if not (transfer_function is kwLinear or
                        transfer_function is kwExponential or
                        transfer_function is kwLogistic or
                        transfer_function is kwSoftMax):
            raise TransferError("Unrecognized function {} specified for kwFunction of {}".
                                format(transfer_function, self.name))

        # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 END

        super().validate_params(request_set=request_set, target_set=target_set, context=context)

    # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 OLD:
    # def instantiate_function(self, context=NotImplemented):
    #     """Instantiate self.function and then call super.instantiate_function()
    #
    #     Override super method to instantiate kwFunction and assign to self.function (rather than self.execute):
    #         Note:
    #         * self.execute will call self.function, but must also carry out other tasks e.g., generate output)
    #         in this respect, it functions as the update() method does for Mechanism, Process, etc.
    #
    #     """
    #
    #     transfer_function = self.function
    #     # Instantiate function
    #     self.function = transfer_function(variable_default=self.variable,
    #                                       # params=self.paramsCurrent[kwFunctionParams])
    #                                       params=self.functionParams)
    #     super().instantiate_function(context=context)


    def __call__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=NotImplemented):
        """Execute Transfer function (currently only trial-level, analytic solution)

        Execute Transfer function on input, and assign to output:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by transfer function in outputState[TransferOuput.ACTIVATION].value
            mean of items in kwTransfer_Output outputState[TransferOuput.ACTIVATION_MEAN].value
            variance of items in kwTransfer_Output outputState[TransferOuput.ACTIVATION_VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + kwTransfer_Bias (float)
            + kwTransfer_Gain (float)
            + kwTransfer_Offset (float)
            + kwTransfer_Range (float)
            + kwTransfer_Length (float)
        - time_scale (TimeScale): determines "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param time_scale: (TimeScale)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        #region ASSIGN PARAMETER VALUES
        # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
        input = (self.inputState.value)
        range = self.paramsCurrent[kwTransfer_Range]
        nunits = len(self.variable)
        #endregion

        #region EXECUTE TRANSFER FUNCTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for Transfer")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.parameterStates[kwBias]
            # Implement terminate() below
        #endregion

        #region EXECUTE TRANSFER FUNCTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Calculate transformation and stats
            # # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 OLD:
            # transformed_vector = self.function.execute(variable=input, params=params)
            # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 NEW:
            transformed_vector = self.function(variable=input, params=params)
            # MODIFIED FOR EXECUTE->FUNCTION 8/29/16 END

            if range.size >= 2:
                maxCapIndices = np.where(transformed_vector > np.max(range))[0]
                minCapIndices = np.where(transformed_vector < np.min(range))[0]
                transformed_vector[maxCapIndices] = np.max(range);
                transformed_vector[minCapIndices] = np.min(range);
            mean = np.mean(transformed_vector)
            variance = np.var(transformed_vector)

            # Map indices of output to outputState(s)
            self.outputStateValueMapping = {}
            self.outputStateValueMapping[kwTransfer_Output] = Transfer_Output.ACTIVATION.value
            self.outputStateValueMapping[kwTransfer_Output_Mean] = Transfer_Output.ACTIVATION_MEAN.value
            self.outputStateValueMapping[kwTransfer_Output_Variance] = Transfer_Output.ACTIVATION_VARIANCE.value

            # Assign output values
            # Get length of output from kwOutputStates
            # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            #       is run (to evaluate output) before outputStates have been instantiated
            output = [None] * len(self.paramsCurrent[kwOutputStates])
            # FIX: USE NP ARRAY
            #     output = np.array([[None]]*len(self.paramsCurrent[kwOutputStates]))
            output[Transfer_Output.ACTIVATION.value] = transformed_vector;
            output[Transfer_Output.ACTIVATION_MEAN.value] = mean
            output[Transfer_Output.ACTIVATION_VARIANCE.value] = variance

            #region Print results
            # if (self.prefs.reportOutputPref and kwFunctionInit not in context):
            import re
            if (self.prefs.reportOutputPref and kwExecuting in context):
                print ("\n{0} execute method:\n- input: {1}\n- params:".
                       format(self.name, self.inputState.value.__str__().strip("[]")))
                print ("    length:", str(nunits).__str__().strip("[]"),
                       "\n    input:", re.sub('[\[,\],\n]','',str(input)),
                       # "\n    gain:", gain,
                       # "\n    bias:", bias,
                       "\n    value range:", re.sub('[\[,\],\n]','',str(range)),
                       "\n- output:",
                       "\n    mean output: {0}".format(output[Transfer_Output.ACTIVATION_MEAN.value]),
                       "\n    output variance: {0}".format(output[Transfer_Output.ACTIVATION_VARIANCE.value]))
                print ("Output: ", re.sub('[\[,\],\n]','',str(output[Transfer_Output.ACTIVATION.value])))
            #endregion

            return output
        #endregion

        else:
            raise MechanismError("time_scale not specified for Transfer")


    def terminate_function(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for Transfer


