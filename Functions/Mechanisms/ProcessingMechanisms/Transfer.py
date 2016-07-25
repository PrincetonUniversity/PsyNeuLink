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
from Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from Functions.Utility import Linear, Exponential, Logistic

# Transfer parameter keywords:
kwTransferFunction = "TransferFunction"
kwTransfer_NUnits = "Transfer_Number_Of_Units"
kwTransfer_Gain = "Transfer_Gain"
kwTransfer_Bias = "Transfer_Bias"
kwTransfer_Range = "Transfer_Range"
kwTransfer_NetInput = "Transfer_Net_Input"

# Transfer outputs (used to create and name outputStates):
kwTransfer_Activation = "Transfer_Activation"
kwTransfer_Activation_Mean = "Transfer_Activation_Mean "
kwTransfer_Activation_Variance = "kwTransfer_Activation_Variance"

# Transfer output indices (used to index output values):
class Transfer_Output(AutoNumber):
    ACTIVATION = ()
    ACTIVATION_MEAN = ()
    ACTIVATION_VARIANCE = ()

# Transfer default parameter values:
Transfer_DEFAULT_NUNITS= 1
Transfer_DEFAULT_GAIN = 1
Transfer_DEFAULT_BIAS = 0
Transfer_DEFAULT_RANGE = np.array([])
Transfer_DEFAULT_NET_INPUT = [0]


class TransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class Transfer(Mechanism_Base):
    """Implement Transfer subclass

    Description:
        Transfer is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the Function class
        It implements a Mechanism that transforms it input variable based on kwTransferFunction (default: Linear)

    Instantiation:
        - A Transfer Mechanism can be instantiated in several ways:
            - directly, by calling Transfer()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), Transfer also implements the following params:
        - params (dict):
            + kwExecuteMethodParams (dict):
                + kwTransferFunction (Utility class or str):   (default: Linear)
                    specifies the function used to transform the input;  can be one of the following:
                    + kwLinear or Linear
                    + kwExponential or Exponential
                    + kwLogistic or Logistic
                + kwTransfer_Gain (float): (default: Transfer_DEFAULT_GAIN)
                    specifies gain of the transfer function:
                        slope for Linear, rate for Exponential, gain for Logistic
                + kwTransfer_Bias (float): (default: Transfer_DEFAULT_BIAS)
                    specifies bias of the transfer function
                        intercept for Linear, scale for Exponential, bias for Logistic
                + kwTransfer_Range ([float, float]): (default: Transfer_DEFAULT_RANGE)
                    specifies the activation range of the units:
                       the first item indicates the minimum activation
                       the second item indicates the maximum activation
                + kwTransfer_NetInput (int):   (default: Transfer_DEFAULT_NUNITS)
                    specifies an array that is added to the input (self.variable) on every call to Transfer.execute()
                # FIX: HOW IS THIS DIFFERENT THAN LENGTH OF self.variable
                + kwTransfer_NUnits (float): (default: Transfer_DEFAULT_NUNITS
                    specifies number of units (length of input array)
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
        - Multiplies input by gain then applies transferFunction and bias; the result is capped by the kwTransfer_Range
        - self.value (and values of outputStates) contain each outcome value
            (e.g., Activation, Activation_Mean, Activation_Variance)
        - self.execute returns self.value
        Notes:
        * Transfer handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the EXECUTE_METHOD_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding MechanismParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): Transfer
        + classPreference (PreferenceSet): Transfer_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
        + variableClassDefault (value):  Transfer_DEFAULT_STARTING_POINT // QUESTION: What to change here
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      kwExecuteMethodParams:{kwTransferFunction: Linear
                                                             kwTransfer_Gain: Transfer_DEFAULT_GAIN
                                                             kwTransfer_Bias: Transfer_DEFAULT_BIAS
                                                             kwTransfer_Range: Transfer_DEFAULT_RANGE
                                                             kwTransfer_NetInput: Transfer_DEFAULT_NET_INPUT
                                                             kwTransfer_NUnits: Transfer_DEFAULT_NUNITS}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value): input to mechanism's execute method (default:  Transfer_DEFAULT_STARTING_POINT) // QUESTION: What to change here
        + value (value): output of execute method
        + transferFunction (Utility): Utility Function used to transform the input
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, a default set is created by copying Transfer_PreferenceSet

    Instance methods:
        • instantiate_execute_method(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        • execute(variable, time_scale, params, context)
            executes kwTransferFunction and returns outcome values (in self.value and values of self.outputStates)

    """

    functionType = "Transfer"

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = Transfer_DEFAULT_NET_INPUT # Sets template for variable (input) to be compatible with Transfer_DEFAULT_STARTING_POINT

    # Transfer parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwExecuteMethod: Linear,
        kwExecuteMethodParams:{
            # kwTransferFunction: Logistic,
            kwTransfer_NetInput: Transfer_DEFAULT_NET_INPUT, # "attentional" component
            kwTransfer_Gain: Transfer_DEFAULT_GAIN,            # used as starting point
            kwTransfer_Bias: Transfer_DEFAULT_BIAS,  # assigned as output
            kwTransfer_Range: Transfer_DEFAULT_RANGE,
            kwTransfer_NUnits: Transfer_DEFAULT_NUNITS,
        },
        kwMechanismOutputStates:[kwTransfer_Activation,
                                 kwTransfer_Activation_Mean,
                                 kwTransfer_Activation_Variance]
    })

    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign type-level preferences, default input value (Transfer_DEFAULT_NET_INPUT) and call super.__init__

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
            default_input_value = Transfer_DEFAULT_NET_INPUT

        super(Transfer, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context,
                                  context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Get (and validate) self.transferFunction from kwExecuteMethod if specified

        Args:
            request_set:
            target_set:
            context:
        Returns:

        """

        try:
            # transfer_function = self.paramsCurrent[kwExecuteMethodParams][kwTransferFunction]
            self.transferFunction = request_set[kwExecuteMethod]
        except KeyError:
            self.transferFunction = Linear
        else:
            # Delete kwExecuteMethod so that it does not supercede self.execute
            del request_set[kwExecuteMethod]
            xferFn = self.transferFunction
            if isclass(xferFn):
                xferFn = xferFn.__name__

            # Validate kwExecuteMethod
            # IMPLEMENTATION:  TEST INSTEAD FOR FUNCTION CATEGORY == TRANSFER
            if not (xferFn is kwLinear or xferFn is kwExponential or xferFn is kwLogistic):
                raise TransferError("Unrecognized function {} specified for kwTransferFunction".format(xferFn))

        super().validate_params(request_set=request_set, target_set=target_set, context=context)

    def instantiate_execute_method(self, context=NotImplemented):
        """Instantiate self.transferFunction and then call super.instantiate_execute_method()

        Override super method to:
            intercept definition of kwExecuteMethod
            assign to self.transferFunction (and leave self.execute intact, that will call transferFunction)
            instantiate self.transferFunction

        """

        gain = self.paramsCurrent[kwExecuteMethodParams][kwTransfer_Gain]
        bias = self.paramsCurrent[kwExecuteMethodParams][kwTransfer_Bias]
        xferFn = self.transferFunction

        if isclass(xferFn):
            xferFn = xferFn.__name__

        if xferFn is kwLinear:
            transfer_function = Linear
            transfer_function_params = {Linear.kwSlope: gain,
                                        Linear.kwIntercept: bias}
        elif xferFn is kwExponential:
            transfer_function = Exponential
            transfer_function_params = {Exponential.kwRate: gain,
                                        # FIX:  IS THIS CORRECT (OR SHOULD EXPONENTIAL INCLUDE AN OFFSET
                                        Exponential.kwScale: bias}
        elif xferFn is kwLogistic:
            transfer_function = Logistic
            transfer_function_params = {Logistic.kwGain: gain,
                                        Logistic.kwBias: bias}

        self.transferFunction = transfer_function(variable_default=self.variable,
                                                  param_defaults=transfer_function_params)

        super().instantiate_execute_method(context=context)


    def execute(self,
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
            mean of items in kwTransfer_Activation outputState[TransferOuput.ACTIVATION_MEAN].value
            variance of items in kwTransfer_Activation outputState[TransferOuput.ACTIVATION_VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + kwTransfer_NetInput (float)
            + kwTransfer_Gain (float)
            + kwTransfer_Bias (float)
            + kwTransfer_Range (float)
            + kwTransfer_NUnits (float)
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
        # - assign convenience names to each param
        net_input = (self.inputState.value)
        gain = float(self.executeMethodParameterStates[kwTransfer_Gain].value *
                     self.executeMethodParameterStates[kwTransfer_Gain].value)
        bias = float(self.executeMethodParameterStates[kwTransfer_Bias].value)
        range = (self.executeMethodParameterStates[kwTransfer_Range].value)
        nunits = float(self.executeMethodParameterStates[kwTransfer_NUnits].value)
        #endregion

        #region EXECUTE INTEGRATOR FUNCTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for Transfer")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.executeMethodParameterStates[kwBias]
            # Implement terminate() below
        #endregion

        #region EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Get length of output from kwMechansimOutputState
            # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            #       is run (to evaluate output) before outputStates have been instantiated
        # FIX: USE LIST:
            output = [None] * len(self.paramsCurrent[kwMechanismOutputStates])
        # FIX: USE NP ARRAY
        #     output = np.array([[None]]*len(self.paramsCurrent[kwMechanismOutputStates]))

            # activation_vector = (net_input * gain + bias)

            activation_vector = self.transferFunction.execute(variable=net_input, params=params)

            if range.size >= 2:
                maxCapIndices = np.where(activation_vector > np.max(range))[0]
                minCapIndices = np.where(activation_vector < np.min(range))[0]
                activation_vector[maxCapIndices] = np.max(range);
                activation_vector[minCapIndices] = np.min(range);

            self.outputStateValueMapping = {}
            self.outputStateValueMapping[kwTransfer_Activation] = \
                Transfer_Output.ACTIVATION.value
            self.outputStateValueMapping[kwTransfer_Activation_Mean] = \
                Transfer_Output.ACTIVATION_MEAN.value
            self.outputStateValueMapping[kwTransfer_Activation_Variance] = \
                Transfer_Output.ACTIVATION_VARIANCE.value

            output[Transfer_Output.ACTIVATION.value] = activation_vector;

            output[Transfer_Output.ACTIVATION_MEAN.value] = \
                np.array(np.mean(output[Transfer_Output.ACTIVATION.value]))
            output[Transfer_Output.ACTIVATION_VARIANCE.value] = \
                np.array(np.var(output[Transfer_Output.ACTIVATION.value]))



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
                       "\n    mean activation: {0}".format(output[Transfer_Output.ACTIVATION_MEAN.value]),
                       "\n    activation variance: {0}".format(output[Transfer_Output.ACTIVATION_VARIANCE.value]))
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


