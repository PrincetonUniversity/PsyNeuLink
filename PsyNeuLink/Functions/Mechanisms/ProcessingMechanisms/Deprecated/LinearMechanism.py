# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  LinearMechanism *******************************************************
#

import numpy as np
# from numpy import sqrt, random, abs, tanh, exp
from numpy import sqrt, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *

# LinearMechanism parameter keywords:
kwLinearMechanism_NUnits = "LinearMechanism_Number_Of_Units"
kwLinearMechanism_Slope = "LinearMechanism_Gain"
kwLinearMechanism_Intercept = "LinearMechanism_Bias"
kwLinearMechanism_Range = "LinearMechanism_Range"
kwLinearMechanism_NetInput = "LinearMechanism_Net_Input"

# LinearMechanism outputs (used to create and name outputStates):
kwLinearMechanism_Activation = "LinearMechanism_Activation"
kwLinearMechanism_Activation_Mean = "LinearMechanism_Activation_Mean "
kwLinearMechanism_Activation_Variance = "kwLinearMechanism_Activation_Variance"

# Linear Layer default parameter values:
LinearMechanism_DEFAULT_NUNITS= 1
LinearMechanism_DEFAULT_SLOPE = 1
LinearMechanism_DEFAULT_INTERCEPT = 0
LinearMechanism_DEFAULT_RANGE = np.array([])
LinearMechanism_DEFAULT_NET_INPUT = [0]

class LinearMechanism_Output(AutoNumber):
    ACTIVATION = ()
    ACTIVATION_MEAN = ()
    ACTIVATION_VARIANCE = ()


class LinearMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LinearMechanism(Mechanism_Base):
    """Implement LinearMechanism subclass

    Description:
        LinearMechanism is a subclass Type of the Mechanism Category of the Function class
        It implements a Mechanism for a single linear neural network layer

    Instantiation:
        - A LinearMechanism mechanism can be instantiated in several ways:
            - directly, by calling LinearMechanism()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), LinearMechanism also implements the following params:
        - params (dict):
            + FUNCTION_PARAMS (dict):
                + kwLinearMechanism_NetInput (int):   (default: LinearMechanism_DEFAULT_NUNITS)
                    specifies net input component that is added to the input (self.variable) on every call to LinearMechanism.execute()
                + kwLinearMechanism_Slope (float): (default: LinearMechanism_DEFAULT_SLOPE)
                    specifies slope of the linear activation function
                + kwLinearMechanism_Intercept (float): (default: LinearMechanism_DEFAULT_INTERCEPT)
                    specifies intercept of th elinear activation function
                + kwLinearMechanism_Range ([float, float]): (default: LinearMechanism_DEFAULT_RANGE)
                    specifies the activation range of the units where the first element indicates the minimum and the second element indicates the maximum activation
                + kwLinearMechanism_NUnits (float): (default: LinearMechanism_DEFAULT_NUNITS
                    specifies number of hidden units
        Notes:
        *  params can be set in the standard way for any Function subclass:
            - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
                 and used for paramsCurrent unless and until the latter are changed in a function call
            - paramInstanceDefaults can be later modified using assign_defaults
            - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of LinearMechanism are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of LinearMechanism can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "DDM" with a hyphenated, indexed suffix ('DDM-n')

    Execution:
        - Multiplies net input times the slope and adds intercept. The result is then capped by the range of the activation function
        - self.value (and values of outputStates) contain each outcome value (e.g., Activation, Activation_Mean, Activation_Variance)
        - self.execute returns self.value
        Notes:
        * LinearMechanism handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): LinearMechanism
        + classPreference (PreferenceSet): LinearMechanism_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
        + variableClassDefault (value):  DDM_DEFAULT_STARTING_POINT // QUESTION: What to change here
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      FUNCTION_PARAMS:{kwLinearMechanism_NetInput: LinearMechanism_DEFAULT_NET_INPUT
                                                                 kwLinearMechanism_Slope: LinearMechanism_DEFAULT_SLOPE
                                                                 kwLinearMechanism_Intercept: LinearMechanism_DEFAULT_INTERCEPT
                                                                 kwLinearMechanism_Range: LinearMechanism_DEFAULT_RANGE
                                                                 kwLinearMechanism_NUnits: LinearMechanism_DEFAULT_NUNITS}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value) - input to mechanism's execute method (default:  DDM_DEFAULT_STARTING_POINT) // QUESTION: What to change here
        + value (value) - output of execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, a default set is created by copying DDM_PreferenceSet

    Instance methods:
        - instantiate_function(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        - execute(variable, time_scale, params, context)
            executes specified version of DDM and returns outcome values (in self.value and values of self.outputStates)

    """

    functionType = "LinearMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'LinearMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = LinearMechanism_DEFAULT_NET_INPUT # Sets template for variable (input) to be compatible with DDM_DEFAULT_STARTING_POINT

    # DDM parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        # function is hard-coded in self.execute, but can be overridden by assigning following param:
        # FUNCTION: None
        FUNCTION_PARAMS:{
            kwLinearMechanism_NetInput: LinearMechanism_DEFAULT_NET_INPUT, # "attentional" component
            kwLinearMechanism_Slope: LinearMechanism_DEFAULT_SLOPE,            # used as starting point
            kwLinearMechanism_Intercept: LinearMechanism_DEFAULT_INTERCEPT,  # assigned as output
            kwLinearMechanism_Range: LinearMechanism_DEFAULT_RANGE,
            kwLinearMechanism_NUnits: LinearMechanism_DEFAULT_NUNITS,
            # TBI:
            # kwDDM_DriftRateVariability: DDM_ParamVariabilityTuple(variability=0, distribution=NotImplemented),
            # kwKwDDM_StartingPointVariability: DDM_ParamVariabilityTuple(variability=0, distribution=NotImplemented),
            # kwDDM_ThresholdVariability: DDM_ParamVariabilityTuple(variability=0, distribution=NotImplemented),
        },
        kwOutputStates:[kwLinearMechanism_Activation,
                                 kwLinearMechanism_Activation_Mean,
                                 kwLinearMechanism_Activation_Variance]
    })

    # Set default input_value to default bias for DDM
    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign type-level preferences, default input value (LinearMechanism_DEFAULT_NET_INPUT) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        if default_input_value is NotImplemented:
            default_input_value = LinearMechanism_DEFAULT_NET_INPUT

        super(LinearMechanism, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context,
                                  context=self)

    def instantiate_function(self, context=NotImplemented):
        """Delete params not in use, call super.instantiate_function
        :param context:
        :return:
        """
        super(LinearMechanism, self).instantiate_function(context=context)

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=NotImplemented):
        """Execute LinearMechanism function (currently only trial-level, analytic solution)

        Execute LinearMechanism and unit activity vector
        Currently implements only trial-level LinearMechanism (analytic solution) and returns:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return current decision variable (self.outputState.value) and other output values (self.outputStates[].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + kwLinearMechanism_NetInput (float)
            + kwLinearMechanism_Slope (float)
            + kwLinearMechanism_Intercept (float)
            + kwLinearMechanism_Range (float)
            + kwLinearMechanism_NUnits (float)
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
        slope = float(self.parameterStates[kwLinearMechanism_Slope].value)
        intercept = float(self.parameterStates[kwLinearMechanism_Intercept].value)
        range = (self.parameterStates[kwLinearMechanism_Range].value)
        nunits = float(self.parameterStates[kwLinearMechanism_NUnits].value)
        #endregion

        #region EXECUTE CASCADED UPDATES (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for DDM")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Implement terminate() below
        #endregion

        #region EXECUTE FULL UPDATE (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Get length of output from kwOutputStates
            # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            #       is run (to evaluate output) before outputStates have been instantiated
        # FIX: USE LIST:
            output = [None] * len(self.paramsCurrent[kwOutputStates])
        # FIX: USE NP ARRAY
        #     output = np.array([[None]]*len(self.paramsCurrent[kwOutputStates]))

            activationVector = (net_input * slope + intercept)

            if range.size >= 2:
                maxCapIndices = np.where(activationVector > np.max(range))[0]
                minCapIndices = np.where(activationVector < np.min(range))[0]
                activationVector[maxCapIndices] = np.max(range);
                activationVector[minCapIndices] = np.min(range);

            self.outputStateValueMapping = {}
            self.outputStateValueMapping[kwLinearMechanism_Activation] = \
                LinearMechanism_Output.ACTIVATION.value
            self.outputStateValueMapping[kwLinearMechanism_Activation_Mean] = \
                LinearMechanism_Output.ACTIVATION_MEAN.value
            self.outputStateValueMapping[kwLinearMechanism_Activation_Variance] = \
                LinearMechanism_Output.ACTIVATION_VARIANCE.value

            output[LinearMechanism_Output.ACTIVATION.value] = activationVector;

            output[LinearMechanism_Output.ACTIVATION_MEAN.value] = \
                np.array(np.mean(output[LinearMechanism_Output.ACTIVATION.value]))
            output[LinearMechanism_Output.ACTIVATION_VARIANCE.value] = \
                np.array(np.var(output[LinearMechanism_Output.ACTIVATION.value]))



            #region Print results
            # if (self.prefs.reportOutputPref and kwFunctionInit not in context):
            import re
            if (self.prefs.reportOutputPref and kwExecuting in context):
                print ("\n{0} execute method:\n- input: {1}\n- params:".
                       format(self.name, self.inputState.value.__str__().strip("[]")))
                print ("    nunits:", str(nunits).__str__().strip("[]"),
                       "\n    net_input:", re.sub('[\[,\],\n]','',str(net_input)),
                       "\n    slope:", slope,
                       "\n    intercept:", intercept,
                       "\n    activation range:", re.sub('[\[,\],\n]','',str(range)),
                       "\n- output:",
                       "\n    mean activation: {0}".format(output[LinearMechanism_Output.ACTIVATION_MEAN.value]),
                       "\n    activation variance: {0}".format(output[LinearMechanism_Output.ACTIVATION_VARIANCE.value]))
                print ("Output: ", re.sub('[\[,\],\n]','',str(output[LinearMechanism_Output.ACTIVATION.value])))
            #endregion

            return output
        #endregion

        else:
            raise MechanismError("time_scale not specified for DDM")


    def terminate_function(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for DDM


