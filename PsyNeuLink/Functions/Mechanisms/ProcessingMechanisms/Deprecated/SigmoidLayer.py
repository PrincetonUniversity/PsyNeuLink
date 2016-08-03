# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***************************************************  SigmoidLayer *************************************************************
#

import numpy as np
# from numpy import sqrt, random, abs, tanh, exp
from numpy import sqrt, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *

# SigmoidLayer parameter keywords:
kwSigmoidLayer_NUnits = "SigmoidLayer_Number_Of_Units"
kwSigmoidLayer_Gain = "SigmoidLayer_Gain"
kwSigmoidLayer_Bias = "SigmoidLayer_Bias"
kwSigmoidLayer_Range = "SigmoidLayer_Range"
kwSigmoidLayer_NetInput = "SigmoidLayer_Net_Input"

# SigmoidLayer outputs (used to create and name outputStates):
kwSigmoidLayer_Activation = "SigmoidLayer_Activation"
kwSigmoidLayer_Activation_Mean = "SigmoidLayer_Activation_Mean "
kwSigmoidLayer_Activation_Variance = "kwSigmoidLayer_Activation_Variance"

# SigmoidLayer default parameter values:
SigmoidLayer_DEFAULT_NUNITS= 1
SigmoidLayer_DEFAULT_GAIN = 1
SigmoidLayer_DEFAULT_BIAS = 0

SigmoidLayer_DEFAULT_RANGE = np.array([0,1])
# SigmoidLayer_DEFAULT_RANGE = np.array([[0]])
# SigmoidLayer_DEFAULT_RANGE = np.array([0])

# SINGLE UNIT INPUT VECTOR:
# I BELIEVE ALL OF THESE WORK AS WELL
# SigmoidLayer_DEFAULT_NET_INPUT = 0                # <- WORKS
SigmoidLayer_DEFAULT_NET_INPUT = [0]              # <- WORKS
# SigmoidLayer_DEFAULT_NET_INPUT = [[0]]            # <- WORKS

# MULTI-UNIT VECTOR (ALL OF THE FOLLOWING ARE SYNONYMS AND WORK):
# SigmoidLayer_DEFAULT_NET_INPUT = [0,0]              # <- WORKS!
# SigmoidLayer_DEFAULT_NET_INPUT = [[0,0]]            # <- WORKS!
# SigmoidLayer_DEFAULT_NET_INPUT = np.array([0, 0])   # <- WORKS!
# SigmoidLayer_DEFAULT_NET_INPUT = np.array([[0, 0]]) # <- WORKS!

# MULTI-STATE INPUT:
# SigmoidLayer_DEFAULT_NET_INPUT = [[0],[0],[1]] # <- GENERATES 3 OUTPUTS, BUT NOT SURE IF MATH IS CORRECT


class SigmoidLayer_Output(AutoNumber):
    ACTIVATION = ()
    ACTIVATION_MEAN = ()
    ACTIVATION_VARIANCE = ()

class SigmoidLayerError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class SigmoidLayer(ProcessingMechanism_Base):
# DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
    """Implement SigmoidLayer subclass (Type) of Mechanism (Category of Function class)

    Description:
        Implements mechanism for SigmoidLayer decision process (for two alternative forced choice)
        Two analytic solutions are implemented (see Parameters below)

    Instantiation:
        - A SigmoidLayer mechanism can be instantiated in several ways:
            - directly, by calling SigmoidLayer()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
         DOCUMENT:

    Parameters:
        SigmoidLayer handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the EXECUTE_METHOD_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    NOTE:  params can be set in the standard way for any Function subclass:
        * params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
             and used for paramsCurrent unless and until the latter are changed in a function call
        * paramInstanceDefaults can be later modified using assign_defaults
        * params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of SigmoidLayer are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of SigmoidLayer can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "SigmoidLayer" with a hyphenated, indexed suffix ('SigmoidLayer-n')

    Class attributes:
        + functionType (str): SigmoidLayer
        + classPreference (PreferenceSet): SigmoidLayer_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
        + variableClassDefault (value):  SigmoidLayer_DEFAULT_BIAS
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      kwExecuteMethodParams:{kwSigmoidLayer_Unitst: kwSigmoidLayer_NetInput, kwControlSignal
                                                                 kwSigmoidLayer_Gain: SigmoidLayer_DEFAULT_GAIN, kwControlSignal
                                                                 kwSigmoidLayer_Bias: SigmoidLayer_DEFAULT_BIAS, kwControlSignal}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value) - input to Mechanism's execute method (default:  SigmoidLayer_DEFAULT_NET_INPUT)
        + value (value) - output of Mechanism's execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, a default set is created by copying SigmoidLayer_PreferenceSet

    Instance methods:
        - execute(time_scale, params, context)
            called by <Mechanism>.update_states_and_execute(); runs the mechanism
            populates outputValue with various values (depending on version run)
            returns decision variable
        # - terminate(context) -
        #     terminates the process
        #     returns outputState.value
    """

    functionType = "SigmoidLayer"

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'SigmoidLayerCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    variableClassDefault = SigmoidLayer_DEFAULT_NET_INPUT # Sets template for variable (input) to be compatible with SigmoidLayer_DEFAULT_NET_INPUT

    # SigmoidLayer parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwExecuteMethodParams:{
            # kwSigmoidLayer_NetInput: ParamValueProjection(SigmoidLayer_DEFAULT_NET_INPUT, kwControlSignal), # input to layer
            # kwSigmoidLayer_Gain: ParamValueProjection(SigmoidLayer_DEFAULT_GAIN, kwControlSignal),            # used as gain of activation function
            # kwSigmoidLayer_Bias: ParamValueProjection(SigmoidLayer_DEFAULT_BIAS, kwControlSignal),  # bias component
            kwSigmoidLayer_NetInput: SigmoidLayer_DEFAULT_NET_INPUT, # input to layer
            kwSigmoidLayer_Gain: SigmoidLayer_DEFAULT_GAIN,            # used as gain of activation function
            kwSigmoidLayer_Bias: SigmoidLayer_DEFAULT_BIAS,  # bias component
            kwSigmoidLayer_NUnits: SigmoidLayer_DEFAULT_NUNITS,
            kwSigmoidLayer_Range: SigmoidLayer_DEFAULT_RANGE,
        },
        kwOutputStates:[kwSigmoidLayer_Activation,
                                 kwSigmoidLayer_Activation_Mean,
                                 kwSigmoidLayer_Activation_Variance]
    })

    # Set default input_value to default bias for SigmoidLayer
    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__

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
            default_input_value = SigmoidLayer_DEFAULT_NET_INPUT

        super(SigmoidLayer, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  context=self)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED

    def instantiate_execute_method(self, context=NotImplemented):
        """Delete params not in use, call super.instantiate_execute_metho
        :param context:
        :return:
        """
        # QUESTION: Check here if input state fits projection

        super(SigmoidLayer, self).instantiate_execute_method(context=context)

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=NotImplemented):
        """Execute SigmoidLayer function (currently only trial-level, analytic solution)

        Executes trial-level SigmoidLayer (analytic solution) which returns Activation, mean Activation across all units and Variance of Activation across all units

        Arguments:
        # IMPLEMENTATION NOTE:
        # variable is implemented, as execute method gets input from Mechanism.inputstate(s)
        # param args not currenlty in use
        # could be restored for potential local use
        # - variable (float): used as template for signal component of drift rate;
        #                     on execution, input is actually provided by self.inputState.value
        # - param (dict):  set of params defined in paramClassDefaults for the subclass
        #     + kwMechanismTimeScale: (default: TimeScale.TRIAL)
        #     + kwNetInput: (param=(0,0,NotImplemented), default: SigmoidLayer_DEFAULT_NET_INPUT)
        #     + kwGain: (param=(0,0,NotImplemented), control_signal=Control.DEFAULT)
        #     + kwBias: (param=(0,0,NotImplemented), control_signal=Control.DEFAULT)
        #     + kwNUnits: # QUESTION: how to write array?
        #     + kwRange:  # QUESTION: how to write array?
        - context (str): optional

        Returns output list with the following items, each of which is also placed in its own outputState:
        - activation
        - mean activation across units
        - activation variance across units

        :param self:
        :param time_scale: (TimeScale)
        # :param variable (float)
        # :param params: (dict)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        #region ASSIGN PARAMETER VALUES
        # - convolve inputState.value (signal) w/ driftRate param value (automatic contribution to the process)
        # - assign convenience names to each param
        # drift_rate = (self.inputState.value * self.executeMethodParameterStates[kwSigmoidLayer_DriftRate].value)
        # drift_rate = (self.variable * self.executeMethodParameterStates[kwSigmoidLayer_DriftRate].value)
        # net_input = (self.variable * self.executeMethodParameterStates[kwSigmoidLayer_NetInput].value)
        net_input = (self.inputState.value * self.executeMethodParameterStates[kwSigmoidLayer_NetInput].value)
        gain = float(self.executeMethodParameterStates[kwSigmoidLayer_Gain].value)
        bias = float(self.executeMethodParameterStates[kwSigmoidLayer_Bias].value)
        range = (self.executeMethodParameterStates[kwSigmoidLayer_Range].value)
        nunits = self.executeMethodParameterStates[kwSigmoidLayer_NUnits].value

        #endregion

        #region EXECUTE INTEGRATOR FUNCTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for SigmoidLayer")
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
            # QUESTION: What is this doing?
            output = [None] * len(self.paramsCurrent[kwOutputStates])


# IMPLEMENTATION VARIANTS **********************************************************************************************

            self.outputStateValueMapping = {}
            self.outputStateValueMapping[kwSigmoidLayer_Activation] = \
                SigmoidLayer_Output.ACTIVATION.value
            self.outputStateValueMapping[kwSigmoidLayer_Activation_Mean] = \
                SigmoidLayer_Output.ACTIVATION_MEAN.value
            self.outputStateValueMapping[kwSigmoidLayer_Activation_Variance] = \
                SigmoidLayer_Output.ACTIVATION_VARIANCE.value

            #region calculate unit activations:
            # IMPLEMENTATION NOTE: OUTPUTS HANDLED AS SIMPLE VARIABLES:  ----------------------------
            # output[SigmoidLayer_Output.ACTIVATION.value] = \
            #     1/(1+np.exp(gain*(net_input-bias))) * (np.max(range)-np.min(range)) + np.min(range)
            # output[SigmoidLayer_Output.ACTIVATION_MEAN.value] = \
            #     np.mean(output[SigmoidLayer_Output.ACTIVATION.value])
            # output[SigmoidLayer_Output.ACTIVATION_VARIANCE.value] = \
            #     np.var(output[SigmoidLayer_Output.ACTIVATION.value])
            # IMPLEMENTATION NOTE: OUTPUTS HANDLED AS SIMPLE VARIABLES:  ----------------------------
            output[SigmoidLayer_Output.ACTIVATION.value] = \
                1/(1+np.exp(gain*(net_input-bias))) * (np.max(range)-np.min(range)) + np.min(range)
            output[SigmoidLayer_Output.ACTIVATION_MEAN.value] = \
                np.array(np.mean(output[SigmoidLayer_Output.ACTIVATION.value]))
            output[SigmoidLayer_Output.ACTIVATION_VARIANCE.value] = \
                np.array(np.var(output[SigmoidLayer_Output.ACTIVATION.value]))
            # IMPLEMENTATION NOTE END VARIANTS
            #endregion

            #region Print results
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
                       "\n    mean activation: {0}".format(output[SigmoidLayer_Output.ACTIVATION_MEAN.value]),
                       "\n    activation variance: {0}".format(output[SigmoidLayer_Output.ACTIVATION_VARIANCE.value]))
                print ("Output: ", re.sub('[\[,\],\n]','',str(output[SigmoidLayer_Output.ACTIVATION.value])))
            #endregion

            # print ("Output: ", output[SigmoidLayer_Output.ACTIVATION.value].__str__().strip("[]"))
            output = np.array(output)
            return output
        #endregion

        else:
            raise MechanismError("time_scale not specified for SigmoidLayer")



    def terminate_function(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for SigmoidLayer




