# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***************************************************  EriksenFlanker *************************************************************
#

import numpy as np
# from numpy import sqrt, random, abs, tanh, exp
from numpy import sqrt, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *

# EriksenFlanker parameter keywords:
kwEriksenFlanker_Spotlight = "EriksenFlanker_Spotlight"
kwEriksenFlanker_MaxOutput = "EriksenFlanker_Max_Output"
kwEriksenFlanker_NetInput = "EriksenFlanker_Net_Input"

# EriksenFlanker outputs (used to create and name outputStates):
kwEriksenFlanker_Activation = "EriksenFlanker_Activation"

# EriksenFlanker default parameter values:
EriksenFlanker_DEFAULT_SPOTLIGHT= 1
EriksenFlanker_DEFAULT_MAX_OUTPUT = 1

# MULTI-UNIT VECTOR (ALL OF THE FOLLOWING ARE SYNONYMS AND WORK):
EriksenFlanker_DEFAULT_NET_INPUT = [1,1,1]              # <- WORKS!
# EriksenFlanker_DEFAULT_NET_INPUT = [[0,0]]            # <- WORKS!
# EriksenFlanker_DEFAULT_NET_INPUT = np.array([0, 0])   # <- WORKS!
# EriksenFlanker_DEFAULT_NET_INPUT = np.array([[0, 0]]) # <- WORKS!

# MULTI-STATE INPUT:
# EriksenFlanker_DEFAULT_NET_INPUT = [[0],[0],[1]] # <- GENERATES 3 OUTPUTS, BUT NOT SURE IF MATH IS CORRECT



class EriksenFlanker_Output(AutoNumber):
    ACTIVATION = ()

# QUESTION: What comes here?
# EriksenFlanker log entry keypaths:
# kpInput = 'DefaultInputState'
# kpDriftRate = kwEriksenFlanker_DriftRate + kwValueSuffix
# kpBias = kwEriksenFlanker_Bias + kwValueSuffix
# kpThreshold = kwEriksenFlanker_Threshold + kwValueSuffix
# kpDecisionVariable = kwEriksenFlanker_DecisionVariable + kwValueSuffix
# kpMeanReactionTime = kwEriksenFlanker_RT_Mean + kwValueSuffix
# kpMeanErrorRate = kwEriksenFlanker_Error_Rate + kwValueSuffix


class EriksenFlanker_Output(AutoNumber):
    ACTIVATION = ()

class EriksenFlankerError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EriksenFlanker(ProcessingMechanism_Base):
# DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
    """Implement EriksenFlanker subclass (Type) of Mechanism (Category of Function class)

    Description:
        Implements mechanism for EriksenFlanker decision process (for two alternative forced choice)
        Two analytic solutions are implemented (see Parameters below)

    Instantiation:
        - A EriksenFlanker mechanism can be instantiated in several ways:
            - directly, by calling EriksenFlanker()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
         DOCUMENT:

    Parameters:
        EriksenFlanker handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the EXECUTE_METHOD_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    NOTE:  params can be set in the standard way for any Function subclass:
        * params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
             and used for paramsCurrent unless and until the latter are changed in a function call
        * paramInstanceDefaults can be later modified using assign_defaults
        * params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of EriksenFlanker are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of EriksenFlanker can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "EriksenFlanker" with a hyphenated, indexed suffix ('EriksenFlanker-n')

    Class attributes:
        + functionType (str): EriksenFlanker
        + classPreference (PreferenceSet): EriksenFlanker_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
        + variableClassDefault (value):  EriksenFlanker_DEFAULT_BIAS
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      kwExecuteMethodParams:{kwEriksenFlanker_Unitst: kwEriksenFlanker_NetInput, kwControlSignal
                                                                 kwEriksenFlanker_Gain: EriksenFlanker_DEFAULT_GAIN, kwControlSignal
                                                                 kwEriksenFlanker_Bias: EriksenFlanker_DEFAULT_BIAS, kwControlSignal}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable - input to mechanism's execute method (default:  EriksenFlanker_DEFAULT_NET_INPUT)
        + executeMethodOutputDefault (value) - sample output of mechanism's execute method
        + executeMethodOutputType (type) - type of output of mechanism's execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, a default set is created by copying EriksenFlanker_PreferenceSet

    Instance methods:
        - execute(time_scale, params, context)
            called by <Mechanism>.update_states_and_execute(); runs the mechanism
            populates outputValue with various values (depending on version run)
            returns decision variable
        # - terminate(context) -
        #     terminates the process
        #     returns outputState.value
    """

    functionType = "EriksenFlanker"

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'EriksenFlankerCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    # classLogEntries = [kpInput,
    #                    kpDriftRate,
    #                    kpBias,
    #                    kpDecisionVariable,
    #                    kpMeanReactionTime,
    #                    kpMeanErrorRate]
    #

    variableClassDefault = EriksenFlanker_DEFAULT_NET_INPUT # Sets template for variable (input) to be compatible with EriksenFlanker_DEFAULT_NET_INPUT

    # EriksenFlanker parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwExecuteMethodParams:{
            kwEriksenFlanker_Spotlight: ParamValueProjection(EriksenFlanker_DEFAULT_SPOTLIGHT, kwControlSignal), # input to layer
            kwEriksenFlanker_MaxOutput: ParamValueProjection(EriksenFlanker_DEFAULT_MAX_OUTPUT, kwControlSignal), # input to layer
        },
        kwOutputStates:[kwEriksenFlanker_Activation,]
    })

    # Set default input_value to default bias for EriksenFlanker
    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
        """Assign type-level preferences, default input value (EriksenFlanker_DEFAULT_BIAS) and call super.__init__

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
            default_input_value = EriksenFlanker_DEFAULT_NET_INPUT

        super(EriksenFlanker, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  context=self)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED
        pass

    def instantiate_execute_method(self, context=NotImplemented):
        """Delete params not in use, call super.instantiate_execute_metho
        :param context:
        :return:
        """
        # QUESTION: Check here if input state fits projection

        super(EriksenFlanker, self).instantiate_execute_method(context=context)

    def execute(self,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=NotImplemented):
        """Execute EriksenFlanker function (currently only trial-level, analytic solution)

        Executes trial-level EriksenFlanker (analytic solution) which returns Activation, mean Activation across all units and Variance of Activation across all units

        Arguments:
        # IMPLEMENTATION NOTE:
        # variable is implemented, as execute method gets input from Mechanism.inputstate(s)
        # param args not currenlty in use
        # could be restored for potential local use
        # - variable (float): used as template for signal component of drift rate;
        #                     on execution, input is actually provided by self.inputState.value
        # - param (dict):  set of params defined in paramClassDefaults for the subclass
        #     + kwMechanismTimeScale: (default: TimeScale.TRIAL)
        #     + kwNetInput: (param=(0,0,NotImplemented), default: EriksenFlanker_DEFAULT_NET_INPUT)
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
        # drift_rate = (self.inputState.value * self.executeMethodParameterStates[kwEriksenFlanker_DriftRate].value)
        # drift_rate = (self.variable * self.executeMethodParameterStates[kwEriksenFlanker_DriftRate].value)
        # net_input = (self.variable * self.executeMethodParameterStates[kwEriksenFlanker_NetInput].value)
        net_input = (self.inputState.value)
        spotlight = (self.executeMethodParameterStates[kwEriksenFlanker_Spotlight].value)
        max_output = (self.executeMethodParameterStates[kwEriksenFlanker_MaxOutput].value)

        # cap spotlight
        if spotlight > 1:
            spotlight = 1;
        if spotlight < 0:
            spotlight = 0;



        #endregion

        #region EXECUTE INTEGRATOR FUNCTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for EriksenFlanker")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.executeMethodParameterStates[kwBias]
            # Implement terminate() below
        #endregion

        #region EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Get length of output from kwOutputStates
            # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            #       is run (to evaluate output) before outputStates have been instantiated
            # QUESTION: What is this doing?
            output = [None] * len(self.paramsCurrent[kwOutputStates])


# IMPLEMENTATION VARIANTS **********************************************************************************************

            #region calculate unit activations:
            # IMPLEMENTATION NOTE: OUTPUTS HANDLED AS SIMPLE VARIABLES:  ----------------------------
            # output[EriksenFlanker_Output.ACTIVATION.value] = \
            #     1/(1+np.exp(gain*(net_input-bias))) * (np.max(range)-np.min(range)) + np.min(range)
            # output[EriksenFlanker_Output.ACTIVATION_MEAN.value] = \
            #     np.mean(output[EriksenFlanker_Output.ACTIVATION.value])
            # output[EriksenFlanker_Output.ACTIVATION_VARIANCE.value] = \
            #     np.var(output[EriksenFlanker_Output.ACTIVATION.value])
            # IMPLEMENTATION NOTE: OUTPUTS HANDLED AS SIMPLE VARIABLES:  ----------------------------
            output[EriksenFlanker_Output.ACTIVATION.value] = max_output * (spotlight * net_input[1] - (1-spotlight) * (net_input[0] + net_input[2]))
            print (spotlight)
            # IMPLEMENTATION NOTE END VARIANTS
            #endregion

            #region Print results
            import re
            if (self.prefs.reportOutputPref and kwFunctionInit not in context):
                print ("\n{0} execute method:\n- input: {1}\n- params:".
                       format(self.name, self.inputState.value.__str__().strip("[]")))
                print ("    spotlight:", str(spotlight).__str__().strip("[]"),
                       "\n    net_input:", re.sub('[\[,\],\n]','',str(net_input)))
                print ("Output: ", re.sub('[\[,\],\n]','',str(output[EriksenFlanker_Output.ACTIVATION.value])))
            #endregion

            # print ("Output: ", output[EriksenFlanker_Output.ACTIVATION.value].__str__().strip("[]"))
            output = np.array(output)
            return output
        #endregion

        else:
            raise MechanismError("time_scale not specified for EriksenFlanker")



    def terminate_function(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for EriksenFlanker




