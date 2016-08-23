# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************   get_param_value_for_keyword ********************************************************
#
def get_param_value_for_keyword(owner, keyword):
    from PsyNeuLink.Functions.Utility import UtilityError
    try:
        return owner.paramsCurrent[kwExecuteMethod].keyword(keyword)
    except UtilityError as e:
        if owner.prefs.verbosePref:
            print ("{} of {}".format(e, owner.name))
        return None
    except AttributeError:
        if owner.prefs.verbosePref:
            print ("Keyword ({}) not recognized for {}".format(keyword, owner.name))
        return None

# ********************************************  Keywords ***************************************************************
#

# **********************************************************************************************************************
# ******************************************    CONSTANTS    ***********************************************************
# **********************************************************************************************************************

ON = True
OFF = False
DEFAULT = False
AUTO = True

#region --------------------------------------------    GENERAL    -----------------------------------------------------
# General

kwSeparator = ': '
kwSeparatorBar = ' | '
kwProgressBarChar = '.'
# kwValueSuffix = '_value'
kwInit = " INITIALIZING "  # Used as context for Log
kwInstantiate = " INSTANTIATING "  # Used as context for Log
kwExecuting = " EXECUTING " # Used in context for Log and ReportOutput pref
kwAssign = ': Assign' # Used in context for Log
kwAggregate = ': Aggregate' # Used in context for Log
kwReceiver = "Receiver"
kwValidate = 'Validate'
#endregion

#region -------------------------------------------    Preferences    --------------------------------------------------

kwPrefs = "Prefs"
kwPrefsOwner = "kwPrefsOwner"
kwPrefLevel = 'kwPrefLevel'
kwPrefCurrentValue = 'kwPrefCurrentValue'
kwPrefBaseValue = 'kwPrefBaseValue'
kwPreferenceSetName = 'kwPreferenceSetName'
kwDefaultPreferenceSetOwner = 'DefaultPreferenceSetOwner'
# kpLogPref = '_log_pref'
# kpParamValidationPref = '_param_validation_pref'
# kpVerbosePref = '_verbose_pref'
#endregion

#region --------------------------------------------    TIME SCALE    --------------------------------------------------

kwCentralClock = "CentralClock"
kwTimeScale = "Time Scale"
#endregion

#region --------------------------------------------    PREFERENCES    -------------------------------------------------

kwPreferenceSet = 'PreferenceSet'
kwFunctionPreferenceSet = 'PreferenceSet'
#endregion

#region ------------------------------------------------   LOG    ------------------------------------------------------

kwTime = 'Time'
kwContext = 'Context'
kwValue = 'Value'
#endregion

#region -----------------------------------------------    MAIN    ---------------------------------------------------

kpMechanismTimeScaleLogEntry = "Mechanism TimeScale"
kpMechanismInputLogEntry = "Mechanism Input"
kpMechanismOutputLogEntry = "Mechanism Output"
kpMechanismControlAllocationsLogEntry = "Mechanism Control Allocations"
#endregion

#region ----------------------------------------------    FUNCTION   ---------------------------------------------------

# General:
kwParamsArg = "params"
kwNameArg = "name"
kwPrefsArg = "prefs"
kwContextArg = "context"

kwFunctionInit = 'Function.__init__'
kwDeferredInit = 'Deferred Init'
kwFunction = 'function'
kwExecuteMethod = "function" # Param name for function, method, or type to instantiate and assign to self.execute
kwExecuteMethodParams  = "function_params" # Params used to instantiate, or to assign to kwExecuteMethod

kwParamClassDefaults = "paramClassDefaults"        # "Factory" default params for a Function
kwParamInstanceDefaults = "paramsInstanceDefaults" # Params used to instantiate a Function; supercede paramClassDefaults
kwParamsCurrent = "paramsCurrent"                  # Params currently in effect for an instance of a Function
                                                   #    in general, this includes params specifed as arg in a
                                                   #    to Function.execute;  however, there are some exceptions
                                                   #    in which those are kept separate from paramsCurrent (see DDM)

kwExecuteMethodCheckArgs = 'super.check_args' # Use for "context" arg
kwFunctionOutputTypeConversion = "FunctionOutputTypeConversion" # Used in Utility Functions to set output type

#endregion

#region ----------------------------------------    FUNCTION SUBCLASSES  -----------------------------------------------

# Function Categories   -----------------

kwProcessFunctionCategory = "Process_Base"
kwMechanismFunctionCategory = "Mechanism_Base"
kwStateFunctionCategory = "State_Base"
kwProjectionFunctionCategory = "Projection_Base"
kwUtilityFunctionCategory = "Utility_Base"

# Function TYPES  -----------------

# Mechanisms:
kwProcessingMechanism = "ProcessingMechanism"
kwMonitoringMechanism = "MonitoringMechanism"
kwControlMechanism = "ControlMechanism"

# States:
kwInputState = "InputState"
kwOutputState = "OutputState"
kwParameterState = "ParameterState"

# Projections:
kwMapping = "Mapping"
kwControlSignal = "ControlSignal"
kwLearningSignal = "LearningSignal"

# Utility:
kwExampleFunction = "EXAMPLE"
kwCombinationFunction = "COMBINATION"
kwTransferFuncton = "TRANSFER"
kwDistributionFunction = "DISTRIBUTION"
kwLearningFunction = 'LEARNING'


# Function SUBTYPES -----------------

# ControlMechanisms:
kwDefaultControlMechanism = "DefaultControlMechanism"
kwEVCMechanism = "EVCMechanism"

# MonitoringMechanisms:
kwComparatorMechanism = "ComparatorMechanism"

# ProcessingMechanisms:
kwDDM = "DDM"
kwLinearMechanism = "LinearMechanism"
kwSigmoidLayer = "SigmoidLayer"
kwAdaptiveIntegrator = "AdaptiveIntegrator"

# Utility:
kwContradiction = "Contradiction"
kwLinearCombination = "LinearCombination"
kwLinear = "Linear"
kwExponential = "Exponential"
kwLogistic = "Logistic"
kwIntegrator = "Integrator"
kwLinearMatrix = "Linear Matrix"
kwBackProp = 'Backpropagation Learning Algorithm'



#endregion

#region ----------------------------------------------    SYSTEM   ----------------------------------------------------

kwSystem = "System"
kwDefaultSystem = "DefaultSystem"
kwController = "Controller"
kwControllerPhaseSpec = 'ControllerPhaseSpec'

#endregion

#region ----------------------------------------------    PROCESS   ----------------------------------------------------

kwProcesses = "Processes"
kwProcess = "PROCESS"
kwConfiguration = "Configuration"
kwLearning = 'Learning'
kwProjections = "Projections"
kwProcessDefaultProjectionFunction = "Default Projection Function"
kwProcessExecute = "ProcessExecute"
kpMechanismExecutedLogEntry = "Mechanism Executed"
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

kwMechanism = "MECHANISM"
kwMechanismName = "MECHANISM NAME"
kwMechanismDefault = "DEFAULT MECHANISM"
kwDefaultProcessingMechanism = "DefaultProcessingMechanism"
kwDefaultMonitoringMechanism = "DefaultMonitoringMechanism"
kwProcessDefaultMechanism = "ProcessDefaultMechanism"
kwMechanismType = "Mechanism Type" # Used in mechanism dict specification (e.g., in process.configuration[])
kwMechanismDefaultInputValue = "Mechanism Default Input Value " # Used in mechanism specification dict
kwMechanismParamValue = "Mechanism Param Value"                 # Used to specify mechanism param value
kwMechanismDefaultParams = "Mechanism Default Params"           # Used in mechanism specification dict

ORIGIN = 'ORIGIN'
INTERNAL = 'INTERNAL'
TERMINAL = 'TERMINAL'

kwStateValue = "State value"   # Used in State specification dict
                                                 #  to specify State value
kwStateParams = "State params" # Used in State specification dict

# ParamClassDefaults:
kwMechanismTimeScale = "Mechanism Time Scale"
kwMechanismExecutionSequenceTemplate = "Mechanism Execution Sequence Template"

# Entries for output OrderedDict, describing the current state of the Mechanism
kwMechanismOutputValue = "MechanismOutputValue" # points to <mechanism>.outputStateValue
kwMechanismConfidence = "MechanismConfidence"   # contains confidence of current kwMechanismValue
kwMechanismPerformance = "MechanismPerformance" # contains value from objective function
kwMechanismDuration = "MechanismDuration"       # contains number of time steps since process.execute was called
kwMechanismParams = "MechanismParams"           # dict of ParameterState objects in <mechanism>.params

kwMechanismExecuteFunction = "MECHANISM EXECUTE FUNCTION"
kwMechanismAdjustFunction = "MECHANISM ADJUST FUNCTION"
kwMechanismInterrogateFunction = "MECHANISM INTERROGATE FUNCTION"
kwMechanismTerminateFunction = "MECHANISM TERMINATE FUNCTION"
# TBI: kwMechanismAccuracyFunction = "MECHANISM ACCURACY FUNCTION"
#endregion

#region ------------------------------------------    CONTROL MECHANISM   ----------------------------------------------

kwMakeDefaultController = "MakeDefaultController"
kwMonitoredOutputStates = "MonitoredOutputStates"
kwPredictionMechanism = "PredictionMechanism"
kwPredictionMechanismType = "PredictionMechanismType"
kwPredictionMechanismParams = "PredictionMechanismParams"
kwPredictionMechanismOutput = "PredictionMechanismOutput"
kwPredictionProcess = "PredictionProcess"
kwControlSignalProjections = 'ControlSignalProjections'
kwValueAggregationFunction = 'ValueAggregationFunction'
kwCostAggregationFunction = 'CostAggregationFunction'
kwCostApplicationFunction = 'CostApplicationFunction'
kwSystemDefaultController = "DefaultController"
kwSaveAllValuesAndPolicies = 'SaveAllPoliciesAndValues'
kwEVCSimulation = 'SIMULATING'

#endregion

#region -------------------------------------------    MECHANISM STATE  ------------------------------------------------

kwState = "State"
# These are use for dict specification of State
kwStateProjections = "StateProjections"  # Used to specify projection list to State
kwStateName = "StateName"
kwStatePrefs = "StatePrefs"
kwStateContext = "StateContext"

kwInputStates = 'InputStates'
kwInputStateParams = 'kwInputStateParams'
kwAddInputState = 'kwAddNewInputState'     # Used by Mechanism.add_projection_to()
kwAddOutputState = 'kwAddNewOutputState'   # Used by Mechanism.add_projection_from()
kwParameterStates = 'ParameterStates'
kwParameterStateParams = 'ParameterStateParams'
kwParamModulationOperation = 'ParameterModulationOperation'

kwOutputStates = 'OutputStates'
kwOutputStateParams = 'kwOutputStatesParams'
#endregion

#region ---------------------------------------------    PROJECTION  ---------------------------------------------------

# Attributes / KVO keypaths / Params
kwProjection = "Projection"
kwProjectionType = "ProjectionType"
kwProjectionParams = "ProjectionParams"
kwMappingParams = "MappingParams"
kwControlSignalParams = "ControlSignalParams"
kwLearningSignalParams = 'LearningSignalParams'
kwProjectionSender = 'ProjectionSender'
kwProjectionSenderValue =  "ProjectDefaultSenderValue"
kwProjectionReceiver = 'ProjectionReceiver'
# kpLog = "ProjectionLog"
kwMonitorForLearning = 'MonitorForLearning'


#endregion

#region ----------------------------------------------    UTILITY  -----------------------------------------------------

kwInitializer = 'initializer'
kwWeights = "weights"
kwExponents = "exponents"
kwOperation = "operation"
kwOffset = "offset"
kwScale = "scale"


kwMatrix = "matrix"
kwIdentityMatrix = "IdentityMatrix"
kwFullConnectivityMatrix = "FullConnectivityMatrix"
kwDefaultMatrix = kwIdentityMatrix

#endregion
