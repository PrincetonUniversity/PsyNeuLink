# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
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
kwExecuteMethod = "kwExecuteMethod" # Param name for function, method, or type to instantiate and assign to self.execute
kwExecuteMethodParams  = "kwExecuteMethodParams" # Params used to instantiate, or to assign to kwExecuteMethod

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
kwMechanismStateFunctionCategory = "MechanismState_Base"
kwProjectionFunctionCategory = "Projection_Base"
kwUtilityFunctionCategory = "Utility_Base"

# Function TYPES  -----------------

# Mechanisms:
kwProcessingMechanism = "ProcessingMechanism"
kwMonitoringMechanism = "MonitoringMechanism"
kwSystemControlMechanism = "SystemControlMechanism"

# MechanismStates:
kwMechanismInputState = "MechanismInputState"
kwMechanismOutputState = "MechanismOutputState"
kwMechanismParameterState = "MechanismParameterState"

# Projections:
kwMapping = "Mapping"
kwControlSignal = "ControlSignal"
# TBI: kwLearning = "Learning"

# Utility:
kwExampleFunction = "EXAMPLE"
kwCombinationFunction = "COMBINATION"
kwTransferFuncton = "TRANSFER"
kwDistributionFunction = "DISTRIBUTION"

# Function SUBTYPES -----------------

# ControlMechanisms:
kwDefaultControlMechanism = "DefaultControlMechanism"
kwEVCMechanism = "EVCMechanism"

# MonitoringMechanisms:
kwLinearComparatorMechanism = "LinearComparatorMechanism"

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
kwProcessDefaultProjection = "Default Projection"
kwProcessDefaultProjectionFunction = "Default Projection Function"
kwProcessExecute = "ProcessExecute"
kpMechanismExecutedLogEntry = "Mechanism Executed"
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

kwMechanism = "MECHANISM"
kwMechanismName = "MECHANISM NAME"
kwMechanismDefault = "DEFAULT MECHANISM"
kwDefaultProcessingMechanism = "DefaultProcessingMechanism"
kwProcessDefaultMechanism = "ProcessDefaultMechanism"
kwMechanismType = "Mechanism Type" # Used in mechanism dict specification (e.g., in process.configuration[])
kwMechanismDefaultInputValue = "Mechanism Default Input Value " # Used in mechanism specification dict
kwMechanismParamValue = "Mechanism Param Value"                 # Used to specify mechanism param value
kwMechanismDefaultParams = "Mechanism Default Params"           # Used in mechanism specification dict

kwMechanismStateValue = "MechanismState value"   # Used in MechanismState specification dict
                                                 #  to specify MechanismState value
kwMechanismStateParams = "MechanismState params" # Used in MechanismState specification dict

# ParamClassDefaults:
kwMechanismTimeScale = "Mechanism Time Scale"
kwMechanismExecutionSequenceTemplate = "Mechanism Execution Sequence Template"

# Entries for output OrderedDict, describing the current state of the Mechanism
kwMechanismOutputValue = "MechanismOutputValue" # points to <mechanism>.outputStateValue
kwMechanismConfidence = "MechanismConfidence"   # contains confidence of current kwMechanismValue
kwMechanismPerformance = "MechanismPerformance" # contains value from objective function
kwMechanismDuration = "MechanismDuration"       # contains number of time steps since process.execute was called
kwMechanismParams = "MechanismParams"           # dict of MechanismParameterState objects in <mechanism>.params

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

kwMechanismState = "MechanismState"
# These are use for dict specification of MechanismState
kwMechanismStateProjections = "MechanismStateProjections"  # Used to specify projection list to MechanismState
kwMechanismStateName = "MechanismStateName"
kwMechanismStatePrefs = "MechanismStatePrefs"
kwMechanismStateContext = "MechanismStateContext"

kwMechanismInputStates = 'MechanismInputStates'
kwMechanismInputStateParams = 'kwMechanismInputStateParams'
kwAddMechanismInputState = 'kwAddNewMechanismInputState'   # Used by Mechanism.add_projection

kwMechanismParameterStates = 'MechanismParameterStates'
kwMechanismParameterStateParams = 'MechanismParameterStateParams'
kwParamModulationOperation = 'MechanismParamValueparamModulationOperation'

kwMechanismOutputStates = 'MechanismOutputStates'
kwMechanismOutputStateParams = 'kwMechanismOutputStatesParams'
#endregion

#region ---------------------------------------------    PROJECTION  ---------------------------------------------------

# Attributes / KVO keypaths / Params
kwProjection = "Projection"
kwProjectionType = "ProjectionType"
kwProjectionParams = "ProjectionParams"
kwMappingParams = "MappingParams"
kwControlSignalParams = "ControlSignalParams"
kwProjectionSender = 'ProjectionSender'
kwProjectionSenderValue =  "ProjectDefaultSenderValue"
kwProjectionReceiver = 'ProjectionReceiver'
# kpLog = "ProjectionLog"

#endregion

#region ----------------------------------------------    UTILITY  -----------------------------------------------------

kwInitializer = 'INITIALIZER'
kwWeights = "WEIGHTS"
kwExponents = "EXPONENTS"
kwOperation = "OPERATION"
kwOffset = "ADDITIVE CONSTANT"
kwScale = "MULTIPLICATIVE SCALE"


kwMatrix = "IdentityMatrix"
kwIdentityMatrix = "IdentityMatrix"
kwFullConnectivityMatrix = "FullConnectivityMatrix"
kwDefaultMatrix = kwIdentityMatrix


#endregion