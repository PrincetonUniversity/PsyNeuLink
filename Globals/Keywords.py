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

#region ----------------------------------------------    SYSTEM   ----------------------------------------------------

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

#region ----------------------------------------------    FUNCTION   ---------------------------------------------------

# General:
kwParamsArg = "params"
kwNameArg = "name"
kwPrefsArg = "prefs"
kwContextArg = "context"

kwFunctionInit = 'Function.__init__'
kwExecuteMethod = "kwExecuteMethod" # Param name for function to assign as class.function
kwParamsCurrent = "paramsCurrent"

kwParamInstanceDefaults = "paramsInstanceDefaults"
kwParamClassDefaults = "paramClassDefaults"

kwExecuteMethodVariable = 'kwExecuteMethodVariable'
kwExecuteMethodParams  = "kwExecuteMethodParams"
kwExecuteMethodInit = 'Function init'
kwExecuteMethodCheckArgs = 'super.check_args'

# Function Categories:
kwProcessFunctionCategory = "Process"
kwMechanismFunctionCategory = "Mechanism_Base"
kwMechanismStateFunctionCategory = "MechanismState_Base"
kwProjectionFunctionCategory = "Projection_Base"
kwLearningFunctionCategory = "Learning"
kwUtilityFunctionCategory = "Utility_Base"

# SUBCLASSES:
kwMechanismInputState = "MechanismInputState"
kwMechanismOutputState = "MechanismOutputState"
kwMechanismParameterState = "MechanismParameterState"
kwMapping = "Mapping"
kwControlSignal = "ControlSignal"

# Utility Function Types:
kwExampleFunction = "EXAMPLE"
kwCombinationFunction = "COMBINATION"
kwTransferFuncton = "TRANSFER"
kwDistributionFunction = "DISTRIBUTION"

# Utility Function Names:
kwContradiction = "CONTRADICTION"
kwLinearCombination = "LINEAR_COMBINATION"
kwLinear = "LINEAR"
kwExponential = "EXPONENTIAL"
kwIntegrator = "INTEGRATOR"
kwLinearMatrix = "LINEAR MATRIX"
kwDDM = "DDM"
kwPDP = "PDP"

kwFunctionOutputTypeConversion = "FunctionOutputTypeConversion" # Used in Utility Functions to set output type
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

kwMechanism = "MECHANISM"
kwMechanismName = "MECHANISM NAME"
kwMechanismDefault = "DEFAULT MECHANISM"
kwSystemDefaultMechanism = "SystemDefaultMechanism"
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

kwSystem = "System"
kwMakeDefaultController = "MakeDefaultController"
kwMonitoredStates = "MonitoredStates"
kwPredictionMechanism = "PredictionMechanism"
kwPredictionProcess = "PredictionProcess"
kwControlSignalProjections = 'ControlSignalProjections'
kwValueAggregationFunction = 'ValueAggregationFunction'
kwCostAggregationFunction = 'CostAggregationFunction'
kwCostApplicationFunction = 'CostApplicationFunction'
kwSystemDefaultController = "SystemDefaultController"
kwEVCMechanism = 'EVCMechanism'
kwSaveAllPoliciesAndValues = 'SaveAllPoliciesAndValues'
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
kwOperation = "OPERATION"
kwOffset = "ADDITIVE CONSTANT"
kwScale = "MULTIPLICATIVE SCALE"


kwMatrix = "IdentityMatrix"
kwIdentityMatrix = "IdentityMatrix"
kwFullConnectivityMatrix = "FullConnectivityMatrix"
kwDefaultMatrix = kwIdentityMatrix


#endregion