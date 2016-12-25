# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ********************************************  Keywords ***************************************************************

# **********************************************************************************************************************
# ******************************************    CONSTANTS    ***********************************************************
# **********************************************************************************************************************

class Keywords:
    """
    Attributes
    ----------

    ORIGIN
        A :doc:`ProcessingMechanism <ProcessingMechanism>` that is the first mechanism of a process and/or system,
        and that receives the input to the process or system when it is :ref:`executed or run <Run>`.  A process may
        have only one :keyword:`ORIGIN` mechanism, but a system may have many.  Note that the :keyword:`ORIGIN`
        mechanism of a process is not necessarily an :keyword:`ORIGIN` of the system to which it belongs, as it may
        receive projections from other processes in the system. The :keyword:`ORIGIN` mechanisms of a process or
        system are listed in its :keyword:`originMechanisms` attribute, and can be displayed using its :keyword:`show`
        method.  For additional details about :keyword:`ORIGIN` mechanisms in processes, see
        :ref:`Process Mechanisms <Process_Mechanisms>` and
        :ref:`Process Input and Output <Process_Input_And_Ouput>`; and for systems see
        :ref:`System Mechanisms <System_Mechanisms>` and
        :ref:`System Input and Initialization <System_Execution_Input_And_Initialization>`.

    INTERNAL
        A :doc:`ProcessingMechanism <ProcessingMechanism>` that is not designated as having any other status.

    CYCLE
        A :doc:`ProcessingMechanism <ProcessingMechanism>` that is *not* an :py:data:`ORIGIN <Keywords.ORIGIN>`
        mechanism, and receives a projection that closes a recurrent loop in a process and/or system.  If it is an
        :py:data:`ORIGIN <Keywords.ORIGIN>` mechanism, then it is designated as an
        :py:data:`INITIALIZE_CYCLE  <Keywords.INITIALIZE_CYCLE>` mechanism.

    INITIALIZE_CYCLE
        A :doc:`ProcessingMechanism <ProcessingMechanism>` that is the ``sender`` of a projection that closes a loop in
        a process or system, and that is not an :py:data:`ORIGIN <Keywords.ORIGIN>` mechanism.  An
        :ref:`initial value  <Run_InitialValues>` can be assigned to such mechanisms, that will be used to initialize
        the process or system when it is first run.  For additional information, see :ref:`Run <Run_Initial_Values>`,
        :ref:`System Mechanisms <System_Mechanisms>` and
        :ref:`System Input and Initialization <System_Execution_Input_And_Initialization>`.

    TERMINAL
        A :doc:`ProcessingMechanism <ProcessingMechanism>` that is the last mechanism of a process and/or system, and
        that provides the output to the process or system when it is :ref:`executed or run <Run>`.  A process may
        have only one :keyword:`TERMINAL` mechanism, but a system may have many.  Note that the :keyword:`TERMINAL`
        mechanism of a process is not necessarily a :keyword:`TERMINAL` mechanism of the system to which it belongs,
        as it may send projections to other processes in the system.  The :keyword:`TERMINAL` mechanisms of a process
        or system are listed in its :keyword:`terminalMechanisms` attribute, and can be displayed using its
        :keyword:`show` method.  For additional details about :keyword:`TERMINAL` mechanisms in processes, see
        :ref:`Process_Mechanisms` and :ref:`Process_Input_And_Ouput`; and for systems see :ref:`System_Mechanisms`.

    SINGLETON
        A :doc:`ProcessingMechanism` that is the only mechanism in a process and/or system.  It can serve the functions
        of an :py:data:`ORIGIN <Keywords.ORIGIN>` and/or a :py:data:`TERMINAL <Keywords.TERMINAL>` mechanism.

    MONITORING
        A :doc:`MonitoringMechanism <MonitoringMechanism>` configured for learning that is not a TARGET; that is, it
        is associated with an :py:data:`INTERNAL <Keywords.INTERNAL>` rather than a :py:data:`TERMINAL
        <Keywords.TERMINAL>` ProcessingMechanism in the process and/or system to which it belongs. For
        :py:class:`backpropagation <Function.BackPropagation>` learning, it is a :doc:WeightedErrorMechanism.
        See :ref:`MonitoringMechanisms <LearningProjection_MonitoringMechanism> for additional details.

    TARGET
        A :doc:`ComparatorMechanism <ComparatorMechanism>` of a process and/or system configured for learning that
        receives a target value from its ``execute`` or ``run`` method.  It must be associated with the
        :py:data:`TERMINAL <Keyword.TERMINAL>` mechanism of the process or system. The :keyword:`TARGET` mechanisms of
        a process or system are listed in its :keyword:`targetMechanisms` attribute, and can be displayed using its
        :keyword:`show` method.  For additional details, see :ref:`TARGET mechanisms <LearningProjection_Targets>`
        and specifying :ref:`target values <Run_Targets>`.


    """
    def __init__(self):
        self.ORIGIN = ORIGIN
        self.INTERNAL = INTERNAL
        self.CYCLE = CYCLE
        self.INITIALIZE_CYCLE = INITIALIZE_CYCLE
        self.TERMINAL = TERMINAL
        self.SINGLETON = SINGLETON
        self.MONITORING = MONITORING
        self.TARGET = TARGET

# parameter_keywords = set()

ON = True
OFF = False
DEFAULT = False
AUTO = True

# Used by initDirective
INIT_FULL_EXECUTE_METHOD = 'init using the full base class execute method'
INIT__EXECUTE__METHOD_ONLY = 'init using only the subclass __execute__ method'
INIT_FUNCTION_METHOD_ONLY = 'init using only the subclass __function__ method'


#region ---------------------------------------------    GENERAL    ----------------------------------------------------
# General

kwSeparator = ': '
SEPARATOR_BAR = ' | '
kwProgressBarChar = '.'
# kwValueSuffix = '_value'
NO_CONTEXT = "NO_CONTEXT"
INITIALIZING = " INITIALIZING "  # Used as context for Log
kwInstantiate = " INSTANTIATING "  # Used as context for Log
EXECUTING = " EXECUTING " # Used in context for Log and ReportOutput pref
kwAssign = ': Assign' # Used in context for Log
kwAggregate = ': Aggregate' # Used in context for Log
kwReceiver = "receiver"
kwValidate = 'Validate'
VALIDATE = kwValidate
COMMAND_LINE = "COMMAND_LINE"
kwParams = 'params'
CHANGED = 'CHANGED'
UNCHANGED = 'UNCHANGED'


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
TIME_SCALE = "time_scale"
#endregion

#region --------------------------------------------    PREFERENCES    -------------------------------------------------

kwPreferenceSet = 'PreferenceSet'
kwComponentPreferenceSet = 'PreferenceSet'
#endregion

#region ------------------------------------------------   LOG    ------------------------------------------------------

kwTime = 'Time'
kwContext = 'Context'
kwValue = 'Value'
#endregion

#region -----------------------------------------------  UTILITIES  ----------------------------------------------------

kpMechanismTimeScaleLogEntry = "Mechanism TimeScale"
kpMechanismInputLogEntry = "Mechanism Input"
kpMechanismOutputLogEntry = "Mechanism Output"
kpMechanismControlAllocationsLogEntry = "Mechanism Control Allocations"
#endregion

#region ----------------------------------------------   COMPONENT   ---------------------------------------------------

# General:
PREFS_ARG = "prefs"
VARIABLE = "variable"
NAME = "name"
PARAMS = "params"
CONTEXT = "context"

kwInitialValues = 'initial_values'

# inputs list/ndarray:
TRIALS_DIM = 0
TIME_STEPS_DIM = 1
PROCESSES_DIM = 2
INPUTS_DIM = 3

COMPONENT_INIT = 'Function.__init__'
DEFERRED_INITIALIZATION = 'Deferred Init'
kwDeferredDefaultName = 'DEFERRED_DEFAULT_NAME'
FUNCTION = "function" # Param name for function, method, or type to instantiate and assign to self.execute
FUNCTION_PARAMS  = "function_params" # Params used to instantiate or assign to a FUNCTION

kwParamClassDefaults = "paramClassDefaults"        # "Factory" default params for a Function
kwParamInstanceDefaults = "paramsInstanceDefaults" # Params used to instantiate a Function; supercede paramClassDefaults
PARAMS_CURRENT = "paramsCurrent"                  # Params currently in effect for an instance of a Function
                                                   #    in general, this includes params specifed as arg in a
                                                   #    to Function.execute;  however, there are some exceptions
                                                   #    in which those are kept separate from paramsCurrent (see DDM)

FUNCTION_CHECK_ARGS = 'super._check_args' # Use for "context" arg
kwFunctionOutputTypeConversion = "FunctionOutputTypeConversion" # Used in Function Components to set output type

#endregion

#region ----------------------------------------    COMPONENT SUBCLASSES  ----------------------------------------------

# Component Categories   -----------------

kwProcessComponentCategory = "Process_Base"
kwMechanismComponentCategory = "Mechanism_Base"
kwStateComponentCategory = "State_Base"
kwProjectionComponentCategory = "Projection_Base"
kwComponentCategory = "Function_Base"

# Component TYPES  -----------------

# Mechanisms:
kwProcessingMechanism = "ProcessingMechanism"
kwMonitoringMechanism = "MonitoringMechanism"
kwControlMechanism = "ControlMechanism"

# States:
kwInputState = "InputState"
OUTPUT_STATE = "OutputState"
kwParameterState = "ParameterState"

# Projections:
MAPPING_PROJECTION = "MappingProjection"
CONTROL_PROJECTION = "ControlProjection"
LEARNING_PROJECTION = "LearningProjection"

# Function:
kwExampleFunction = "EXAMPLE FUNCTION"
kwUserDefinedFunctionType = "USER DEFINED FUNCTION TYPE"
kwCombinationFunction = "COMBINATION FUNCTION"
kwIntegratorFunction = "INTEGRATOR FUNCTION"
kwTransferFunction = "TRANSFER FUNCTION"
kwDistributionFunction = "DISTRIBUTION FUNCTION"
LEARNING_FUNCTION = 'LEARNING FUNCTION'


# Component SUBTYPES -----------------

# ControlMechanisms:
kwDefaultControlMechanism = "DefaultControlMechanism"
kwEVCMechanism = "EVCMechanism"

# MonitoringMechanisms:
kwComparatorMechanism = "ComparatorMechanism"

# ProcessingMechanisms:
kwDDM = "DDM"
kwLinearMechanism = "LinearMechanism"
kwSigmoidLayer = "SigmoidLayer"
kwIntegratorMechanism = "IntegratorMechanism"

# Function:
kwContradiction = "Contradiction"
kwUserDefinedFunction = "USER DEFINED FUNCTION"
kwReduce = "Reduce"
kwLinearCombination = "LinearCombination"
kwLinear = "Linear"
kwExponential = "Exponential"
kwLogistic = "Logistic"
kwSoftMax = 'SoftMax'
kwIntegrator = "Integrator"
kwLinearMatrix = "Linear Matrix"
kwBackProp = 'Backpropagation Learning Algorithm'
kwRL = 'Reinforcement Learning Algorithm'


#endregion

#region ----------------------------------------------    SYSTEM   ----------------------------------------------------

SYSTEM = "System"
kwSystemInit = 'System.__init__'
kwDefaultSystem = "DefaultSystem"
kwController = "controller"
kwEnableController = "enable_controller"
kwControllerPhaseSpec = 'ControllerPhaseSpec'

RUN = 'Run'

#endregion

#region ----------------------------------------------    PROCESS   ----------------------------------------------------

PROCESS = "PROCESS"
kwProcesses = "processes"
kwProcessInit = 'Process.__init__'
PATHWAY = "pathway"
CLAMP_INPUT = "clamp_input"
SOFT_CLAMP = "soft_clamp"
HARD_CLAMP = "hard_clamp"
LEARNING = 'learning'
kwProjections = "projections"
kwProcessDefaultProjectionFunction = "Default Projection Function"
kwProcessExecute = "ProcessExecute"
kpMechanismExecutedLogEntry = "Mechanism Executed"
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

MECHANISM = 'MECHANISM'
kwMechanism = "MECHANISM"
kwMechanismName = "MECHANISM NAME"
kwMechanismDefault = "DEFAULT MECHANISM"
DEFAULT_PROCESSING_MECHANISM = "DefaultProcessingMechanism"
kwDefaultMonitoringMechanism = "DefaultMonitoringMechanism"
kwProcessDefaultMechanism = "ProcessDefaultMechanism"
kwMechanismType = "Mechanism Type" # Used in mechanism dict specification (e.g., in process.pathway[])
kwMechanismDefaultInputValue = "Mechanism Default Input Value " # Used in mechanism specification dict
kwMechanismParamValue = "Mechanism Param Value"                 # Used to specify mechanism param value
kwMechanismDefaultParams = "Mechanism Default Params"           # Used in mechanism specification dict

ORIGIN = 'ORIGIN'
INTERNAL = 'INTERNAL'
CYCLE = 'CYCLE'
INITIALIZE_CYCLE = 'INITIALIZE_CYCLE'
TERMINAL = 'TERMINAL'
SINGLETON = 'ORIGIN AND TERMINAL'
MONITORING = 'MONITORING'
TARGET = 'TARGET'

kwStateValue = "State value"   # Used in State specification dict
                                                 #  to specify State value
STATE_PARAMS = "State params" # Used in State specification dict

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

MAKE_DEFAULT_CONTROLLER = "make_default_controller"
MONITOR_FOR_CONTROL = "monitor_for_control"
PREDICTION_MECHANISM = "PredictionMechanism"
PREDICTION_MECHANISM_TYPE = "prediction_mechanism_type"
PREDICTION_MECHANISM_PARAMS = "prediction_mechanism_params"
PREDICTION_MECHANISM_OUTPUT = "PredictionMechanismOutput"
kwPredictionProcess = "PredictionProcess"
CONTROL_SIGNAL = 'control_signal'
CONTROL_PROJECTIONS = 'ControlProjections'
kwValueAggregationFunction = 'ValueAggregationFunction'
OUTCOME_AGGREGATION_FUNCTION = 'outcome_aggregation_function'
COST_AGGREGATION_FUNCTION = 'cost_aggregation_function'
COMBINE_OUTCOMES_AND_COSTS_FUNCTION = 'combine_outcomes_and_costs_function'
VALUE_FUNCTION = 'value_function'
SAVE_ALL_VALUES_AND_POLICIES = 'save_all_values_and_policies'
kwSystemDefaultController = "DefaultController"
EVC_SIMULATION = 'SIMULATING'
ALLOCATION_SAMPLES = "allocation_samples"

#endregion

#region ----------------------------------------------    STATES  ------------------------------------------------------

kwState = "State"
# These are use for dict specification of State
STATE_PROJECTIONS = "StateProjections"  # Used to specify projection list to State
kwStateName = "StateName"
kwStatePrefs = "StatePrefs"
kwStateContext = "StateContext"

INPUT_STATES = 'input_states'
INPUT_STATE_PARAMS = 'input_state_params'
kwAddInputState = 'kwAddNewInputState'     # Used by Mechanism._add_projection_to()
kwAddOutputState = 'kwAddNewOutputState'   # Used by Mechanism._add_projection_from()
PARAMETER_STATES = 'parameter_states'
PARAMETER_STATE_PARAMS = 'parameter_state_params'
PARAMETER_MODULATION_OPERATION = 'parameter_modulation_operation'
OUTPUT_STATES = 'output_states'
OUTPUT_STATE_PARAMS = 'output_states_params'
INDEX = 'index'
CALCULATE = 'calculate'
#endregion

#region ---------------------------------------------    PROJECTION  ---------------------------------------------------

# Attributes / KVO keypaths / Params
PROJECTION = "Projection"
PROJECTION_TYPE = "ProjectionType"
PROJECTION_PARAMS = "ProjectionParams"
MAPPING_PROJECTION_PARAMS = "MappingProjectionParams"
CONTROL_PROJECTION_PARAMS = "ControlProjectionParams"
LEARNING_PROJECTION_PARAMS = 'LearningProjectionParams'
PROJECTION_SENDER = 'projectionSender'
kwSenderArg = 'sender'
PROJECTION_SENDER_VALUE =  "projectionSenderValue"
kwProjectionReceiver = 'ProjectionReceiver'
kwReceiverArg = 'receiver'
# kpLog = "ProjectionLog"
MONITOR_FOR_LEARNING = 'monitor_for_learning'


#endregion

#region ----------------------------------------------    FUNCTION   ---------------------------------------------------


SUM = 'sum'
DIFFERENCE = 'difference'
PRODUCT = 'product'
QUOTIENT = 'quotient'
SUBTRACTION = 'subtraction'
DIVISION = 'division'
SCALAR = 'scalar'
VECTOR = 'vector'

GAIN = 'gain'
BIAS = 'bias'
SLOPE = 'slope'
INTERCEPT = 'intercept'
RATE = 'rate'
SCALE = 'scale'
NOISE = 'noise'

WEIGHTING = "weighting"

OUTPUT_TYPE = 'output'
ALL = 'all'
MAX_VAL = 'max_val'
MAX_INDICATOR = 'max_indicator'
PROB = 'prob'
MUTUAL_ENTROPY = 'mutual entropy'

kwInitializer = 'initializer'
WEIGHTS = "weights"
EXPONENTS = "exponents"
OPERATION = "operation"
OFFSET = "offset"
LINEAR = 'linear'
CONSTANT = 'constant'
SIMPLE = 'scaled'
ADAPTIVE = 'apaptive'


MATRIX = "matrix"
IDENTITY_MATRIX = "IdentityMatrix"
FULL_CONNECTIVITY_MATRIX = "FullConnectivityMatrix"
RANDOM_CONNECTIVITY_MATRIX = "RandomConnectivityMatrix"
AUTO_ASSIGN_MATRIX = 'AutoAssignMatrix'
# DEFAULT_MATRIX = AUTO_ASSIGN_MATRIX
DEFAULT_MATRIX = IDENTITY_MATRIX

#endregion
