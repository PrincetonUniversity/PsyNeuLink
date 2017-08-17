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
# ******************************************    CLASSES    *************************************************************
# **********************************************************************************************************************

# IMPLEMENTATION NOTE:
#  These classes are used for documentation purposes only.
#  The attributes of each are assigned to constants (listed in the next section of this module)
#    that are the ones actually used by the code.

class Keywords:
    """
    Attributes
    ----------

    ORIGIN
        A `ProcessingMechanism <ProcessingMechanism>` that is the first Mechanism of a `Process` and/or `System`,
        and that receives the input to the Process or System when it is :ref:`executed or run <Run>`.  A Process may
        have only one `ORIGIN` Mechanism, but a System may have many.  Note that the `ORIGIN`
        Mechanism of a Process is not necessarily an `ORIGIN` of the System to which it belongs, as it may receive
        `Projections <Projection>` from other Processes in the System. The `ORIGIN` Mechanisms of a Process or
        System are listed in its :keyword:`origin_mechanisms` attribute, and can be displayed using its :keyword:`show`
        method.  For additional details about `ORIGIN` Mechanisms in Processes, see
        `Process Mechanisms <Process_Mechanisms>` and `Process Input and Output <Process_Input_And_Output>`;
        and for Systems see `System Mechanisms <System_Mechanisms>` and
        `System Input and Initialization <System_Execution_Input_And_Initialization>`.

    INTERNAL
        A `ProcessingMechanism <ProcessingMechanism>` that is not designated as having any other status.

    CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is *not* an `ORIGIN` Mechanism, and receives a `Projection`
        that closes a recurrent loop in a `Process` and/or `System`.  If it is an `ORIGIN` Mechanism, then it is simply
        designated as such (since it will be assigned input and therefore be initialized in any event).

    INITIALIZE_CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is the `sender <Projection.Projection.sender>` of a
        `Projection` that closes a loop in a `Process` or `System`, and that is not an `ORIGIN` Mechanism (since in
        that case it will be initialized in any event). An `initial value  <Run_InitialValues>` can be assigned to such
        Mechanisms, that will be used to initialize the Process or System when it is first run.  For additional
        information, see `Run <Run_Initial_Values>`, `System Mechanisms <System_Mechanisms>` and
        `System Input and Initialization <System_Execution_Input_And_Initialization>`.

    TERMINAL
        A `ProcessingMechanism <ProcessingMechanism>` that is the last Mechanism of a `Process` and/or `System`, and
        that provides the output to the Process or System when it is `executed or run <Run>`.  A Process may
        have only one `TERMINAL` Mechanism, but a System may have many.  Note that the `TERMINAL`
        Mechanism of a process is not necessarily a `TERMINAL` Mechanism of the System to which it belongs,
        as it may send projections to other processes in the System (see `example
        <LearningProjection_Target_vs_Terminal_Figure>`).  The `TERMINAL` Mechanisms of a Process or System are listed in
        its :keyword:`terminalMechanisms` attribute, and can be displayed using its :keyword:`show` method.  For
        additional details about `TERMINAL` Mechanisms in Processes, see `Process_Mechanisms` and
        `Process_Input_And_Output`; and for Systems see `System_Mechanisms`.

    SINGLETON
        A `ProcessingMechanism` that is the only Mechanism in a `Process` and/or `System`.  It can serve the
        functions of an `ORIGIN` and/or a `TERMINAL` Mechanism.

    LEARNING
        A `LearningMechanism <LearningMechanism>` in a `Process` and/or `System`.

    TARGET
        A `ComparatorMechanism` of a `Process` and/or `System` configured for learning that receives a target value
        from its `execute <ComparatorMechanism.ComparatorMechanism.execute>` or
        `run <ComparatorMechanism.ComparatorMechanism.execute>` method.  It must be associated with the `TERMINAL`
        Mechanism of the Process or System. The `TARGET` Mechanisms of a Process or System are listed in its
        :keyword:`target_mechanisms` attribute, and can be displayed using its :keyword:`show` method.  For additional
        details, see `TARGET Mechanisms <LearningMechanism_Targets>` and specifying `target values <Run_Targets>`.


    """
    def __init__(self):
        self.ORIGIN = ORIGIN
        self.INTERNAL = INTERNAL
        self.CYCLE = CYCLE
        self.INITIALIZE_CYCLE = INITIALIZE_CYCLE
        self.TERMINAL = TERMINAL
        self.SINGLETON = SINGLETON
        self.LEARNING = LEARNING
        self.TARGET = TARGET


class MatrixKeywords:
    """
    Attributes
    ----------

    IDENTITY_MATRIX
        a square matrix of 1's along the diagnoal, 0's elsewhere; this requires that the length of the sender and 
        receiver values are the same.

    HOLLOW_MATRIX
        a square matrix of 0's along the diagnoal, 1's elsewhere; this requires that the length of the sender and 
        receiver values are the same.

    FULL_CONNECTIVITY_MATRIX
        a matrix that has a number of rows equal to the length of the sender's value, and a number of columns equal
        to the length of the receiver's value, all the elements of which are 1's.

    RANDOM_CONNECTIVITY_MATRIX
        a matrix that has a number of rows equal to the length of the sender's value, and a number of columns equal
        to the length of the receiver's value, all the elements of which are filled with random values uniformly
        distributed between 0 and 1.

    AUTO_ASSIGN_MATRIX
        if the sender and receiver are of equal length, an `IDENTITY_MATRIX` is assigned;  otherwise, a
        `FULL_CONNECTIVITY_MATRIX` is assigned.

    DEFAULT_MATRIX
        used if no matrix specification is provided in the constructor;  it presently assigns an `IDENTITY_MATRIX`.

    """
    def __init__(self):
        # self.MATRIX = MATRIX
        self.IDENTITY_MATRIX = IDENTITY_MATRIX
        self.HOLLOW_MATRIX = HOLLOW_MATRIX
        self.FULL_CONNECTIVITY_MATRIX = FULL_CONNECTIVITY_MATRIX
        self.RANDOM_CONNECTIVITY_MATRIX = RANDOM_CONNECTIVITY_MATRIX
        self.AUTO_ASSIGN_MATRIX = AUTO_ASSIGN_MATRIX
        self.DEFAULT_MATRIX = DEFAULT_MATRIX

    def _values(self):
        return list(self.__dict__.values())

    def _set(self):
        return set(self.__dict__.values())

    def _names(self):
        return list(self.__dict__)

MATRIX = "matrix"
IDENTITY_MATRIX = "IdentityMatrix"
HOLLOW_MATRIX = "HollowMatrix"
FULL_CONNECTIVITY_MATRIX = "FullConnectivityMatrix"
RANDOM_CONNECTIVITY_MATRIX = "RandomConnectivityMatrix"
AUTO_ASSIGN_MATRIX = 'AutoAssignMatrix'
DEFAULT_MATRIX = AUTO_ASSIGN_MATRIX
# DEFAULT_MATRIX = IDENTITY_MATRIX

MATRIX_KEYWORDS = MatrixKeywords()
MATRIX_KEYWORD_SET = MATRIX_KEYWORDS._set()
MATRIX_KEYWORD_VALUES = MATRIX_KEYWORDS._values()
MATRIX_KEYWORD_NAMES = MATRIX_KEYWORDS._names()
# MATRIX_KEYWORD_VALUES = list(MATRIX_KEYWORDS.__dict__.values())
# MATRIX_KEYWORD_NAMES = list(MATRIX_KEYWORDS.__dict__)


# **********************************************************************************************************************
# ******************************************    CONSTANTS  *************************************************************
# **********************************************************************************************************************

ON = True
OFF = False
DEFAULT = False
# AUTO = True  # MODIFIED 7/14/17 CW


# Used by initDirective
INIT_FULL_EXECUTE_METHOD = 'init using the full base class execute method'
INIT__EXECUTE__METHOD_ONLY = 'init using only the subclass _execute method'
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
kwAssign = '| Assign' # Used in context for Log
ASSIGN_VALUE = ': Assign value'
kwAggregate = ': Aggregate' # Used in context for Log
RECEIVER = "receiver"
VALIDATE = 'Validate'
COMMAND_LINE = "COMMAND_LINE"
SET_ATTRIBUTE = "SET ATTRIBUTE"
kwParams = 'params'
CHANGED = 'CHANGED'
UNCHANGED = 'UNCHANGED'
ENABLED = 'ENABLED'


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

CENTRAL_CLOCK = "CentralClock"
TIME_SCALE = "time_scale"
CLOCK = "clock"
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

# Standard arg / attribute names:
VARIABLE = "variable"
VALUE = "value"
PARAMS = "params"
NAME = "name"
PREFS_ARG = "prefs"
CONTEXT = "context"
STANDARD_ARGS = {NAME, VARIABLE, VALUE, PARAMS, PREFS_ARG, CONTEXT}

INITIAL_VALUES = 'initial_values'

# inputs list/ndarray:
TRIALS_DIM = 0
TIME_STEPS_DIM = 1
PROCESSES_DIM = 2
INPUTS_DIM = 3

COMPONENT_INIT = 'Component.__init__'
DEFERRED_INITIALIZATION = 'Deferred Init'
DEFERRED_ASSIGNMENT = 'Deferred Assignment'
DEFERRED_DEFAULT_NAME = 'DEFERRED_DEFAULT_NAME'
USER_PARAMS = 'user_params' # Params available to user for inspection in user_params dict
FUNCTION = "function" # Param name for function, method, or type to instantiate and assign to self.execute
FUNCTION_PARAMS  = "function_params" # Params used to instantiate or assign to a FUNCTION

PARAM_CLASS_DEFAULTS = "paramClassDefaults"        # "Factory" default params for a Function
PARAM_INSTANCE_DEFAULTS = "paramsInstanceDefaults" # Params used to instantiate a Function; supercede paramClassDefaults
PARAMS_CURRENT = "paramsCurrent"                  # Params currently in effect for an instance of a Function
                                                   #    in general, this includes params specifed as arg in a
                                                   #    to Function.execute;  however, there are some exceptions
                                                   #    in which those are kept separate from paramsCurrent (see DDM)
FUNCTION_CHECK_ARGS = 'super._check_args' # Use for "context" arg
FUNCTION_OUTPUT_TYPE_CONVERSION = "FunctionOutputTypeConversion" # Used in Function Components to set output type

#endregion

#region ----------------------------------------    COMPONENT SUBCLASSES  ----------------------------------------------

# Component Categories   -----------------

kwSystemComponentCategory = "System_Base"
kwProcessComponentCategory = "Process_Base"
kwMechanismComponentCategory = "Mechanism_Base"
kwStateComponentCategory = "State_Base"
kwProjectionComponentCategory = "Projection_Base"
kwComponentCategory = "Function_Base"

# Component TYPES  -----------------

# Mechanisms:
PROCESSING_MECHANISM = "ProcessingMechanism"
ADAPTIVE_MECHANISM = "AdpativeMechanism"
LEARNING_MECHANISM = "LearningMechanism"
CONTROL_MECHANISM = "ControlMechanism"
GATING_MECHANISM = 'GatingMechanism'

# States:
INPUT_STATE = "InputState"
PARAMETER_STATE = "ParameterState"
OUTPUT_STATE = "OutputState"
MODULATORY_SIGNAL = 'ModulatorySignal'

# Projections:
MAPPING_PROJECTION = "MappingProjection"
AUTO_ASSOCIATIVE_PROJECTION = "AutoAssociativeProjection"
LEARNING_PROJECTION = "LearningProjection"
CONTROL_PROJECTION = "ControlProjection"
GATING_PROJECTION = "GatingProjection"
PATHWAY_PROJECTION = "PathwayProjection"
MODULATORY_PROJECTION = "ModulatoryProjection"


# Function:
EXAMPLE_FUNCTION_TYPE = "EXAMPLE FUNCTION"
USER_DEFINED_FUNCTION_TYPE = "USER DEFINED FUNCTION TYPE"
COMBINATION_FUNCTION_TYPE = "COMBINATION FUNCTION TYPE"
DIST_FUNCTION_TYPE = "DIST FUNCTION TYPE"
INTEGRATOR_FUNCTION_TYPE = "INTEGRATOR FUNCTION TYPE"
TRANSFER_FUNCTION_TYPE = "TRANSFER FUNCTION TYPE"
DISTRIBUTION_FUNCTION_TYPE = "DISTRIBUTION FUNCTION TYPE"
OBJECTIVE_FUNCTION_TYPE = "OBJECTIVE FUNCTION TYPE"
LEARNING_FUNCTION_TYPE = 'LEARNING FUNCTION TYPE'


# Component SUBTYPES -----------------

# ControlMechanisms:
DEFAULT_CONTROL_MECHANISM = "DefaultControlMechanism"
EVC_MECHANISM = "EVCMechanism"

# ObjectiveMechanisms:
OBJECTIVE_MECHANISM = "ObjectiveMechanism"
COMPARATOR_MECHANISM = "ComparatorMechanism"

# ProcessingMechanisms:
TRANSFER_MECHANISM = "TransferMechanism"
RECURRENT_TRANSFER_MECHANISM = "RecurrentTransferMechanism"
LCA = "LCA"
KWTA = "KWTA"
INTEGRATOR_MECHANISM = "IntegratorMechanism"
DDM_MECHANISM = "DDM"
COMPOSITION_INTERFACE_MECHANISM = "CompositionInterfaceMechanism"

# Functions:
ARGUMENT_THERAPY_FUNCTION = "Contradiction Function"
USER_DEFINED_FUNCTION = "USER DEFINED FUNCTION"
REDUCE_FUNCTION = "Reduce Function"
LINEAR_COMBINATION_FUNCTION = "LinearCombination Function"
LINEAR_FUNCTION = "Linear Function"
EXPONENTIAL_FUNCTION = "Exponential Function"
LOGISTIC_FUNCTION = "Logistic Function"
SOFTMAX_FUNCTION = 'SoftMax Function'
INTEGRATOR_FUNCTION = "Integrator Function"
SIMPLE_INTEGRATOR_FUNCTION = "SimpleIntegrator Function"
CONSTANT_INTEGRATOR_FUNCTION = "ConstantIntegrator Function"
ACCUMULATOR_INTEGRATOR_FUNCTION = "AccumulatorIntegrator Function"
ACCUMULATOR_INTEGRATOR = "AccumulatorIntegrator"  # (7/19/17 CW) added for MappingProjection.py
ADAPTIVE_INTEGRATOR_FUNCTION = "AdaptiveIntegrator Function"
DRIFT_DIFFUSION_INTEGRATOR_FUNCTION = "DriftDiffusionIntegrator Function"
ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION = "OU Integrator Function"
LINEAR_MATRIX_FUNCTION = "LinearMatrix Function"
BACKPROPAGATION_FUNCTION = 'Backpropagation Learning Function'
RL_FUNCTION = 'ReinforcementLearning Function'
ERROR_DERIVATIVE_FUNCTION = 'Error Derivative Function'

# Distribution functions

NORMAL_DIST_FUNCTION = "Normal Distribution Function"
UNIFORM_DIST_FUNCTION = "Uniform Distribution Function"
EXPONENTIAL_DIST_FUNCTION = "Exponential Distribution Function"
GAMMA_DIST_FUNCTION = "Gamma Distribution Function"
WALD_DIST_FUNCTION = "Wald Distribution Function"

# Objective functions
STABILITY_FUNCTION = 'Stability Function'
DISTANCE_FUNCTION = 'Distance Function'

ENERGY = 'energy'
ENTROPY = 'entropy'

DIFFERENCE = 'difference'
EUCLIDEAN = 'euclidean'
ANGLE = 'angle'
CORRELATION = 'correlation'
PEARSON = 'Pearson'
CROSS_ENTROPY = 'cross-entropy'
DISTANCE_METRICS = {DIFFERENCE, EUCLIDEAN, ANGLE, CORRELATION, PEARSON, CROSS_ENTROPY}

#endregion

#region ----------------------------------------------    SYSTEM   ----------------------------------------------------

SYSTEM = "System"
SCHEDULER = "scheduler"
SYSTEM_INIT = 'System.__init__'
DEFAULT_SYSTEM = "DefaultSystem"
CONTROLLER = "controller"
ENABLE_CONTROLLER = "enable_controller"
CONROLLER_PHASE_SPEC = 'ControllerPhaseSpec'

RUN = 'Run'

#endregion

#region ----------------------------------------------    PROCESS   ----------------------------------------------------

PROCESS = "PROCESS"
PROCESSES = "processes"
PROCESS_INIT = 'Process.__init__'
PATHWAY = "pathway"
CLAMP_INPUT = "clamp_input"
SOFT_CLAMP = "soft_clamp"
HARD_CLAMP = "hard_clamp"
LEARNING = 'LEARNING'
LEARNING_RATE = "learning_rate"
CONTROL = 'control'
GATING = 'gating'
kwProjections = "projections"
kwProcessDefaultProjectionFunction = "Default Projection Function"
kwProcessExecute = "ProcessExecute"
kpMechanismExecutedLogEntry = "Mechanism Executed"
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

MECHANISM = 'MECHANISM'
kwMechanismName = "MECHANISM NAME"
kwMechanismDefault = "DEFAULT MECHANISM"
DEFAULT_PROCESSING_MECHANISM = "DefaultProcessingMechanism"
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
SAMPLE = 'SAMPLE'
TARGET = 'TARGET'

RESULT = 'RESULT'
MEAN = 'MEAN'
MEDIAN = 'MEDIAN'
VARIANCE = 'VARIANCE'
SIZE = 'size'
K_VALUE = 'k_value'
THRESHOLD = 'threshold'
RATIO = 'ratio'

STATE_VALUE = "State value"   # Used in State specification dict
                                                 #  to specify State value
STATE_PARAMS = "State params" # Used in State specification dict

# ParamClassDefaults:
MECHANISM_TIME_SCALE = "Mechanism Time Scale"
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

#DDM
kwThreshold = 'thresh'
kwInitialPoint = 'initial_point'
#endregion

#region ----------------------------------------    MODULATORY MECHANISMS ----------------------------------------------

MODULATION = 'modulation'

# ControlMechanism / EVCMechanism
MAKE_DEFAULT_CONTROLLER = "make_default_controller"
MONITOR_FOR_CONTROL = "monitor_for_control"
PREDICTION_MECHANISM = "Prediction Mechanism"
PREDICTION_MECHANISM_TYPE = "prediction_mechanism_type"
PREDICTION_MECHANISM_PARAMS = "prediction_mechanism_params"
PREDICTION_MECHANISM_OUTPUT = "PredictionMechanismOutput"
LEARNING_SIGNAL = 'learning_signal'
LEARNING_SIGNALS = 'learning_signals'
LEARNING_SIGNAL_SPECS = 'LEARNING_SIGNAL_SPECS'
LEARNED_PARAM = 'learned_param'
CONTROL_SIGNAL = 'control_signal'
CONTROL_SIGNALS = 'control_signals'
CONTROL_SIGNAL_SPECS = 'CONTROL_SIGNAL_SPECS'
CONTROLLED_PARAM = 'controlled_param'
CONTROL_PROJECTIONS = 'ControlProjections'
OUTCOME_FUNCTION = 'outcome_function'
COST_FUNCTION = 'cost_function'
COMBINE_OUTCOME_AND_COST_FUNCTION = 'combine_outcome_and_cost_function'
VALUE_FUNCTION = 'value_function'
SAVE_ALL_VALUES_AND_POLICIES = 'save_all_values_and_policies'
SYSTEM_DEFAULT_CONTROLLER = "DefaultController"
EVC_SIMULATION = 'SIMULATING'
ALLOCATION_SAMPLES = "allocation_samples"

# GatingMechanism
MAKE_DEFAULT_GATING_MECHANISM = "make_default_gating_mechanism"
GATING_SIGNAL = 'gating_signal'
GATING_SIGNALS = 'gating_signals'
GATING_SIGNAL_SPECS = 'GATING_SIGNAL_SPECS'
GATE = 'GATE'
GATING_PROJECTIONS = 'GatingProjections'
GATING_POLICY = 'gating_policy'

#endregion

#region ----------------------------------------------    STATES  ------------------------------------------------------

STATE = "State"
# These are used as keys in State specification dictionaries
STATES = "STATES"
STATE_TYPE = "state_type"
PROJECTIONS = "projections"  # Used to specify projection list to State
kwStateName = "StateName"
kwStatePrefs = "StatePrefs"
kwStateContext = "StateContext"
kwAddInputState = 'kwAddNewInputState'     # Used by Mechanism._add_projection_to()
kwAddOutputState = 'kwAddNewOutputState'   # Used by Mechanism._add_projection_from()
FULL = 'FULL'
OWNER = 'owner'
REFERENCE_VALUE = 'reference_value'

# InputStates:
PRIMARY = 'Primary'
INPUT_STATES = 'input_states'
INPUT_STATE_PARAMS = 'input_state_params'
WEIGHT = 'weight'
EXPONENT = 'exponent'

# ParameterStates:
PARAMETER_STATES = 'parameter_states'
PARAMETER_STATE_PARAMS = 'parameter_state_params'

# OutputStates:
OUTPUT_STATES = 'output_states'
OUTPUT_STATE_PARAMS = 'output_states_params'
STANDARD_OUTPUT_STATES = 'standard_output_states'
INDEX = 'index'
CALCULATE = 'calculate'


#endregion

#region ---------------------------------------------    PROJECTION  ---------------------------------------------------

# Attributes / KVO keypaths / Params
PROJECTION = "Projection"
PROJECTION_TYPE = "ProjectionType"
PROJECTION_PARAMS = "ProjectionParams"
MAPPING_PROJECTION_PARAMS = "MappingProjectionParams"
LEARNING_PROJECTION_PARAMS = 'LearningProjectionParams'
CONTROL_PROJECTION_PARAMS = "ControlProjectionParams"
GATING_PROJECTION_PARAMS = 'GatingProjectionParams'
PROJECTION_SENDER = 'projection_sender'
SENDER = 'sender'
PROJECTION_SENDER_VALUE =  "projection_sender_value"
kwProjectionReceiver = 'projection_receiver'
kwReceiverArg = 'receiver'
# kpLog = "ProjectionLog"
MONITOR_FOR_LEARNING = 'monitor_for_learning'
AUTO = 'auto'
HETERO = 'hetero'


#endregion

#region ----------------------------------------------    FUNCTION   ---------------------------------------------------


FUNCTION_OUTPUT_TYPE = 'functionOutputType'

SUM = 'sum'
DIFFERENCE = DIFFERENCE # Defined above for DISTANCE_METRICS
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

AUTO_DEPENDENT='auto_dependent'
DRIFT_RATE = 'drift_rate'
INCREMENT = 'increment'
INTEGRATOR_FUNCTION = 'integrator_function'
INTEGRATION_TYPE = "integration_type"
TIME_STEP_SIZE = 'time_step_size'
DECAY = 'decay'

LOW = 'low'
HIGH = 'high'

BETA = 'beta'

DIST_SHAPE = 'dist_shape'

STANDARD_DEVIATION = 'standard_dev'
DIST_MEAN = 'mean'

OUTPUT_TYPE = 'output'
ALL = 'all'
MAX_VAL = 'max_val'
MAX_INDICATOR = 'max_indicator'
PROB = 'prob'
MUTUAL_ENTROPY = 'mutual entropy'

INITIALIZER = 'initializer'
WEIGHTS = "weights"
EXPONENTS = "exponents"
OPERATION = "operation"
OFFSET = "offset"
LINEAR = 'linear'
CONSTANT = 'constant'
SIMPLE = 'scaled'
ADAPTIVE = 'adaptive'
DIFFUSION = 'diffusion'

#endregion
