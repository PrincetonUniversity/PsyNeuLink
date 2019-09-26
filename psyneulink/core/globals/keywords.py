# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ********************************************  Keywords ***************************************************************

"""
This module provides the string keywords used throughout psyneulink

https://princetonuniversity.github.io/PsyNeuLink/Keywords.html
"""

# **********************************************************************************************************************
# ******************************************    CLASSES    *************************************************************
# **********************************************************************************************************************

# IMPLEMENTATION NOTE:
#  These classes are used for documentation purposes only.
#  The attributes of each are assigned to constants (listed in the next section of this module)
#    that are the ones actually used by the code.

__all__ = [
    'ACCUMULATOR_INTEGRATOR', 'ACCUMULATOR_INTEGRATOR_FUNCTION',
    'ADAPTIVE', 'ADAPTIVE_INTEGRATOR_FUNCTION', 'ADAPTIVE_MECHANISM',
    'ADDITIVE', 'ADDITIVE_PARAM', 'AFTER', 'ALL', 'ALLOCATION_SAMPLES', 'ANGLE',
    'ARGUMENT_THERAPY_FUNCTION', 'ARRANGEMENT', 'ASSERT', 'ASSIGN', 'ASSIGN_VALUE', 'AUTO','AUTO_ASSIGN_MATRIX',
    'AUTO_ASSOCIATIVE_PROJECTION', 'HAS_INITIALIZERS', 'AUTOASSOCIATIVE_LEARNING_MECHANISM',
    'BACKPROPAGATION_FUNCTION', 'BEFORE', 'BETA', 'BIAS', 'BOLD', 'BOTH', 'BOUNDS', 'BUFFER_FUNCTION',
    'CHANGED', 'CLAMP_INPUT', 'COMBINATION_FUNCTION_TYPE', 'COMBINE', 'COMBINE_MEANS_FUNCTION',
    'COMBINE_OUTCOME_AND_COST_FUNCTION', 'COMMAND_LINE', 'COMPARATOR_MECHANISM', 'COMPONENT', 'COMPONENT_INIT',
    'COMPOSITION', 'COMPOSITION_INTERFACE_MECHANISM', 'CONCATENATE_FUNCTION', 'CONDITION', 'CONDITIONS', 'CONSTANT',
    'ContentAddressableMemory_FUNCTION', 'CONTEXT',
    'CONTROL', 'CONTROL_MECHANISM', 'CONTROL_PROJECTION', 'CONTROL_PROJECTION_PARAMS', 'CONTROL_PROJECTIONS',
    'CONTROL_SIGNAL', 'CONTROL_SIGNAL_SPECS', 'CONTROL_SIGNALS', 'CONTROLLED_PARAMS', 'CONTROLLER',
    'CORRELATION', 'COSINE', 'COST_FUNCTION', 'COUNT', 'CROSS_ENTROPY', 'CURRENT_EXECUTION_TIME', 'CUSTOM_FUNCTION',
    'CYCLE',
    'DDM_MECHANISM', 'DECAY', 'DEFAULT', 'DEFAULT_CONTROL_MECHANISM', 'DEFAULT_MATRIX',
    'DEFAULT_PROCESSING_MECHANISM', 'DEFAULT_SYSTEM',
    'DEFERRED_ASSIGNMENT', 'DEFERRED_DEFAULT_NAME', 'DEFERRED_INITIALIZATION',
    'DIFFERENCE', 'DIFFERENCE', 'DIFFUSION', 'DISABLE', 'DISABLE_PARAM', 'DIST_FUNCTION_TYPE', 'DIST_MEAN',
    'DIST_SHAPE', 'DISTANCE_FUNCTION', 'DISTANCE_METRICS', 'DISTRIBUTION_FUNCTION_TYPE', 'DIVISION',
    'DRIFT_DIFFUSION_INTEGRATOR_FUNCTION', 'DUAL_ADAPTIVE_INTEGRATOR_FUNCTION',
    'EID_SIMULATION', 'EID_FROZEN', 'ENABLE_CONTROLLER', 'ENABLED', 'ENERGY', 'ENTROPY', 'ERROR_DERIVATIVE_FUNCTION',
    'EUCLIDEAN', 'EVC_MECHANISM', 'EVC_SIMULATION', 'EXAMPLE_FUNCTION_TYPE',
    'EXECUTING', 'EXECUTION', 'EXECUTION_ID', 'EXECUTION_PHASE',
    'EXPONENTIAL', 'EXPONENT', 'EXPONENTIAL_DIST_FUNCTION', 'EXPONENTIAL_FUNCTION', 'EXPONENTS',
    'FITZHUGHNAGUMO_INTEGRATOR_FUNCTION', 'FINAL', 'FLAGS', 'FULL', 'FULL_CONNECTIVITY_MATRIX',
    'FUNCTION', 'FUNCTIONS', 'FUNCTION_CHECK_ARGS', 'FUNCTION_OUTPUT_TYPE', 'FUNCTION_OUTPUT_TYPE_CONVERSION',
    'FUNCTION_PARAMS',
    'GAIN', 'GAMMA_DIST_FUNCTION', 'GATE', 'GATING', 'GATING_MECHANISM', 'GATING_ALLOCATION', 'GATING_PROJECTION',
    'GATING_PROJECTION_PARAMS', 'GATING_PROJECTIONS', 'GATING_SIGNAL', 'GATING_SIGNAL_SPECS', 'GATING_SIGNALS',
    'GAUSSIAN', 'GAUSSIAN_FUNCTION', 'GILZENRAT_INTEGRATOR_FUNCTION',
    'GRADIENT_OPTIMIZATION_FUNCTION', 'GRID_SEARCH_FUNCTION',
    'HARD_CLAMP', 'HEBBIAN_FUNCTION', 'HETERO', 'HIGH', 'HOLLOW_MATRIX', 'IDENTITY_MATRIX', 'INCREMENT', 'INDEX',
    'INIT_EXECUTE_METHOD_ONLY', 'INIT_FULL_EXECUTE_METHOD', 'INIT_FUNCTION_METHOD_ONLY', 'INITIAL_VALUES',
    'INITIALIZE_CYCLE', 'INITIALIZATION', 'INITIALIZED', 'INITIALIZER', 'INITIALIZING', 'INITIALIZATION_STATUS',
    'INPUT', 'INPUT_LABELS_DICT', 'INPUT_STATE', 'INPUT_STATES', 'INPUT_STATE_PARAMS', 'INPUT_STATE_VARIABLES',
    'INPUTS_DIM', 'INSTANTANEOUS_MODE_VALUE', 'INTEGRATION_TYPE', 'INTEGRATOR_FUNCTION', 'INTEGRATOR_FUNCTION',
    'INTEGRATOR_FUNCTION_TYPE', 'INTEGRATOR_MECHANISM', 'INTEGRATOR_MODE_VALUE', 'INTERCEPT', 'INTERNAL',
    'INTERNAL_ONLY', 'K_VALUE', 'KOHONEN_FUNCTION', 'KOHONEN_MECHANISM', 'KOHONEN_LEARNING_MECHANISM', 'KWTA_MECHANISM',
    'kpMechanismControlAllocationsLogEntry', 'kpMechanismExecutedLogEntry', 'kpMechanismInputLogEntry',
    'kpMechanismOutputLogEntry', 'kpMechanismTimeScaleLogEntry', 'kwAddInputState', 'kwAddOutputState',
    'kwAggregate', 'kwAssign', 'kwComponentCategory', 'kwComponentPreferenceSet', 'kwDefaultPreferenceSetOwner',
    'kwInitialPoint', 'kwInstantiate', 'kwMechanismAdjustFunction', 'kwMechanismComponentCategory',
    'kwMechanismConfidence', 'kwMechanismDefault', 'kwMechanismDefaultInputValue', 'kwMechanismDefaultParams',
    'kwMechanismDuration', 'kwMechanismExecuteFunction', 'kwMechanismExecutionSequenceTemplate',
    'kwMechanismInterrogateFunction', 'kwMechanismName', 'kwMechanismOutputValue', 'kwMechanismParams',
    'kwMechanismParamValue', 'kwMechanismPerformance', 'kwMechanismTerminateFunction', 'kwMechanismType', 'kwParams',
    'kwPrefBaseValue', 'kwPrefCurrentValue', 'kwPreferenceSet', 'kwPreferenceSetName', 'kwPrefLevel', 'kwPrefs',
    'kwPrefsOwner', 'kwProcessComponentCategory', 'kwProcessDefaultMechanism', 'kwProcessDefaultProjectionFunction',
    'kwProcessExecute', 'kwProgressBarChar', 'kwProjectionComponentCategory', 'kwProjectionReceiver', 'kwProjections',
    'kwReceiverArg', 'kwSeparator', 'kwStateComponentCategory',
    'kwSystemComponentCategory', 'kwThreshold',
    'LABELS', 'LCA_MECHANISM', 'LEAKY_COMPETING_INTEGRATOR_FUNCTION', 'LEAK', 'LEARNING', 'LEARNED_PARAM',
    'LEARNED_PROJECTION', 'LEARNING_FUNCTION_TYPE', 'LEARNING_MECHANISM', 'LEARNING_PROJECTION',
    'LEARNING_PROJECTION_PARAMS', 'LEARNING_RATE', 'LEARNING_SIGNAL', 'LEARNING_SIGNAL_SPECS', 'LEARNING_SIGNALS',
    'LINEAR', 'LINEAR_COMBINATION_FUNCTION', 'LINEAR_FUNCTION', 'LINEAR_MATRIX_FUNCTION', 'LOG_ENTRIES',
    'LOGISTIC_FUNCTION', 'LOW', 'LVOC_CONTROL_MECHANISM', 'L0', 'L1',
    'MAKE_DEFAULT_GATING_MECHANISM', 'MAPPING_PROJECTION', 'MAPPING_PROJECTION_PARAMS', 'MASKED_MAPPING_PROJECTION',
    'MATRIX', 'MATRIX_KEYWORD_NAMES', 'MATRIX_KEYWORD_SET', 'MATRIX_KEYWORD_VALUES', 'MATRIX_KEYWORDS','MatrixKeywords',
    'MAX_ABS_VAL', 'MAX_ABS_INDICATOR', 'MAX_ABS_DIFF', 'MAX_INDICATOR', 'MAX_VAL', 'MAYBE', 'MECHANISM', 'METRIC',
    'MECHANISM_VALUE', 'MIN_VAL', 'MODE',
    'MODULATES','MODULATION', 'MODULATORY_PROJECTION', 'MODULATORY_SIGNAL', 'MODULATORY_SIGNALS', 'MODULATORY_MECHANISM',
    'MONITOR', 'MONITOR_FOR_CONTROL', 'MONITOR_FOR_LEARNING', 'MONITOR_FOR_MODULATION',
    'MULTIPLICATIVE', 'MULTIPLICATIVE_PARAM', 'MUTUAL_ENTROPY',
    'NAME', 'NEWEST',  'NODE', 'NodeRoles', 'NOISE', 'NORMAL_DIST_FUNCTION', 'NORMED_L0_SIMILARITY',
    'OBJECTIVE_FUNCTION_TYPE', 'OBJECTIVE_MECHANISM', 'OBJECTIVE_MECHANISM_OBJECT', 'OFF', 'OFFSET', 'OLDEST',
    'ON',  'ONLINE', 'OPERATION', 'OPTIMIZATION_FUNCTION_TYPE', 'ORIGIN','ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION',
    'OUTCOME', 'OUTCOME_FUNCTION', 'OUTPUT', 'OUTPUT_LABELS_DICT', 'OUTPUT_MEAN', 'OUTPUT_MEDIAN', 'OUTPUT_STD_DEV',
    'OUTPUT_STATE', 'OUTPUT_STATE_PARAMS', 'output_state_spec_to_parameter_name', 'OUTPUT_STATES', 'OUTPUT_TYPE',
    'OUTPUT_VARIANCE', 'OVERRIDE', 'OVERRIDE_PARAM', 'OVERWRITE',
    'OWNER', 'OWNER_EXECUTION_COUNT', 'OWNER_EXECUTION_TIME', 'OWNER_VALUE', 'OWNER_VARIABLE',
    'PARAM_CLASS_DEFAULTS', 'PARAM_INSTANCE_DEFAULTS', 'PARAMETER_STATE', 'PARAMETER_STATE_PARAMS',
    'PARAMETER_STATES', 'PARAMS', 'PARAMS_DICT', 'PARAMS_CURRENT', 'PATHWAY', 'PATHWAY_PROJECTION', 'PEARSON',
    'PREDICTION_MECHANISM', 'PREDICTION_MECHANISMS', 'PREDICTION_MECHANISM_OUTPUT', 'PREDICTION_MECHANISM_PARAMS',
    'PREDICTION_MECHANISM_TYPE', 'PREFS_ARG', 'PREVIOUS_VALUE', 'PRIMARY', 'PROB', 'PROB_INDICATOR',
    'PROCESS', 'PROCESSING', 'PROCESS_INIT', 'PROCESSES', 'PROCESSES_DIM', 'PROCESSING_MECHANISM', 'PRODUCT',
    'PROJECTION', 'PROJECTION_DIRECTION', 'PROJECTION_PARAMS', 'PROJECTION_SENDER', 'PROJECTION_TYPE', 'PROJECTIONS',
    'QUOTIENT', 'RANDOM', 'RANDOM_CONNECTIVITY_MATRIX', 'RATE', 'RATIO', 'REARRANGE_FUNCTION', 'RECEIVER',
    'RECURRENT_TRANSFER_MECHANISM', 'REDUCE_FUNCTION', 'REFERENCE_VALUE', 'REINITIALIZE', 'REINITIALIZE_WHEN',
    'RELU_FUNCTION', 'REST', 'RESULT', 'RESULTS', 'ROLES', 'RL_FUNCTION', 'RUN',
    'SAMPLE', 'SAVE_ALL_VALUES_AND_POLICIES', 'SCALAR', 'SCALE', 'SCHEDULER', 'SELF', 'SENDER',
    'SEPARATOR_BAR', 'SIMPLE', 'SIMPLE_INTEGRATOR_FUNCTION', 'SINGLETON', 'SIZE', 'SLOPE', 'SOFT_CLAMP',
    'SOFTMAX_FUNCTION', 'SOURCE', 'STABILITY_FUNCTION', 'STANDARD_ARGS', 'STANDARD_DEVIATION', 'STANDARD_OUTPUT_STATES',
    'STATE', 'STATE_CONTEXT', 'STATE_NAME', 'STATE_PARAMS', 'STATE_PREFS', 'STATE_TYPE', 'STATE_VALUE', 'STATES',
    'SUBTRACTION', 'SUM', 'SYSTEM', 'SYSTEM_DEFAULT_CONTROLLER', 'SYSTEM_INIT',
    'TARGET', 'TARGET_MECHANISM', 'TARGET_LABELS_DICT', 'TERMINAL', 'THRESHOLD', 'TIME', 'TIME_STEP_SIZE',
    'TIME_STEPS_DIM', 'TRAINING_SET', 'TRANSFER_FUNCTION_TYPE', 'TRANSFER_MECHANISM', 'TRANSFER_WITH_COSTS_FUNCTION',
    'TRIAL', 'TRIALS_DIM',
    'UNCHANGED', 'UNIFORM_DIST_FUNCTION', 'USER_DEFINED_FUNCTION', 'USER_DEFINED_FUNCTION_TYPE', 'USER_PARAMS',
    'VALUES', 'VALIDATE', 'VALIDATION', 'VALUE', 'VALUE_ASSIGNMENT', 'VALUE_FUNCTION', 'VARIABLE', 'VARIANCE',
    'VECTOR', 'WALD_DIST_FUNCTION', 'WEIGHT', 'WEIGHTS', 'X_0'
]


class NodeRoles:
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
        A `ProcessingMechanism <ProcessingMechanism>` that is *not* an `ORIGIN` Mechanism, and receives a `Projection
        <Projection>` that closes a recurrent loop in a `Process` and/or `System`.  If it is an `ORIGIN` Mechanism, then
        it is simply designated as such (since it will be assigned input and therefore be initialized in any event).

    INITIALIZE_CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is the `sender <Projection_Base.sender>` of a
        `Projection <Projection>` that closes a loop in a `Process` or `System`, and that is not an `ORIGIN` Mechanism
        (since in that case it will be initialized in any event). An `initial value  <Run_InitialValues>` can be
        assigned to such Mechanisms, that will be used to initialize the Process or System when it is first run.  For
        additional information, see `Run <Run_Initial_Values>`, `System Mechanisms <System_Mechanisms>` and
        `System Input and Initialization <System_Execution_Input_And_Initialization>`.

    TERMINAL
        A `ProcessingMechanism <ProcessingMechanism>` that is the last Mechanism of a `Process` and/or `System`, and
        that provides the output to the Process or System when it is `executed or run <Run>`.  A Process may
        have only one `TERMINAL` Mechanism, but a System may have many.  Note that the `TERMINAL`
        Mechanism of a process is not necessarily a `TERMINAL` Mechanism of the System to which it belongs,
        as it may send projections to other processes in the System (see `example
        <LearningProjection_Output_vs_Terminal_Figure>`).  The `TERMINAL` Mechanisms of a Process or System are listed in
        its :keyword:`terminalMechanisms` attribute, and can be displayed using its :keyword:`show` method.  For
        additional details about `TERMINAL` Mechanisms in Processes, see `Process_Mechanisms` and
        `Process_Input_And_Output`; and for Systems see `System_Mechanisms`.

    SINGLETON
        A `ProcessingMechanism <ProcessingMechanism>` that is the only Mechanism in a `Process` and/or `System`.
        It can serve the functions of an `ORIGIN` and/or a `TERMINAL` Mechanism.

    LEARNING
        A `LearningMechanism <LearningMechanism>` in a `Process` and/or `System`.

    TARGET
        A `ComparatorMechanism` of a `Process` and/or `System` configured for learning that receives a target value
        from its `execute <ComparatorMechanism.ComparatorMechanism.execute>` or
        `run <ComparatorMechanism.ComparatorMechanism.execute>` method.  It is usually (but not necessarily)
        associated with the `TERMINAL` Mechanism of the Process or System. The `TARGET` Mechanisms of a Process or
        System are listed in its :keyword:`target_nodes` attribute, and can be displayed using its
        :keyword:`show` method.  For additional details, see `TARGET Mechanisms <LearningMechanism_Targets>`,
        `learning sequence <Process_Learning_Sequence>`, and specifying `target values <Run_Targets>`.


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

NODE = 'NODE'

class MatrixKeywords:
    """
    Attributes
    ----------

    IDENTITY_MATRIX
        a square matrix of 1's along the diagonal, 0's elsewhere; this requires that the length of the sender and
        receiver values are the same.

    HOLLOW_MATRIX
        a square matrix of 0's along the diagonal, 1's elsewhere; this requires that the length of the sender and
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
        self.INVERSE_HOLLOW_MATRIX = INVERSE_HOLLOW_MATRIX
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
INVERSE_HOLLOW_MATRIX = "InverseHollowMatrix"
FULL_CONNECTIVITY_MATRIX = "FullConnectivityMatrix"
RANDOM_CONNECTIVITY_MATRIX = "RandomConnectivityMatrix"
AUTO_ASSIGN_MATRIX = 'AutoAssignMatrix'
DEFAULT_MATRIX = AUTO_ASSIGN_MATRIX
# DEFAULT_MATRIX = IDENTITY_MATRIX

MATRIX_KEYWORDS = MatrixKeywords()
MATRIX_KEYWORD_SET = MATRIX_KEYWORDS._set()
MATRIX_KEYWORD_VALUES = MATRIX_KEYWORDS._values()
MATRIX_KEYWORD_NAMES = MATRIX_KEYWORDS._names()


class DistanceMetrics:
    """Distance between two arrays.

    Each keyword specifies a metric for the distance between two arrays, :math:`a_1` and :math:`a_2`, of equal length
    for which *len* is their length, :math:`\\bar{a}` is the mean of an array, :math:`\\sigma_{a}` the standard
    deviation of an array, and :math:`w_{a_1a_2}` a coupling coefficient ("weight") between a pair of elements,
    one from each array:

    Attributes
    ----------

    MAX_ABS_DIFF
        :math:`d = \\max(|a_1-a_2|)`

    DIFFERENCE
        (can also be referenced as L0)\n
        :math:`d = \\sum\\limits^{len}(|a_1-a_2|)`

    EUCLIDEAN
        (can also be referenced as L1)\n
        :math:`d = \\sum\\limits^{len}\\sqrt{(a_1-a_2)^2}`

    COSINE
        :math:`d = 1 - \\frac{\\sum\\limits^{len}a_1a_2}{\\sqrt{\\sum\\limits^{len}a_1^2}\\sqrt{\\sum\\limits^{len}a_2^2}}`

    CORRELATION
        :math:`d = 1 - \\left|\\frac{\\sum\\limits^{len}(a_1-\\bar{a}_1)(a_2-\\bar{a}_2)}{(len-1)\\sigma_{a_1}\\sigma_{
        a_2}}\\right|`

    COMMENT:
    PEARSON
        <Description>
    COMMENT

    ENTROPY and CROSS_ENTROPY
        :math:`d = \\sum\\limits^{len}a_1log(a_2)`

    ENERGY:
        :math:`d = -\\frac{1}{2}\\sum\\limits_{i,j}a_{1_i}a_{2_j}w_{ij}`

    """
    def __init__(self):
        self.MAX_ABS_DIFF = MAX_ABS_DIFF
        self.DIFFERENCE = DIFFERENCE
        self.L0 = L0
        self.NORMED_L0_SIMILARITY = NORMED_L0_SIMILARITY
        self.EUCLIDEAN = EUCLIDEAN
        self.L1 = L1
        self.ANGLE = ANGLE
        self.CORRELATION = CORRELATION
        # self.PEARSON = PEARSON
        self.COSINE = COSINE
        self.ENTROPY = CROSS_ENTROPY
        self.CROSS_ENTROPY = CROSS_ENTROPY
        self.ENERGY = ENERGY

    def _values(self):
        return list(self.__dict__.values())

    def _set(self):
        return set(self.__dict__.values())

    def _names(self):
        return list(self.__dict__)

    def _is_metric(metric):
        if metric in DISTANCE_METRICS_SET:
            return True
        else:
            return False


METRIC = 'metric'
DIFFERENCE = 'difference'
L0 = DIFFERENCE
NORMED_L0_SIMILARITY = 'normed_L0_similarity'
MAX_ABS_DIFF = 'max_abs_diff'
EUCLIDEAN = 'euclidean'
L1 = EUCLIDEAN
ANGLE = 'angle'
CORRELATION = 'correlation'
COSINE = 'cosine'
PEARSON = 'Pearson'
ENTROPY = 'cross-entropy'
CROSS_ENTROPY = 'cross-entropy'
ENERGY = 'energy'

DISTANCE_METRICS = DistanceMetrics()
DISTANCE_METRICS_SET = DISTANCE_METRICS._set()
DISTANCE_METRICS_VALUES = DISTANCE_METRICS._values()
DISTANCE_METRICS_NAMES = DISTANCE_METRICS._names()

ENERGY = 'energy'
ENTROPY = 'entropy'


# **********************************************************************************************************************
# ******************************************    CONSTANTS  *************************************************************
# **********************************************************************************************************************

ON = True
OFF = False
DEFAULT = False
# AUTO = True  # MODIFIED 7/14/17 CW
ASSERT = True

# Used by initDirective
INIT_FULL_EXECUTE_METHOD = 'init using the full base class execute method'
INIT_EXECUTE_METHOD_ONLY = 'init using only the subclass _execute method'
INIT_FUNCTION_METHOD_ONLY = 'init using only the subclass __function__ method'


#region ---------------------------------------------    GENERAL    ----------------------------------------------------
# General

ALL = 'all'
BOTH = 'both'
MAYBE = 0.5

kwSeparator = ': '
SEPARATOR_BAR = ' | '
kwProgressBarChar = '.'
# kwValueSuffix = '_value'
SELF = 'self'
FLAGS = 'flags'
INITIALIZATION_STATUS = 'initialization_status'
EXECUTION_PHASE = 'execution_phase'
SOURCE = 'source'
INITIALIZING = " INITIALIZING "  # Used as status and context for Log
INITIALIZED = " INITIALIZED "  # Used as status
kwInstantiate = " INSTANTIATING "  # Used as context for Log
EXECUTING = " EXECUTING " # Used in context for Log and ReportOutput pref
kwAssign = '| Assign' # Used in context for Log
ASSIGN_VALUE = ': Assign value'
kwAggregate = ': Aggregate' # Used in context for Log
VALIDATE = 'Validate'
COMMAND_LINE = "COMMAND_LINE"
kwParams = 'params'
CHANGED = 'CHANGED'
UNCHANGED = 'UNCHANGED'
ENABLED = 'ENABLED'
STATEFUL_ATTRIBUTES = 'stateful_attributes'
COUNT = 'COUNT'
BOLD = 'bold'
BEFORE = 'before'
AFTER = 'after'
ONLINE = 'online'
INPUT = 'input'
OUTPUT = 'output'
RANDOM =  'random'
OLDEST = 'oldest'
NEWEST = 'newest'

EID_SIMULATION = '-sim'
EID_FROZEN = '-frozen'

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

#region --------------------------------------------    PREFERENCES    -------------------------------------------------

kwPreferenceSet = 'PreferenceSet'
kwComponentPreferenceSet = 'PreferenceSet'
#endregion

#region ------------------------------------------------   LOG    ------------------------------------------------------

TIME = 'time'
LOG_ENTRIES = 'LOG_ENTRIES'
INITIALIZATION = 'INITIALIZATION'
VALIDATION = 'VALIDATION'
EXECUTION = 'EXECUTION'
PROCESSING = 'PROCESSING'
VALUE_ASSIGNMENT = 'VALUE_ASSIGNMENT'
FINAL = 'FINAL'


#endregion

#region -----------------------------------------------  UTILITIES  ----------------------------------------------------

kpMechanismTimeScaleLogEntry = "Mechanism TimeScale"
kpMechanismInputLogEntry = "Mechanism Input"
kpMechanismOutputLogEntry = "Mechanism Output"
kpMechanismControlAllocationsLogEntry = "Mechanism Control Allocations"
#endregion

#region ----------------------------------------------   COMPONENT   ---------------------------------------------------

COMPOSITION = 'COMPOSITION'

#endregion

#region ----------------------------------------------   COMPONENT   ---------------------------------------------------

COMPONENT = 'COMPONENT'

# Standard arg / attribute names:
VARIABLE = "variable"
DEFAULT_VARIABLE = "default_variable"
VALUE = "value"
PREVIOUS_VALUE = 'previous_value'
LABELS = 'labels'
PARAMS = "params"
NAME = "name"
PREFS_ARG = "prefs"
CONTEXT = "context"
STANDARD_ARGS = {NAME, VARIABLE, VALUE, PARAMS, PREFS_ARG, CONTEXT}

INITIAL_VALUES = 'initial_values'
CURRENT_EXECUTION_TIME = 'current_execution_time'
EXECUTION_ID = 'execution_id'

# inputs list/ndarray:
TRIALS_DIM = 0
TIME_STEPS_DIM = 1
PROCESSES_DIM = 2
INPUTS_DIM = 3

COMPONENT_INIT = 'Component.__init__'
DEFERRED_INITIALIZATION = 'Deferred Init'
DEFERRED_ASSIGNMENT = 'Deferred Assignment'
DEFERRED_DEFAULT_NAME = 'DEFERRED_DEFAULT_NAME'
USER_PARAMS = 'user_params' # Parameters available to user for inspection in user_params dict
FUNCTION = "function" # Parameter name for function, method, or type to instantiate and assign to self.execute
FUNCTION_PARAMS  = "function_params" # Parameters used to instantiate or assign to a FUNCTION

PARAM_CLASS_DEFAULTS = "paramClassDefaults"        # "Factory" default params for a Function
PARAM_INSTANCE_DEFAULTS = "paramInstanceDefaults" # Parameters used to instantiate a Function; supercede paramClassDefaults
PARAMS_CURRENT = "paramsCurrent"                  # Parameters currently in effect for an instance of a Function
                                                   #    in general, this includes params specifed as arg in a
                                                   #    to Function.execute;  however, there are some exceptions
                                                   #    in which those are kept separate from paramsCurrent (see DDM)
FUNCTION_CHECK_ARGS = 'super._check_args' # Use for "context" arg
FUNCTION_OUTPUT_TYPE_CONVERSION = "enable_output_type_conversion"  # Used in Function Components to set output type

#endregion

#region ----------------------------------------    COMPONENT SUBCLASSES  ----------------------------------------------

# Component Categories   -----------------

kwSystemComponentCategory = "System"
kwProcessComponentCategory = "Process"
kwMechanismComponentCategory = "Mechanism_Base"
kwStateComponentCategory = "State_Base"
kwProjectionComponentCategory = "Projection_Base"
kwComponentCategory = "Function_Base"

# Component TYPES  -----------------

# Mechanisms:
PROCESSING_MECHANISM = "ProcessingMechanism"
ADAPTIVE_MECHANISM = "AdaptiveMechanism"
LEARNING_MECHANISM = "LearningMechanism"
CONTROL_MECHANISM = "ControlMechanism"
TARGET_MECHANISM = "TargetMechanism"
GATING_MECHANISM = 'GatingMechanism'
MODULATORY_MECHANISM = 'ModulatoryMechanism'
AUTOASSOCIATIVE_LEARNING_MECHANISM = 'AutoAssociativeLearningMechanism'
KOHONEN_LEARNING_MECHANISM = 'KohonenLearningMechanism'

# States:
INPUT_STATE = "InputState"
PROCESS_INPUT_STATE = "ProcessInputState"
SYSTEM_INPUT_STATE = "SystemInputState"
PARAMETER_STATE = "ParameterState"
OUTPUT_STATE = "OutputState"
MODULATORY_SIGNAL = 'ModulatorySignal'
LEARNING_SIGNAL = 'LearningSignal'
CONTROL_SIGNAL = 'ControlSignal'
GATING_SIGNAL = 'GatingSignal'

# Projections:
MAPPING_PROJECTION = "MappingProjection"
AUTO_ASSOCIATIVE_PROJECTION = "AutoAssociativeProjection"
MASKED_MAPPING_PROJECTION = 'MaskedMappingProjection'
LEARNING_PROJECTION = "LearningProjection"
CONTROL_PROJECTION = "ControlProjection"
GATING_PROJECTION = "GatingProjection"
PATHWAY_PROJECTION = "PathwayProjection"
PATHWAY_PROJECTIONS = "PathwayProjections"
MODULATORY_PROJECTION = "ModulatoryProjection"
MODULATORY_PROJECTIONS = "ModulatoryProjections"


# Function:
EXAMPLE_FUNCTION_TYPE = "EXAMPLE FUNCTION"
USER_DEFINED_FUNCTION_TYPE = "USER DEFINED FUNCTION TYPE"
COMBINATION_FUNCTION_TYPE = "COMBINATION FUNCTION TYPE"
DIST_FUNCTION_TYPE = "DIST FUNCTION TYPE"
STATEFUL_FUNCTION_TYPE = "STATEFUL FUNCTION TYPE"
MEMORY_FUNCTION_TYPE = "MEMORY FUNCTION TYPE"
INTEGRATOR_FUNCTION_TYPE = "INTEGRATOR FUNCTION TYPE"
TRANSFER_FUNCTION_TYPE = "TRANSFER FUNCTION TYPE"
LEABRA_FUNCTION_TYPE = "LEABRA FUNCTION TYPE"
DISTRIBUTION_FUNCTION_TYPE = "DISTRIBUTION FUNCTION TYPE"
OBJECTIVE_FUNCTION_TYPE = "OBJECTIVE FUNCTION TYPE"
OPTIMIZATION_FUNCTION_TYPE = "OPTIMIZATION FUNCTION TYPE"
LEARNING_FUNCTION_TYPE = 'LEARNING FUNCTION TYPE'
NORMALIZING_FUNCTION_TYPE = "NORMALIZING FUNCTION TYPE"
INTERFACE_FUNCTION_TYPE = "INTERFACE FUNCTION TYPE"
SELECTION_FUNCTION_TYPE = "SELECTION FUNCTION TYPE"


# Component SUBTYPES -----------------

# ControlMechanism:
DEFAULT_CONTROL_MECHANISM = "DefaultControlMechanism"
OPTIMIZATION_CONTROL_MECHANISM = 'OptimizationControlMechanism'
EVC_MECHANISM = "EVCControlMechanism"
LVOC_CONTROL_MECHANISM = 'LVOCControlMechanism'

# ObjectiveMechanisms:
OBJECTIVE_MECHANISM_OBJECT = "ObjectiveMechanism"
COMPARATOR_MECHANISM = "ComparatorMechanism"
PREDICTION_ERROR_MECHANISM = "PredictionErrorMechanism"

# ProcessingMechanisms:
TRANSFER_MECHANISM = "TransferMechanism"
LEABRA_MECHANISM = "LeabraMechanism"
RECURRENT_TRANSFER_MECHANISM = "RecurrentTransferMechanism"
CONTRASTIVE_HEBBIAN_MECHANISM = "ContrastiveHebbianMechanism"
LCA_MECHANISM = "LCAMechanism"
KOHONEN_MECHANISM = 'KohonenMechanism'
KWTA_MECHANISM = "KWTAMechanism"
INTEGRATOR_MECHANISM = "IntegratorMechanism"
DDM_MECHANISM = "DDM"
COMPOSITION_INTERFACE_MECHANISM = "CompositionInterfaceMechanism"
PROCESSING_MECHANISM = "ProcessingMechanism"

# Functions:
ARGUMENT_THERAPY_FUNCTION = "Contradiction Function"
USER_DEFINED_FUNCTION = "USER DEFINED FUNCTION"

# CombinationFunctions:
REDUCE_FUNCTION = "Reduce Function"
CONCATENATE_FUNCTION = "Concatenate Function"
REARRANGE_FUNCTION = 'Rearrange Function'
LINEAR_COMBINATION_FUNCTION = "LinearCombination Function"
COMBINE_MEANS_FUNCTION = "CombineMeans Function"

# TransferFunctions:
IDENTITY_FUNCTION = 'Identity Function'
LINEAR_FUNCTION = "Linear Function"
LEABRA_FUNCTION = "Leabra Function"
EXPONENTIAL_FUNCTION = "Exponential Function"
LOGISTIC_FUNCTION = "Logistic Function"
RELU_FUNCTION = "ReLU Function"
GAUSSIAN_FUNCTION = "Gaussian Function"
GAUSSIAN_DISTORT_FUNCTION = "GaussianDistort Function"
SOFTMAX_FUNCTION = 'SoftMax Function'
LINEAR_MATRIX_FUNCTION = "LinearMatrix Function"
TRANSFER_WITH_COSTS_FUNCTION = "TransferWithCosts Function"

# SelectionFunctions:
ONE_HOT_FUNCTION = "OneHot Function"

# IntegratorFunctions:
INTEGRATOR_FUNCTION = "IntegratorFunction Function"
SIMPLE_INTEGRATOR_FUNCTION = "SimpleIntegrator Function"
INTERACTIVE_ACTIVATION_INTEGRATOR_FUNCTION = "Interactive Activation IntegratorFunction Function"
ACCUMULATOR_INTEGRATOR_FUNCTION = "AccumulatorIntegrator Function"
FITZHUGHNAGUMO_INTEGRATOR_FUNCTION = "FitzHughNagumoIntegrator Function"
DUAL_ADAPTIVE_INTEGRATOR_FUNCTION = "DualAdaptiveIntegrator Function"
ACCUMULATOR_INTEGRATOR = "AccumulatorIntegrator"  # (7/19/17 CW) added for MappingProjection.py
LEAKY_COMPETING_INTEGRATOR_FUNCTION = 'LeakyCompetingIntegrator Function'
ADAPTIVE_INTEGRATOR_FUNCTION = "AdaptiveIntegrator Function"
GILZENRAT_INTEGRATOR_FUNCTION = "GilzenratDecisionIntegrator Function"
DRIFT_DIFFUSION_INTEGRATOR_FUNCTION = "DriftDiffusionIntegrator Function"
ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION = "OU IntegratorFunction Function"

# MemoryFunctions:
BUFFER_FUNCTION = 'Buffer Function'
ContentAddressableMemory_FUNCTION = 'ContentAddressableMemory Function'

# OptimizationFunctions:
GRADIENT_OPTIMIZATION_FUNCTION = "GradientOptimization Function"
GRID_SEARCH_FUNCTION = 'GridSearch Function'

# LearningFunctions:
GAUSSIAN_PROCESS_FUNCTION = 'GaussianProcess Function'
HEBBIAN_FUNCTION = 'Hebbian Learning Function'
CONTRASTIVE_HEBBIAN_FUNCTION = 'ContrastiveHebbian Learning Function'
KOHONEN_FUNCTION = 'Kohonen Learning Function'
RL_FUNCTION = 'ReinforcementLearning Function'
BACKPROPAGATION_FUNCTION = 'Backpropagation Learning Function'
TDLEARNING_FUNCTION = "TD Learning Function"
PREDICTION_ERROR_DELTA_FUNCTION = "PredictionErrorDelta Function"
ERROR_DERIVATIVE_FUNCTION = 'Error Derivative Function'

# Distributionfunctions
NORMAL_DIST_FUNCTION = "Normal Distribution Function"
UNIFORM_DIST_FUNCTION = "Uniform Distribution Function"
EXPONENTIAL_DIST_FUNCTION = "Exponential Distribution Function"
GAMMA_DIST_FUNCTION = "Gamma Distribution Function"
WALD_DIST_FUNCTION = "Wald Distribution Function"
DRIFT_DIFFUSION_ANALYTICAL_FUNCTION = "Drift Diffusion Analytical Function"

# ObjectiveFunctions
STABILITY_FUNCTION = 'Stability Function'
DISTANCE_FUNCTION = 'Distance Function'

# Interface Functions:
STATE_MAP_FUNCTION = 'State Map Function'

#endregion

#region ---------------------------------------    SYSTEM / COMPOSITION   ----------------------------------------------

SYSTEM = "System"
SCHEDULER = "scheduler"
SYSTEM_INIT = 'System.__init__'
DEFAULT_SYSTEM = "DefaultSystem"
CONTROLLER = "controller"
ENABLE_CONTROLLER = "enable_controller"

RUN = 'run'
TRIAL = 'trial'

ROLES = 'roles'
CONDITIONS = 'conditions'
VALUES = 'values'
FUNCTIONS = 'functions'

#endregion

#region ---------------------------------------    AUTODIFF COMPOSITION   ----------------------------------------------

TRAINING_SET = 'training set'

#endregion

#region ----------------------------------------------    PROCESS   ----------------------------------------------------

PROCESS = "PROCESS"
PROCESSES = "processes"
PROCESS_INIT = 'Process.__init__'
PATHWAY = "pathway"
CLAMP_INPUT = "clamp_input"
SOFT_CLAMP = "soft_clamp"
HARD_CLAMP = "hard_clamp"
PULSE_CLAMP = "pulse_clamp"
NO_CLAMP = "no_clamp"
LEARNING_RATE = "learning_rate"
CONTROL = 'CONTROL'
GATING = 'gating'
kwProjections = "projections"
kwProcessDefaultProjectionFunction = "Default Projection Function"
kwProcessExecute = "ProcessExecute"
kpMechanismExecutedLogEntry = "Mechanism Executed"
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

MECHANISM = 'MECHANISM'
MECHANISMS = 'MECHANISMS'
kwMechanismName = "MECHANISM NAME"
kwMechanismDefault = "DEFAULT MECHANISM"
DEFAULT_PROCESSING_MECHANISM = "DefaultProcessingMechanism"
kwProcessDefaultMechanism = "ProcessDefaultMechanism"
kwMechanismType = "Mechanism Type" # Used in mechanism dict specification (e.g., in process.pathway[])
kwMechanismDefaultInputValue = "Mechanism Default Input Value " # Used in mechanism specification dict
kwMechanismParamValue = "Mechanism Parameter Value"                 # Used to specify mechanism param value
kwMechanismDefaultParams = "Mechanism Default Parameters"           # Used in mechanism specification dict
CONDITION = 'condition'

# Keywords for OUTPUT_STATE_VARIABLE dict:
OWNER_VARIABLE = 'OWNER_VARIABLE'
OWNER_VALUE = 'OWNER_VALUE'
OWNER_EXECUTION_COUNT = 'EXECUTION_COUNT'
OWNER_EXECUTION_TIME = 'EXECUTION_TIME'
INPUT_STATE_VARIABLES = 'INPUT_STATE_VARIABLES'
PARAMS_DICT = 'PARAMS_DICT'

# this exists because kws like OWNER_VALUE are set as properties
# on Mechanisms, so you can't just change the string value to map
# as you want - the property "value" will be overwritten then
output_state_spec_to_parameter_name = {
    OWNER_VARIABLE: VARIABLE,
    OWNER_VALUE: VALUE,
    INPUT_STATE_VARIABLES: 'input_state_variables',
    OWNER_EXECUTION_COUNT: 'execution_count',
    OWNER_EXECUTION_TIME: 'current_execution_time'
}

# Dictionaries of labels for input, output and target arrays
INPUT_LABELS_DICT = 'input_labels_dict'
OUTPUT_LABELS_DICT = 'output_labels_dict'
TARGET_LABELS_DICT = 'target_labels_dict'

ORIGIN = 'ORIGIN'
INTERNAL = 'INTERNAL'
CYCLE = 'CYCLE'
INITIALIZE_CYCLE = 'INITIALIZE_CYCLE'
# AUTO_LEARNING = 'AUTO_LEARNING'
TERMINAL = 'TERMINAL'
SINGLETON = 'ORIGIN AND TERMINAL'
LEARNING = 'LEARNING'
SAMPLE = 'SAMPLE'
TARGET = 'TARGET'
ERROR = 'ERROR'

RESULTS = 'RESULTS'
RESULT = 'RESULT'
OUTPUT_MEAN = 'OUTPUT_MEAN'
OUTPUT_MEDIAN = 'OUTPUT_MEDIAN'
OUTPUT_VARIANCE = 'OUTPUT_VARIANCE'
OUTPUT_STD_DEV = 'OUTPUT_STD_DEV'
MECHANISM_VALUE = 'MECHANISM_VALUE'
SIZE = 'size'
K_VALUE = 'k_value'
THRESHOLD = 'threshold'
RATIO = 'ratio'

STATE_VALUE = "State value"   # Used in State specification dict
                                                 #  to specify State value
STATE_PARAMS = "State params" # Used in State specification dict

# ParamClassDefaults:
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

# LearningMechanism:
LEARNING_SIGNALS = 'learning_signals'
LEARNING_SIGNAL_SPECS = 'LEARNING_SIGNAL_SPECS'
LEARNED_PARAM = 'learned_param'
LEARNED_PROJECTION = 'learned_projection'

# ControlMechanism / EVCControlMechanism / ObjectiveMechanism
SIMULATIONS = 'simulations'
OBJECTIVE_MECHANISM = "objective_mechanism"
OUTCOME = 'OUTCOME'
MONITOR = "monitor"
MONITOR_FOR_CONTROL = "monitor_for_control"
PREDICTION_MECHANISM = "Prediction Mechanism"
PREDICTION_MECHANISMS = "prediction_mechanisms"
PREDICTION_MECHANISM_TYPE = "prediction_mechanism_type"
PREDICTION_MECHANISM_PARAMS = "prediction_mechanism_params"
PREDICTION_MECHANISM_OUTPUT = "PredictionMechanismOutput"

MODULATORY_SIGNALS = 'modulatory_signals'
CONTROL_SIGNALS = 'control_signals'
CONTROL_SIGNAL_SPECS = 'CONTROL_SIGNAL_SPECS'
CONTROLLED_PARAMS = 'CONTROLLED_PARAMS'
CONTROL_PROJECTIONS = 'ControlProjections'
GATING_SIGNALS = 'gating_signals'
OUTCOME_FUNCTION = 'outcome_function'
COST_FUNCTION = 'cost_function'
COMBINE_OUTCOME_AND_COST_FUNCTION = 'combine_outcome_and_cost_function'
VALUE_FUNCTION = 'value_function'
SAVE_ALL_VALUES_AND_POLICIES = 'save_all_values_and_policies'
SYSTEM_DEFAULT_CONTROLLER = "DefaultController"
EVC_SIMULATION = 'CONTROL SIMULATION'
ALLOCATION_SAMPLES = "allocation_samples"


# GatingMechanism
MAKE_DEFAULT_GATING_MECHANISM = "make_default_gating_mechanism"
GATING_SIGNALS = 'gating_signals'
GATING_SIGNAL_SPECS = 'GATING_SIGNAL_SPECS'
GATE = 'GATE'
GATED_STATES = 'GATED_STATES'
GATING_PROJECTIONS = 'GatingProjections'
GATING_ALLOCATION = 'gating_allocation'

MODULATORY_SPEC_KEYWORDS = {LEARNING, LEARNING_SIGNAL, LEARNING_PROJECTION, LEARNING_MECHANISM,
                            CONTROL, CONTROL_SIGNAL, CONTROL_PROJECTION, CONTROL_MECHANISM,
                            GATING, GATING_SIGNAL, GATING_PROJECTION, GATING_MECHANISM,
                            MODULATORY_MECHANISM}

MODULATED_PARAMETER_PREFIX = 'mod_'

#endregion

#region ----------------------------------------------    STATES  ------------------------------------------------------

STATE = "State"
STATE_TYPE = "state_type"
# These are used as keys in State specification dictionaries
STATES = "STATES"
MODULATES = "modulates"
PROJECTIONS = "projections"  # Used to specify projection list to State DEPRECATED;  REPLACED BY CONNECTIONS
CONNECTIONS = 'CONNECTIONS'
STATE_NAME = "StateName"
STATE_PREFS = "StatePrefs"
STATE_CONTEXT = "StateContext"
kwAddInputState = 'kwAddNewInputState'     # Used by Mechanism._add_projection_to()
kwAddOutputState = 'kwAddNewOutputState'   # Used by Mechanism._add_projection_from()
FULL = 'FULL'
OWNER = 'owner'
REFERENCE_VALUE = 'reference_value'
REFERENCE_VALUE_NAME = 'reference_value_name'

# InputStates:
PRIMARY = 'Primary'
INPUT_STATES = 'input_states'
INPUT_STATE_PARAMS = 'input_state_params'
WEIGHT = 'weight'
EXPONENT = 'exponent'
INTERNAL_ONLY = 'internal_only'

# ParameterStates:
PARAMETER_STATES = 'parameter_states'
PARAMETER_STATE_PARAMS = 'parameter_state_params'

# OutputStates:
OUTPUT_STATES = 'output_states'
OUTPUT_STATE_PARAMS = 'output_states_params'
STANDARD_OUTPUT_STATES = 'standard_output_states'
INDEX = 'index'       # For backward compatibility with INDEX and ASSIGN
ASSIGN = 'assign'     # For backward compatibility with INDEX and ASSIGN
CALCULATE = 'assign'  # For backward compatibility with CALCULATE

# Modulation
MODULATION = 'modulation'
MONITOR_FOR_MODULATION = 'monitor_for_modulation'
ADDITIVE = ADDITIVE_PARAM = 'additive_param'
MULTIPLICATIVE = MULTIPLICATIVE_PARAM = 'multiplicative_param'
OVERRIDE = OVERRIDE_PARAM = 'OVERRIDE'
DISABLE = DISABLE_PARAM = 'DISABLE'

#endregion

#region ---------------------------------------------    PROJECTION  ---------------------------------------------------

# Attributes / KVO keypaths / Parameters
PROJECTION = "Projection"
PROJECTION_TYPE = "PROJECTION_TYPE"
PROJECTION_PARAMS = "ProjectionParams"
MAPPING_PROJECTION_PARAMS = "MappingProjectionParams"
LEARNING_PROJECTION_PARAMS = 'LearningProjectionParams'
CONTROL_PROJECTION_PARAMS = "ControlProjectionParams"
GATING_PROJECTION_PARAMS = 'GatingProjectionParams'
PROJECTION_SENDER = 'projection_sender'
SENDER = 'sender'
RECEIVER = "receiver"
PROJECTION_DIRECTION = {SENDER: 'to',
                        RECEIVER: 'from'}
kwProjectionReceiver = 'projection_receiver'
kwReceiverArg = 'receiver'
# kpLog = "ProjectionLog"
MONITOR_FOR_LEARNING = 'monitor_for_learning'
AUTO = 'auto'
HETERO = 'hetero'


#endregion

#region ----------------------------------------------    FUNCTION   ---------------------------------------------------


# General ------------------------------------------------

FUNCTION_PARAMETER_PREFIX = 'func_'
CUSTOM_FUNCTION = 'custom_function'
FUNCTION_OUTPUT_TYPE = 'output_type'
OUTPUT_TYPE = 'output'
OVERWRITE = 'overwrite'
REINITIALIZE = "reinitialize"
REINITIALIZE_WHEN = "reinitialize_when"

LOW = 'low'
HIGH = 'high'
BOUNDS = 'bounds'
MODE = 'mode'
REST = "rest"

# Function-specific ---------------------------------------

STATEFUL_FUNCTION = 'stateful_function'
INTEGRATOR_FUNCTION = 'integrator_function'
MEMORY_FUNCTION = 'memory_function'
HAS_INITIALIZERS='has_initializers'
INCREMENT = 'increment'
INTEGRATION_TYPE = "integration_type"
TIME_STEP_SIZE = 'time_step_size'
DECAY = 'decay'
INTEGRATOR_MODE_VALUE = "integrator_mode_value"
INSTANTANEOUS_MODE_VALUE = "instantaneous_mode_value"
LINEAR = 'linear'
CONSTANT = 'constant'
SIMPLE = 'scaled'
ADAPTIVE = 'adaptive'
DIFFUSION = 'diffusion'
EXPONENTIAL = 'exponential'
GAUSSIAN = 'gaussian'
SINUSOID = 'sinusoid'

COMBINE = 'combine'
SUM = 'sum'
DIFFERENCE = DIFFERENCE # Defined above for DISTANCE_METRICS
PRODUCT = 'product'
QUOTIENT = 'quotient'
SUBTRACTION = 'subtraction'
DIVISION = 'division'
SCALAR = 'scalar'
VECTOR = 'vector'
ARRANGEMENT = 'arrangement'

GAIN = 'gain'
BIAS = 'bias'
X_0 = "x_0"
LEAK = 'leak'
SLOPE = 'slope'
INTERCEPT = 'intercept'
RATE = 'rate'
SCALE = 'scale'
NOISE = 'noise'

BETA = 'beta'

DIST_SHAPE = 'dist_shape'

STANDARD_DEVIATION = 'standard_deviation'
VARIANCE = 'variance'
DIST_MEAN = 'mean'

MAX_VAL = 'MAX_VAL'
MAX_ABS_VAL = 'MAX_ABS_VAL'
MAX_INDICATOR = 'MAX_INDICATOR'
MAX_ABS_INDICATOR = 'MAX_ABS_INDICATOR'
MIN_VAL = 'MIN_VAL'
MIN_ABS_VAL = 'MIN_ABS_VAL'
MIN_INDICATOR = 'MIN_INDICATOR'
MIN_ABS_INDICATOR = 'MIN_ABS_INDICATOR'
PROB = 'PROB'
PROB_INDICATOR = 'PROB_INDICATOR'
MUTUAL_ENTROPY = 'mutual entropy'
PER_ITEM = 'per_item'

INITIALIZER = 'initializer'
INITIAL_V = 'initial_v'
INITIAL_W = 'initial_w'
WEIGHTS = "weights"
EXPONENTS = "exponents"
OPERATION = "operation"
OFFSET = "offset"

REWARD = 'reward'
NETWORK = 'network'

GAMMA = 'gamma'
#endregion
