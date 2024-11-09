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
    'ADAPTIVE', 'ADAPTIVE_INTEGRATOR_FUNCTION', 'ADAPTIVE_MECHANISM', 'ADD_INPUT_PORT', 'ADD_OUTPUT_PORT', 'ADDITIVE',
    'ADDITIVE_PARAM', 'AFTER', 'ALL', 'ALLOCATION_SAMPLES', 'ALLOW_PROBES', 'ANGLE', 'ANGLE_FUNCTION','ANY',
    'ARG_MAX', 'ARG_MAX_ABS', 'ARG_MAX_INDICATOR', 'ARG_MAX_ABS_INDICATOR',
    'ARG_MIN', 'ARG_MIN_ABS', 'ARG_MIN_INDICATOR', 'ARG_MIN_ABS_INDICATOR',
    'ARGUMENT_THERAPY_FUNCTION', 'ARRANGEMENT', 'ASSERT', 'ASSIGN', 'ASSIGN_VALUE', 'AUTO','AUTO_ASSIGN_MATRIX',
    'AUTO_ASSOCIATIVE_PROJECTION', 'HAS_INITIALIZERS', 'AUTOASSOCIATIVE_LEARNING_MECHANISM',
    'AUTODIFF_COMPOSITION', 'AUTODIFF_RESULTS', 'BACKPROPAGATION_FUNCTION', 'BINOMIAL_DISTORT_FUNCTION',
    'BEFORE', 'BETA', 'BIAS', 'BOLD', 'BOTH', 'BOUNDS', 'BUFFER_FUNCTION',
    'CHANGED', 'CLAMP_INPUT', 'COMBINATION_FUNCTION_TYPE', 'COMBINE', 'COMBINE_MEANS_FUNCTION',
    'COMBINE_OUTCOME_AND_COST_FUNCTION', 'COMMAND_LINE', 'comparison_operators', 'COMPARATOR_MECHANISM', 'COMPONENT',
    'COMPONENT_INIT', 'COMPONENT_PREFERENCE_SET', 'COMPOSITION', 'COMPOSITION_FUNCTION_APPROXIMATOR',
    'COMPOSITION_INTERFACE_MECHANISM', 'CONCATENATE', 'CONCATENATE_FUNCTION', 'CONDITION', 'CONDITIONS', 'CONSTANT',
    'ContentAddressableMemory_FUNCTION', 'CONTEXT',
    'CONTROL', 'CONTROL_MECHANISM', 'CONTROL_PATHWAY', 'CONTROL_PROJECTION',  'CONTROL_PROJECTION_PARAMS',
    'CONTROL_PROJECTIONS', 'CONTROL_SIGNAL', 'CONTROL_SIGNAL_SPECS', 'CONTROL_SIGNALS', 'CONTROLLED_PARAMS',
    'CONTROLLER', 'CONTROLLER_OBJECTIVE', 'CORRELATION', 'CPU', 'COSINE', 'COSINE_SIMILARITY',
    'COST_FUNCTION', 'COUNT', 'CROSS_ENTROPY', 'CURRENT_EXECUTION_TIME', 'CUSTOM_FUNCTION', 'CUDA', 'CYCLE',
    'DDM_MECHANISM', 'DECAY', 'DEFAULT', 'DEFAULT_CONTROL_MECHANISM', 'DEFAULT_INPUT', 'DEFAULT_MATRIX',
    'DEFAULT_PREFERENCE_SET_OWNER', 'DEFAULT_PROCESSING_MECHANISM', 'DEFAULT_VARIABLE',
    'DEFERRED_ASSIGNMENT', 'DEFERRED_DEFAULT_NAME', 'DEFERRED_INITIALIZATION', 'DETERMINISTIC',
    'DICT', 'DictionaryMemory_FUNCTION', 'DIFFERENCE', 'DIFFERENCE',
    'DIFFUSION', 'DIRECT', 'DISABLE', 'DISABLE_PARAM',
    'DIST_FUNCTION_TYPE', 'DIST_MEAN', 'DIST_SHAPE', 'DISTANCE_FUNCTION', 'DISTANCE_METRICS',
    'DISTRIBUTION_FUNCTION_TYPE', 'DIVISION', 'DOT_PRODUCT',
    'DRIFT_DIFFUSION_INTEGRATOR_FUNCTION', 'DRIFT_ON_A_SPHERE_INTEGRATOR_FUNCTION', 'DROPOUT_FUNCTION',
    'DUAL_ADAPTIVE_INTEGRATOR_FUNCTION',
    'EFFERENTS', 'EID_SIMULATION', 'EID_FROZEN', 'EITHER', 'ENABLE_CONTROLLER', 'ENABLED', 'ENERGY', 'ENTROPY',
    'EM_COMPOSITION', 'EM_STORAGE_FUNCTION', 'EM_STORAGE_MECHANISM', 'EPISODIC_MEMORY_MECHANISM', 'EPOCH', 'EPOCHS',
    'EQUAL', 'ERROR_DERIVATIVE_FUNCTION', 'EUCLIDEAN', 'EVC_MECHANISM', 'EVC_SIMULATION',  'EXAMPLE_FUNCTION_TYPE',
    'EXECUTE_UNTIL_FINISHED', 'EXECUTING',
    'EXECUTION', 'EXECUTION_COUNT', 'EXECUTION_ID', 'EXECUTION_MODE', 'EXECUTION_PHASE',
    'EXPONENTIAL', 'EXPONENT', 'EXPONENTIAL_DIST_FUNCTION', 'EXPONENTIAL_FUNCTION', 'EXPONENTS',
    'FEEDBACK', 'FITZHUGHNAGUMO_INTEGRATOR_FUNCTION', 'FINAL', 'FLAGS', 'FULL', 'FULL_CONNECTIVITY_MATRIX',
    'FUNCTION', 'FUNCTIONS', 'FUNCTION_COMPONENT_CATEGORY','FUNCTION_CHECK_ARGS',
    'FUNCTION_OUTPUT_TYPE', 'FUNCTION_OUTPUT_TYPE_CONVERSION', 'FUNCTION_PARAMS',
    'GAIN', 'GAMMA_DIST_FUNCTION', 'GATE', 'GATING_MECHANISM', 'GATING_ALLOCATION', 'GATING_PROJECTION',
    'GATING_PROJECTION_PARAMS', 'GATING_PROJECTIONS', 'GATING_SIGNAL', 'GATING_SIGNAL_SPECS', 'GATING_SIGNALS',
    'GAUSSIAN', 'GAUSSIAN_FUNCTION', 'GILZENRAT_INTEGRATOR_FUNCTION',
    'GREATER_THAN', 'GREATER_THAN_OR_EQUAL', 'GRADIENT_OPTIMIZATION_FUNCTION', 'GRID_SEARCH_FUNCTION',
    'HARD_CLAMP', 'HEBBIAN_FUNCTION', 'HETERO', 'HIGH', 'HOLLOW_MATRIX', 'IDENTITY_MATRIX', 'INCREMENT', 'INDEX',
    'INIT_EXECUTE_METHOD_ONLY', 'INIT_FULL_EXECUTE_METHOD', 'INIT_FUNCTION_METHOD_ONLY',
    'INITIALIZE', 'INITIALIZED', 'INITIALIZER', 'INITIALIZE_CYCLE', 'INITIALIZE_CYCLE_VALUES',
    'INITIALIZING', 'INITIALIZATION', 'INITIALIZATION_STATUS',
    'INPUT', 'INPUTS', 'INPUT_CIM_NAME', 'INPUT_LABELS_DICT', 'INPUT_PORT', 'INPUT_PORTS', 'INPUT_PORT_PARAMS',
    'INPUT_PORT_VARIABLES', 'INPUTS_DIM', 'INSET', 'CURRENT_VALUE', 'INTEGRATION_TYPE',
    'INTEGRATOR_FUNCTION','INTEGRATOR_FUNCTION', 'INTEGRATOR_FUNCTION_TYPE', 'INTEGRATOR_MECHANISM',
    'LAST_INTEGRATED_VALUE', 'INTERCEPT', 'INTERNAL', 'INTERNAL_ONLY',
    'K_VALUE', 'KOHONEN_FUNCTION', 'KOHONEN_MECHANISM', 'KOHONEN_LEARNING_MECHANISM', 'KWTA_MECHANISM', 'L0',
    'LABELS', 'LCA_MECHANISM', 'LEAKY_COMPETING_INTEGRATOR_FUNCTION', 'LEAK', 'LEARNABLE', 'LEARNED_PROJECTIONS',
    'LEARNING', 'LEARNING_FUNCTION', 'LEARNING_FUNCTION_TYPE', 'LEARNING_OBJECTIVE', 'LEARNING_MECHANISM',
    'LEARNING_MECHANISMS', 'LEARNING_PATHWAY', 'LEARNING_PROJECTION', 'LEARNING_PROJECTION_PARAMS', 'LEARNING_RATE',
    'LEARNING_SCALE', 'LEARNING_SCALE_LITERALS', 'LEARNING_SCALE_NAMES', 'LEARNING_SIGNAL', 'LEARNING_SIGNAL_SPECS',
    'LEARNING_SIGNALS', 'LESS_THAN', 'LESS_THAN_OR_EQUAL', 'LINEAR', 'LINEAR_COMBINATION_FUNCTION', 'LINEAR_FUNCTION',
    'LOG_ENTRIES', 'LOGISTIC_FUNCTION', 'Loss', 'LOSSES', 'LOW', 'LVOC_CONTROL_MECHANISM',
    'MAPPING_PROJECTION', 'MAPPING_PROJECTION_PARAMS', 'MASKED_MAPPING_PROJECTION',
    'MATRIX', 'MATRIX_KEYWORD_NAMES', 'MATRIX_KEYWORD_SET', 'MATRIX_KEYWORD_VALUES', 'MATRIX_KEYWORDS',
    'MatrixKeywords', 'MATRIX_TRANSFORM_FUNCTION',  'MATRIX_WEIGHTS',
    'MAX_ABS_DIFF', 'MAX_ABS_INDICATOR', 'MAX_ONE_HOT', 'MAX_ABS_ONE_HOT', 'MAX_ABS_VAL',
    'MAX_VS_NEXT', 'MAX_VS_AVG', 'MAX_EXECUTIONS_BEFORE_FINISHED', 'MAX_INDICATOR', 'MAX_VAL', 'MAYBE', 'MEAN',
    'MECHANISM', 'MECHANISM_COMPONENT_CATEGORY', 'MECHANISM_DEFAULT', 'MECHANISM_DEFAULT_INPUT_VALUE',
    'MECHANISM_DEFAULTParams', 'MECHANISM_EXECUTED_LOG_ENTRY', 'MECHANISM_NAME', 'MECHANISM_PARAM_VALUE',
    'MECHANISM_TYPE', 'MECHANISM_VALUE', 'MEDIAN', 'METRIC', 'MIN_VAL', 'MIN_ABS_VAL', 'MIN_ABS_INDICATOR', 'MINIBATCH',
    'MOD_AFFERENTS', 'MODE', 'MODULATES','MODULATION', 'MODULATORY_PROJECTION', 'MODULATORY_SIGNAL',
    'MODULATORY_SIGNALS', 'MONITOR', 'MONITOR_FOR_CONTROL', 'MONITOR_FOR_LEARNING', 'MONITOR_FOR_MODULATION',
    'MODEL_SPEC_ID_GENERIC', 'MODEL_SPEC_ID_INPUT_PORTS', 'MODEL_SPEC_ID_OUTPUT_PORTS',
    'MODEL_SPEC_ID_PSYNEULINK', 'MODEL_SPEC_ID_SENDER_MECH', 'MODEL_SPEC_ID_SENDER_PORT',
    'MODEL_SPEC_ID_RECEIVER_MECH', 'MODEL_SPEC_ID_RECEIVER_PORT',
    'MODEL_SPEC_ID_PARAMETER_INITIAL_VALUE', 'MODEL_SPEC_ID_PARAMETER_SOURCE', 'MPS',
    'MODEL_SPEC_ID_PARAMETER_VALUE', 'MODEL_SPEC_ID_TYPE',
    'MULTIPLICATIVE', 'MULTIPLICATIVE_PARAM', 'MUTUAL_ENTROPY',
    'NAME', 'NESTED', 'NEWEST',  'NODE', 'NODES', 'NODE_VALUES', 'NODE_VARIABLES', 'NOISE',
    'NORMAL_DIST_FUNCTION', 'NORMALIZE', 'NORMED_L0_SIMILARITY',
    'NOT_EQUAL', 'NUM_EXECUTIONS_BEFORE_FINISHED',
    'OBJECTIVE_FUNCTION_TYPE', 'OBJECTIVE_MECHANISM', 'OBJECTIVE_MECHANISM_OBJECT', 'OFF', 'OFFSET', 'OLDEST', 'ON',
    'ONLINE', 'ONLY', 'OPERATION', 'OPTIMIZATION_FUNCTION_TYPE', 'OPTIMIZATION_STEP', 'ORIGIN',
    'ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION',
    'OUTCOME', 'OUTCOME_FUNCTION', 'OUTPUT', 'OUTPUTS', 'OUTPUT_CIM_NAME', 'OUTPUT_LABELS_DICT', 'OUTPUT_MECHANISM',
    'OUTPUT_PORT', 'OUTPUT_PORT_PARAMS', 'output_port_spec_to_parameter_name', 'OUTPUT_PORTS', 'OUTPUT_TYPE',
    'OVERRIDE', 'OVERRIDE_PARAM', 'OVERWRITE', 'OWNER', 'OWNER_EXECUTION_COUNT', 'OWNER_EXECUTION_TIME',
    'OWNER_VALUE', 'OWNER_VARIABLE',
    'PARAMETER', 'PARAMETER_CIM_NAME', 'PARAMETER_PORT', 'PARAMETER_PORT_PARAMS', 'PARAMETER_PORTS',
    'PARAMETERS', 'PARAMS', 'PARAMS_DICT', 'PATH_AFFERENTS', 'PATHWAY',  'PATHWAY_PROJECTION', 'PEARSON',
    'PORT', 'PORT_COMPONENT_CATEGORY', 'PORT_CONTEXT', 'Port_Name', 'port_params',
    'PORT_PREFS', 'PORT_TYPE', 'port_value', 'PORTS',
    'PREDICTION_ERROR_DELTA_FUNCTION', 'PREDICTION_MECHANISM', 'PREDICTION_MECHANISMS',
    'PREDICTION_MECHANISM_OUTPUT', 'PREDICTION_MECHANISM_PARAMS', 'PREDICTION_MECHANISM_TYPE',
    'PREFS_ARG', 'PREF_BASE_VALUE', 'PREF_CURRENT_VALUE',
    'PREFERENCE_SET', 'PREFERENCE_SET_NAME', 'PREF_LEVEL', 'PREFS', 'PREFS_OWNER', 'PREVIOUS_VALUE', 'PRIMARY',
    'PROB', 'PROB_INDICATOR', 'PROBABILISTIC',
    'PROCESS', 'PROCESS_COMPONENT_CATEGORY', 'PROCESS_DEFAULT_MECHANISM', 'PROCESS_DEFAULT_PROJECTION_FUNCTION',
    'PROCESS_EXECUTE', 'PROCESS_INIT', 'PROCESSES', 'PROCESSES_DIM', 'PROCESSING', 'PROCESSING_MECHANISM',
    'PROCESSING_PATHWAY', 'PRODUCT', 'PROGRESS_BAR_CHAR', 'PROJECTION', 'PROJECTION_DIRECTION', 'PROJECTION_PARAMS',
    'PROJECTION_RECEIVER', 'PROJECTION_SENDER', 'PROJECTION_TYPE', 'PROJECTIONS', 'PROJECTION_COMPONENT_CATEGORY',
    'QUOTIENT', 'RANDOM', 'RANDOM_CONNECTIVITY_MATRIX', 'RATE', 'RATIO', 'REARRANGE_FUNCTION', 'RECEIVER',
    'RECEIVER_ARG', 'RECURRENT_TRANSFER_MECHANISM', 'REDUCE_FUNCTION', 'REFERENCE_VALUE', 'RESET',
    'RESET_STATEFUL_FUNCTION_WHEN', 'RELU_FUNCTION', 'REST', 'RESULT', 'RESULT', 'ROLES', 'RL_FUNCTION', 'RUN',
    'SAMPLE', 'SAVE_ALL_VALUES_AND_POLICIES', 'SCALAR', 'SCALE', 'SCHEDULER', 'SELF', 'SENDER', 'SEPARATE',
    'SEPARATOR_BAR', 'SHADOW_INPUT_NAME', 'SHADOW_INPUTS', 'SIMPLE', 'SIMPLE_INTEGRATOR_FUNCTION', 'SIMULATIONS',
    'SINGLE', 'SINGLETON', 'INPUT_SHAPES', 'SLOPE', 'SOFT_CLAMP', 'SOFTMAX_FUNCTION', 'SOURCE', 'STABILITY_FUNCTION',
    'STANDARD_ARGS', 'STANDARD_DEVIATION', 'STANDARD_OUTPUT_PORTS', 'STORE', 'SUBTRACTION', 'SUM',
    'TARGET', 'TARGET_MECHANISM', 'TARGET_LABELS_DICT', 'TERMINAL', 'TARGETS',
    'TERMINATION_MEASURE', 'TERMINATION_THRESHOLD', 'TERMINATION_COMPARISION_OP', 'TERSE', 'TEXT', 'THRESHOLD',
    'TIME', 'TIME_STEP_SIZE', 'TIME_STEPS_DIM', 'TRAINED_OUTPUTS', 'TRAINING_SET',
    'TRANSFER_FUNCTION_TYPE', 'TRANSFER_MECHANISM', 'TRANSFER_WITH_COSTS_FUNCTION',
    'TRIAL', 'TRIALS_DIM',
    'UNCHANGED', 'UNIFORM_DIST_FUNCTION', 'UPDATE', 'USER_DEFINED_FUNCTION', 'USER_DEFINED_FUNCTION_TYPE',
    'VALUES', 'VALIDATE', 'VALIDATION', 'VALUE', 'VALUE_ASSIGNMENT', 'VALUE_FUNCTION', 'VARIABLE', 'VARIANCE',
    'VECTOR', 'WALD_DIST_FUNCTION', 'WEIGHT', 'WEIGHTS', 'X_0', 'ZEROS_MATRIX', 'SHARED_COMPONENT_TYPES',
]

# **********************************************************************************************************************
# ******************************************  KEYWORD CLASSES **********************************************************
# **********************************************************************************************************************
import operator
from enum import Enum, auto

from psyneulink._typing import Literal


#region ----------------------------------------- MATRICES   -----------------------------------------------------------

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

    ZEROS_MATRIX
        a matrix that has a number of rows equal to the length of the sender's value, and a number of columns equal
        to the length of the receiver's value, all the elements of which are 0's.

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
        self.ZEROS_MATRIX = ZEROS_MATRIX
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
ZEROS_MATRIX = "ZerosMatrix"
RANDOM_CONNECTIVITY_MATRIX = "RandomConnectivityMatrix"
AUTO_ASSIGN_MATRIX = 'AutoAssignMatrix'
DEFAULT_MATRIX = AUTO_ASSIGN_MATRIX
# DEFAULT_MATRIX = IDENTITY_MATRIX

MatrixKeywordLiteral = Literal["IdentityMatrix", "HollowMatrix", "InverseHollowMatrix", "FullConnectivityMatrix",
                               "RandomConnectivityMatrix", "AutoAssignMatrix"]

MATRIX_KEYWORDS = MatrixKeywords()
MATRIX_KEYWORD_SET = MATRIX_KEYWORDS._set()
MATRIX_KEYWORD_VALUES = MATRIX_KEYWORDS._values()
MATRIX_KEYWORD_NAMES = MATRIX_KEYWORDS._names()
#endregion

#region ----------------------------------------  DISTANCE METRICS  ----------------------------------------------------

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
        (can also be referenced as *L0*)\n
        :math:`d = \\sum\\limits^{len}(|a_1-a_2|)`

    DOT_PRODUCT
        :math:`d = \\sum\\limits^{len}\\(a_1 * a_2)}`

    EUCLIDEAN
        (can also be referenced as *L1*)\n
        :math:`d = \\sum\\limits^{len}\\sqrt{(a_1-a_2)^2}`

    COSINE
        :math:`d = 1 - \\frac{\\sum\\limits^{len}a_1a_2}{\\sqrt{\\sum\\limits^{len}a_1^2}\\sqrt{\\sum\\limits^{len}a_2^2}}`

    CORRELATION
        :math:`d = 1 - \\left|\\frac{\\sum\\limits^{len}(a_1-\\bar{a}_1)(a_2-\\bar{a}_2)}{(len-1)\\sigma_{a_1}\\sigma_{
        a_2}}\\right|`

    CROSS_ENTROPY
        (can also be referenced as *ENTROPY*)\n
        :math:`d = \\sum\\limits^{len}a_1log(a_2)`

    ENERGY:
        :math:`d = -\\frac{1}{2}\\sum\\limits_{i,j}a_{1_i}a_{2_j}w_{ij}`

    """
    def __init__(self):
        self.MAX_ABS_DIFF = MAX_ABS_DIFF
        self.DIFFERENCE = DIFFERENCE
        self.L0 = L0
        self.NORMED_L0_SIMILARITY = NORMED_L0_SIMILARITY
        self.DOT_PRODUCT = DOT_PRODUCT
        self.EUCLIDEAN = EUCLIDEAN
        self.L1 = L1
        self.ANGLE = ANGLE
        self.CORRELATION = CORRELATION
        # self.PEARSON = PEARSON
        self.COSINE = COSINE
        self.COSINE_SIMILARITY = COSINE
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
DOT_PRODUCT = 'dot_product'
EUCLIDEAN = 'euclidean'
L1 = EUCLIDEAN
ANGLE = 'angle'
CORRELATION = 'correlation'
COSINE = 'cosine'
COSINE_SIMILARITY = 'cosine'
PEARSON = 'Pearson'
ENTROPY = 'entropy'
CROSS_ENTROPY = 'cross-entropy'
ENERGY = 'energy'

DISTANCE_METRICS = DistanceMetrics()
DISTANCE_METRICS_SET = DISTANCE_METRICS._set()
DISTANCE_METRICS_VALUES = DISTANCE_METRICS._values()
DISTANCE_METRICS_NAMES = DISTANCE_METRICS._names()

# ENTROPY = 'entropy'
CONVERGENCE = 'CONVERGENCE'
#endregion

#region -------------------------------------------    LEARNING    -----------------------------------------------------


class LearningScale:
    """Scales at which `learning <Composition_Learning>` occurs

    Used to specify the scales over which learning-related events occur when `learning <Composition_Learning>` is
    executed in a `Composition`.

    Attributes
    ----------

    OPTIMIZATION_STEP
        a single step of gradient calculation, of which there can be one or more in a `minibatch
        <LearningScale.minibatch>`, based on a Composition's `mini_batch_size <Composition.mini_batch_size>`
        Parameter.

    TRIAL
        identical to MINIBACH when `minibatch_size <Composition.minibatch_size>`= 1; otherwise a warning is raised,
        and unanticipated results can occur.

    MINIBATCH
        a subset of the training set used to calculate an `error_signal <Composition.error_signal>`
        (i.e. one step along the gradient) used to  and update the weights of a MappingProjection's
        `matrix <MappingProjection.matrix>` Parameter.

    EPOCH
        a complete pass through the training set;  the number of gradient calculations and weight updates that occur
        in an epoch depends on the `mini_batch_size <Composition.mini_batch_size>` and `optimizations_per_minibatch
        <Composition.optimizations_per_minibatch>` Parameters of the Composition.

    RUN
        a complete execution of the `learn <Composition.learn>` method of the Composition, involving
        `num_epochs <Composition.num_epochs>` epochs.

    """
    def __init__(self):
        self.OPTIMIZATION_STEP = OPTIMIZATION_STEP
        self.TRIAL = MINIBATCH
        self.MINIBATCH = MINIBATCH
        self.EPOCH = EPOCH
        self.RUN = RUN

    def _values(self):
        return list(self.__dict__.values())

    def _set(self):
        return set(self.__dict__.values())

    def _names(self):
        return list(self.__dict__)


OPTIMIZATION_STEP = 'optimization_step'
# TRIAL = 'trial'  # defined below in section on Composition
MINIBATCH = 'minibatch'
EPOCH = 'epoch'
RUN = 'run'

LEARNING_SCALE = LearningScale()
LEARNING_SCALE_SET = LEARNING_SCALE._set()
LEARNING_SCALE_VALUES = LEARNING_SCALE._values()
LEARNING_SCALE_NAMES = LEARNING_SCALE._names()
LEARNING_SCALE_LITERALS = Literal[tuple(LEARNING_SCALE_VALUES)] # Used for type hinting


class Loss(Enum):
    """Loss function used for `learning <Composition_Learning>`.

    Used to specify the **loss_spec** argument of the constructor for an `AutodiffComposition`,
    or in the `learning methods <Composition_Learning_Methods>` used to construct `learning pathways
    <Composition_Learning_Pathway>` in a `Composition`. Each keyword specifies a loss function used
    for learning, and the comparable `loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
    used by `PyTorch` when an AutodiffComposition is executed in `ExecutionMode.PyTorch` mode.
    COMMENT:
    Get latex for remaining equations from https://blmoistawinde.github.io/ml_equations_latex/#cross-entropy
    COMMENT

    Attributes
    ----------

    L0
        sum of errors:   :math:`\\sum\\limits^{len}_i|target_i - output_i|`

    COMMENT:
    L1
        mean
    COMMENT

    SSE
        sum of squared errors:    :math:`\\sum\\limits^{len}_i(target_i - output_i)^2`

    MSE
        mean of squared errors:   :math:`\\frac{\\sum\\limits^{len}_i(target_i - output_i)^2}{len}`

    CROSS_ENTROPY
        cross entropy:   :math:`\\sum\\limits^{len}_ioutput_i\\log(target_i)`

    KL_DIV
        `Kullback-Leibler (KL) divergence
        <https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kullback>`_:
        :math:`\\sum\\limits^{len}_itarget_i\\log{(\\frac{target_i}{output_i})}`

    NLL
        `Negative log likelihood loss <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html?highlight=nll>`_:
        :math:`-{\\log(target_i)}`

    POISSON_NLL
        `Poisson negative log likelihood loss <https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html>`_
    """
    L0 = auto()
    L1 = auto()
    SSE = auto()
    MSE = auto()
    CROSS_ENTROPY = auto()
    BINARY_CROSS_ENTROPY = auto()
    KL_DIV = auto()
    NLL = auto()
    POISSON_NLL = auto()
    SUM = L0


LOSSES = 'losses'

#endregion

# **********************************************************************************************************************
# ******************************************    CONSTANTS  *************************************************************
# **********************************************************************************************************************

#region ---------------------------------------------    DEVICES    ----------------------------------------------------
# Devices
CUDA = 'cuda'
CPU = 'cpu'
MPS = 'mps'
#endregion

#region ---------------------------------------------    GENERAL    ----------------------------------------------------
# General


ON = True
OFF = False
DEFAULT = False
# AUTO = True  # MODIFIED 7/14/17 CW
ASSERT = True

# Used by initDirective
INIT_FULL_EXECUTE_METHOD = 'init using the full base class execute method'
INIT_EXECUTE_METHOD_ONLY = 'init using only the subclass _execute method'
INIT_FUNCTION_METHOD_ONLY = 'init using only the subclass __function__ method'


ALL = 'all'
ANY = 'any'
ONLY = 'only'
EITHER = 'either'
BOTH = 'both'
MAYBE = 0.5

SEPARATOR_BAR = ' | '
PROGRESS_BAR_CHAR = '.'
# VALUE_SUFFIX = '_value'
SELF = 'self'
FLAGS = 'flags'
INITIALIZATION_STATUS = 'initialization_status'
EXECUTION_PHASE = 'execution_phase'
EXECUTION_MODE = 'execution_mode'
SOURCE = 'source'
INITIALIZE = "initialize"  # Used as instruction to some methods
INITIALIZING = " INITIALIZING "  # Used as status and context for Log
INITIALIZED = " INITIALIZED "  # Used as status
EXECUTING = " EXECUTING " # Used in context for Log and ReportOutput pref
ASSIGN_VALUE = ': Assign value'
UPDATE = 'update'
STORE = 'store'
VALIDATE = 'Validate'
COMMAND_LINE = "COMMAND_LINE"
CHANGED = 'CHANGED'
UNCHANGED = 'UNCHANGED'
ENABLED = 'ENABLED'
STATEFUL_ATTRIBUTES = 'stateful_attributes'
BOLD = 'bold'
ONLINE = 'online'
COUNT = 'COUNT'
INPUT = 'input'
OUTPUT = 'output'
OUTPUTS = 'outputs'
PARAMETER = 'parameter'
RANDOM = 'random'
FIRST= 'first'
LAST = 'last'
BEFORE = 'before'
AFTER = 'after'
LOW = 'low'
HIGH = 'high'
MAX = 'max'
MIN = 'min'
OLDEST = 'oldest'
NEWEST = 'newest'
FULL = 'full'
TERSE = 'terse'
DIRECT = 'direct'
DICT = 'dict'
TEXT = 'text'

ADD = 'add'
SUBTRACT = 'subtract'
LESS_THAN = '<'
LESS_THAN_OR_EQUAL = '<='
EQUAL = '=='
GREATER_THAN = '>'
GREATER_THAN_OR_EQUAL = '>='
NOT_EQUAL = '!='

comparison_operators = {LESS_THAN : operator.lt,
                        LESS_THAN_OR_EQUAL : operator.le,
                        EQUAL : operator.eq,
                        GREATER_THAN : operator.gt,
                        GREATER_THAN_OR_EQUAL : operator.ge,
                        NOT_EQUAL : operator.ne}

EID_SIMULATION = '-sim'
EID_FROZEN = '-frozen'

#endregion

#region --------------------------------------------    PREFERENCES    -------------------------------------------------

PREFS = "Prefs"
PREFS_OWNER = "PrefsOwner"
PREF_LEVEL = 'PrefLevel'
PREF_CURRENT_VALUE = 'PrefCurrentValue'
PREF_BASE_VALUE = 'PrefBaseValue'
PREFERENCE_SET_NAME = 'PreferenceSetName'
DEFAULT_PREFERENCE_SET_OWNER = 'DefaultPreferenceSetOwner'

PREFERENCE_SET = 'PreferenceSet'
COMPONENT_PREFERENCE_SET = 'BasePreferenceSet'
# COMPONENT_PREFERENCE_SET = 'PreferenceSet'
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

#region ----------------------------------------------   COMPOSITION   -------------------------------------------------

# 11/15/21: FIX - CHANGE TO LOWER CASE FOR USE WITH componentCategory (OR CHANGE THAT?);  MAY NEED TO CHANGE TESTS
# Composition Categories
COMPOSITION = 'Composition'
AUTODIFF_COMPOSITION = 'AutodiffComposition'
AUTODIFF_RESULTS = 'AutodiffResults'
COMPOSITION_FUNCTION_APPROXIMATOR = 'CompositionFunctionApproximator'
EM_COMPOSITION = 'EMComposition'

INPUT_CIM_NAME = 'INPUT_CIM'
OUTPUT_CIM_NAME = 'OUTPUT_CIM'
PARAMETER_CIM_NAME = 'PARAMETER_CIM'
SHADOW_INPUTS = 'shadow_inputs'
SHADOW_INPUT_NAME = 'Shadowed input of '
PATHWAY = "pathway"
PROCESSING_PATHWAY = "processing_pathway"
CONTROL_PATHWAY = "control_pathway"
LEARNING_PATHWAY = "learning_pathway"
NODE = 'NODE'
NODES = 'NODES'
NODE_VARIABLES = 'node_variables'
NODE_VALUES = 'node_values'
INPUTS = 'inputs'
TARGETS = 'targets'
EPOCHS = 'epochs'

# Used in show_graph for show_nested
NESTED = 'nested'
INSET = 'inset'

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
PARAMETERS = 'params'
NAME = "name"
PREFS_ARG = "prefs"
CONTEXT = "context"
STANDARD_ARGS = {NAME, VARIABLE, VALUE, PARAMS, PREFS_ARG, CONTEXT}
EXECUTION_COUNT = 'execution_count' # Total number of executions of a Component
EXECUTE_UNTIL_FINISHED = 'execute_until_finished' # Specifies mode of execution
NUM_EXECUTIONS_BEFORE_FINISHED = 'num_executions_before_finished' # Number of executions since last finished
MAX_EXECUTIONS_BEFORE_FINISHED = 'max_executions_before_finished'

INITIALIZE_CYCLE_VALUES = 'initialize_cycle_values'
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
FUNCTION = "function" # Parameter name for function, method, or type to instantiate and assign to self.execute
FUNCTION_PARAMS  = "function_params" # Parameters used to instantiate or assign to a FUNCTION

FUNCTION_CHECK_ARGS = 'super._check_args' # Use for "context" arg
FUNCTION_OUTPUT_TYPE_CONVERSION = "enable_output_type_conversion"  # Used in Function Components to set output type

#endregion

#region ----------------------------------------    COMPONENT SUBCLASSES  ----------------------------------------------

# Component Categories   -----------------

PROCESS_COMPONENT_CATEGORY = "Process"
MECHANISM_COMPONENT_CATEGORY = "Mechanism_Base"
PORT_COMPONENT_CATEGORY = "Port_Base"
PROJECTION_COMPONENT_CATEGORY = "Projection_Base"
FUNCTION_COMPONENT_CATEGORY = "Function_Base"

# Component TYPES  -----------------

# Mechanisms:
PROCESSING_MECHANISM = "ProcessingMechanism"
ADAPTIVE_MECHANISM = "ModulatoryMechanism"
LEARNING_MECHANISM = "LearningMechanism"
CONTROL_MECHANISM = "ControlMechanism"
GATING_MECHANISM = 'GatingMechanism'
AUTOASSOCIATIVE_LEARNING_MECHANISM = 'AutoAssociativeLearningMechanism'
EM_STORAGE_MECHANISM = 'EMStorageMechanism'
KOHONEN_LEARNING_MECHANISM = 'KohonenLearningMechanism'

# Ports:
INPUT_PORT = "InputPort"
PARAMETER_PORT = "ParameterPort"
OUTPUT_PORT = "OutputPort"
MODULATORY_SIGNAL = 'ModulatorySignal'
LEARNING_SIGNAL = 'LearningSignal'
CONTROL_SIGNAL = 'ControlSignal'
GATING_SIGNAL = 'GatingSignal'

PATH_AFFERENTS = 'path_afferents'
MOD_AFFERENTS = 'mod_afferents'
EFFERENTS = 'efferents'

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
EPISODIC_MEMORY_MECHANISM = "EpisodicMemoryMechanism"
COMPOSITION_INTERFACE_MECHANISM = "CompositionInterfaceMechanism"
PROCESSING_MECHANISM = "ProcessingMechanism"

# Functions:
ARGUMENT_THERAPY_FUNCTION = "Contradiction Function"
USER_DEFINED_FUNCTION = "USER DEFINED FUNCTION"

# Transformfunctions:
CONCATENATE_FUNCTION = "Concatenate Function"
REARRANGE_FUNCTION = 'Rearrange Function'
REDUCE_FUNCTION = "Reduce Function"
LINEAR_COMBINATION_FUNCTION = "LinearCombination Function"
COMBINE_MEANS_FUNCTION = "CombineMeans Function"
MATRIX_TRANSFORM_FUNCTION = "MatrixTransform Function"

# TransferFunctions:
IDENTITY_FUNCTION = 'Identity Function'
LINEAR_FUNCTION = "Linear Function"
LEABRA_FUNCTION = "Leabra Function"
EXPONENTIAL_FUNCTION = "Exponential Function"
LOGISTIC_FUNCTION = "Logistic Function"
TANH_FUNCTION = "Tanh Function"
RELU_FUNCTION = "ReLU Function"
ANGLE_FUNCTION = 'Angle Function'
GAUSSIAN_FUNCTION = "Gaussian Function"
GAUSSIAN_DISTORT_FUNCTION = "GaussianDistort Function"
BINOMIAL_DISTORT_FUNCTION = 'BinomialDistort Function'
DROPOUT_FUNCTION = 'Dropout Function'
SOFTMAX_FUNCTION = 'SoftMax Function'
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
DRIFT_ON_A_SPHERE_INTEGRATOR_FUNCTION = "DriftOnASphereIntegrator Function"
ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION = "OU IntegratorFunction Function"

# MemoryFunctions:
BUFFER_FUNCTION = 'Buffer Function'
DictionaryMemory_FUNCTION = 'DictionaryMemory Function'
ContentAddressableMemory_FUNCTION = 'ContentAddressableMemory Function'

# OptimizationFunctions:
GRADIENT_OPTIMIZATION_FUNCTION = "GradientOptimization Function"
GRID_SEARCH_FUNCTION = 'GridSearch Function'

# LearningFunctions:
EM_STORAGE_FUNCTION = "EMStorage Function"
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
PORT_MAP_FUNCTION = 'Port Map Function'

#endregion

#region -------------------------------------------    COMPOSITION   ---------------------------------------------------

SCHEDULER = "scheduler"
CONTROLLER = "controller"
ENABLE_CONTROLLER = "enable_controller"

RUN = 'run'
TRIAL = 'trial'

ROLES = 'roles'
ALLOW_PROBES = 'allow_probes'
CONDITIONS = 'conditions'
VALUES = 'values'
FUNCTIONS = 'functions'

#endregion

#region ------------------------------------------  AUTODIFF COMPOSITION  ----------------------------------------------

LEARNING_RATE = "learning_rate"
TRAINING_SET = 'training set'
TRAINED_OUTPUTS = 'trained_outputs'

#endregion

#region ----------------------------------------------    PROCESS   ----------------------------------------------------

PROCESS = "PROCESS"
PROCESSES = "processes"
PROCESS_INIT = 'Process.__init__'
CLAMP_INPUT = "clamp_input"
SOFT_CLAMP = "soft_clamp"
HARD_CLAMP = "hard_clamp"
PULSE_CLAMP = "pulse_clamp"
NO_CLAMP = "no_clamp"
# CONTROL = 'CONTROL'
PROCESS_DEFAULT_PROJECTION_FUNCTION = "Default Projection Function"
PROCESS_EXECUTE = "ProcessExecute"
MECHANISM_EXECUTED_LOG_ENTRY = "Mechanism Executed"
#endregion

#region ---------------------------------------------    MECHANISM   ---------------------------------------------------

MECHANISM = 'MECHANISM'
MECHANISMS = 'MECHANISMS'
MECHANISM_NAME = "MECHANISM NAME"
MECHANISM_DEFAULT = "DEFAULT MECHANISM"
DEFAULT_PROCESSING_MECHANISM = "DefaultProcessingMechanism"
PROCESS_DEFAULT_MECHANISM = "ProcessDefaultMechanism"
MECHANISM_TYPE = "Mechanism Type" # Used in mechanism dict specification (e.g., in process.pathway[])
MECHANISM_DEFAULT_INPUT_VALUE = "Mechanism Default Input Value " # Used in mechanism specification dict
MECHANISM_PARAM_VALUE = "Mechanism Parameter Value"                 # Used to specify mechanism param value
MECHANISM_DEFAULTParams = "Mechanism Default Parameters"           # Used in mechanism specification dict
CONDITION = 'condition'

# Keywords for OUTPUT_PORT_VARIABLE dict:
OWNER_VARIABLE = 'OWNER_VARIABLE'
OWNER_VALUE = 'OWNER_VALUE'
OWNER_EXECUTION_COUNT = EXECUTION_COUNT
OWNER_EXECUTION_TIME = 'EXECUTION_TIME'
INPUT_PORT_VARIABLES = 'INPUT_PORT_VARIABLES'
PARAMS_DICT = 'PARAMS_DICT'

# this exists because keywords like OWNER_VALUE are set as properties
# on Mechanisms, so you can't just change the string value to map
# as you want - the property "value" will be overwritten then
output_port_spec_to_parameter_name = {
    OWNER_VARIABLE: VARIABLE,
    OWNER_VALUE: VALUE,
    INPUT_PORT_VARIABLES: 'input_port_variables',
    OWNER_EXECUTION_COUNT: EXECUTION_COUNT,
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
CONTROLLER_OBJECTIVE = 'CONTROLLER_OBJECTIVE'
LEARNING_OBJECTIVE = 'LEARNING_OBJECTIVE'

RESULTS = 'RESULTS'
RESULT = 'RESULT'
MEAN = 'MEAN'
MEDIAN = 'MEDIAN'
MECHANISM_VALUE = 'MECHANISM_VALUE'
INPUT_SHAPES = 'input_shapes'
K_VALUE = 'k_value'
RATIO = 'ratio'

THRESHOLD = 'threshold'
TERMINATION_MEASURE = 'termination_measure'
TERMINATION_THRESHOLD = 'termination_threshold'
TERMINATION_COMPARISION_OP = 'termination_comparison_op'

port_value = "Port value"   # Used in Port specification dict to specify Port value
port_params = "Port params" # Used in Port specification dict to specify Port params

OWNER_MECH = 'owner_mech'
#endregion

#region ----------------------------------------    MODULATORY MECHANISMS ----------------------------------------------

# LearningMechanism:
LEARNING_SIGNALS = 'learning_signals'
LEARNING_SIGNAL_SPECS = 'LEARNING_SIGNAL_SPECS'
LEARNING_FUNCTION = 'learning_function'
LEARNED_PROJECTIONS = 'LEARNED_PROJECTIONS'
LEARNING_MECHANISMS = "LEARNING_MECHANISMS"
OUTPUT_MECHANISM = "OUTPUT_MECHANISM"
TARGET_MECHANISM = "TARGET_MECHANISM"

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
CONTROL = 'control'
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
EVC_SIMULATION = 'CONTROL SIMULATION'
ALLOCATION_SAMPLES = "allocation_samples"

# GatingMechanism
GATE = 'gate'
GATING_SIGNALS = 'gating_signals'
GATING_SIGNAL_SPECS = 'GATING_SIGNAL_SPECS'
GATED_PORTS = 'GATED_PORTS'
GATING_PROJECTIONS = 'GatingProjections'
GATING_ALLOCATION = 'gating_allocation'

MODULATORY_SPEC_KEYWORDS = {LEARNING, LEARNING_SIGNAL, LEARNING_PROJECTION, LEARNING_MECHANISM,
                            CONTROL, CONTROL_SIGNAL, CONTROL_PROJECTION, CONTROL_MECHANISM,
                            GATE, GATING_SIGNAL, GATING_PROJECTION, GATING_MECHANISM}

MODULATED_PARAMETER_PREFIX = 'mod_'

#endregion

#region ----------------------------------------------    PORTS  ------------------------------------------------------

PORT = "Port"
PORT_TYPE = "port_type"
# These are used as keys in Port specification dictionaries
PORTS = "PORTS"
MODULATES = "modulates"
PROJECTIONS = "projections"  # Used to specify projection list to Port DEPRECATED;  REPLACED BY CONNECTIONS
CONNECTIONS = 'CONNECTIONS'
Port_Name = "PortName"
PORT_PREFS = "PortPrefs"
PORT_CONTEXT = "PortContext"
ADD_INPUT_PORT = 'AddNewInputPort'     # Used by Mechanism._add_projection_to()
ADD_OUTPUT_PORT = 'AddNewOutputPort'   # Used by Mechanism._add_projection_from()
OWNER = 'owner'
REFERENCE_VALUE = 'reference_value'
REFERENCE_VALUE_NAME = 'reference_value_name'

# InputPorts:
PRIMARY = 'Primary'
INPUT_PORTS = 'input_ports'
INPUT_PORT_PARAMS = 'input_port_params'
WEIGHT = 'weight'
EXPONENT = 'exponent'
INTERNAL_ONLY = 'internal_only'
DEFAULT_INPUT = 'default_input'

# ParameterPorts:
PARAMETER_PORTS = 'parameter_ports'
PARAMETER_PORT_PARAMS = 'parameter_port_params'

# OutputPorts:
OUTPUT_PORTS = 'output_ports'
OUTPUT_PORT_PARAMS = 'output_ports_params'
STANDARD_OUTPUT_PORTS = 'standard_output_ports'
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
PROJECTION_PARAMS = "PROJECTION_PARAMS"
MAPPING_PROJECTION_PARAMS = 'MAPPING_PROJECTION_PARAMS'
LEARNING_PROJECTION_PARAMS = 'LEARNING_PROJECTION_PARAMS'
CONTROL_PROJECTION_PARAMS = "CONTROL_PROJECTION_PARAMS"
GATING_PROJECTION_PARAMS = 'GATING_PROJECTION_PARAMS'


SENDER = 'sender'
RECEIVER = "receiver"
PROJECTION_SENDER = 'projection_sender'
PROJECTION_RECEIVER = 'projection_receiver'
PROJECTION_DIRECTION = {SENDER: 'to',
                        RECEIVER: 'from'}
RECEIVER_ARG = 'receiver'
FEEDBACK = 'feedback'
MONITOR_FOR_LEARNING = 'monitor_for_learning'
LEARNABLE = 'learnable'
MATRIX_WEIGHTS = 'matrix_weights'
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
RESET = "reset"
RESET_STATEFUL_FUNCTION_WHEN = "reset_stateful_function_when"

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
LAST_INTEGRATED_VALUE = "last_integrated_value"
CURRENT_VALUE = "current_value"
LINEAR = 'linear'
CONSTANT = 'constant'
SIMPLE = 'scaled'
ADAPTIVE = 'adaptive'
DIFFUSION = 'diffusion'
EXPONENTIAL = 'exponential'
GAUSSIAN = 'gaussian'
SINUSOID = 'sinusoid'

COMBINE = 'combine'
CONCATENATE = 'concatenate'
SEPARATE = 'separate'
SUM = 'sum'
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
DIST_MEAN = 'mean'

SINGLE = 'single'
WEIGHTED = 'weighted'

# Note:  These are used as both arg names (hence lower case) and names of StandardOutputPorts
STANDARD_DEVIATION = 'standard_deviation'
VARIANCE = 'variance'

DETERMINISTIC = 'deterministic'
PROBABILISTIC = 'probabilistic'

ARG_MAX = 'arg_max'
ARG_MAX_ABS = 'arg_max_abs'
ARG_MAX_INDICATOR = 'arg_max_indicator'
ARG_MAX_ABS_INDICATOR = 'arg_max_abs_indicator'
ARG_MIN = 'arg_min'
ARG_MIN_ABS = 'arg_min_abs'
ARG_MIN_INDICATOR = 'arg_min_indicator'
ARG_MIN_ABS_INDICATOR = 'arg_min_abs_indicator'
MAX_VS_NEXT = 'max_vs_next'
MAX_VS_AVG = 'max_vs_avg'

# Note:  These are used only as names of StandardOutputPorts (hence upper case)
MAX_VAL = 'MAX_VAL'
MAX_ABS_VAL = 'MAX_ABS_VAL'
MAX_ONE_HOT = 'MAX_ONE_HOT'
MAX_ABS_ONE_HOT = 'MAX_ABS_ONE_HOT'
MAX_INDICATOR = 'MAX_INDICATOR'
MAX_ABS_INDICATOR = 'MAX_ABS_INDICATOR'
MIN_VAL = 'MIN_VAL'
MIN_ABS_VAL = 'MIN_ABS_VAL'
MIN_INDICATOR = 'MIN_INDICATOR'
MIN_ABS_INDICATOR = 'MIN_ABS_INDICATOR'
PROB = 'PROB'
PROB_INDICATOR = 'PROB_INDICATOR'
MUTUAL_ENTROPY = 'MUTUAL_ENTROPY'
PER_ITEM = 'per_item'

INITIALIZER = 'initializer'
INITIAL_V = 'initial_v'
INITIAL_W = 'initial_w'
WEIGHTS = "weights"
EXPONENTS = "exponents"
OPERATION = "operation"
OFFSET = "offset"
NORMALIZE = "normalize"

REWARD = 'reward'
NETWORK = 'network'

GAMMA = 'gamma'

#endregion

# model spec keywords
MODEL_SPEC_ID_TYPE = 'type'
MODEL_SPEC_ID_PSYNEULINK = 'PNL'
MODEL_SPEC_ID_GENERIC = 'generic'
MODEL_SPEC_ID_METADATA = 'metadata'

MODEL_SPEC_ID_INPUT_PORTS = 'input_ports'
MODEL_SPEC_ID_OUTPUT_PORTS = 'output_ports'

MODEL_SPEC_ID_SENDER_MECH = 'sender'
MODEL_SPEC_ID_SENDER_PORT = 'sender_port'
MODEL_SPEC_ID_RECEIVER_MECH = 'receiver'
MODEL_SPEC_ID_RECEIVER_PORT = 'receiver_port'

MODEL_SPEC_ID_PARAMETER_SOURCE = 'source'
MODEL_SPEC_ID_PARAMETER_VALUE = 'value'
MODEL_SPEC_ID_PARAMETER_INITIAL_VALUE = 'default_initial_value'

MODEL_SPEC_ID_MDF_VARIABLE = 'variable0'

MODEL_SPEC_ID_SHAPE = 'shape'

MODEL_SPEC_ID_INPUT_PORT_COMBINATION_FUNCTION = 'input_combination_function'

SHARED_COMPONENT_TYPES = 'shared_component_types'
