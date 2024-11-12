from psyneulink.core.llvm import ExecutionMode
from psyneulink.core.globals.keywords import ALL, ADAPTIVE, CONTROL, CPU, Loss, MPS, OPTIMIZATION_STEP, RUN, TRIAL

model_params = dict(

    # Names:
    name = "EGO Model CSW",
    state_input_layer_name = "STATE",
    previous_state_layer_name = "PREVIOUS STATE",
    context_layer_name = 'CONTEXT',
    em_name = "EM",
    prediction_layer_name = "PREDICTION",

    # Structural
    state_d = 11, # length of state vector
    previous_state_d = 11, # length of state vector
    context_d = 11, # length of context vector
    memory_capacity = ALL, # number of entries in EM memory; ALL=> match to number of stims
    memory_init = (0,.0001),  # Initialize memory with random values in interval
    # memory_init = None,  # Initialize with zeros
    concatenate_queries = False,
    # concatenate_queries = True,

    # environment
    # curriculum_type = 'Interleaved',
    curriculum_type = 'Blocked',
    num_stims = 7,  # Integer or ALL
    # num_stims = ALL,  # Integer or ALL

    # Processing
    integration_rate = .69, # rate at which state is integrated into new context
    state_weight = 1, # weight of the state used during memory retrieval
    context_weight = 1, # weight of the context used during memory retrieval
    # normalize_field_weights = False, # whether to normalize the field weights during memory retrieval
    normalize_field_weights = True, # whether to normalize the field weights during memory retrieval
    # softmax_temperature = None, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    softmax_temperature = .1, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_temperature = ADAPTIVE, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_temperature = CONTROL, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_threshold = None, # threshold used to mask out small values in softmax
    softmax_threshold = .001, # threshold used to mask out small values in softmax
    enable_learning=[True, False, False], # Enable learning for PREDICTION (STATE) but not CONTEXT or PREVIOUS STATE
    # enable_learning=[True, True, True]
    # enable_learning=True,
    # enable_learning=False,
    learn_field_weights = True,
    # learn_field_weights = False,
    loss_spec = Loss.BINARY_CROSS_ENTROPY,
    # loss_spec = Loss.CROSS_ENTROPY,
    # loss_spec = Loss.MSE,
    learning_rate = .5,
    num_optimization_steps = 10,
    synch_weights = RUN,
    synch_values = RUN,
    synch_results = RUN,
    execution_mode = ExecutionMode.Python,
    # execution_mode = ExecutionMode.PyTorch,
    device = CPU,
    # device = MPS,
)
#endregion