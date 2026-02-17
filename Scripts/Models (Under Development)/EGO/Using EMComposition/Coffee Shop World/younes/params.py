from psyneulink import *

STATE_D = 11  # dimensionality of the state input
CONTEXT_D = 11  # dimensionality of the learned context representations
OUTPUT_D = 11

MEMORY_INIT = 0.01  # initial value for the memory entries

SOFTMAX_THRESHOLD = 1e-10
INTEGRATION_RATE = 0.5
NUM_OPTIM_STEPS = 10
LEARNING_RATE = 0.2

TEMPERATURE = 1.

params_data = dict(
    n_participants=1,
    probs=[1, 1, 1],
    seed=0,
)

params_torch = dict(
    state_d=STATE_D,
    context_d=CONTEXT_D,
    output_d=OUTPUT_D,

    memory_init=MEMORY_INIT,  # initial value for the memory entries

    learning_rate=LEARNING_RATE,  # learning rate for the episodic pathway
    softmax_threshold=SOFTMAX_THRESHOLD,

    integration_rate=INTEGRATION_RATE,

    temperature=TEMPERATURE,  # temperature for EM retrieval (lower is more argmax-like)
    n_optimization_steps=NUM_OPTIM_STEPS,  # number of optimization steps to take for each state
)

params_ego = dict(

    # Names:
    name="EGO Model CSW",
    em_name="EM",
    state_input_layer_name="STATE",
    previous_state_layer_name="PREVIOUS STATE",
    context_layer_name='CONTEXT',
    prediction_layer_name="PREDICTION",

    # Structural
    state_d=STATE_D,  # length of state vector
    previous_state_d=STATE_D,  # length of state vector
    context_d=CONTEXT_D,  # length of context vector
    memory_capacity=ALL,  # number of entries in EM memory; ALL=> match to number of stims
    memory_init=MEMORY_INIT,  # .001,  # Initialize memory with random values in interval
    concatenate_queries=False,  # whether to concatenate queries before retrieval

    # environment
    curriculum_type='Blocked',
    num_stims=ALL,  # Integer or ALL

    # Processing
    integration_rate=INTEGRATION_RATE,  # don't change this
    previous_state_weight=1.,  # weight of the state used during memory retrieval
    context_weight=1.,  # weight of the context used during memory retrieval
    state_weight=None,  # weight of the state used during memory retrieval
    normalize_field_weights=False,  # whether to normalize the field weights during memory retrieval
    normalize_memories=False,  # whether to normalize the memory during memory retrieval
    softmax_temperature=TEMPERATURE,
    # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    softmax_threshold=SOFTMAX_THRESHOLD,  # threshold used to mask out small values in softmax
    enable_learning=True,
    loss_spec=Loss.BINARY_CROSS_ENTROPY,
    learning_rate=LEARNING_RATE,
    num_optimization_steps=NUM_OPTIM_STEPS,
    synch_weights=RUN,
    synch_values=RUN,
    synch_results=RUN,
    execution_mode=ExecutionMode.PyTorch,  # Use PyTorch for execution
    device=CPU,
)
