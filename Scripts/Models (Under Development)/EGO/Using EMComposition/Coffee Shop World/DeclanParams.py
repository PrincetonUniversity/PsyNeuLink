"""
DECLAN Params: **************************************************************************
√ episodic_lr = 1  # learning rate for the episodic pathway
√ temperature = 0.1  # temperature for EM retrieval (lower is more argmax-like)
√ n_optimization_steps = 10  # number of update steps
sim_thresh = 0.8 # threshold for discarding bad seeds -- can probably ignore this for now
Filter runs whose context representations are too uniform (i.e. not similar to "checkerboard" foil)

May need to pad the context reps because there will be 999 reps
def filter_run(run_em, thresh=0.8):
    foil = np.zeros([4,4])
    foil[::2, ::2] = 1
    foil[1::2, 1::2] = 1
    run_em = run_em.reshape(200, 5, 11).mean(axis=1)
    mat = cosine_similarity(run_em, run_em)
    vec = mat[:160, :160].reshape(4, 40, 4, 40).mean(axis=(1, 3)).ravel()
    return cosine_similarity(foil.reshape(1, -1), vec.reshape(1, -1))[0][0]

# Stack the model predictions (should be 999x11), pad with zeros, and reshape into trials for averaging.
em_preds = np.vstack([em_preds, np.zeros([1,11])]).reshape(-1,5,11)

# Stack the ground truth states (should be 999x11), pad with zeros, and reshape into trials for averaging.
ys = np.vstack([data_loader.dataset.ys.cpu().numpy(), np.zeros([1,11])]).reshape(-1,5,11)

# compute the probability as a performance metric
def calc_prob(em_preds, test_ys):
    em_preds, test_ys = em_preds[:, 2:-1, :], test_ys[:, 2:-1, :]
    em_probability = (em_preds*test_ys).sum(-1).mean(-1)
    trial_probs = (em_preds*test_ys)
    return em_probability, trial_probs

Calculate the retrieval probability of the correct response as a performance metric (probs)
probs, trial_probs = calc_prob(em_preds, test_ys)
"""
from psyneulink.core.llvm import ExecutionMode
from psyneulink.core.globals.keywords import ALL, ADAPTIVE, CONTROL, CPU, Loss, MPS, OPTIMIZATION_STEP, RUN, TRIAL

model_params = dict(

    # Names:
    name = "EGO Model CSW",
    em_name = "EM",
    state_input_layer_name = "STATE",
    previous_state_layer_name = "PREVIOUS STATE",
    context_layer_name = 'CONTEXT',
    prediction_layer_name = "PREDICTION",

    # Structural
    state_d = 11, # length of state vector
    previous_state_d = 11, # length of state vector
    context_d = 11, # length of context vector
    memory_capacity = ALL, # number of entries in EM memory; ALL=> match to number of stims
    memory_init = (0,.0001),  # Initialize memory with random values in interval
    # memory_init = None,  # Initialize with zeros
    # concatenate_queries = False,
    concatenate_queries = True,

    # environment
    # curriculum_type = 'Interleaved',
    curriculum_type = 'Blocked',
    # num_stims = 100,  # Integer or ALL
    num_stims = ALL,  # Integer or ALL

    # Processing
    integration_rate = .69, # rate at which state is integrated into new context
    # state_weight =normalize_field_weightsnormalize_field_weights 1, # weight of the state used during memory retrieval
    # context_weight = 1, # weight of the context used during memory retrieval
    previous_state_weight = .5, # weight of the state used during memory retrieval
    context_weight = .5, # weight of the context used during memory retrieval
    state_weight = None, # weight of the state used during memory retrieval
    # normalize_field_weights = False, # whether to normalize the field weights during memory retrieval
    normalize_field_weights = True, # whether to normalize the field weights during memory retrieval
    normalize_memories = False, # whether to normalize the memory during memory retrieval
    # normalize_memories = True, # whether to normalize the memory during memory retrieval
    # softmax_temperature = None, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    softmax_temperature = .1, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_temperature = ADAPTIVE, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_temperature = CONTROL, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_threshold = None, # threshold used to mask out small values in softmax
    softmax_threshold = .001, # threshold used to mask out small values in softmax
    # target_fields=[True, False, False], # Enable learning for PREDICTION (STATE) but not CONTEXT or PREVIOUS STATE
    enable_learning = True,
    loss_spec = Loss.BINARY_CROSS_ENTROPY,
    # loss_spec = Loss.MSE,
    learning_rate = .5,
    # num_optimization_steps = 1,
    num_optimization_steps = 10,
    synch_weights = RUN,
    synch_values = RUN,
    synch_results = RUN,
    # execution_mode = ExecutionMode.Python,
    execution_mode = ExecutionMode.PyTorch,
    device = CPU,
    # device = MPS,
)
#endregion