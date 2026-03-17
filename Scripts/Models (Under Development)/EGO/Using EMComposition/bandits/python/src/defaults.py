from src.phase_config import PhaseConfig

# ===== DEFAULT CONFIGURATION ==== #

STATE_SIZE = 7  # length of state vector (number of unique states)
CONTEXT_SIZE = STATE_SIZE  # length of the context vector (usually the same as state vector)

REWARD_SIZE = 1
TASK_SIZE = 1  # (only used in PNL version)

# === DRIFT ON A SPHERE === #
# Used to simulate time: the higher noise, the less 2 subsequent time points are correlated
TIME_SIZE = 25  # length of the time vector (arbitrary choice, the more dimensions, the less likely `wrap-around`)
TIME_DRIFT_RATE = 0.  # constant drift in one "direction"
TIME_DRIFT_NOISE = .05  # Brownian motion noise on the sphere .2

# === MODEL === #
# time retrieval weight (influences how fast to relearn)
TIME_RETRIEVAL_WEIGHT = .6
# state integration rate (how fast working memory is updated)
STATE_INTEGRATION_RATE = .8  # .6

TEMPERATURE = .02  # temperature of the softmax used during memory retrieval (smaller means more argmax-like)
MEMORY_INIT = 1.
SOFTMAX_THRESHOLD = .01  # 80
N_STEPS = 3  # number of steps per rollout (not used in PNL version, use 3 for equal match)

# how much to retrieve context from memory (instead of simulating it) in rollout.
# This can simulate "model free" vs "model based" like behavior:
# - "model free" (0, no retrieval of context)
# - "model based"  (1, use retrieved context)
CONTEXT_RETRIEVE_IN_SIM = 0.
MODEL_BASED_NESS = CONTEXT_RETRIEVE_IN_SIM  # Alias used in python version

# Metric to use for match (only in python version) In pnl: dot_product
METRIC = 'dot_product'
# Metric to use for sample mode (only in python version) In pnl: softmax
SAMPLE_MODE = 'softmax'
# Strategy to combine retrieval matches: Multiplicative (AND) or Additive (OR)
RETRIEVAL_STRATEGY = 'additive'

# === EXPERIMENT === #
N_BASELINE_TRIALS = 20  # number of baseline trials per participant (one sequence = one trial of each stimulus sequence)
N_REVALUATION_TRIALS = 20  # number of revaluation trials per participant (one sequence = one trial of each stimulus sequence)
N_EXPERIENCE_SEQS = 3 * N_BASELINE_TRIALS + 2 * N_REVALUATION_TRIALS

REWARD_BASELINE_1 = 10  # reward state 1 in baseline (10 in original)
REWARD_BASELINE_2 = 1  # reward state 2 in baseline (1 in original)

# --- Baseline Phase -- #
STATE_SEQ_1_BASELINE = [1, 3, 5]  # Stimulus sequence 1
STATE_SEQ_2_BASELINE = [2, 4, 6]  # Stimulus sequence 2

REWARD_MAPPING_BASELINE = {
    1: 0, 3: 0, 5: REWARD_BASELINE_1,  # Rewards for sequence 1 / stimulus 5 is rewarded with 10
    2: 0, 4: 0, 6: REWARD_BASELINE_2  # Rewards for sequence 2 / stimulus 6 is rewarded with 1
}

BASELINE_PHASE = PhaseConfig(
    state_seq_1=STATE_SEQ_1_BASELINE,
    state_seq_2=STATE_SEQ_2_BASELINE,
    num_sequences=N_BASELINE_TRIALS,
    reward_by_state=REWARD_MAPPING_BASELINE,
)

# --- Reward Reval Phase --- #
STATE_SEQ_1_REWARD_REVAL = STATE_SEQ_1_BASELINE[1:]  # Stimulus sequence (no change but exclude first state)
STATE_SEQ_2_REWARD_REVAL = STATE_SEQ_2_BASELINE[1:]  # Stimulus sequence (no change but exclude first state)

REWARD_MAPPING_REWARD_REVAL = {
    1: 0, 3: 0, 5: REWARD_BASELINE_2,  # Rewards for sequence 1 / stimulus 5 is rewarded with 1 (changed from 10)
    2: 0, 4: 0, 6: REWARD_BASELINE_1  # Rewards for sequence 2 / stimulus 6 is rewarded with 10 (changed from 1)
}

REWARD_REVAL_PHASE = PhaseConfig(
    state_seq_1=STATE_SEQ_1_REWARD_REVAL,
    state_seq_2=STATE_SEQ_2_REWARD_REVAL,
    num_sequences=N_REVALUATION_TRIALS,
    reward_by_state=REWARD_MAPPING_REWARD_REVAL,
)

# --- Transition Reval Phase --- #
STATE_SEQ_1_TRANSITION_REVAL = [3, 6]  # Stimulus sequence 1 (changed transition 3->6 )
STATE_SEQ_2_TRANSITION_REVAL = [4, 5]  # Stimulus sequence 2 (changed transition 4->5)

REWARD_MAPPING_TRANSITION_REVAL = REWARD_MAPPING_BASELINE  # Rewards remain the same as baseline

TRANSITION_REVAL_PHASE = PhaseConfig(
    state_seq_1=STATE_SEQ_1_TRANSITION_REVAL,
    state_seq_2=STATE_SEQ_2_TRANSITION_REVAL,
    num_sequences=N_REVALUATION_TRIALS,
    reward_by_state=REWARD_MAPPING_TRANSITION_REVAL,
)

# === SIMULATION === #
N_PARTICIPANTS = 1  # number of participants to simulate (58 in original)
N_SIMULATIONS = 1  # number of simulations per participant (with fixed trial sequences)
RANDOM_SEED = None
