import random

STATE_SIZE = 7  # length of state vector (number of unique states)
CONTEXT_SIZE = STATE_SIZE  # length of the context vector (usually the same as state vector)

REWARD_SIZE = 1

# Params for drift on a sphere
TIME_SIZE = 25  # length of the time vector (arbitrary choice, the more dimensions, the less likely `wrap-around`)
TIME_DRIFT_RATE = 0.  # constant drift in one "direction"
TIME_DRIFT_NOISE = 0.1  # Brownian motion noise on the sphere

# MODEL parameters

# retrieval weight (weight the matches by these weights when retrieving from memory)
TIME_RETRIEVAL_WEIGHT = 0.1  # .1#.03#.2
REWARD_RETRIEVAL_WEIGHT = None

# integration rates (how much to integrate old context, state and retrieved context into the new context)
STATE_INTEGRATION_RATE = .6  # .45

TEMPERATURE = .05  # temperature of the softmax used during memory retrieval (smaller means more argmax-like)
MEMORY_INIT = .001
SOFTMAX_THRESHOLD = .001

# SIMULATION parameters
N_PARTICIPANTS = 10  # 58 # 58 # number of participants to simulate
N_BASELINE_TRIALS = 20  # number of baseline trials per participant (one sequence = one trial of each stimulus sequence)
N_REVALUATION_TRIALS = 20  # number of revaluation trials per participant (one sequence = one trial of each stimulus sequence)
N_EXPERIENCE_SEQS = 3 * N_BASELINE_TRIALS + 2 * N_REVALUATION_TRIALS

N_SIMULATIONS = 100  # number of rollouts per participant
N_STEPS = 3  # number of steps per rollout

REWARD_BASELINE_1 = 10
REWARD_BASELINE_2 = 1  # 1

RANDOM_SEED = None  # 1234

# Only "nob" to replicate the full behavior (how much to use retrieved vs simulated context)
MODEL_BASED_NESS = 0.
