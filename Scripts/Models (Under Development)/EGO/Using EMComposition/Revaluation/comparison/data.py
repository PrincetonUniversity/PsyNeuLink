"""
Data generation for comparison experiments
"""

from typing import Optional
from functools import partial

import numpy as np
import psyneulink as pnl

from comparison.params import *
from comparison.utils import one_hot_encode

import torch
import comparison.utils as utils

# **** Experiment parameters **** #

# * Baseline experiment parameters * #

STIM_SEQ_1_BASELINE = [1, 3, 5]  # Stimulus sequence 1
STIM_SEQ_2_BASELINE = [2, 4, 6]  # Stimulus sequence 2

REWARD_MAPPING_BASELINE = {
    1: 0, 3: 0, 5: REWARD_BASELINE_1,  # Rewards for sequence 1 / stimulus 5 is rewarded with 1
    2: 0, 4: 0, 6: REWARD_BASELINE_2  # Rewards for sequence 2 / stimulus 6 is rewarded with 10
}

# * Reward revaluation experiment parameters * #

STIM_SEQ_1_REWARD_REVAL = STIM_SEQ_1_BASELINE[1:]  # Stimulus sequence (no change but exclude first state)
STIM_SEQ_2_REWARD_REVAL = STIM_SEQ_2_BASELINE[1:]  # Stimulus sequence

REWARD_MAPPING_REWARD_REVAL = {
    1: 0, 3: 0, 5: REWARD_BASELINE_2,  # Rewards for sequence 1 / stimulus 5 is rewarded with 1 (changed from 10)
    2: 0, 4: 0, 6: REWARD_BASELINE_1  # Rewards for sequence 2 / stimulus 6 is rewarded with 10 (changed from 1)
}

# * Transition revaluation experiment parameters * #

STIM_SEQ_1_TRANSITION_REVAL = [3, 6]  # Stimulus sequence 1 (changed transition 3->6 )
STIM_SEQ_2_TRANSITION_REVAL = [4, 5]  # Stimulus sequence 2 (changed transition 4->5)

REWARD_MAPPING_TRANSITION_REVAL = REWARD_MAPPING_BASELINE  # Rewards remain the same as baseline

# *** Time Generation Parameters *** #

TIME_DRIFT_RATE = 0.01
TIME_DRIFT_NOISE = 0.0


def gen_trials(
        state_seq_1: list,
        state_seq_2: list,
        num_seqs: int,
        reward_mapping: dict,
        random_state: Optional[int] = RANDOM_SEED):
    """
    Generate baseline trials with random selection of stimulus sequences
    """

    visited_states = []
    rewards = []
    rng = np.random.default_rng(random_state)
    for _ in range(num_seqs):
        if rng.random() < .5:
            visited_states.extend(state_seq_1)
            rewards.extend([reward_mapping[s] for s in state_seq_1])
        else:
            visited_states.extend(state_seq_2)
            rewards.extend([reward_mapping[s] for s in state_seq_2])

    visited_states = np.array(visited_states)
    visited_states = one_hot_encode(visited_states, STATE_SIZE)
    rewards = np.array(rewards)

    return visited_states, rewards


get_baseline_trials = partial(
    gen_trials,
    state_seq_1=STIM_SEQ_1_BASELINE,
    state_seq_2=STIM_SEQ_2_BASELINE,
    reward_mapping=REWARD_MAPPING_BASELINE,
)

get_reward_revaluation_trials = partial(
    gen_trials,
    state_seq_1=STIM_SEQ_1_REWARD_REVAL,
    state_seq_2=STIM_SEQ_2_REWARD_REVAL,
    reward_mapping=REWARD_MAPPING_REWARD_REVAL,
)

get_transition_revaluation_trials = partial(
    gen_trials,
    state_seq_1=STIM_SEQ_1_TRANSITION_REVAL,
    state_seq_2=STIM_SEQ_2_TRANSITION_REVAL,
    reward_mapping=REWARD_MAPPING_TRANSITION_REVAL,
)


# def get_time_sequence(num_trials: int,
#                       time_drift_rate: float = TIME_DRIFT_RATE,
#                       noise: float = TIME_DRIFT_NOISE,
#                       random_state: Optional[int] = None) -> np.ndarray:
#     time_code = torch.zeros((TIME_SIZE,), dtype=torch.float) + .01
#     times = [time_code.clone()]
#     for _ in range(1, num_trials):
#         time_code += torch.randn_like(time_code) * .01
#         times.append(time_code.clone())
#     return np.array(times)

#
def get_time_sequence(num_trials: int,
                      time_drift_rate: float = TIME_DRIFT_RATE,
                      noise: float = TIME_DRIFT_NOISE,
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate time sequence as drift on a sphere
    """
    rng = np.random.default_rng(random_state)
    time_fct = pnl.DriftOnASphereIntegrator(initializer=rng.random(TIME_SIZE),
                                            noise=0.02,
                                            dimension=TIME_SIZE)
    # return an arrays of .01
    # return np.array([np.zeros(25) + .01 for _ in range(num_trials)])
    time = np.array([time_fct(0.) for _ in range(num_trials)])
    return time



# if __name__ == '__main__':
#     states, rewards = (random_state=42)
#     print('States:\n', states)
#     print('Rewards:\n', rewards)


#
#
# def build_experience_inputs(**params):
#     # time_drift_rate: float = TIME_DRIFT_RATE,
#     # num_baseline_seqs: int = NUM_BASELINE_SEQS,
#     # num_revaluation_seqs: int = NUM_REVALUATION_SEQS,
#     # reward_vals: list = REWARD_VALS,
#     # sampling_type: Literal[Union['random', 'alternating']] = SAMPLING_TYPE,
#     # ratio: int = RATIO,
#     # stim_seqs: list = STIM_SEQS) -> tuple:
#     """
#     Build inputs for full sequence of trials (with one stim per trial) for EGO MDP model
#     Return tuple in which each item is list of all trials for a layer of the model: (time, task, state, reward)
#     """
#
#     def gen_baseline_states_and_rewards(state_size: int = state_size,
#                                         stim_seqs: list = stim_seqs,
#                                         reward_vals: list = reward_vals,
#                                         num_seqs: int = num_baseline_seqs,
#                                         sampling_type: Literal[Union['random', 'alternating']] = sampling_type,
#                                         ratio: int = ratio,
#                                         ) -> tuple:
#         """Generate states and rewards for reward revaluation phase of Experiment 1
#         Return tuple with one-hot representations of (states, rewards, length of a single sequence)
#         """
#         # Generate one-hots
#         state_reps = get_states(state_size)
#
#         # Generate sequence of states
#         visited_states, rewards = [], []
#         seq_len = len(stim_seqs[0])
#         for i in range(num_seqs):
#             seq_0 = np.random.random() < (ratio / (ratio + 1)) if sampling_type == 'random' else i % (ratio + 1)
#             if seq_0:
#                 visited_states.extend(stim_seqs[0])
#                 rewards.extend([0] * (seq_len - 1) + [reward_vals[0]])
#             else:
#                 visited_states.extend(stim_seqs[1])
#                 rewards.extend([0] * (seq_len - 1) + [reward_vals[1]])
#
#         # Pick one-hots corresponding to each state
#         visited_states = state_reps[visited_states]
#         rewards = np.array(rewards)
#
#         return visited_states, rewards, seq_len
#
#     def gen_reward_revaluation_states_and_reward(state_size: int = STATE_SIZE,
#                                                  stim_seqs: list = stim_seqs,
#                                                  reward_vals: list = reward_vals,
#                                                  num_seqs: int = num_revaluation_seqs,
#                                                  sampling_type: Literal[Union['random', 'alternating']] = sampling_type,
#                                                  ratio: int = ratio,
#                                                  ) -> tuple:
#         """Generate states and rewards for reward revaluation phase of Experiment 1
#         Return tuple with one-hot representations of (states, rewards, length of a single sequence)
#         """
#
#         # Generate one-hots
#         state_reps = get_states(state_size)
#
#         # Generate sequence of states
#         visited_states, rewards = [], []
#         seq_len = len(stim_seqs[0][1:])
#         for trial_idx in range(num_seqs):
#             seq_0 = np.random.random() < (ratio / (ratio + 1)) if sampling_type == 'random' else trial_idx % (ratio + 1)
#             if seq_0:
#                 visited_states.extend(stim_seqs[0][1:])
#                 rewards.extend([0] * (seq_len - 1) + [reward_vals[0]])
#             else:
#                 visited_states.extend(stim_seqs[1][1:])
#                 rewards.extend([0] * (seq_len - 1) + [reward_vals[1]])
#
#         # Pick one-hots corresponding to each state
#         visited_states = state_reps[visited_states]
#         rewards = np.array(rewards)
#
#         return visited_states, rewards, seq_len
