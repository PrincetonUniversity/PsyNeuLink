"""
Data generation for comparison experiments
"""

from typing import Optional
from functools import partial

import numpy as np
import psyneulink as pnl

from params import *
from utils import one_hot_encode

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


def gen_trials(
        state_seq_1: list,
        state_seq_2: list,
        num_seqs: int,
        reward_mapping: dict):
    """
    Generate baseline trials with random selection of stimulus sequences
    """

    visited_states = []
    rewards = []
    rng = np.random.default_rng(None)
    for _ in range(num_seqs):
        _rand = rng.random()
        if np.random.random() < .5:
            visited_states.extend(state_seq_2)
            rewards.extend([reward_mapping[s] for s in state_seq_2])
        else:
            visited_states.extend(state_seq_1)
            rewards.extend([reward_mapping[s] for s in state_seq_1])

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


def get_time_sequence(num_trials: int,
                      time_drift_rate: float = TIME_DRIFT_RATE,
                      noise: float = TIME_DRIFT_NOISE,
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate time sequence as drift on a sphere
    """
    rng = np.random.default_rng(random_state)
    time_fct = pnl.DriftOnASphereIntegrator(initializer=rng.random(TIME_SIZE),
                                            noise=noise,
                                            dimension=TIME_SIZE)
    time = np.array([time_fct(time_drift_rate) for _ in range(num_trials)])
    return time
