"""
Data generation for comparison experiments
"""

from typing import Optional
from functools import partial

import numpy as np
import psyneulink as pnl

import params as params
from utils import one_hot_encode


def _gen_trials(
        state_seq_1: list,
        state_seq_2: list,
        num_seqs: int,
        reward_mapping: dict):
    """
    Generate trials with random selection of stimulus sequences
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
    visited_states = one_hot_encode(visited_states, params.STATE_SIZE)
    rewards = np.array(rewards)

    return visited_states, rewards

def _get_trials_variant(kind, **kwargs):
    defaults = {
        name: getattr(params, name.upper() + f'_{kind.upper()}')
        for name in ['state_seq_1', 'state_seq_2', 'num_seqs', 'reward_mapping']
    }
    defaults.update({k: v for k, v in kwargs.items() if k in defaults})
    return _gen_trials(**defaults)

get_baseline_trials = partial(_get_trials_variant, kind='baseline')
get_reward_reval_trials = partial(_get_trials_variant, kind='reward_reval')
get_transition_reval_trials = partial(_get_trials_variant, kind='transition_reval')

#
# def get_baseline_trials(**kwargs):
#     defaults = {
#         name: getattr(params, name.upper() + "_BASELINE")
#         for name in ["state_seq_1", "state_seq_2", "num_seqs", "reward_mapping"]
#     }
#     defaults.update({k: v for k, v in kwargs.items() if k in defaults})
#     return _gen_trials(**defaults)
#
# def get_baseline_trials(**kwargs):
#     state_seq_1 = params.STATE_SEQ_1_BASELINE
#     state_seq_2 = params.STATE_SEQ_2_BASELINE
#     num_seqs = params.N_BASELINE_TRIALS
#     reward_mapping = params.REWARD_MAPPING_BASELINE
#
#     if 'state_seq_1' in kwargs:
#         state_seq_1 = kwargs['state_seq_baseline']
#     if 'state_seq_2' in kwargs:
#         state_seq2_baseline = kwargs['state_seq2_baseline']
#     if 'num_seqs' in kwargs:
#         num_seqs = kwargs['num_seqs']
#     if 'reward_mapping' in kwargs:
#         reward_mapping = kwargs['reward_mapping']
#     return _gen_trials(
#         state_seq_1=state_seq_1,
#         state_seq_2=state_seq_2,
#         num_seqs=num_seqs,
#         reward_mapping=reward_mapping
#     )
#
#
# get_baseline_trials = partial(
#     gen_trials,
#     state_seq_1=STIM_SEQ_1_BASELINE,
#     state_seq_2=STIM_SEQ_2_BASELINE,
#     reward_mapping=REWARD_MAPPING_BASELINE,
# )
#
# get_reward_revaluation_trials = partial(
#     gen_trials,
#     state_seq_1=STIM_SEQ_1_REWARD_REVAL,
#     state_seq_2=STIM_SEQ_2_REWARD_REVAL,
#     reward_mapping=REWARD_MAPPING_REWARD_REVAL,
# )
#
# get_transition_revaluation_trials = partial(
#     gen_trials,
#     state_seq_1=STIM_SEQ_1_TRANSITION_REVAL,
#     state_seq_2=STIM_SEQ_2_TRANSITION_REVAL,
#     reward_mapping=REWARD_MAPPING_TRANSITION_REVAL,
# )


def get_time_sequence(num_trials: int,
                      time_drift_rate: float = params.TIME_DRIFT_RATE,
                      noise: float = params.TIME_DRIFT_NOISE,
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate time sequence as drift on a sphere
    """
    rng = np.random.default_rng(random_state)
    time_fct = pnl.DriftOnASphereIntegrator(initializer=rng.random(params.TIME_SIZE),
                                            noise=noise,
                                            dimension=params.TIME_SIZE)
    time = np.array([time_fct(time_drift_rate) for _ in range(num_trials)])
    return time


def _gen_experiment(
        get_reval_trials,
        n_seq_baseline=params.N_BASELINE_TRIALS,
        n_seq_reval=params.N_REVALUATION_TRIALS,

):
    n_baseline_trials = N_SEQ_BASELINE * N_STATES_BASELINE
    n_reval_trials = N_SEQ_REVAL * N_STATES_REVAL
    memory_capacity = n_baseline_trials + n_reval_trials

    states_baseline, rewards_baseline = data.get_baseline_trials(num_seqs=N_SEQ_BASELINE)
    states_reward_reval, rewards_reward_reval = get_reval_trials(num_seqs=N_SEQ_REVAL)

    times = get_time_sequence(num_trials=memory_capacity)

    return (states_baseline, rewards_baseline, times[:n_baseline_trials], [0] * n_baseline_trials,
            states_reward_reval, rewards_reward_reval, times[n_baseline_trials:], [0] * n_reval_trials,
            memory_capacity)


get_experiment_reward_reval = partial(
    _gen_experiment,
    get_reval_trials=get_reward_revaluation_trials
)

get_experiment_transition_reval = partial(
    _gen_experiment,
    get_reval_trials=get_transition_revaluation_trials
)
