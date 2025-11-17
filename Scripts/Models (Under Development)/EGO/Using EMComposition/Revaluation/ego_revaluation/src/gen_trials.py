"""
Trial sequence generation for the ego revaluation task.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import psyneulink as pnl

from ..config.phase_config import PhaseConfig
from ..config import defaults
from ..src.utils import one_hot_encode


# ---------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------

@dataclass
class TrialData:
    states: np.ndarray
    rewards: np.ndarray


@dataclass
class ExperimentTrials:
    states_baseline: np.ndarray
    rewards_baseline: np.ndarray
    times_baseline: np.ndarray

    states_reval: np.ndarray
    rewards_reval: np.ndarray
    times_reval: np.ndarray

    @property
    def n_trials(self):
        return self.states_baseline.shape[0] + self.states_reval.shape[0]

    @property
    def states(self):
        return np.vstack((self.states_baseline, self.states_reval))

    @property
    def rewards(self):
        return np.concatenate((self.rewards_baseline, self.rewards_reval))

    @property
    def times(self):
        return np.concatenate((self.times_baseline, self.times_reval))

    @property
    def tasks_baseline(self):
        return np.zeros(self.states_baseline.shape[0], dtype=int)

    @property
    def tasks_reval(self):
        return np.zeros(self.states_reval.shape[0], dtype=int)

    @property
    def tasks(self):
        return np.zeros(self.n_trials, dtype=int)


# ---------------------------------------------------------
# Trial generation
# ---------------------------------------------------------

def gen_trials(phase: PhaseConfig, counterbalance: bool = False) -> TrialData:
    """
    Generate trials for a given PhaseConfig.
    """

    seq1 = phase.state_seq_1
    seq2 = phase.state_seq_2
    num_seqs = phase.num_sequences
    reward_by_state = phase.reward_by_state

    # -------------------------------------------
    # Build sequence blocks
    # -------------------------------------------
    if not counterbalance:
        sequence_blocks = [
            seq1 if np.random.random() < 0.5 else seq2
            for _ in range(num_seqs)
        ]
    else:
        assert not num_seqs % 2, "Number of sequences must be even"
        half = num_seqs // 2
        sequence_blocks = [seq1] * half + [seq2] * half
        np.random.shuffle(sequence_blocks)

    # -------------------------------------------
    # Flatten states and rewards
    # -------------------------------------------
    visited_states = []
    rewards = []

    for seq in sequence_blocks:
        visited_states.extend(seq)
        rewards.extend(reward_by_state[s] for s in seq)

    visited_states = np.array(visited_states)
    rewards = np.array(rewards)

    visited_states = one_hot_encode(visited_states, defaults.STATE_SIZE)

    return TrialData(visited_states, rewards)


# Partials
get_baseline_trials = partial(gen_trials, phase=defaults.BASELINE_PHASE)
get_reward_reval_trials = partial(gen_trials, phase=defaults.REWARD_REVAL_PHASE)
get_transition_reval_trials = partial(gen_trials, phase=defaults.TRANSITION_REVAL_PHASE)


# ---------------------------------------------------------
# Time generation
# ---------------------------------------------------------

def get_time_sequence(
        num_trials: int,
        time_drift_rate: float = defaults.TIME_DRIFT_RATE,
        noise: float = defaults.TIME_DRIFT_NOISE,
) -> np.ndarray:
    """
    Generate time sequence as drift on a sphere.
    """
    time_fct = pnl.DriftOnASphereIntegrator(
        initializer=np.random.random(defaults.TIME_SIZE),
        noise=noise,
        dimension=defaults.TIME_SIZE,
    )
    return np.array([time_fct(time_drift_rate) for _ in range(num_trials)])


# ---------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------

def gen_experiment(
        baseline_phase: PhaseConfig,
        reval_phase: PhaseConfig,
        counterbalance=False,
) -> ExperimentTrials:
    baseline = gen_trials(baseline_phase, counterbalance)
    reval = gen_trials(reval_phase, counterbalance)

    n_b = baseline.states.shape[0]
    n_r = reval.states.shape[0]
    memory_capacity = n_b + n_r

    times = get_time_sequence(memory_capacity)
    times_baseline = times[:n_b]
    times_reval = times[n_b:]

    return ExperimentTrials(
        states_baseline=baseline.states,
        rewards_baseline=baseline.rewards,
        times_baseline=times_baseline,

        states_reval=reval.states,
        rewards_reval=reval.rewards,
        times_reval=times_reval,
    )


gen_experiment_reward_reval = partial(
    gen_experiment,
    baseline_phase=defaults.BASELINE_PHASE,
    reval_phase=defaults.REWARD_REVAL_PHASE,
)

gen_experiment_transition_reval = partial(
    gen_experiment,
    baseline_phase=defaults.BASELINE_PHASE,
    reval_phase=defaults.TRANSITION_REVAL_PHASE,
)
