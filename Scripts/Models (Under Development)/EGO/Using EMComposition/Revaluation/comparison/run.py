"""
main scripts to run the experiments
"""

from functools import partial
from typing import Callable

from data import (get_baseline_trials, get_reward_revaluation_trials,
                  get_transition_revaluation_trials, get_time_sequence)

import model_python as model_python

import params as params
import utils as utils

import torch
import numpy as np


def gen_trials(
        num_seqs_baseline,
        num_seqs_revaluation,
        num_trials,
):
    trials = []
    # ** GENERATE TRIALS ** #
    for trial in range(num_trials):
        # Baseline phase
        baseline_trials, baseline_rewards = get_baseline_trials(
            num_seqs=num_seqs_baseline)

        # Reward revaluation phase

        # get only the revaluation trials
        reward_revaluation_trials_only, reward_revaluation_rewards_only = get_reward_revaluation_trials(
            num_seqs=num_seqs_revaluation)

        # combine with baseline trials
        reward_revaluation_trials = np.concatenate(
            [baseline_trials, reward_revaluation_trials_only], axis=0)
        reward_revaluation_rewards = np.concatenate(
            [baseline_rewards, reward_revaluation_rewards_only], axis=0)

        # Transition revaluation phase

        # get only the revaluation trials
        transition_revaluation_trials_only, transition_revaluation_rewards_only = get_transition_revaluation_trials(
            num_seqs=num_seqs_revaluation)

        # combine with baseline trials
        transition_revaluation_trials = np.concatenate(
            [baseline_trials, transition_revaluation_trials_only], axis=0)
        transition_revaluation_rewards = np.concatenate(
            [baseline_rewards, transition_revaluation_rewards_only], axis=0)

        # Time code for all trials
        time_sequence = get_time_sequence(
            num_trials=len(reward_revaluation_trials)
        )


        trials.append({
            'baseline_trials': baseline_trials,
            'baseline_rewards': baseline_rewards,
            'reward_revaluation_trials': reward_revaluation_trials,
            'reward_revaluation_rewards': reward_revaluation_rewards,
            'transition_revaluation_trials': transition_revaluation_trials,
            'transition_revaluation_rewards': transition_revaluation_rewards,
            'time_sequence': time_sequence})

    return trials


def run(
        estimate_reward_from_starting_state: Callable,
        num_participants: int = params.N_PARTICIPANTS,
        num_seqs_baseline: int = params.N_BASELINE_TRIALS,
        num_seqs_revaluation: int = params.N_REVALUATION_TRIALS,
        n_simulations: int = params.N_SIMULATIONS,  # number of simulation trajectories
        n_steps: int = params.N_STEPS,  # number of steps per simulation trajectory
        state_retrieval_weight: float = params.STATE_RETRIEVAL_WEIGHT,
        context_retrieval_weight: float = params.CONTEXT_RETRIEVAL_WEIGHT,
        time_retrieval_weight: float = params.TIME_RETRIEVAL_WEIGHT,
        old_context_integration_rate: float = params.OLD_CONTEXT_INTEGRATION_RATE,
        state_integration_rate: float = params.STATE_INTEGRATION_RATE,
        new_context_integration_rate: float = params.NEW_CONTEXT_INTEGRATION_RATE,
        context_d=params.STATE_SIZE,
        state_d=params.STATE_SIZE,
        time_d=params.TIME_SIZE,

):
    utils.set_random_seed(params.RANDOM_SEED)
    # initialize revaluation scores
    data = {
        'reval_scores_reward': [],
        'reval_scores_transition': [],
        'estimated_reward_state_1_baseline': [],
        'estimated_reward_state_2_baseline': [],
        'estimated_reward_state_1_reward_reval': [],
        'estimated_reward_state_2_reward_reval': [],
        'estimated_reward_state_1_transition_reval': [],
        'estimated_reward_state_2_transition_reval': [],
    }

    trials = gen_trials(num_seqs_baseline=num_seqs_baseline,
                        num_seqs_revaluation=num_seqs_revaluation,
                        num_trials=num_participants)

    for participant_idx in range(num_participants):
        trial = trials[participant_idx]
        baseline_trials = trial['baseline_trials']
        baseline_rewards = trial['baseline_rewards']
        reward_revaluation_trials = trial['reward_revaluation_trials']
        reward_revaluation_rewards = trial['reward_revaluation_rewards']
        transition_revaluation_trials = trial['transition_revaluation_trials']
        transition_revaluation_rewards = trial['transition_revaluation_rewards']
        time_sequence = trial['time_sequence']

        # ** GENERATE MEMORIES ** #

        # convenience partial function to generate memories with fixed parameters
        _gen_memories = partial(
            model_python.gen_memories,
            old_context_integration_rate=old_context_integration_rate,
            state_integration_rate=state_integration_rate,
            retrieved_context_integration_rate=new_context_integration_rate,
            state_retrieval_weight=state_retrieval_weight,
            context_retrieval_weight=context_retrieval_weight,
            time_retrieval_weight=time_retrieval_weight,
            context_d=context_d)

        # memories baseline only
        memories_baseline = _gen_memories(
            visited_states=baseline_trials,
            rewards=baseline_rewards,
            time_sequence=time_sequence[:len(baseline_trials)],  # only the time codes for baseline trials
        )

        # memories reward revaluation
        memories_reward_reval = _gen_memories(
            visited_states=reward_revaluation_trials,
            rewards=reward_revaluation_rewards,
            time_sequence=time_sequence,  # all time codes
        )

        # memories transition revaluation
        memories_transition_reval = _gen_memories(
            visited_states=transition_revaluation_trials,
            rewards=transition_revaluation_rewards,
            time_sequence=time_sequence,  # all time codes
        )

        # ** ESTIMATE REWARDS FROM STARTING STATES ** #
        starting_state_1 = torch.eye(7)[1]
        starting_state_2 = torch.eye(7)[2]

        _estimated_reward_from_starting_state = partial(
            estimate_reward_from_starting_state,
            n_simulations=n_simulations,
            n_steps=n_steps,
            state_retrieval_weight=state_retrieval_weight,
            context_retrieval_weight=context_retrieval_weight,
            time_retrieval_weight=time_retrieval_weight,
            old_context_integration_rate=old_context_integration_rate,
            state_integration_rate=state_integration_rate,
            new_context_integration_rate=new_context_integration_rate,
            context_d=context_d,
            state_d=state_d,
            time_d=time_d
        )

        estimated_reward_state_1_baseline = _estimated_reward_from_starting_state(
            memories=memories_baseline,
            starting_state=starting_state_1,
        )

        estimated_reward_state_2_baseline = _estimated_reward_from_starting_state(
            memories=memories_baseline,
            starting_state=starting_state_2)

        estimated_reward_state_1_reward_reval = _estimated_reward_from_starting_state(
            memories=memories_reward_reval,
            starting_state=starting_state_1,
        )

        estimated_reward_state_2_reward_reval = _estimated_reward_from_starting_state(
            memories=memories_reward_reval,
            starting_state=starting_state_2)

        estimated_reward_state_1_transition_reval = _estimated_reward_from_starting_state(
            memories=memories_transition_reval,
            starting_state=starting_state_1,
        )

        estimated_reward_state_2_transition_reval = _estimated_reward_from_starting_state(
            memories=memories_transition_reval,
            starting_state=starting_state_2)

        state_one_preference_baseline = estimated_reward_state_1_baseline - estimated_reward_state_2_baseline
        state_one_preference_reward_reval = estimated_reward_state_1_reward_reval - estimated_reward_state_2_reward_reval
        state_one_preference_transition_reval = estimated_reward_state_1_transition_reval - estimated_reward_state_2_transition_reval

        reward_reval_reward = state_one_preference_baseline - state_one_preference_reward_reval
        reward_transition_reval = state_one_preference_baseline - state_one_preference_transition_reval

        data['reval_scores_reward'].append(reward_reval_reward)
        data['reval_scores_transition'].append(reward_transition_reval)
        data['estimated_reward_state_1_baseline'].append(estimated_reward_state_1_baseline)
        data['estimated_reward_state_2_baseline'].append(estimated_reward_state_2_baseline)
        data['estimated_reward_state_1_reward_reval'].append(estimated_reward_state_1_reward_reval)
        data['estimated_reward_state_2_reward_reval'].append(estimated_reward_state_2_reward_reval)
        data['estimated_reward_state_1_transition_reval'].append(estimated_reward_state_1_transition_reval)
        data['estimated_reward_state_2_transition_reval'].append(estimated_reward_state_2_transition_reval)

    return data


run_original = partial(
    run,
    estimate_reward_from_starting_state=model_python.estimate_reward_from_starting_state
)

if __name__ == '__main__':
    run_original()
