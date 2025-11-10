"""
main scripts to run the experiments
"""

from functools import partial
from typing import Callable

from comparison.data import (get_baseline_trials, get_reward_revaluation_trials,
                              get_transition_revaluation_trials, get_time_sequence)

import comparison.model_to_evaluate as model_to_evaluate

import comparison.params as params

import torch
import numpy as np


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
    # initialize revaluation scores
    revaluation_scores = np.zeros((num_participants, 3))
    for participant_idx in range(num_participants):
        # ** GENERATE TRIALS ** #

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

        # ** GENERATE MEMORIES ** #

        # convenience partial function to generate memories with fixed parameters
        _gen_memories = partial(
            model_to_evaluate.gen_memories,
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

        revaluation_scores[participant_idx, 0] = state_one_preference_baseline - state_one_preference_reward_reval
        revaluation_scores[participant_idx, 1] = state_one_preference_baseline - state_one_preference_transition_reval
        revaluation_scores[participant_idx, 2] = state_one_preference_baseline

    return revaluation_scores


run_to_evaluate = partial(
    run,
    estimate_reward_from_starting_state=model_to_evaluate.estimate_reward_from_starting_state
)

run_to_evaluate()
