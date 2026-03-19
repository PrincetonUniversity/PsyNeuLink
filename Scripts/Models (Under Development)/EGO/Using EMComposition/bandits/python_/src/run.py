"""
Convenience functions to run the python version of the model
"""
import random
from src import gen_trials
from scripts.src.ego_model import gen_memories, estimate_reward_from_starting_state
from src import defaults

import numpy as np


def get_reward_estimates(
        states, rewards, times,
        state_1=np.array([0, 1, 0, 0, 0, 0, 0]),
        state_2=np.array([0, 0, 1, 0, 0, 0, 0]),
        metric=defaults.METRIC,
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        ego_softmax_temperature=defaults.TEMPERATURE,
        ego_softmax_threshold=defaults.SOFTMAX_THRESHOLD,
):
    memories = gen_memories(
        states=states,
        rewards=rewards,
        times=times,
        state_integration_rate=state_integration_rate
    )

    # query one memory
    reward_estimate_state_1 = estimate_reward_from_starting_state(
        memories, state_1, times[-1], metric=metric, model_based_ness=model_based_ness,
        state_integration_rate=state_integration_rate,
        time_retrieval_weight=time_retrieval_weight,
        ego_temperature=ego_softmax_temperature,
        ego_threshold=ego_softmax_threshold
    )
    reward_estimate_state_2 = estimate_reward_from_starting_state(
        memories, state_2, times[-1], metric=metric, model_based_ness=model_based_ness,
        state_integration_rate=state_integration_rate,
        time_retrieval_weight=time_retrieval_weight,
        ego_temperature=ego_softmax_temperature,
        ego_threshold=ego_softmax_threshold
    )
    return reward_estimate_state_1, reward_estimate_state_2


def run(
        num_participants: int = defaults.N_PARTICIPANTS,
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,
        time_noise=defaults.TIME_DRIFT_NOISE,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        ego_softmax_temperature=defaults.TEMPERATURE,
        ego_softmax_threshold=defaults.SOFTMAX_THRESHOLD,
        seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

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
    # initialize revaluation scores
    revaluation_scores = np.zeros((num_participants, 3))
    for participant_idx in range(num_participants):
        # ** GENERATE TRIALS ** #

        # reward_reval
        exp_rr = gen_trials.gen_experiment_reward_reval(time_drift_noise=time_noise)
        exp_tr = gen_trials.gen_experiment_transition_reval(time_drift_noise=time_noise)

        estimate_rr_baseline_state_1, estimate_rr_baseline_state_2 = get_reward_estimates(
            exp_rr.states_baseline, exp_rr.rewards_baseline, exp_rr.times_baseline,
            metric=metric, model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
            ego_softmax_temperature=ego_softmax_temperature,
            ego_softmax_threshold=ego_softmax_threshold
        )

        estimate_rr_state_1, estimate_rr_state_2 = get_reward_estimates(
            exp_rr.states, exp_rr.rewards, exp_rr.times,
            metric=metric, model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
            ego_softmax_temperature=ego_softmax_temperature,
            ego_softmax_threshold=ego_softmax_threshold
        )

        # exp_tr = gen_trials.gen_experiment_transition_reval()

        estimate_tr_baseline_state_1, estimate_tr_baseline_state_2 = get_reward_estimates(
            exp_tr.states_baseline, exp_tr.rewards_baseline, exp_tr.times_baseline,
            metric=metric, model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
            ego_softmax_temperature=ego_softmax_temperature,
            ego_softmax_threshold=ego_softmax_threshold
        )

        estimate_tr_state_1, estimate_tr_state_2 = get_reward_estimates(
            exp_tr.states, exp_tr.rewards, exp_tr.times,
            metric=metric, model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
            ego_softmax_temperature=ego_softmax_temperature,
            ego_softmax_threshold=ego_softmax_threshold
        )

        estimated_reward_state_1_baseline = (
                (estimate_rr_baseline_state_1 + estimate_tr_baseline_state_1) / 2)
        estimated_reward_state_2_baseline = (
                (estimate_rr_baseline_state_2 + estimate_tr_baseline_state_2) / 2
        )

        state_one_preference_baseline_rr = estimate_rr_baseline_state_1 - estimate_rr_baseline_state_2
        state_one_preference_rr = estimate_rr_state_1 - estimate_rr_state_2

        state_one_preference_baseline_tr = estimate_tr_baseline_state_1 - estimate_tr_baseline_state_2
        state_one_preference_tr = estimate_tr_state_1 - estimate_tr_state_2

        reward_reval_reward = state_one_preference_baseline_rr - state_one_preference_rr
        reward_transition_reval = state_one_preference_baseline_tr - state_one_preference_tr
        revaluation_scores[participant_idx, 2] = (
                (state_one_preference_baseline_rr + state_one_preference_baseline_tr) / 2)

        data['reval_scores_reward'].append(reward_reval_reward)
        data['reval_scores_transition'].append(reward_transition_reval)
        data['estimated_reward_state_1_baseline'].append(estimated_reward_state_1_baseline)
        data['estimated_reward_state_2_baseline'].append(estimated_reward_state_2_baseline)
        data['estimated_reward_state_1_reward_reval'].append(estimate_rr_state_1)
        data['estimated_reward_state_2_reward_reval'].append(estimate_rr_state_2)
        data['estimated_reward_state_1_transition_reval'].append(estimate_tr_state_1)
        data['estimated_reward_state_2_transition_reval'].append(estimate_tr_state_2)

    return data
