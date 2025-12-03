from random import random

from .src import gen_trials

from .model_python import gen_memories, estimate_reward_from_starting_state

from .config import defaults

import numpy as np


def get_reward_estimates(
        states, rewards, times,
        state_1=np.array([0, 1, 0, 0, 0, 0, 0]),
        state_2=np.array([0, 0, 1, 0, 0, 0, 0]),
        metric=defaults.METRIC,
        model_based_ness=defaults.MODEL_BASED_NESS,
):
    memories = gen_memories(
        states=states,
        rewards=rewards,
        times=times
    )

    # query one memory
    reward_estimate_state_1 = estimate_reward_from_starting_state(
        memories, state_1, times[-1], metric=metric, model_based_ness=model_based_ness,
        state_d=9,
        context_d=9,
    )
    reward_estimate_state_2 = estimate_reward_from_starting_state(
        memories, state_2, times[-1], metric=metric, model_based_ness=model_based_ness,
        state_d=9,
        context_d=9,
    )
    return reward_estimate_state_1, reward_estimate_state_2


def gen_trials_base(
        n=20,
        common_prob=.8):
    state_1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    state_2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])

    state_3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    state_4 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    state_5 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    state_6 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])

    state_7 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    state_8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

    common_seqs = {}
    rare_seqs = {}

    # state_1 -> state_3 is common
    common_seqs['0_reward'] = [state_1, state_3, state_5]  # rewarded
    common_seqs['0_no_reward'] = [state_1, state_3, state_7]  # no reward

    # state_1 -> state_4 is rare
    rare_seqs['0_reward'] = [state_1, state_4, state_6]  # rewarded
    rare_seqs['0_no_reward'] = [state_1, state_4, state_8]  # no reward

    # state_2 -> state_4 is common
    common_seqs['1_reward'] = [state_2, state_4, state_6]  # rewarded
    common_seqs['1_no_reward'] = [state_2, state_4, state_6]  # no reward

    # state_2 -> state_3 is rare
    rare_seqs['1_reward'] = [state_2, state_3, state_5]  # rewarded
    rare_seqs['1_no_reward'] = [state_2, state_3, state_7]  # no rewarded

    visited_states = []
    rewards = []

    for seq in range(n):
        seq = 0
        if random() < .5:
            seq = 1
        is_reward = 'reward'
        if random() < .5:
            is_reward = 'no_reward'

        if random() < common_prob:  # common
            chosen_seq = common_seqs[f'{seq}_{is_reward}']
        else:
            chosen_seq = rare_seqs[f'{seq}_{is_reward}']

        for idx, sq in enumerate(chosen_seq):
            visited_states.append(sq)
            if idx == 2:
                if is_reward == 'reward':
                    rewards.append(1)
                else:
                    rewards.append(0)
            else:
                rewards.append(0)

    times = gen_trials.get_time_sequence(n*3+ 3)

    return np.array(visited_states), np.array(rewards), times



def run(
        num_participants: int = defaults.N_PARTICIPANTS,
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,

):
    vs_bl, rw_bl, t = gen_trials_base(20)

    t_bl = t[:-3]

    state_1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    state_2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])

    state_3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    state_4 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    state_5 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    state_6 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])

    state_7 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    state_8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

    estimate_bl = get_reward_estimates(vs_bl, rw_bl, t_bl, state_1, state_2)

    # add a common rewarded state: Expectation: estimated_reward_state_1 goes up
    vs_cr = np.vstack([vs_bl, state_1, state_3, state_5])

    rw_cr = np.concatenate([rw_bl, [0, 0, 1]])

    estimate_common_rewarded = get_reward_estimates(vs_cr, rw_cr, t, state_1, state_2)

    print('Baseline', estimate_bl)
    print('Common reward', estimate_common_rewarded)







    # print(vs_bl)
    # print(rw_bl)
    # print(t_bl)









   # print(estimate_1)


# data = {
#         'reval_scores_reward': [],
#         'reval_scores_transition': [],
#         'estimated_reward_state_1_baseline': [],
#         'estimated_reward_state_2_baseline': [],
#         'estimated_reward_state_1_reward_reval': [],
#         'estimated_reward_state_2_reward_reval': [],
#         'estimated_reward_state_1_transition_reval': [],
#         'estimated_reward_state_2_transition_reval': [],
#     }
#     # initialize revaluation scores
#     revaluation_scores = np.zeros((num_participants, 3))
#     for participant_idx in range(num_participants):
#         # ** GENERATE TRIALS ** #
#
#         # reward_reval
#         exp_rr = gen_trials.gen_experiment_reward_reval()
#         exp_tr = gen_trials.gen_experiment_transition_reval()
#
#         estimate_rr_baseline_state_1, estimate_rr_baseline_state_2 = get_reward_estimates(
#             exp_rr.states_baseline, exp_rr.rewards_baseline, exp_rr.times_baseline,
#             metric=metric, model_based_ness=model_based_ness
#         )
#
#         estimate_rr_state_1, estimate_rr_state_2 = get_reward_estimates(
#             exp_rr.states, exp_rr.rewards, exp_rr.times,
#             metric=metric, model_based_ness=model_based_ness
#         )
#
#         # exp_tr = gen_trials.gen_experiment_transition_reval()
#
#         estimate_tr_baseline_state_1, estimate_tr_baseline_state_2 = get_reward_estimates(
#             exp_tr.states_baseline, exp_tr.rewards_baseline, exp_tr.times_baseline,
#             metric=metric, model_based_ness=model_based_ness
#         )
#
#         estimate_tr_state_1, estimate_tr_state_2 = get_reward_estimates(
#             exp_tr.states, exp_tr.rewards, exp_tr.times,
#             metric=metric, model_based_ness=model_based_ness
#         )
#
#         estimated_reward_state_1_baseline = (
#                 (estimate_rr_baseline_state_1 + estimate_tr_baseline_state_1) / 2)
#         estimated_reward_state_2_baseline = (
#                 (estimate_rr_baseline_state_2 + estimate_tr_baseline_state_2) / 2
#         )
#
#         state_one_preference_baseline_rr = estimate_rr_baseline_state_1 - estimate_rr_baseline_state_2
#         state_one_preference_rr = estimate_rr_state_1 - estimate_rr_state_2
#
#         state_one_preference_baseline_tr = estimate_tr_baseline_state_1 - estimate_tr_baseline_state_2
#         state_one_preference_tr = estimate_tr_state_1 - estimate_tr_state_2
#
#         reward_reval_reward = state_one_preference_baseline_rr - state_one_preference_rr
#         reward_transition_reval = state_one_preference_baseline_tr - state_one_preference_tr
#         revaluation_scores[participant_idx, 2] = (
#                 (state_one_preference_baseline_rr + state_one_preference_baseline_tr) / 2)
#
#         data['reval_scores_reward'].append(reward_reval_reward)
#         data['reval_scores_transition'].append(reward_transition_reval)
#         data['estimated_reward_state_1_baseline'].append(estimated_reward_state_1_baseline)
#         data['estimated_reward_state_2_baseline'].append(estimated_reward_state_2_baseline)
#         data['estimated_reward_state_1_reward_reval'].append(estimate_rr_state_1)
#         data['estimated_reward_state_2_reward_reval'].append(estimate_rr_state_2)
#         data['estimated_reward_state_1_transition_reval'].append(estimate_tr_state_1)
#         data['estimated_reward_state_2_transition_reval'].append(estimate_tr_state_2)
#
#     return data
