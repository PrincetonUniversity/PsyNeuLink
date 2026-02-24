import numpy as np
from random import random
import itertools
import pandas as pd


from .src import gen_trials
from .model_python import gen_memories, estimate_reward_from_starting_state
from .config import defaults

STATES = {
    1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
    2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
    3: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),  # second-stage A
    4: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),  # second-stage B
    5: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),  # reward marker for 3
    6: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),  # reward marker for 4
    7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),  # no-reward marker for 3
    8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),  # no-reward marker for 4
}

def drift(p, sigma=0.025, lo=0.25, hi=0.75):
    """Gaussian random walk for reward probabilities, clipped."""
    p = p + np.random.normal(0, sigma)
    return float(np.clip(p, lo, hi))


def gen_trials_base(
        n=200,
        common_prob=0.7,
):
    """
    - First-stage states: 1, 2  (rockets)
    - Second-stage states: 3, 4 (planets)
    - Third-stage states: 5, 6, 7, 8 (aliens/reward)
    - Reward depends on second-stage state:
        p3(t), p4(t) drift independently over trials.
    """

    # one-hot state vectors
    states = {
        1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
        2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
        3: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),  # second-stage A
        4: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),  # second-stage B
        5: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),  # reward marker for 3
        6: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),  # reward marker for 4
        7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),  # no-reward marker for 3
        8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),  # no-reward marker for 4
    }

    # independent drifting reward probabilities for the 2 second-stage states
    p1 = np.random.uniform(0.25, 0.75)
    p2 = np.random.uniform(0.25, 0.75)
    p3 = np.random.uniform(0.25, 0.75)
    p4 = np.random.uniform(0.25, 0.75)

    # p1 = 0.75
    # p2 = 0.75
    # p3 = 0.25
    # p4 = 0.25

    visited_states = []
    rewards = []
    trial_log = []

    for trial in range(n):

        # -----------------------------
        # 1. First-stage choice (rocket)
        # -----------------------------
        if trial == 0:
            start_id = 1 if random() < 0.5 else 2
        else:
            # start_id = 1 if random() < max(p1,p2)/(max(p1,p2)+max(p3,p4)) else 2
            start_id = 1 if random() < 0.5 else 2
        start_state = states[start_id]

        # -----------------------------
        # 2. Common vs rare transition
        #    and second-stage state
        # -----------------------------
        if start_id == 1:
            # state 1: common->3, rare->4
            if random() < common_prob:
                transition = "common"
                second_id = 3
            else:
                transition = "rare"
                second_id = 4
        else:
            # state 2: common->4, rare->3
            if random() < common_prob:
                transition = "common"
                second_id = 4
            else:
                transition = "rare"
                second_id = 3

        second_state = states[second_id]

        # -----------------------------
        # 3. Reward depends on second_id
        # -----------------------------
        if second_id == 3:
            # reward_prob, terminal_id = (p1, 5) if random() < 0.5 else (p2, 7)
            reward_prob, terminal_id = (p1, 5) if random() < p1 / (p1 + p2) else (p2, 7)
        else:  # second_id == 4
            # reward_prob, terminal_id = (p3, 6) if random() < 0.5 else (p4, 8)
            reward_prob, terminal_id = (p3, 6) if random() < p3 / (p3 + p4) else (p4, 8)

        reward = 1 if random() < reward_prob else 0

        # # terminal state is just a marker for reward/no reward
        # if second_id == 3:
        #     terminal_id = 5 if reward == 1 else 7
        # else:  # second_id == 4
        #     terminal_id = 6 if reward == 1 else 8

        terminal_state = states[terminal_id]

        # append full 3-step sequence for memory
        visited_states.extend([start_state, second_state, terminal_state])
        rewards.extend([0, 0, reward])

        # -----------------------------
        # 4. Drift reward probabilities
        # -----------------------------
        p1 = drift(p1)
        p2 = drift(p2)
        p3 = drift(p3)
        p4 = drift(p4)

        # -----------------------------
        # 5. Log trial info
        # (value estimates filled in later)
        # -----------------------------
        trial_log.append({
            "trial": trial,
            "start_state": start_id,
            "second_state": second_id,
            "transition": transition,  # "common" or "rare"
            "reward": reward,  # 0/1
            "estimate_state_1": None,
            "estimate_state_2": None,
            "pred_next_state": None,  # 1 or 2 based on estimates
            "stay": None,  # 0/1 w.r.t. previous start_state
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
        })

    # one time-step per event (3 events per trial | n is number of trials)
    # times = gen_trials.get_time_sequence(n * 3)
    times = gen_trials.get_time_sequence_event(n, 3)

    return np.array(visited_states), np.array(rewards), times, trial_log


def initial_reward_probabilities(is_random=True, low=.25, high=.75):
    return {
        5: np.random.uniform(low, high) if is_random else high,
        6: np.random.uniform(low, high) if is_random else high,
        7: np.random.uniform(low, high) if is_random else low,
        8: np.random.uniform(low, high) if is_random else low
    }





# softmax value-to-choice rule with perseveration bias from Daw et al, 2011
# simply replace Q-value with the value retrieved from Episodic Memory
def val_to_choice(
        r1: float,  # value/reward retrieved from memory for candidate choice "1"
        r2: float,  # value/reward retrieved from memory for candidate choice "2"
        prev_choice: int,  # np.array, identity of previous 1st stage choice
        temp: float = 2.,  # temperature for softmax
        bias: float = 0.5,  # perseveration bias, only for 1st stage ('rocket') choice
):
    # determine if choice in question is repetition from previous trial
    repeat1 = int(prev_choice == 1)
    repeat2 = int(prev_choice == 2)

    # numerator of softmax for each candidate choice
    choice1 = np.exp(temp * (r1 + bias * repeat1))
    choice2 = np.exp(temp * (r2 + bias * repeat2))

    # calculate softmax probability for each candidate choice
    p_choice1 = choice1 / (choice1 + choice2)
    p_choice2 = choice2 / (choice1 + choice2)

    return p_choice1, p_choice2


def common_v_rare_transition(start_id, common_prob):
    if start_id == 1:
        # state 1: common->3, rare->4
        if random() < common_prob:
            transition = "common"
            second_id = 3
        else:
            transition = "rare"
            second_id = 4
    else:
        # state 2: common->4, rare->3
        if random() < common_prob:
            transition = "common"
            second_id = 4
        else:
            transition = "rare"
            second_id = 3
    return second_id, transition


def gen_trials_base_persevere(
        n=200,
        common_prob=0.7,
        repeat_bias=0.0,
        model_free=True,
        random_alien=True,
        is_random_initial_probs=False,
        is_drifting=True,
):
    """
    - First-stage states: 1, 2  (rockets)
    - Second-stage states: 3, 4 (planets)
    - Third-stage states: 5, 6, 7, 8 (aliens/reward)
    - Reward depends on second-stage state:
        p3(t), p4(t) drift independently over trials.
    """

    reward_probs = initial_reward_probabilities(is_random=is_random_initial_probs)

    visited_states = []
    rewards = []
    trial_log = []

    for trial in range(n):

        # -----------------------------
        # 1. First-stage choice (rocket)
        # -----------------------------
        if model_free:
            if trial == 0:
                start_id = 1 if random() < 0.5 else 2
            else:
                prev_start_id = trial_log[-1]["start_id"]
                rew = trial_log[-1]["reward"]

                if prev_start_id == 1 and rew == 1:
                    start_id = 1 if random() < 0.5 + repeat_bias else 2
                elif prev_start_id == 1 and rew == 0:
                    start_id = 1 if random() < 0.5 - repeat_bias else 2
                elif prev_start_id == 2 and rew == 1:
                    start_id = 2 if random() < 0.5 + repeat_bias else 1
                elif prev_start_id == 2 and rew == 0:
                    start_id = 2 if random() < 0.5 - repeat_bias else 1
        else:
            if trial == 0:
                start_id = 1 if random() < 0.5 else 2
            else:
                # start_id = 1 if random() < max(p1,p2)/(max(p1,p2)+max(p3,p4)) else 2
                # generate start_state "choices" that are biased towards repeating last "choice"
                prev_start_id = trial_log[-1]["start_id"]
                if prev_start_id == 1:
                    start_id = 1 if random() < 0.5 + repeat_bias else 2
                else:
                    start_id = 2 if random() < 0.5 + repeat_bias else 1

        # -----------------------------
        # 2. Common vs rare transition
        #    and second-stage state
        # -----------------------------
        second_id, transition = common_v_rare_transition(start_id, common_prob)

        # -----------------------------
        # 3. Reward depends on second_id
        # -----------------------------
        p1 = reward_probs[5]
        p2 = reward_probs[6]
        p3 = reward_probs[7]
        p4 = reward_probs[8]
        if not random_alien:
            if second_id == 3:
                reward_prob, terminal_id = (p1, 5) if random() < p1 / (p1 + p2) else (p2, 6)
            else:  # second_id == 4
                reward_prob, terminal_id = (p3, 7) if random() < p3 / (p3 + p4) else (p4, 8)
        else:
            if second_id == 3:
                reward_prob, terminal_id = (p1, 5) if random() < 0.5 else (p2, 6)
            else:  # second_id == 4
                reward_prob, terminal_id = (p3, 7) if random() < 0.5 else (p4, 8)

        reward = 1 if random() < reward_prob else 0

        # append full 3-step sequence for memory
        visited_states.extend([STATES[start_id], STATES[second_id], STATES[terminal_id]])
        rewards.extend([0, 0, reward])

        # -----------------------------
        # 4. Drift reward probabilities
        # -----------------------------
        if is_drifting:
            for i in range(5, 9):
                reward_probs[i] = drift(reward_probs[i])

        # -----------------------------
        # 5. Log trial info
        # (value estimates filled in later)
        # -----------------------------
        trial_log.append({
            "trial": trial,
            "start_id": start_id,
            "second_id": second_id,
            "terminal_id": terminal_id,
            "transition": transition,  # "common" or "rare"
            "reward": reward,  # 0/1
            "estimate_reward_state1": None,
            "estimate_reward_state2": None,
            "pred_next_state": None,  # 1 or 2 based on estimates
            "stay": None,  # 0/1 w.r.t. previous start_state
            "reward_prob_a5": p1,
            "reward_prob_a6": p2,
            "reward_prob_a7": p3,
            "reward_prob_a8": p4,
        })

    # one time-step per event (3 events per trial | n is number of trials)
    # times = gen_trials.get_time_sequence(n * 3)
    times = gen_trials.get_time_sequence_event(n, 3)

    return np.array(visited_states), np.array(rewards), times, trial_log

def gen_trials_from_data(
        subj_data
):
    """
    - First-stage states: 1, 2  (rockets)
    - Second-stage states: 3, 4 (planets)
    - Third-stage states: 5, 6, 7, 8 (aliens/reward)
    - Reward depends on second-stage state:
        p3(t), p4(t) drift independently over trials.
    """


    visited_states = []
    _rewards = subj_data['r'].tolist()
    __rewards = [[0,0,1] if r == 1 else [0,0,0] for r in _rewards]
    # flatten rewards
    rewards = list(itertools.chain.from_iterable(__rewards))

    rockets_choice = subj_data['c1'].tolist()
    landed_planet = subj_data['s'].tolist()
    alien_choice = subj_data['c2'].tolist()

    n = len(rockets_choice)



    trial_log = []

    for trial in range(n):
        start_id = rockets_choice[trial]
        second_id = 3 if landed_planet[trial] == 1 else 4
        transition = 'common' if rockets_choice[trial] == landed_planet[trial] else 'rare'

        if second_id == 3:
            terminal_id = 5 if alien_choice[trial] == 1 else 6
        else:
            terminal_id = 7 if landed_planet[trial] == 1 else 8

        visited_states.extend([STATES[start_id], STATES[second_id], STATES[terminal_id]])


        reward = 1 if _rewards[trial] == 1 else 0
        # append full 3-step sequence for memory
        rewards.extend([0, 0, reward])

        # -----------------------------
        # 4. Drift reward probabilities
        # -----------------------------


        # -----------------------------
        # 5. Log trial info
        # (value estimates filled in later)
        # -----------------------------
        trial_log.append({
            "trial": trial,
            "start_id": start_id,
            "second_id": second_id,
            "terminal_id": terminal_id,
            "transition": transition,  # "common" or "rare"
            "reward": reward,  # 0/1
            "estimate_reward_state1": None,
            "estimate_reward_state2": None,
            "pred_next_state": None,  # 1 or 2 based on estimates
            "stay": None,  # 0/1 w.r.t. previous start_state
            "reward_prob_a5": None,
            "reward_prob_a6": None,
            "reward_prob_a7": None,
            "reward_prob_a8": None,
        })

    # one time-step per event (3 events per trial | n is number of trials)
    # times = gen_trials.get_time_sequence(n * 3)
    times = gen_trials.get_time_sequence_event(n, 3)

    return np.array(visited_states), np.array(rewards), times, trial_log


def get_reward_estimates(
        states, rewards, times,
        state_1=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
        state_2=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
        metric=defaults.METRIC,
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        n_steps=3  # number of states for rollout
):
    memories = gen_memories(
        states=states,
        rewards=rewards,
        times=times,
        state_integration_rate=state_integration_rate,
    )

    r1 = estimate_reward_from_starting_state(
        memories, state_1, times[-1],
        n_steps=n_steps,
        metric=metric,
        model_based_ness=model_based_ness,
        state_d=9,
        context_d=9,
        time_retrieval_weight=time_retrieval_weight,
        state_integration_rate=state_integration_rate,
    )
    r2 = estimate_reward_from_starting_state(
        memories, state_2, times[-1],
        n_steps=n_steps,
        metric=metric,
        model_based_ness=model_based_ness,
        state_d=9,
        context_d=9,
        time_retrieval_weight=time_retrieval_weight,
        state_integration_rate=state_integration_rate,
    )

    return r1, r2


def run(
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        n_base_trials=200,
        common_prob=0.7,
):
    # vs, rw, t, trial_log = gen_trials_base(
    vs, rw, t, trial_log = gen_trials_base_persevere(
        n=n_base_trials,
        common_prob=common_prob,
    )

    for i, tr in enumerate(trial_log):
        # use memory up to and including this trial
        end = (i + 1) * 3  # 3 events per trial
        r1, r2 = get_reward_estimates(
            states=vs[:end],
            rewards=rw[:end],
            times=t[:end],
            state_1=STATES[1],
            state_2=STATES[2],
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        tr["estimate_reward_state1"] = r1
        tr["estimate_reward_state2"] = r2

        # “policy”: pick start state with higher estimated reward
        pred_next = 1 if r1 > r2 else 2
        tr["pred_next_state"] = pred_next

        # "stay" = would the model repeat the current first-stage state on the next trial?
        tr["stay"] = int(pred_next == tr["start_state"])

    return trial_log


def run_random_choices(
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        n_base_trials=200,
        common_prob=0.7,
        is_random_initial_probs=False,
        is_drifting=False
):
    """
    run with data generated from random choices
    """
    # vs, rw, t, trial_log = gen_trials_base(
    vs, rw, t, trial_log = gen_trials_base_persevere(
        n=n_base_trials,
        common_prob=common_prob,
        is_random_initial_probs=is_random_initial_probs,
        is_drifting=is_drifting
    )

    for i, tr in enumerate(trial_log):
        # use memory up to and including this trial
        end = (i + 1) * 3  # 3 events per trial
        r1, r2 = get_reward_estimates(
            states=vs[:end],
            rewards=rw[:end],
            times=t[:end],
            n_steps=3,
            state_1=STATES[1],
            state_2=STATES[2],
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        tr["estimate_rew_state1"] = r1
        tr["estimate_rew_state2"] = r2

        # get previous choice, only valid for 2nd trial onwards
        prev_choice = tr["start_id"]
        # get softmax choice probabilities
        p_choice1, p_choice2 = val_to_choice(r1, r2, prev_choice)
        # “policy”: pick start state with higher estimated reward
        pred_next = 1 if random() < p_choice1 else 2

        # pred_next = val_to_choice2(r1, r2)
        tr["pred_next_state"] = pred_next

        # "stay" = would the model repeat the current first-stage state on the next trial?
        tr["stay"] = int(pred_next == tr["start_id"])

    return trial_log


def run_human_choices(
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        subj_data=None,
):
    """
    run with data generated with human choices
    """
    # vs, rw, t, trial_log = gen_trials_base(


    vs, rw, t, trial_log = gen_trials_from_data(subj_data)



    for i, tr in enumerate(trial_log):
        # use memory up to and including this trial
        end = (i + 1) * 3  # 3 events per trial
        r1, r2 = get_reward_estimates(
            states=vs[:end],
            rewards=rw[:end],
            times=t[:end],
            n_steps=3,
            state_1=STATES[1],
            state_2=STATES[2],
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        tr["estimate_rew_state1"] = r1
        tr["estimate_rew_state2"] = r2

        # get previous choice, only valid for 2nd trial onwards
        prev_choice = tr["start_id"]
        # get softmax choice probabilities
        p_choice1, p_choice2 = val_to_choice(r1, r2, prev_choice)
        # “policy”: pick start state with higher estimated reward
        pred_next = 1 if random() < p_choice1 else 2

        # pred_next = val_to_choice2(r1, r2)
        tr["pred_next_state"] = pred_next

        # "stay" = would the model repeat the current first-stage state on the next trial?
        tr["stay"] = int(pred_next == tr["start_id"])

    return trial_log

def run_model_choices(
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        n_base_trials=200,
        common_prob=0.7,
        is_random_initial_probs=False,
        is_drifting=True
):
    # generate timestamps for each trial for EM
    times = gen_trials.get_time_sequence_event(n_base_trials, 3)

    # generate initial reward probabilities
    rew_probs = initial_reward_probabilities(is_random=is_random_initial_probs)

    # -----------------------------
    # 1. First-stage choice (rocket)
    # -----------------------------
    start_id = 1 if random() < 0.5 else 2
    # -----------------------------
    # 2. Second-stage state (planet)
    # -----------------------------
    second_id, transition = common_v_rare_transition(start_id, common_prob)
    # -----------------------------
    # 3a. Terminal-stage state (alien)
    # -----------------------------
    if second_id == 3:
        reward_prob, terminal_id = (rew_probs[5], 5) if random() < 0.5 else (rew_probs[6], 6)
    else:  # second_id == 4
        reward_prob, terminal_id = (rew_probs[7], 7) if random() < 0.5 else (rew_probs[8], 8)
    # -----------------------------
    # 3b. Terminal-stage state (reward)
    # -----------------------------
    reward = 1 if random() < reward_prob else 0
    # -----------------------------
    # Create trial log
    # -----------------------------
    visited_states = [STATES[start_id], STATES[second_id], STATES[terminal_id]]
    rewards = [0, 0, reward]
    first_trial = {
        "trial": 0,
        "start_id": start_id, "second_id": second_id, "terminal_id": terminal_id,
        "prev_transition": None,
        "prev_reward": None,
        "transition": transition,  # "common" or "rare"
        "reward": reward,  # 0/1
        "estimate_rew_state1": None, "estimate_rew_state2": None,
        "estimate_rew_alien5": None, "estimate_rew_alien6": None,
        "estimate_rew_alien7": None, "estimate_rew_alien8": None,
        "pred_next_rocket": None,  # 1 or 2 based on estimates
        "pred_next_alien": None,
        "stay": None,  # 0/1 w.r.t. previous start_state
        "reward_prob_a5": rew_probs[5], "reward_prob_a6": rew_probs[6],
        "reward_prob_a7": rew_probs[7], "reward_prob_a8": rew_probs[8],
    }

    trial_log = [first_trial]
    # -----------------------------
    # Drift reward probabilities
    # -----------------------------
    if is_drifting:
        for i in range(5, 9):
            rew_probs[i] = drift(rew_probs[i])

    for trial in range(1, n_base_trials):
        # -----------------------------
        # 1. First-stage choice (rocket)
        # -----------------------------
        end = trial * 3  # 3 events per trial
        r1, r2 = get_reward_estimates(
            states=np.array(visited_states),
            rewards=rewards,
            times=times[:end],
            n_steps=3,  # rollout of 3 for starting state
            state_1=STATES[1],
            state_2=STATES[2],
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        # get previous choice, only valid for 2nd trial onwards
        prev_rocket = trial_log[-1]["start_id"]
        # get softmax choice probabilities
        p_choice1, p_choice2 = val_to_choice(r1, r2, prev_rocket)
        # “policy”: pick start state with higher estimated reward
        # pred_next_rocket = 1 if random() < p_choice1 else 2
        pred_next_rocket_model = 1 if random() < p_choice1 else 2

        # pred_next_rocket_random = 1 if random() < .5 else 2


        # make a choice
        start_id = pred_next_rocket_model
        # start_id = pred_next_rocket_random



        # -----------------------------
        # 2. Second-stage state (planet)
        # -----------------------------
        # Cache previous transition
        prev_transition = transition
        second_id, transition = common_v_rare_transition(start_id, common_prob)

        # -----------------------------
        # 3a. Terminal-stage state (alien)
        # -----------------------------
        if second_id == 3:
            alien_states = [5, 6]
        else:
            alien_states = [7, 8]

        r_alien1, r_alien2 = get_reward_estimates(
            states=np.array(visited_states),
            rewards=rewards,
            times=times[:end],
            n_steps=2,  # "rollout" of 1 for terminal state
            state_1=STATES[alien_states[0]],
            state_2=STATES[alien_states[1]],
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        if second_id == 3:
            estimate_rew_alien5 = r_alien1
            estimate_rew_alien6 = r_alien2
            estimate_rew_alien7 = None
            estimate_rew_alien8 = None
        else:
            estimate_rew_alien7 = r_alien1
            estimate_rew_alien8 = r_alien2
            estimate_rew_alien5 = None
            estimate_rew_alien6 = None

        p_choice_alien1, p_choice_alien2 = val_to_choice(r_alien1, r_alien2, prev_choice=0)

        pred_next_alien = alien_states[0] if random() < p_choice_alien1 else alien_states[1]
        terminal_id = pred_next_alien

        # pred_next_alien_random = alien_states[0] if random() < .5 else alien_states[1]
        # terminal_id = pred_next_alien_random

        reward_prob = rew_probs[terminal_id]

        # Cache previous reward for logging
        prev_reward = reward
        reward = 1 if random() < reward_prob else 0

        # -----------------------------
        # Create trial log
        # -----------------------------
        visited_states.extend(
            [STATES[start_id], STATES[second_id], STATES[terminal_id]])
        rewards.extend([0, 0, reward])
        # reward_prob = rew_probs[pred_next_alien]
        # reward = 1 if random() < reward_prob else 0

        trial_ = [{
            "trial": trial,
            "start_id": start_id, "second_id": second_id, "terminal_id": terminal_id,
            "prev_transition": prev_transition,  # "common" or "rare"
            "prev_reward": prev_reward,  # 0/1
            "transition": transition,  # "common" or "rare"
            "reward": reward,  # 0/1
            "estimate_rew_state1": r1, "estimate_rew_state2": r2,
            "estimate_rew_alien5": estimate_rew_alien5, "estimate_rew_alien6": estimate_rew_alien6,
            "estimate_rew_alien7": estimate_rew_alien7, "estimate_rew_alien8": estimate_rew_alien8,
            "pred_next_rocket": pred_next_rocket_model,  # 1 or 2 based on estimates
            "pred_next_alien": pred_next_alien,
            # "stay": int(start_id == prev_rocket),  # 0/1 w.r.t. previous start_state
            "stay": int(pred_next_rocket_model == prev_rocket),  # 0/1 w.r.t. previous start_state
            "reward_prob_a5": rew_probs[5], "reward_prob_a6": rew_probs[6],
            "reward_prob_a7": rew_probs[7], "reward_prob_a8": rew_probs[8],
        }]
        trial_log.extend(trial_)


        # -----------------------------
        # 4. Drift reward probabilities
        # -----------------------------
        if is_drifting:
            for i in range(5, 9):
                rew_probs[i] = drift(rew_probs[i])

    return trial_log
