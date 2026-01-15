import numpy as np
from random import random

from .src import gen_trials
from .model_python import gen_memories, estimate_reward_from_starting_state
from .config import defaults


def drift(p, sigma=0.125, lo=0.25, hi=0.75):
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
            start_id = 1 if random() < max(p1,p2)/(max(p1,p2)+max(p3,p4)) else 2
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
            reward_prob, terminal_id = (p1, 5) if random() < p1/(p1+p2) else (p2, 7)
        else:  # second_id == 4
            # reward_prob, terminal_id = (p3, 6) if random() < 0.5 else (p4, 8)
            reward_prob, terminal_id = (p3, 6) if random() < p3/(p3+p4) else (p4, 8)

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
    times = gen_trials.get_time_sequence(n * 3)

    return np.array(visited_states), np.array(rewards), times, trial_log


def get_reward_estimates(
        states, rewards, times,
        state_1=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
        state_2=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
        metric=defaults.METRIC,
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
):
    memories = gen_memories(
        states=states,
        rewards=rewards,
        times=times,
        state_integration_rate=state_integration_rate,
    )

    r1 = estimate_reward_from_starting_state(
        memories, state_1, times[-1],
        metric=metric,
        model_based_ness=model_based_ness,
        state_d=9,
        context_d=9,
        time_retrieval_weight=time_retrieval_weight,
    )
    r2 = estimate_reward_from_starting_state(
        memories, state_2, times[-1],
        metric=metric,
        model_based_ness=model_based_ness,
        state_d=9,
        context_d=9,
        time_retrieval_weight=time_retrieval_weight,
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
    vs, rw, t, trial_log = gen_trials_base(
        n=n_base_trials,
        common_prob=common_prob,
    )

    state_1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    state_2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])


    for i, tr in enumerate(trial_log):
        # use memory up to and including this trial
        end = (i + 1) * 3  # 3 events per trial
        r1, r2 = get_reward_estimates(
            states=vs[:end],
            rewards=rw[:end],
            times=t[:end],
            state_1=state_1,
            state_2=state_2,
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        tr["estimate_state_1"] = r1
        tr["estimate_state_2"] = r2

        # “policy”: pick start state with higher estimated reward
        pred_next = 1 if r1 > r2 else 2
        tr["pred_next_state"] = pred_next

        # "stay" = would the model repeat the current first-stage state on the next trial?
        tr["stay"] = int(pred_next == tr["start_state"])

    return trial_log

def get_state_one_hot(state_id):
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
    return states[state_id]

# softmax value-to-choice rule with perseveration bias from Daw et al, 2011
# simply replace Q-value with the value retrieved from Episodic Memory
def val_to_choice(
        candstate1: int, # candidate state 1
        candstate2: int, # candidate state 2
        val1: float, # value retrieved from memory for candidate choice "1"
        val2: float, # value retrieved from memory for candidate choice "2"
        prev_choice = np.array([]), # np.array, identity of previous 1st stage choice
        temp: float = 1, # temperature for softmax
        bias: float = 0, # perseveration bias, only for 1st stage ('rocket') choice
        first_stage_choice: bool = True):

    # determine if choice in question is repetition from previous trial
    if prev_choice.size == 0: # if first trial, no previous choice
        repeat1 = 0; repeat2 = 0
    else:
        repeat1 = all(get_state_one_hot(candstate1) == prev_choice)
        repeat2 = all(get_state_one_hot(candstate2) == prev_choice)

    # numerator of softmax for each candidate choice
    choice1 = np.exp(temp*(val1 + bias*repeat1))
    choice2 = np.exp(temp*(val2 + bias*repeat2))

    # calculate softmax probability for each candidate choice
    p_choice1 = choice1 / (choice1 + choice2)
    p_choice2 = choice2 / (choice1 + choice2)

    return p_choice1, p_choice2

def run2(
        metric='dot_product',
        model_based_ness=defaults.MODEL_BASED_NESS,
        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
        n_base_trials=200,
        common_prob=0.7,
):
    vs, rw, t, trial_log = gen_trials_base(
        n=n_base_trials,
        common_prob=common_prob,
    )

    state_1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    state_2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])


    for i, tr in enumerate(trial_log):
        # use memory up to and including this trial
        end = (i + 1) * 3  # 3 events per trial
        r1, r2 = get_reward_estimates(
            states=vs[:end],
            rewards=rw[:end],
            times=t[:end],
            state_1=state_1,
            state_2=state_2,
            metric=metric,
            model_based_ness=model_based_ness,
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
        )

        tr["estimate_state_1"] = r1
        tr["estimate_state_2"] = r2

        # get previous choice, only valid for 2nd trial onwards
        prev_choice = np.array([]) if i == 0 else get_state_one_hot(trial_log[i-1]["start_state"])
        # get softmax choice probabilities
        candstate1 = 1; candstate2 = 2
        p_choice1, p_choice2 = val_to_choice(candstate1, candstate2, r1, r2, prev_choice)

        # “policy”: pick start state with higher estimated reward
        pred_next = 1 if random() < p_choice1 else 2
        tr["pred_next_state"] = pred_next

        # "stay" = would the model repeat the current first-stage state on the next trial?
        tr["stay"] = int(pred_next == tr["start_state"])

    return trial_log
