from functools import partial

import numpy as np

import psyneulink as pnl

import params as params
import data as data
from model_pnl import construct_model

N_SEQ_BASELINE = 20
N_SEQ_REVAL = 20

N_STATES_BASELINE = 3
N_STATES_REVAL = 2


def get_trials(
        get_reval_trials
):
    n_baseline_trials = N_SEQ_BASELINE * N_STATES_BASELINE
    n_reval_trials = N_SEQ_REVAL * N_STATES_REVAL
    memory_capacity = n_baseline_trials + n_reval_trials

    states_baseline, rewards_baseline = data.get_baseline_trials(num_seqs=N_SEQ_BASELINE)
    states_reward_reval, rewards_reward_reval = get_reval_trials(num_seqs=N_SEQ_REVAL)

    times = data.get_time_sequence(num_trials=memory_capacity)

    return (states_baseline, rewards_baseline, times[:n_baseline_trials], [0] * n_baseline_trials,
            states_reward_reval, rewards_reward_reval, times[n_baseline_trials:], [0] * n_reval_trials,
            memory_capacity)


get_trials_reward_reval = partial(
    get_trials,
    get_reval_trials=data.get_reward_revaluation_trials
)

get_trials_transition_reval = partial(
    get_trials,
    get_reval_trials=data.get_transition_revaluation_trials
)

def get_prediction(
        model,
        state,
        time,
        _state_input,
        _time_input,
        _reward_input,
        _task_input
):
    prediction_input = {
        _task_input: [1, 3, 2, 3, 2],
        _state_input: [state] * 5,
        _time_input: [time] * 5,
        _reward_input: [0] * 5,
    }

    model.run(inputs=prediction_input)

    return model.results[-1][1]



def run_reval(
        get_trials_func,
):
    (states_baseline, rewards_baseline, times_baseline, tasks_baseline,
     states_reval, rewards_reval, times_reval, tasks_reval,
     memory_capacity) = get_trials_func()

    model, _state_input, _time_input, _reward_input, _task_input = construct_model(
        capacity=memory_capacity,
        context_retrieval_in_sim=0,
        time_retrieval_weight=.2
    )

    inputs_baseline = {
        _state_input: states_baseline,
        _time_input: times_baseline,
        _reward_input: rewards_baseline,
        _task_input: tasks_baseline,
    }

    model.run(inputs=inputs_baseline)

    _get_prediction = partial(
        get_prediction,
        model=model,
        _state_input=_state_input,
        _time_input=_time_input,
        _reward_input=_reward_input,
        _task_input=_task_input
    )
    state_one = [0, 1, 0, 0, 0, 0, 0]
    state_two = [0, 0, 1, 0, 0, 0, 0]

    estimate_one_baseline = _get_prediction(state=state_one, time=times_baseline[-1])
    estimate_two_baseline = _get_prediction(state=state_two, time=times_baseline[-1])

    print(estimate_one_baseline, estimate_two_baseline)

    inputs_reval = {
        _state_input: states_reval,
        _time_input: times_reval,
        _reward_input: rewards_reval,
        _task_input: tasks_reval,
    }

    model.run(inputs=inputs_reval)

    estimate_one_reval = _get_prediction(state=state_one, time=times_reval[-1])
    estimate_two_reval = _get_prediction(state=state_two, time=times_reval[-1])

    print(estimate_one_reval, estimate_two_reval)



run_reward_reval = partial(
    run_reval,
    get_trials_func=get_trials_reward_reval,
)

run_transition_reval = partial(
    run_reval,
    get_trials_func=get_trials_transition_reval,
)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    run_reward_reval()
    # run_transition_reval()
    # seq_baseline = 20
    #
    # memory_capacity = seq_baseline * 3
    # model, _state_input, _time_input, _reward_input, _task_input = construct_model(memory_capacity)
    #
    #
    #
    # states, rewards = data.get_baseline_trials(num_seqs=seq_baseline)
    #
    # times = data.get_time_sequence(num_trials=len(states))
    #
    # inputs = {
    #     _state_input: states,
    #     _time_input: times,
    #     _reward_input: rewards,
    #     _task_input: [0] * len(states),
    # }
    #
    #
    # def _cb():
    #     print('*' * 10)
    #
    #
    # def _ca():
    #     print('*' * 10)
    #
    #
    # model.run(inputs=inputs,
    #           # call_before_trial=_cb,
    #           # call_after_trial=_ca
    #           )
    # _memory = model.nodes[EM_NAME].parameters.memory.get(MODEL_NAME)
    #
    # print('*' * 10)
    # print()
    # print(model.results)
    # #
    # prediction_task_baseline = {
    #     _task_input: [1, 3, 2, 3, 2],
    #     _state_input: [[0, 1, 0, 0, 0, 0, 0]] * 5,
    #     _time_input: [times[-1]] * 5,
    #     _reward_input: [0] * 5,
    # }
    #
    # model.run(
    #     inputs=prediction_task_baseline,
    #     call_before_trial=_cb,
    #     call_after_trial=_ca
    # )
    # print('*' * 10)
    # print('Estimated Reward State 1 (~10)')
    # print(model.results[-1])
    # print('*' * 10)
