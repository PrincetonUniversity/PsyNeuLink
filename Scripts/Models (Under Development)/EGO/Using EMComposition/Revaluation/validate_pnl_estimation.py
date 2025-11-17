"""
Validate pnl model vs python model by reward estimation
"""

import warnings

warnings.filterwarnings('ignore')

from ego_revaluation.src.gen_trials import gen_experiment_reward_reval
from ego_revaluation.model_pnl import construct_model

from ego_revaluation import model_python

import numpy as np

RUN_PY = True
RUN_PNL = True


def main():
    # === Trial Data === #
    experiment = gen_experiment_reward_reval()

    state_1 = [0, 1, 0, 0, 0, 0, 0]
    state_2 = [0, 0, 1, 0, 0, 0, 0]

    if RUN_PY:
        # === PY Model (custom python) === #
        memories_python = model_python.gen_memories(
            states=experiment.states,
            rewards=experiment.rewards,
            times=experiment.times,
        )

        # memories for comparison
        mem_py_state = memories_python[0]
        mem_py_context = memories_python[1]
        mem_py_time = memories_python[2]
        mem_py_reward = memories_python[3]

        # query one memory
        reward_estimate_state_1_py = model_python.estimate_reward_from_starting_state(
            memories_python, np.array(state_1), experiment.times[-1]
        )
        reward_estimate_state_2_py = model_python.estimate_reward_from_starting_state(
            memories_python, np.array(state_2), experiment.times[-1]
        )

    if RUN_PNL:
        # === PY Model (custom python) === #
        model_pnl, state_input, time_input, reward_input, task_input = construct_model(
            capacity=experiment.n_trials,
        )

        pnl_input = {
            state_input: experiment.states,
            time_input: experiment.times,
            reward_input: experiment.rewards,
            task_input: experiment.tasks,
        }

        model_pnl.run(inputs=pnl_input)

        memories_pnl = model_pnl.nodes['EM'].parameters.memory.get(model_pnl.name)

        mem_pnl_state = np.asarray([mem[0] for mem in memories_pnl])
        mem_pnl_time = np.asarray([mem[1] for mem in memories_pnl])
        mem_pnl_context = np.asarray([mem[2] for mem in memories_pnl])
        mem_pnl_reward = np.asarray([mem[3] for mem in memories_pnl])

        # run a query
        prediction_state_1 = {
            task_input: [1, 3, 2, 3, 2],
            state_input: [state_1] * 5,
            time_input: [experiment.times[-1]] * 5,
            reward_input: [0] * 5,
        }
        model_pnl.run(inputs=prediction_state_1)

        _res = model_pnl.results
        reward_estimate_state_1_pnl = _res[-1][1] + _res[-3][1] + _res[-5][1]

        prediction_state_2 = {
            task_input: [1, 3, 2, 3, 2],
            state_input: [state_2] * 5,
            time_input: [experiment.times[-1]] * 5,
            reward_input: [0] * 5,
        }

        model_pnl.run(inputs=prediction_state_2)

        _res = model_pnl.results
        reward_estimate_state_2_pnl = _res[-1][1] + _res[-3][1] + _res[-5][1]

    if RUN_PNL and RUN_PY:
        _TOL_MEM = 1e-24

        # === ASSERT MEMORIES ARE THE SAME === #
        assert np.allclose(mem_py_state, mem_pnl_state, atol=_TOL_MEM), "State memory mismatch"
        assert np.allclose(mem_py_context, mem_pnl_context, atol=_TOL_MEM), "Context memory mismatch"
        assert np.allclose(mem_py_time, mem_pnl_time, atol=_TOL_MEM), "Time memory mismatch"
        assert np.allclose(mem_py_reward, mem_pnl_reward, atol=_TOL_MEM), "Reward memory mismatch"

        print(f'All memories are close with a precision of {_TOL_MEM}')

        _TOL_ESTIMATE = 1e-3

        assert np.isclose(reward_estimate_state_1_py, reward_estimate_state_1_pnl[0],
                          atol=_TOL_ESTIMATE), f"Reward state mismatch {reward_estimate_state_1_py}, {reward_estimate_state_1_pnl}"
        assert np.isclose(reward_estimate_state_2_py, reward_estimate_state_2_pnl[0],
                          atol=_TOL_ESTIMATE), f"Reward state mismatch {reward_estimate_state_2_py}, {reward_estimate_state_2_pnl}"

        print(f'All rewards are close with a precision of {_TOL_ESTIMATE}')


if __name__ == "__main__":
    main()
