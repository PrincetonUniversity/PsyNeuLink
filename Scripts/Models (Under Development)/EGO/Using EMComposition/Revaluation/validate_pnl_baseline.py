"""
Validate pnl model vs python model by
(1) memory
(2) retrieval with arbitrary cue
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
    experiment_1 = gen_experiment_reward_reval()

    test_query_state = [0, 1, 0, 0, 0, 0, 0]
    test_query_context = [0, .6, 0, 0, 0, 0, .2]
    test_query_time = experiment_1.times[30]

    if RUN_PY:
        # === PY Model (custom python) === #
        memories_python = model_python.gen_memories(
            states=experiment_1.states,
            rewards=experiment_1.rewards,
            times=experiment_1.times,
        )

        # memories for comparison
        mem_py_state = memories_python[0]
        mem_py_context = memories_python[1]
        mem_py_time = memories_python[2]
        mem_py_reward = memories_python[3]

        # query one memory
        sample_py = model_python.sample_memory(
            time_retrieval_weight=.1,
            context_retrieval_weight=0,
            state_retrieval_weight=.9,
            memories=memories_python,
            query=(test_query_state, test_query_context, test_query_time, 0),
            metric='dot_product',
            mode='softmax'
        )

        (sample_py_state, sample_py_context,
         sample_py_time, sample_py_reward, _) = sample_py

    if RUN_PNL:
        # === PY Model (custom python) === #
        model_pnl, state_input, time_input, reward_input, task_input = construct_model(
            capacity=experiment_1.n_trials,
        )

        pnl_input = {
            state_input: experiment_1.states,
            time_input: experiment_1.times,
            reward_input: experiment_1.rewards,
            task_input: experiment_1.tasks,
        }

        model_pnl.run(inputs=pnl_input)

        memories_pnl = model_pnl.nodes['EM'].parameters.memory.get(model_pnl.name)

        mem_pnl_state = np.asarray([mem[0] for mem in memories_pnl])
        mem_pnl_time = np.asarray([mem[1] for mem in memories_pnl])
        mem_pnl_context = np.asarray([mem[2] for mem in memories_pnl])
        mem_pnl_reward = np.asarray([mem[3] for mem in memories_pnl])

        # run a query
        query_input = {
            state_input: [test_query_state],
            time_input: [test_query_time],
            reward_input: [0],
            task_input: [0],
        }
        model_pnl.run(inputs=query_input)

        RETRIEVED = ' [RETRIEVED]'
        sample_pnl_state = model_pnl.nodes['EM'].nodes['STATE' + RETRIEVED].value
        sample_pnl_context = model_pnl.nodes['EM'].nodes['CONTEXT' + RETRIEVED].value
        sample_pnl_time = model_pnl.nodes['EM'].nodes['TIME' + RETRIEVED].value
        sample_pnl_reward = model_pnl.nodes['EM'].nodes['REWARD' + RETRIEVED].value

    if RUN_PNL and RUN_PY:
        _TOL = 1e-24

        # === ASSERT MEMORIES ARE THE SAME === #
        assert np.allclose(mem_py_state, mem_pnl_state, atol=_TOL), "State memory mismatch"
        assert np.allclose(mem_py_context, mem_pnl_context, atol=_TOL), "Context memory mismatch"
        assert np.allclose(mem_py_time, mem_pnl_time, atol=_TOL), "Time memory mismatch"
        assert np.allclose(mem_py_reward, mem_pnl_reward, atol=_TOL), "Reward memory mismatch"

        print(f'All memories are close with a precision of {_TOL}')

        assert np.allclose(sample_py_state, sample_pnl_state, atol=_TOL), "State sample mismatch"
        assert np.allclose(sample_py_context, sample_pnl_context, atol=_TOL), "Context sample mismatch"
        assert np.allclose(sample_py_time, sample_pnl_time, atol=_TOL), "Time sample mismatch"
        assert np.allclose(sample_py_reward, sample_pnl_reward, atol=_TOL), "Reward sample mismatch"

        print(f'Sample memories are close with a precision of {_TOL}')


if __name__ == "__main__":
    main()
