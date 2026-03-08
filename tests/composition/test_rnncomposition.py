import pytest

import psyneulink as pnl
from psyneulink.library.compositions.autodiffcomposition import torch_available

import numpy as np


if torch_available:
    from psyneulink.library.compositions.rnncomposition.rnncomposition import RNNComposition


    @pytest.mark.pytorch
    @pytest.mark.composition
    def test_forward_linear_passthrough_exact():
        rnn = pnl.RNNComposition(
            input_size=1,
            hidden_size=1,
            bias=False,
            hidden_function=pnl.Linear(slope=1.0, intercept=0.0),
            state_integration_function=pnl.LinearCombination(weights=[1.0, 0.0]),
        )

        rnn.wts_ih.parameters.matrix.set(np.array([[2.0]]))
        rnn.wts_hh.parameters.matrix.set(np.array([[3.0]]))

        inputs = [[1.0], [2.0], [0.0]]
        results = rnn.run(inputs={rnn.input_node: inputs}, num_trials=3)

        expected = [
            np.array([[2.0]]),
            np.array([[10.0]]),
            np.array([[30.0]]),
        ]

        assert len(rnn.results) == 3
        for result, exp in zip(rnn.results, expected):
            np.testing.assert_allclose(result, exp, atol=1e-8)

        np.testing.assert_allclose(results[-1], expected[-1], atol=1e-8)