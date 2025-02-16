import numpy as np
import pytest

import psyneulink as pnl

from psyneulink.library.compositions.grucomposition.grucomposition import GRUComposition, GRUCompositionError

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of GRUComposition class

@pytest.mark.pytorch
@pytest.mark.composition
class TestExecution:
    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_pytorch_execution_identicality(self, bias):
        import torch
        inputs = [[1,2,3]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5

        h0 = torch.tensor(np.array([[0,0,0,0,0]]).astype(np.float32))
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        result, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),h0)
        torch_results = [result.detach().numpy()]
        result, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),hn)
        torch_results.append(result.detach().numpy())

        gru = GRUComposition(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        gru.set_weights_from_torch_gru(torch_gru)
        gru.run(inputs={gru.input_node:inputs}, num_trials=2)

        np.testing.assert_allclose(torch_results, gru.results, atol=1e-6)

    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_pytorch_learning_identicality(self, bias):
        import torch
        inputs = [[1,2,3]]
        seed = 42
        BIAS = True
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5

        # Set up models
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=BIAS)
        gru = GRUComposition(input_size=3, hidden_size=5, bias=True)
        gru.set_weights_from_torch_gru(torch_gru)
        targets = gru.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)

        # Execute Torch GRU Node
        h0 = torch.tensor([[0,0,0,0,0]], dtype=torch.float32)
        torch_result_before_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),h0)
        # print("\nTORCH before learning:         ",
        #       [float(f"{value:.4f}") for value in torch_result_before_learning.flatten()])
        torch_result_before_learning.backward(torch.tensor([[1,1,1,1,1]], dtype=torch.float32),retain_graph=True)
        torch_result_after_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),hn)
        # torch_result_after_learning.backward(torch.tensor([[1,1,1,1,1]], dtype=torch.float32))
        # print("TORCH after learning:         ",
        #       [float(f"{value:.4f}") for value in torch_result_after_learning.flatten()])

        # Execute PNL GRUComposition
        # result = gru.learn(inputs={gru.input_node:[[1,2,3]],
        #                            targets[0]: [[1,1,1,1,1]]},
        #                    execution_mode=ExecutionMode.PyTorch)
        pnl_result_before_learning = gru.run(inputs={gru.input_node:[[1,2,3]]})
        # print("\nPNL before learning: ", pnl_result_before_learning)
        pnl_result_after_learning = gru.learn(inputs={gru.input_node:[[1,2,3]],
                                   targets[0]: [[1,1,1,1,1]]},
                           execution_mode=pnl.ExecutionMode.PyTorch)
        # pnl_result_after_learning = gru.learn(inputs={gru.input_node:[[1,2,3],[1,2,3]],
        #                            targets[0]: [[1,1,1,1,1],[1,1,1,1,1]]},
        #                    execution_mode=ExecutionMode.PyTorch)
        print("PNL after learning: ", pnl_result_after_learning)

        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(), pnl_result_before_learning, atol=1e-6)
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(), pnl_result_after_learning, atol=1e-6)
