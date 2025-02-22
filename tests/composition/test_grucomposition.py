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
    def test_pytorch_execution_identicality_with_pytorch(self, bias):
        import torch
        inputs = [[1,2,3]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5

        h0 = torch.tensor(np.array([[0,0,0,0,0]]).astype(np.float32))
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_weights_from_torch_gru(torch_gru)
        result, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),h0)
        torch_results = [result.detach().numpy()]
        result, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),hn)
        torch_results.append(result.detach().numpy())

        gru = GRUComposition(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        # gru.set_weights_from_torch_gru(torch_gru)
        gru.set_weights(*torch_gru_initial_weights)
        gru.run(inputs={gru.input_node:inputs}, num_trials=2)

        np.testing.assert_allclose(torch_results, gru.results, atol=1e-6)

    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_pytorch_learning_identicality_with_pytorch(self, bias):
        import torch
        inputs = [[1,2,3]]
        targets = [[1,1,1,1,1]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001

        bias = True

        # Set up TORCH GRU model
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_gru.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_weights_from_torch_gru(torch_gru)
        # Execute Torch model
        h0 = torch.tensor([[0,0,0,0,0]], dtype=torch.float32)
        torch_result_before_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),h0)
        torch_optimizer.zero_grad()
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets, dtype=torch.float32))
        torch_loss.backward()
        torch_optimizer.step()
        torch_result_after_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),hn)

        # Set up PNL model
        # Initialize GRU Node of PNL with starting weights from Torch GRU, so that they start identically
        pnl_gru = GRUComposition(input_size=3, hidden_size=5, bias=bias, learning_rate=LEARNING_RATE)
        pnl_gru.set_weights(*torch_gru_initial_weights)
        target_node = pnl_gru.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
        # Execute PNL GRUComposition
        pnl_result_before_learning = pnl_gru.run(inputs={pnl_gru.input_node:[[1,2,3]]})
        pnl_gru.learn(inputs={pnl_gru.input_node:[[1,2,3]], target_node[0]: [[1,1,1,1,1]]},
                      execution_mode=pnl.ExecutionMode.PyTorch)
        # Need to run it one more time (due to lazy updating) to see the effects of learning
        pnl_result_after_learning = pnl_gru.run(inputs={pnl_gru.input_node:[[1,2,3]]})

        # Compare results from before learning:
        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(), pnl_result_before_learning, atol=1e-6)
        # Compare results from after learning:
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(), pnl_result_after_learning, atol=1e-6)
