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
        targets = [[1,1,1,1,1]]
        BIAS = True
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5

        # Set up models
        # PNL:
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=BIAS)
        gru = GRUComposition(input_size=3, hidden_size=5, bias=True)
        gru.set_weights_from_torch_gru(torch_gru)
        target_node = gru.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
        torch_optimizer = torch.optim.SGD(lr=gru.learning_rate, params=torch_gru.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')

        # Save weights of torch GRU before learning, to initialize PNL GRU with same weights below
        torch_weights_before_learning = {'wts_hh': torch_gru.weight_hh_l0.data.detach().clone(),
                                        'wts_ih': torch_gru.weight_ih_l0.data.detach().clone(),
                                        'bias_hh': torch_gru.bias_hh_l0.data.detach().clone(),
                                        'bias_ih': torch_gru.bias_ih_l0.data.detach().clone()}

        # Execute Torch GRU Node
        h0 = torch.tensor([[0,0,0,0,0]], dtype=torch.float32)
        torch_result_before_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),h0)
        # print("\nTORCH before learning:         ",
        #       [float(f"{value:.4f}") for value in torch_result_before_learning.flatten()])
        torch_optimizer.zero_grad()  # Zero the gradients before each optimization step.
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets, dtype=torch.float32))
        torch_loss.backward()
        torch_optimizer.step()  # backprop to update context module weights.
        torch_result_after_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),hn)

        gru.gru_mech.function.weight_hh_l0.data.copy_(torch_weights_before_learning['wts_hh'])
        gru.gru_mech.function.weight_ih_l0.data.copy_(torch_weights_before_learning['wts_ih'])
        gru.gru_mech.function.bias_hh_l0.data.copy_(torch_weights_before_learning['bias_hh'])
        gru.gru_mech.function.bias_ih_l0.data.copy_(torch_weights_before_learning['bias_ih'])

        pnl_result_before_learning = gru.run(inputs={gru.input_node:[[1,2,3]]})
        x = gru.learn(inputs={gru.input_node:[[1,2,3]], target_node[0]: [[1,1,1,1,1]]},
                  execution_mode=pnl.ExecutionMode.PyTorch)
        pnl_result_after_learning = gru.run(inputs={gru.input_node:[[1,2,3]]})

        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(), pnl_result_before_learning, atol=1e-6)
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(), pnl_result_after_learning, atol=1e-6)
