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
        # Test identicality of learning results of PyTorch GRU against native Pytorch GRU
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
    def test_pytorch_unnested_learning_identicality_with_pytorch(self, bias):
        # Test identicality of learning results of GRUComposition against native Pytorch GRU

        import torch
        inputs = [[1,2,3]]
        targets = [[1,1,1,1,1]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001

        # Set up and run torch model -------------------------------------

        # Set up torch model
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_gru.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_weights_from_torch_gru(torch_gru)

        # Execute model without learning
        h0 = torch.tensor([[0,0,0,0,0]], dtype=torch.float32)
        torch_result_before_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),h0)

        # Compute loss and update weights
        torch_optimizer.zero_grad()
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets, dtype=torch.float32))
        torch_loss.backward()
        torch_optimizer.step()

        # Compute loss and update weights
        torch_result_after_learning, hn = torch_gru(torch.tensor(np.array(inputs).astype(np.float32)),hn)

        # Set up and run PNL Autodiff model -------------------------------------

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

    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_nested_gru_composition_learning(self, bias):
        # Test identicality of results of nested GRUComposition and pure pytorch version

        import torch
        import torch.optim as optim
        inputs = [[1,2,3]]
        targets = [[1,1,1,1,1]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001

        GRU_node_values = {}

        # Set up and run TORCH GRU model
        class TorchModel(torch.nn.Module):
            def __init__(self):
                super(TorchModel, self).__init__()
                # FIX: MAKE SURE BIASES ARE NOT USED FOR THESE MODULES:
                self.input = torch.nn.Linear(INPUT_SIZE, INPUT_SIZE, bias=False)
                self.input.weight.requires_grad = False # Since weights to INPUT nodes of Composition are not learnable
                self.gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
                self.output = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
                self.gru.register_forward_hook(TorchModel._pytorch_GRU_module_values_hook)

            def forward(self, x, hidden):
                after_input = self.input(x)
                after_gru, hidden_state = self.gru(after_input, hidden)
                after_output = self.output(after_gru)
                return after_output, hidden_state

            def _pytorch_GRU_module_values_hook(module, input, output):
                in_len = module.input_size
                hid_len = module.hidden_size
                z_idx = hid_len
                n_idx = 2 * hid_len

                ih = module.weight_ih_l0
                hh = module.weight_hh_l0
                if module.bias:
                    b_ih = module.bias_ih_l0
                    b_hh = module.bias_hh_l0
                else:
                    b_ih = torch.tensor(np.array([0] * 3 * hid_len))
                    b_hh = torch.tensor(np.array([0] * 3 * hid_len))

                w_ir = ih[:z_idx].T
                w_iz = ih[z_idx:n_idx].T
                w_in = ih[n_idx:].T
                w_hr = hh[:z_idx].T
                w_hz = hh[z_idx:n_idx].T
                w_hn = hh[n_idx:].T

                b_ir = b_ih[:z_idx]
                b_iz = b_ih[z_idx:n_idx]
                b_in = b_ih[n_idx:]
                b_hr = b_hh[:z_idx]
                b_hz = b_hh[z_idx:n_idx]
                b_hn = b_hh[n_idx:]

                # assert len(input) > 1, (f"PROGRAM ERROR: _pytorch_GRU_module_values_hook hook received only one tensor "
                #                         f"in its input argument: {input}; either the input or hidden state is missing.")
                x = input[0]
                h = input[1] if len(input) > 1 else torch.tensor([[0] * module.hidden_size], dtype=torch.float32)
                # h = input[1]

                # Reproduce GRU forward calculations
                r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
                z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
                n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
                h_t = (1 - z_t) * n_t + z_t * h

                # Put internal calculations in dict with corresponding node names as keys
                GRU_node_values['RESET'] = r_t.detach()
                GRU_node_values['UPDATE'] = z_t.detach()
                GRU_node_values['NEW'] = n_t.detach()
                GRU_node_values['HIDDEN'] = h_t.detach()

        torch_model = TorchModel()
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_model.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')

        # Get initial weights (to initialize autodiff below with same initial conditions)
        torch_input_initial_weights = torch_model.state_dict()['input.weight'].T.detach().cpu().numpy().copy()
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_weights_from_torch_gru(torch_model.gru)
        torch_output_initial_weights = torch_model.state_dict()['output.weight'].T.detach().cpu().numpy().copy()

        # Execute Torch model without learning
        hidden_init = torch.tensor([[0,0,0,0,0]], dtype=torch.float32)
        torch_result_before_learning, hidden_state = torch_model(torch.tensor(np.array(inputs).astype(np.float32)),
                                                                 hidden_init)
        torch_result_before_learning, hidden_state = torch_model(torch.tensor(np.array(inputs).astype(np.float32)),
                                                                 hidden_state)
        # Compute loss and update weights
        torch_optimizer.zero_grad()
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets, dtype=torch.float32))
        torch_loss.backward()
        torch_optimizer.step()

        # Get results after learning
        torch_result_after_learning, hidden_state = torch_model(torch.tensor(np.array(inputs).astype(np.float32)),
                                                                hidden_state)

        # Set up and run PNL Autodiff model
        input_mech = pnl.ProcessingMechanism(name='INPUT MECH', input_shapes=3)
        output_mech = pnl.ProcessingMechanism(name='OUTPUT MECH', input_shapes=5)
        gru = GRUComposition(name='GRU COMP',
                             input_size=3, hidden_size=5, bias=bias, learning_rate = LEARNING_RATE)
        autodiff_comp = pnl.AutodiffComposition(name='OUTER COMP',
                                   pathways=[input_mech, gru, output_mech],
                                                learning_rate = LEARNING_RATE)
        autodiff_comp.projections[0].learnable = False
        autodiff_comp.set_weights(autodiff_comp.nodes[0].path_afferents[0], torch_input_initial_weights)
        autodiff_comp.nodes['GRU COMP'].set_weights(*torch_gru_initial_weights)
        autodiff_comp.set_weights(autodiff_comp.projections[1], torch_output_initial_weights)
        target_mechs = autodiff_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)

        # Execute autodiff without learning
        autodiff_result_before_learning = autodiff_comp.run(inputs={input_mech:inputs}, num_trials=2)
        totals = [i.sum().item() for i in list(autodiff_comp._build_pytorch_representation().parameters())]

        # Execute autodiff with learning
        autodiff_comp.learn(inputs={input_mech:inputs, target_mechs[0]: targets},
                            execution_mode=pnl.ExecutionMode.PyTorch)
        # Get results after learning
        autodiff_result_after_learning = autodiff_comp.run(inputs={input_mech:inputs})
        new_totals = [i.sum().item() for i in list(autodiff_comp.pytorch_representation.parameters())]

        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(),
                                   autodiff_result_before_learning, atol=1e-6)

        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(),
                                   autodiff_result_after_learning, atol=1e-6)
