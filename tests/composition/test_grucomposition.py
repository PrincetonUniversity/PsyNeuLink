import numpy as np
import pytest

import psyneulink as pnl
from psyneulink import CompositionError

from psyneulink.library.compositions.grucomposition.grucomposition import GRUComposition

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# @pytest.mark.pytorch
# @pytest.fixture(scope='module')
# def global_torch_dtype():
#     import torch
#     torch_dtype = torch.float64
#     entry_torch_dtype = torch.get_default_dtype()
#     torch.set_default_dtype(torch_dtype)
#     yield torch_dtype
#     torch.set_default_dtype(entry_torch_dtype)

# ---------------------
# HOOK FOR torch.GRU module for use in debugging internal calculations

def _pytorch_gru_module_values_hook(module, input, output):
    import torch
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
        b_ih = torch.tensor([0] * 3 * hid_len)
        b_hh = torch.tensor([0] * 3 * hid_len)

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
    h = input[1] if len(input) > 1 else torch.tensor([[0] * module.hidden_size])
    # h = input[1]

    # Reproduce GRU forward calculations
    r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
    z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
    n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h

    module.gru_hook_values = {}

    # Put internal calculations in dict with corresponding node names as keys
    module.gru_hook_values = {}  # IF THIS FAILS, MAKE gru_hook_values A GLOBAL
    module.gru_hook_values['RESET'] = r_t.detach()
    module.gru_hook_values['UPDATE'] = z_t.detach()
    module.gru_hook_values['NEW'] = n_t.detach()
    module.gru_hook_values['HIDDEN'] = h_t.detach()

# TEMPLATE FOR HOOK TO BE INCLUDED IN TEST:
# torch_gru = <pytorchGRUMechanismWrapper.function.function>
# torch_gru.register_forward_hook(_pytorch_GRU_module_values_hook)

# RESULTS WILL BE IN torch_gru.gru_hook_values

# HANDLES FOR HOOK (NEEDS TO BE REFINED)
# torch_gru._node_variables_hook_handle = None
# torch_gru._node_values_hook_handle = None
# # Set hooks here if they will always be in use
# if torch_gru._composition.parameters.synch_node_variables_with_torch.get(kwargs[CONTEXT]) == ALL:
#     torch_gru._node_variables_hook_handle = torch_gru._add_pytorch_hook(self._copy_pytorch_node_inputs_to_pnl_variables)
# if torch_gru._composition.parameters.synch_node_values_with_torch.get(kwargs[CONTEXT]) == ALL:
#     torch_gru._node_values_hook_handle = torch_gru._add_pytorch_hook(self._copy_pytorch_node_outputs_to_pnl_values)

# ----------------------------------

# Unit tests for functions of GRUComposition class

@pytest.mark.pytorch
@pytest.mark.composition
class TestConstruction:
    def test_disallow_modification(self):
        gru = GRUComposition()
        mech = pnl.ProcessingMechanism()
        with pytest.raises(CompositionError) as error_text:
            gru.add_node(mech)
        assert 'Nodes cannot be added to GRU Composition.' in str(error_text.value)


@pytest.mark.pytorch
@pytest.mark.composition
class TestExecution:
    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_pytorch_execution_identicality_with_pytorch(self, bias):
        # Test identicality of learning results of PyTorch GRU against native Pytorch GRU

        import torch
        entry_torch_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        inputs = [[1.0,2.0,3.0]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5

        h0 = torch.tensor(np.array([[0.,0.,0.,0.,0.]]))
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(torch_gru)
        result, hn = torch_gru(torch.tensor(np.array(inputs)),h0)
        torch_results = [result.detach().numpy()]
        result, hn = torch_gru(torch.tensor(np.array(inputs)),hn)
        torch_results.append(result.detach().numpy())

        gru = GRUComposition(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        # gru.set_weights_from_torch_gru(torch_gru)
        gru.set_weights(*torch_gru_initial_weights)
        gru.run(inputs={gru.input_node:inputs}, num_trials=2)

        np.testing.assert_allclose(torch_results, gru.results, atol=1e-6)

        torch.set_default_dtype(entry_torch_dtype)

    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_pytorch_unnested_learning_identicality_with_pytorch(self, bias):
        # Test identicality of learning results of GRUComposition against native Pytorch GRU

        import torch
        entry_torch_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        inputs = [[1.,2.,3.]]
        targets = [[1.,1.,1.,1.,1.]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001

        # Set up and run torch model -------------------------------------

        # Set up torch model
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_gru.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(torch_gru)

        # Execute model without learning
        h0 = torch.tensor([[0.,0.,0.,0.,0.]])
        torch_result_before_learning, hn = torch_gru(torch.tensor(np.array(inputs)),h0)

        # Compute loss and update weights
        torch_optimizer.zero_grad()
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets))
        torch_loss.backward()
        torch_optimizer.step()

        # Get output after learning
        torch_result_after_learning, hn = torch_gru(torch.tensor(np.array(inputs)),hn)

        # Set up and run PNL Autodiff model -------------------------------------

        # Initialize GRU Node of PNL with starting weights from Torch GRU, so that they start identically
        pnl_gru = GRUComposition(input_size=3, hidden_size=5, bias=bias, learning_rate=LEARNING_RATE)
        pnl_gru.set_weights(*torch_gru_initial_weights)
        target_node = pnl_gru.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)

        # Execute PNL GRUComposition
        # Corresponds to forward and then backward passes of torch model, but results do not yet reflect weight updates
        pnl_result_before_learning = pnl_gru.learn(inputs={pnl_gru.input_node:[[1.,2.,3.]],
                                                           target_node[0]: [[1.,1.,1.,1.,1.]]},
                                                   execution_mode=pnl.ExecutionMode.PyTorch)

        # Need to run it one more time (due to lazy updating) to see the effects of learning
        pnl_result_after_learning = pnl_gru.run(inputs={pnl_gru.input_node:[[1.,2.,3.]]})

        # Compare results from before learning:
        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(), pnl_result_before_learning, atol=1e-6)
        # Compare results from after learning:
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(),
                                   pnl_result_after_learning, atol=1e-6)

        torch.set_default_dtype(entry_torch_dtype)

    @pytest.mark.parametrize('bias', [False, True], ids=['no_bias','bias'])
    def test_nested_gru_composition_learning_and_copy_values(self, bias):
        # Test identicality of results of nested GRUComposition and pure pytorch version

        import torch
        entry_torch_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        inputs = [[1.,2.,3.]]
        targets = [[1.,1.,1.,1.,1.]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001

        # Set up and run TORCH GRU model
        class TorchModel(torch.nn.Module):
            def __init__(self):
                super(TorchModel, self).__init__()
                # Note:  input and output modules don't use biases in PNL version, so match that here
                self.input = torch.nn.Linear(INPUT_SIZE, INPUT_SIZE, bias=False)
                self.input.weight.requires_grad = False # Since weights to INPUT nodes of Composition are not learnable
                self.gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
                self.output = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

            def forward(self, x, hidden):
                after_input = self.input(x)
                after_gru, hidden_state = self.gru(after_input, hidden)
                after_output = self.output(after_gru)
                return after_output, hidden_state

        torch_model = TorchModel()
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_model.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')

        # Get initial weights (to initialize autodiff below with same initial conditions)
        torch_input_initial_weights = torch_model.state_dict()['input.weight'].T.detach().cpu().numpy().copy()
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(torch_model.gru)
        torch_output_initial_weights = torch_model.state_dict()['output.weight'].T.detach().cpu().numpy().copy()

        # Execute Torch model without learning
        hidden_init = torch.tensor([[0.,0.,0.,0.,0.]])
        torch_result_before_learning, hidden_state = torch_model(torch.tensor(np.array(inputs)), hidden_init)

        # Compute loss and update weights
        torch_optimizer.zero_grad()
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets))
        torch_loss.backward()
        torch_optimizer.step()

        # Get results after learning
        torch_result_after_learning, hidden_state = torch_model(torch.tensor(np.array(inputs)), hidden_state)

        # Set up and run PNL Autodiff model
        input_mech = pnl.ProcessingMechanism(name='INPUT MECH', input_shapes=3)
        output_mech = pnl.ProcessingMechanism(name='OUTPUT MECH', input_shapes=5)
        gru = GRUComposition(name='GRU COMP',
                             input_size=3, hidden_size=5, bias=bias, learning_rate = LEARNING_RATE)
        autodiff_comp = pnl.AutodiffComposition(name='OUTER COMP',
                                   pathways=[input_mech, gru, output_mech],
                                                learning_rate = LEARNING_RATE)
        # FIX: 3/15/25 - NEED TO BE HARDWIRED IN CONSTRUCTION OF ?AUTODIFF OR GRUCOMPOSITION:
        autodiff_comp.projections[0].learnable = False
        autodiff_comp.set_weights(autodiff_comp.nodes[0].efferents[0], torch_input_initial_weights)
        autodiff_comp.nodes['GRU COMP'].set_weights(*torch_gru_initial_weights)
        autodiff_comp.set_weights(autodiff_comp.projections[1], torch_output_initial_weights)
        target_mechs = autodiff_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)

        # Execute autodiff with learning (but not weight updates yet)
        autodiff_result_before_learning = autodiff_comp.learn(inputs={input_mech:inputs, target_mechs[0]: targets},
                                                              execution_mode=pnl.ExecutionMode.PyTorch)
        # Get results after learning (with weight updates)
        autodiff_result_after_learning = autodiff_comp.run(inputs={input_mech:inputs},
                                                           execution_mode=pnl.ExecutionMode.PyTorch)

        # Test of forward pass (without effects of learning yet):
        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(),
                                   autodiff_result_before_learning, atol=1e-6)

        # Test of execution after backward pass (learning):
        np.testing.assert_allclose(torch_loss.detach().numpy(), autodiff_comp.torch_losses.squeeze())
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(),
                                   autodiff_result_after_learning, atol=1e-6)

        if not bias:
            # Test synchronization of values
            # Outer Comp
            np.testing.assert_allclose(autodiff_comp.nodes['INPUT MECH'].parameters.variable.get('OUTER COMP'),
                                       np.array([[1., 2., 3.]]))
            np.testing.assert_allclose(autodiff_comp.nodes['OUTPUT MECH'].parameters.variable.get('OUTER COMP'),
                                       np.array([[-0.2371911, 0.09483196, 0.08101949, -0.32086433, 0.17566031]]),
                                       atol=1e-8)
            # GRU Comp
            GRU_comp_nodes = autodiff_comp.nodes['GRU COMP'].nodes
            np.testing.assert_allclose(GRU_comp_nodes['INPUT'].parameters.value.get('OUTER COMP'),
                                       np.array([[0.88200826,  1.82932232, -0.43319262]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['RESET'].parameters.value.get('OUTER COMP'),
                                       np.array([[0.52969068, 0.42252881, 0.54034619, 0.64740737, 0.34754141]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['UPDATE'].parameters.value.get('OUTER COMP'),
                                       np.array([[0.4842834,  0.65262676, 0.73368542, 0.32401945, 0.51233801]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['NEW'].parameters.value.get('OUTER COMP'),
                                       np.array([[-0.2679114,  -0.01421539,  0.67555595,  0.76259181, -0.81329808]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['HIDDEN\nLAYER'].parameters.value.get('OUTER COMP'),
                                       np.array([[-0.21075669, -0.0222539, 0.32382497, 0.57810654, -0.51770585]]),
                                       atol=1e-8)
            torch_gru = autodiff_comp.pytorch_representation._wrapped_nodes[2]._wrapped_nodes[0]
            np.testing.assert_allclose(GRU_comp_nodes['OUTPUT'].parameters.value.get('OUTER COMP'),
                                       GRU_comp_nodes['HIDDEN\nLAYER'].parameters.value.get('OUTER COMP'))
            # MODIFIED 3/16/25 OLD:
            # np.testing.assert_allclose(GRU_comp_nodes['OUTPUT'].parameters.value.get('OUTER COMP'),
            #                           torch_gru.hidden_state.detach().numpy(), atol=1e-8)
            # MODIFIED 3/16/25 END

        torch.set_default_dtype(entry_torch_dtype)
