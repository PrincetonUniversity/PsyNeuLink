import numpy as np
import pytest

import psyneulink as pnl
from psyneulink import CompositionError, AutodiffComposition, HIDDEN_TO_HIDDEN

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

def _pytorch_gru_module_values_hook(module, input):
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
        with pytest.raises(CompositionError) as error_text:
            gru.add_node(pnl.ProcessingMechanism())
        assert 'Nodes cannot be added to a GRUComposition' in str(error_text.value)
        with pytest.raises(CompositionError) as error_text:
            gru.add_projection(pnl.MappingProjection())
        assert 'Projections cannot be added to a GRUComposition' in str(error_text.value)

    # BREADCRUMB:  ADD TEST OF enable_learning SETTINGS FOR OUTER AND NESTED: True/False vs. False/True
    @pytest.mark.parametrize('enable_learning', ['outer_false', 'gru_false', 'both_false'])
    @pytest.mark.parametrize('execution_type', ['run', 'learn'])
    @pytest.mark.parametrize('pathway_type', ['solo', 'gru_as_input', 'gru_as_hidden', 'gru_as_output'])
    def test_gru_as_solo_input_hidden_output_node_in_nested(self, pathway_type, execution_type, enable_learning):
        gru_enable_learning = True if 'outer' in enable_learning else False
        outer_enable_learning = True if 'gru' in enable_learning else False

        input_mech = pnl.ProcessingMechanism(input_shapes=3)
        output_mech = pnl.ProcessingMechanism(input_shapes=5)
        gru = pnl.GRUComposition(input_size=3, hidden_size=5, bias=False, enable_learning=gru_enable_learning)
        if pathway_type == 'solo':
            pathway = [gru]
            input_node = gru
            target_node = gru.gru_mech
        elif pathway_type == 'gru_as_input':
            pathway = [gru, output_mech]
            input_node = gru
            target_node = output_mech
        elif pathway_type == 'gru_as_hidden':
            pathway = [input_mech, gru, output_mech]
            input_node = input_mech
            target_node = output_mech
        elif pathway_type == 'gru_as_output':
            pathway = [input_mech, gru]
            input_node = input_mech
            target_node = gru.gru_mech
        else:
            raise ValueError("Invalid pathway_type")
        outer_comp = pnl.AutodiffComposition(pathway, enable_learning=outer_enable_learning)
        inputs = {input_node: [[.1, .2, .3]]}
        targets = {target_node: [[1,1,1,1,1]]}
        if execution_type == 'run':
            outer_comp.run(inputs=inputs)
        else:
            outer_comp.learn(inputs=inputs, targets=targets)
        assert True


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
        LEARNING_RATE = .1

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
        # Note: tests learning of input weights to GRU (in both PNL and torch) as well as GRU itself

        import torch
        entry_torch_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        inputs = [[1.,2.,3.]]
        targets = [[1.,1.,1.,1.,1.]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = 6.1

        # Set up and run TORCH GRU model
        class TorchModel(torch.nn.Module):
            def __init__(self):
                super(TorchModel, self).__init__()
                # Note:  input and output modules don't use biases in PNL version, so match that here
                self.input = torch.nn.Linear(INPUT_SIZE, INPUT_SIZE, bias=False)
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
        autodiff_comp.set_weights( autodiff_comp.projections[0], torch_input_initial_weights)
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
                                   autodiff_result_before_learning, atol=1e-8)

        # Test of execution after backward pass (learning):
        np.testing.assert_allclose(torch_loss.detach().numpy(), autodiff_comp.torch_losses.squeeze())
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(),
                                   autodiff_result_after_learning, atol=1e-8)

        if not bias:
            # Test synchronization of values
            # Outer Comp
            np.testing.assert_allclose(autodiff_comp.nodes['INPUT MECH'].parameters.variable.get('OUTER COMP'),
                                       np.array([[1., 2., 3.]]))
            np.testing.assert_allclose(autodiff_comp.nodes['OUTPUT MECH'].parameters.variable.get('OUTER COMP'),
                                       np.array([[0.8578478, 1.0700558, 0.46894508, 0.73393936, 0.35604749]]),
                                       atol=1e-8)
            # GRU Comp
            GRU_comp_nodes = autodiff_comp.nodes['GRU COMP'].nodes
            np.testing.assert_allclose(GRU_comp_nodes['INPUT'].parameters.value.get('OUTER COMP'),
                                       np.array([[0.90319426,  3.09532846, 11.46387165]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['RESET'].parameters.value.get('OUTER COMP'),
                                       np.array([[0.03300054, 0.40847909, 0.04398941, 0.10445835, 0.30668056]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['UPDATE'].parameters.value.get('OUTER COMP'),
                                       np.array([[0.66483663, 0.11608262, 0.13153948, 0.97442451, 0.97814372]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['NEW'].parameters.value.get('OUTER COMP'),
                                       np.array([[-7.10650456, -0.64283666, -4.00117146, 4.19362381, 4.91683601]]),
                                       atol=1e-8)
            np.testing.assert_allclose(GRU_comp_nodes['HIDDEN\nLAYER'].parameters.value.get('OUTER COMP'),
                                       np.array([[-0.43922832, -0.50410907, -0.83792679, 0.45777554, -0.34143725]]),
                                       atol=1e-8)
            torch_gru = autodiff_comp.pytorch_representation.node_wrappers[2].node_wrappers[0]
            np.testing.assert_allclose(GRU_comp_nodes['OUTPUT'].parameters.value.get('OUTER COMP'),
                                       GRU_comp_nodes['HIDDEN\nLAYER'].parameters.value.get('OUTER COMP'))

        torch.set_default_dtype(entry_torch_dtype)

    constructor_expected = [[ 0.23619161, 0.18558876, 0.16821693, 0.27253839, -0.18351431]]
    learn_method_expected = [[0.32697333, 0.22005074, 0.28091698, 0.4033476, -0.10994711]]
    continued_learning_expected = [[0.44543197, 0.47387584, 0.25515581, 0.34837884, -0.07662127]]
    none_expected = [[0.19536549, 0.04794166, 0.14910019, 0.3058192, -0.35057197]]
    test_specs = [('constructor', pnl.INPUT_TO_HIDDEN, constructor_expected),
                  ('constructor', "HIDDEN TO UPDATE WEIGHTS", None),
                  ('learn_method', pnl.HIDDEN_TO_HIDDEN, learn_method_expected),
                  ('learn_method', "HIDDEN TO UPDATE WEIGHTS", None),
                  ('constructor', pnl.BIAS_INPUT_TO_HIDDEN, None),
                  ('learn_method', pnl.BIAS_HIDDEN_TO_HIDDEN, None),
                  ('both', pnl.HIDDEN_TO_HIDDEN, learn_method_expected),
                  ('specs_to_nested', pnl.INPUT_TO_HIDDEN, constructor_expected),
                  ('none', pnl.HIDDEN_TO_HIDDEN, none_expected)]
    @pytest.mark.parametrize("condition, gru_proj, expected", test_specs,
                             ids=[f"{x[0]}_{x[1]}" for x in test_specs])
    def test_learning_rate_assignments(self, condition, gru_proj, expected):
        input_mech = pnl.ProcessingMechanism(input_shapes=3)
        output_mech = pnl.ProcessingMechanism(input_shapes=5)
        gru = pnl.GRUComposition(input_size=3, hidden_size=5, bias=False,
                                 learning_rate={gru_proj: 0.3} if condition=='specs_to_nested' else None
                                 )
        input_proj = pnl.MappingProjection(input_mech, gru.input_node)
        output_proj = pnl.MappingProjection(gru.output_node, output_mech)
        constructor_learning_rates = {gru_proj: .3,
                                        input_proj: 2.9,
                                        output_proj: .5}
        learning_method_learning_rates = {gru_proj: .95,
                                          input_proj: .66,
                                          output_proj: 1.5}

        # Test for error on attempt to set individual Projection learning rate
        if gru_proj == "HIDDEN TO UPDATE WEIGHTS":
            error_msg = ("GRUComposition does not support setting of learning rates for individual "
                         "hidden_to_hidden Projections (HIDDEN TO UPDATE WEIGHTS); use 'HIDDEN_TO_HIDDEN' "
                         "to set learning rate for all such weights.")
            with pytest.raises(pnl.GRUCompositionError) as error_text:
                outer = pnl.AutodiffComposition(
                    [input_mech, input_proj, gru, output_proj, output_mech],
                    learning_rate=constructor_learning_rates if condition in {'constructor'} else None
                )
                outer.learn(
                    inputs={input_mech: [[.1, .2, .3]]}, targets={output_mech: [[1,1,1,1,1]]},
                    learning_rate=learning_method_learning_rates if condition in {'learn_method'} else None)
            assert error_msg in str(error_text.value)

        # Test for error on attempt to set BIAS learning rate if bias option is False
        elif gru_proj in {pnl.BIAS_INPUT_TO_HIDDEN, pnl.BIAS_HIDDEN_TO_HIDDEN}:
            bias_spec = 'BIAS_INPUT_TO_HIDDEN' if gru_proj == pnl.BIAS_INPUT_TO_HIDDEN else "BIAS_HIDDEN_TO_HIDDEN"
            error_msg = (f"Attempt to set learning rate for bias(es) of GRU using '{bias_spec}' in the "
                         f"'learning_rate' arg of the learn() method for 'GRU Composition' when its bias option "
                         f"is set to False; the spec(s) must be removed or bias set to True.")
            with pytest.raises(pnl.GRUCompositionError) as error_text:
                outer = pnl.AutodiffComposition(
                    [input_mech, input_proj, gru, output_proj, output_mech],
                    learning_rate=constructor_learning_rates if condition in {'constructor'} else None
                )
                outer.learn(
                    inputs={input_mech: [[.1, .2, .3]]}, targets={output_mech: [[1,1,1,1,1]]},
                    learning_rate=learning_method_learning_rates if condition in {'learn_method'} else None)
            assert error_msg in str(error_text.value)

        # Test for assignment of learning_rates to nested Composition on its construction
        elif condition == 'specs_to_nested':
            outer = pnl.AutodiffComposition(
                [input_mech, input_proj, gru, output_proj, output_mech],
                # Exclude gru_proj from learning_rate since it was set on nested gru above
                learning_rate={input_proj: 2.9, output_proj: .5})
            results = outer.learn(
                inputs={input_mech: [[.1, .2, .3]]}, targets={output_mech: [[1,1,1,1,1]]},
                num_trials=2
            )
            np.testing.assert_allclose(expected, results)

        else:
            # Test assignment of learning_rate on construction
            outer = pnl.AutodiffComposition(
                [input_mech, input_proj, gru, output_proj, output_mech],
                learning_rate=constructor_learning_rates if condition in {'constructor'} else None
            )
            # Test assignment of learning_Rate on learning
            results = outer.learn(
                inputs={input_mech: [[.1, .2, .3]]}, targets={output_mech: [[1,1,1,1,1]]},
                learning_rate=learning_method_learning_rates if condition in {'learn_method', 'both'} else None,
                num_trials=2)
            np.testing.assert_allclose(expected, results)

    @pytest.mark.parametrize("bias", [False, True])
    def test_pytorch_identicality_of_learning_rates_unnested(self, bias):
        import torch
        entry_torch_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        inputs = [[1.,2.,3.]]
        targets = [[1.,1.,1.,1.,1.]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001
        W_IH_LEARNING_RATE = 1.414
        W_HH_LEARNING_RATE = 6.02
        B_IH_LEARNING_RATE = 6.26
        B_HH_LEARNING_RATE = 2.7

        # Set up and run torch model -------------------------------------

        # Set up torch model
        torch_gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_gru.parameters())
        del torch_optimizer.param_groups[0]
        w_ih_param = [p[1] for p in torch_gru.named_parameters() if p[0]==pnl.W_IH_NAME][0]
        w_hh_param = [p[1] for p in torch_gru.named_parameters() if p[0]==pnl.W_HH_NAME][0]
        torch_optimizer.add_param_group({'params': [w_ih_param], 'lr': W_IH_LEARNING_RATE})
        torch_optimizer.add_param_group({'params': [w_hh_param], 'lr': W_HH_LEARNING_RATE})
        if bias:
            b_ih_param = [p[1] for p in torch_gru.named_parameters() if p[0]==pnl.B_IH_NAME][0]
            b_hh_param = [p[1] for p in torch_gru.named_parameters() if p[0]==pnl.B_HH_NAME][0]
            torch_optimizer.add_param_group({'params': [b_ih_param], 'lr': B_IH_LEARNING_RATE})
            torch_optimizer.add_param_group({'params': [b_hh_param], 'lr': B_HH_LEARNING_RATE})
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
        learning_rate={pnl.INPUT_TO_HIDDEN: W_IH_LEARNING_RATE, pnl.HIDDEN_TO_HIDDEN:W_HH_LEARNING_RATE}
        if bias:
            learning_rate.update({pnl.DEFAULT_LEARNING_RATE: LEARNING_RATE,
                                  pnl.BIAS_INPUT_TO_HIDDEN: B_IH_LEARNING_RATE,
                                  pnl.BIAS_HIDDEN_TO_HIDDEN: B_HH_LEARNING_RATE})
        pnl_gru = GRUComposition(input_size=3, hidden_size=5, bias=bias, learning_rate=learning_rate)
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

    # Note:  if this is ever deprecated or removed, ensure version in test_autodiffcomposition is in use
    @pytest.mark.parametrize("bias", [False, True], ids=['no_bias','bias'])
    def test_pytorch_identicality_of_learning_rates_nested(self, bias):
        import torch
        entry_torch_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        inputs = [[1.,2.,3.]]
        targets = [[1.,1.,1.,1.,1.]]
        INPUT_SIZE = 3
        HIDDEN_SIZE = 5
        LEARNING_RATE = .001
        W_IH_LEARNING_RATE = 1.414
        W_HH_LEARNING_RATE = 6.02
        B_IH_LEARNING_RATE = 6.26
        B_HH_LEARNING_RATE = 2.7

        # Set up and run torch model ----------------------------------------------------------------------

        # Set up torch model
        class TorchModel(torch.nn.Module):
            def __init__(self, bias):
                super(TorchModel, self).__init__()
                # Note:  input and output modules don't use biases in PNL version, so match that here
                self.input = torch.nn.Linear(INPUT_SIZE, INPUT_SIZE, bias=False)
                self.gru = torch.nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bias=bias)
                self.output = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

            def forward(self, x, hidden):
                after_input = self.input(x)
                after_gru, hidden_state = self.gru(after_input, hidden)
                after_output = self.output(after_gru)
                return after_output, hidden_state

        torch_model = TorchModel(bias)
        torch_optimizer = torch.optim.SGD(lr=LEARNING_RATE, params=torch_model.parameters())
        loss_fct = torch.nn.MSELoss(reduction='mean')

        # Assign learning rates to IH and HH parameters
        _torch_param_short_to_long_names_map = {k.split('.')[-1]:k
                                               for k in [p[0] for p in torch_model.named_parameters()]}
        w_ih_param = [p[1] for p in torch_model.named_parameters()
                      if p[0]== _torch_param_short_to_long_names_map[pnl.W_IH_NAME]][0]
        w_hh_param = [p[1] for p in torch_model.named_parameters()
                      if p[0]== _torch_param_short_to_long_names_map[pnl.W_HH_NAME]][0]
        param_group = torch_optimizer.param_groups[0]
        for i, p in enumerate(param_group['params'].copy()):
            if p is w_ih_param:
                del param_group['params'][i]
        for i, p in enumerate(param_group['params'].copy()):
            if p is w_hh_param:
                del param_group['params'][i]
        torch_optimizer.add_param_group({'params': [w_ih_param], 'lr': W_IH_LEARNING_RATE})
        torch_optimizer.add_param_group({'params': [w_hh_param], 'lr': W_HH_LEARNING_RATE})
        if bias:
            b_ih_param = [p[1] for p in torch_model.named_parameters()
                      if p[0]== _torch_param_short_to_long_names_map[pnl.B_IH_NAME]][0]
            b_hh_param = [p[1] for p in torch_model.named_parameters()
                      if p[0]== _torch_param_short_to_long_names_map[pnl.B_HH_NAME]][0]
            for i, p in enumerate(param_group['params'].copy()):
                if p is b_ih_param:
                    del param_group['params'][i]
            for i, p in enumerate(param_group['params'].copy()):
                if p is b_hh_param:
                    del param_group['params'][i]
            torch_optimizer.add_param_group({'params': [b_ih_param], 'lr': B_IH_LEARNING_RATE})
            torch_optimizer.add_param_group({'params': [b_hh_param], 'lr': B_HH_LEARNING_RATE})

        # Get initial weights (to initialize autodiff below with same initial conditions)
        torch_input_initial_weights = torch_model.state_dict()['input.weight'].T.detach().cpu().numpy().copy()
        torch_gru_initial_weights = pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(torch_model.gru)
        torch_output_initial_weights = torch_model.state_dict()['output.weight'].T.detach().cpu().numpy().copy()

        # Execute model without learning
        hidden_init = torch.tensor([[0.,0.,0.,0.,0.]])
        torch_result_before_learning, hidden_state = torch_model(torch.tensor(np.array(inputs)), hidden_init)

        # Compute loss and update weights
        torch_optimizer.zero_grad()
        torch_loss = loss_fct(torch_result_before_learning, torch.tensor(targets))
        torch_loss.backward()
        torch_optimizer.step()

        # Get output after learning
        torch_result_after_learning, hidden_state = torch_model(torch.tensor(np.array(inputs)), hidden_state)

        # Set up and run PNL Autodiff model ------------------------------------------------------------

        input_mech = pnl.ProcessingMechanism(name='INPUT MECH', input_shapes=3)
        output_mech = pnl.ProcessingMechanism(name='OUTPUT MECH', input_shapes=5)
        learning_rate={pnl.INPUT_TO_HIDDEN: W_IH_LEARNING_RATE, pnl.HIDDEN_TO_HIDDEN:W_HH_LEARNING_RATE}
        if bias:
            learning_rate.update({pnl.DEFAULT_LEARNING_RATE: LEARNING_RATE,
                                  pnl.BIAS_INPUT_TO_HIDDEN: B_IH_LEARNING_RATE,
                                  pnl.BIAS_HIDDEN_TO_HIDDEN: B_HH_LEARNING_RATE})
        gru = GRUComposition(name='GRU COMP',
                             input_size=3, hidden_size=5, bias=bias, learning_rate = LEARNING_RATE)
        autodiff_comp = pnl.AutodiffComposition(name='OUTER COMP',
                                                pathways=[input_mech, gru, output_mech],
                                                learning_rate=learning_rate)
        autodiff_comp.set_weights(autodiff_comp.projections[0], torch_input_initial_weights)
        autodiff_comp.nodes['GRU COMP'].set_weights(*torch_gru_initial_weights)
        autodiff_comp.set_weights(autodiff_comp.projections[1], torch_output_initial_weights)
        target_mechs = autodiff_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)

        # Execute autodiff with learning (but not weight updates yet)
        autodiff_result_before_learning = autodiff_comp.learn(inputs={input_mech:inputs, target_mechs[0]: targets},
                                                              execution_mode=pnl.ExecutionMode.PyTorch)
        # Get results after learning (with weight updates)
        autodiff_result_after_learning = autodiff_comp.run(inputs={input_mech:inputs},
                                                           execution_mode=pnl.ExecutionMode.PyTorch)

        # Compare results ------------------------------------------------------------

        # Test of forward pass (without effects of learning yet):
        np.testing.assert_allclose(torch_result_before_learning.detach().numpy(),
                                   autodiff_result_before_learning, atol=1e-8)

        # Test of execution after backward pass (learning):
        np.testing.assert_allclose(torch_loss.detach().numpy(), autodiff_comp.torch_losses.squeeze())
        np.testing.assert_allclose(torch_result_after_learning.detach().numpy(),
                                   autodiff_result_after_learning, atol=1e-8)

        torch.set_default_dtype(entry_torch_dtype)
