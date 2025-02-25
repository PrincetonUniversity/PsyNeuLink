# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for GRUComposition"""

import torch
import numpy as np
from typing import Union, Optional, Literal

from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper, \
    PytorchProjectionWrapper, PytorchFunctionWrapper
from psyneulink.core.globals.context import handle_external_context, ContextFlags
from psyneulink.core.globals.keywords import ALL, INPUTS
from psyneulink.core.globals.log import LogCondition

__all__ = ['PytorchGRUCompositionWrapper', 'GRU_NODE_NAME', 'TARGET_NODE_NAME']

GRU_NODE_NAME = 'PYTORCH GRU NODE'
TARGET_NODE_NAME = 'GRU TARGET NODE'

# Dict that captures internal computations of GRU node in _node_values_hook
node_values = {}

class PytorchGRUCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for GRUComposition as a Pytorch Module
    Manage the exchange of the Composition's Projection `Matrices <MappingProjection_Matrix>`
    and the Pytorch GRU Module's parameters, and return its output value.
    """

    torch_dtype = torch.float32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _instantiate_pytorch_mechanism_wrappers(self, composition, device, context):
        """Instantiate PytorchMechanismWrapper for GRU Node"""
        node = composition.gru_mech
        pytorch_node = PytorchGRUMechanismWrapper(node, self, 0, device, context)
        self.torch_gru = pytorch_node.function.function
        self._nodes_map[node] = pytorch_node
        # MODIFIED 2/22/25 NEW:
        # Assign map from input_node and output_node to GRU Node, since it serves their functions in pytorch graph
        #   (for use by show_graph(show_pytorch=True)
        self._nodes_map[composition.input_node] = pytorch_node
        self._nodes_map[composition.output_node] = pytorch_node
        # MODIFIED 2/22/25 END
        self._wrapped_nodes.append(pytorch_node)
        if not composition.is_nested:
            node._is_input = True

    def _instantiate_pytorch_projection_wrappers(self, composition, device, context):
        """Create PytorchGRUProjectionWrappers for each learnable Projection of GRUComposition
        For each PytorchGRUProjectionWrapper, assign the current weight matrix of the PNL Projection
        to the corresponding part of the tensor in the parameter of the Pytorch GRU module.
        """

        pnl = self._composition
        torch_gru = self.torch_gru
        self.torch_parameters = torch_gru.parameters
        self._projection_map = {}

        # Pytorch parameter info
        torch_params = torch_gru.state_dict()
        hid_len = pnl.hidden_size
        z_idx = hid_len
        n_idx = 2 * hid_len

        w_ih = torch_params['weight_ih_l0']
        w_hh = torch_params['weight_hh_l0']
        torch_gru_wts_indices = [(w_ih, slice(None, z_idx)), (w_ih, slice(z_idx, n_idx)),(w_ih, slice(n_idx, None)),
                                 (w_hh, slice(None, z_idx)), (w_hh, slice(z_idx, n_idx)), (w_hh, slice(n_idx, None))]
        pnl_proj_wts = [pnl.wts_ir, pnl.wts_iu, pnl.wts_in, pnl.wts_hr, pnl.wts_hu, pnl.wts_hn]
        for pnl_proj, torch_matrix in zip(pnl_proj_wts, torch_gru_wts_indices):
            self._projection_map[pnl_proj] = PytorchGRUProjectionWrapper(pnl_proj, torch_matrix, device, context)
        self._pnl_refs_to_torch_params_map = {'w_ih': w_ih, 'w_hh':  w_hh}

        if pnl.bias:
            assert torch_gru.bias, f"PROGRAM ERROR: '{pnl.name}' has bias=True but {GRU_NODE_NAME}.bias=False. "
            b_ih = torch_params['bias_ih_l0']
            b_hh = torch_params['bias_hh_l0']
            torch_gru_bias_indices = [(b_ih, slice(None, z_idx)), (b_ih, slice(z_idx, n_idx)),(b_ih, slice(n_idx, None)),
                                      (b_hh, slice(None, z_idx)), (b_hh, slice(z_idx, n_idx)), (b_hh, slice(n_idx, None))]
            pnl_biases = [pnl.bias_ir, pnl.bias_iu, pnl.bias_in, pnl.bias_hr, pnl.bias_hu, pnl.bias_hn]
            for pnl_bias_proj, torch_bias in zip(pnl_biases, torch_gru_bias_indices):
                self._projection_map[pnl_bias_proj] = PytorchGRUProjectionWrapper(pnl_bias_proj, torch_bias,
                                                                                  device, context)
            self._pnl_refs_to_torch_params_map.update({'b_ih': b_ih, 'b_hh':  b_hh})

        self.copy_weights_to_torch_gru(context)

    # # MODIFIED 2/22/25 OLD:
    # def _get_nodes_map(self, context):
    #     """Return only Node for gru_mech in _nodes_map"""
    #     gru_mech = self._composition.gru_mech
    #     if context.flags & ContextFlags.DISPLAYING:
    #         return {gru_mech: self._nodes_map[gru_mech]}
    #     else:
    #         return self._nodes_map
    # # MODIFIED 2/22/25 END

    @handle_external_context()
    def forward(self, inputs, optimization_num, context=None)->dict:
        """Forward method of the model for PyTorch modes
        Returns a dictionary {output_node:value} with the output value for the module in case it is run as a
        standalone Composition; otherwise, those will be ignored and the outputs will be used by the aggregate_afferents
        method(s) of the other node(s) that receive Projections from the GRUComposition.
        """
        # Reshape input for GRU module (from float64 to float32)
        inputs = torch.tensor(np.array(inputs[self._composition.input_node]).astype(np.float32))
        hidden_state = self._composition.hidden_state
        output, self.hidden_state = self._wrapped_nodes[0].execute([inputs, hidden_state], context)
        # Assign output to the OUTPUT Node of the GRUComposition
        self._composition.output_node.parameters.value._set(output.detach().cpu().numpy(), context)
        self._composition.gru_mech.parameters.value._set(output.detach().cpu().numpy(), context)
        return {self._composition.output_node: output}

    def log_weights(self):
        for proj_wrapper in self._projection_wrappers:
            proj_wrapper.log_matrix()

    # FIX ALONG LINES OF _copy_internal_nodes_values_to_pnl
    def copy_node_variables_to_psyneulink(self, nodes:Optional[Union[list,Literal[ALL, INPUTS]]]=ALL, context=None):
        """Copy input to Pytorch nodes to variable of AutodiffComposition nodes.
        IMPLEMENTATION NOTE:  list included in nodes arg to allow for future specification of specific nodes to copy
        """
        if nodes == ALL:
            nodes = self._nodes_map.items()
        for pnl_node, pytorch_node in nodes:
            # First get variable in numpy format
            if isinstance(pytorch_node.input, list):
                variable = np.array([val.detach().cpu().numpy() for val in pytorch_node.input], dtype=object)
            else:
                variable = pytorch_node.input.detach().cpu().numpy()
            # Set pnl_node's value to value
            pnl_node.parameters.variable._set(variable, context)

    def _copy_internal_nodes_values_to_pnl(self, nodes, context):
        # FIX: 2/22/25: FLESH THIS OUT TO BE SURE ONLY NODES ARE GRU, INPUT AND OUTPUT
        #               AND EXPLICITY ASSIGN pnl_node TO ONE IN MAP
        node_map = list(nodes)
        pnl_comp = self._composition
        pnl_node = list(nodes)[0][0]
        pytorch_node = list(nodes)[0][1]

        assert len(nodes) == 3, \
            (f"PROGRAM ERROR: PytorchGRUCompositionWrapper should have only one node, "
             f"but has {len(nodes)}: {[node.name for node in nodes]}")
        assert pnl_node == pnl_comp.gru_mech, \
            f"PROGRAM ERROR: Bad mechanism passed ({pnl_node}); should be: {pnl_node.name}."
        assert pytorch_node == self._wrapped_nodes[0], \
            f"PROGRAM ERROR: Bad PyTorchMechanismWrapper passed ({pytorch_node}); should be: {pytorch_node.name}."

        # Update  node's value with the output of the corresponding wrapper in the PyTorch representation
        if pytorch_node.output is None:
            assert pytorch_node.cexclude_from_gradient_calc, \
                (f"PROGRAM ERROR: Value of PyTorch wrapper for '{pnl_node.name}' is None during forward pass, "
                 f"but it is not excluded from gradient calculation.")
        torch_gru_output = pytorch_node.output[0].detach().cpu().numpy()
        h = pytorch_node.output[1][0].detach()

        torch_gru_parameters = self.__class__.get_weights_from_torch_gru(self.torch_gru)
        torch_weights = torch_gru_parameters[0]
        torch_weights = list(torch_weights)
        for i, weight in enumerate(torch_weights):
            torch_weights[i] = torch.tensor(weight, dtype=torch.float32)
        w_ir, w_iz, w_in, w_hr, w_hz, w_hn = torch_weights
        if pnl_comp.bias:
            assert len(torch_gru_parameters) > 1, \
                (f"PROGRAM ERROR: '{pnl_comp.name}' has bias set to True, "
                 f"but no bias weights were returned for torch_gru_parameters.")
            b_ir, b_iz, b_in, b_hr, b_hz, b_hn = torch_gru_parameters[1]
        else:
            b_ir = b_iz = b_in = b_hr = b_hz = b_hn = 0.0

        x = self._wrapped_nodes[0].input[0][0]

        r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
        z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
        n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
        h_t = (1 - z_t) * n_t + z_t * h

        # Set values of nodes in pnl composition to the result of the corresponding computations in the PyTorch module
        pnl_comp.reset_node.parameters.value._set(r_t.detach().cpu().numpy(), context)
        pnl_comp.update_node.parameters.value._set(z_t.detach().cpu().numpy(), context)
        pnl_comp.new_node.parameters.value._set(n_t.detach().cpu().numpy(), context)
        pnl_comp.hidden_layer_node.parameters.value._set(h_t.detach().cpu().numpy(), context)
        pnl_comp.output_node.parameters.value._set(h_t.detach().cpu().numpy(), context)

        # KEEP FOR FUTURE DEBUGGING
        # result = self._composition(inputs={self._composition.input_node: x.detach().numpy()})

        # # KEEP THIS FOR REFERENCE IN CASE hidden_layer_node IS REPLACED WITH RecurrentTransferMechanism
        # # If pnl_node's function is Stateful, assign value to its previous_value parameter
        # #   so that if Python implementation is run it picks up where PyTorch execution left off
        # if isinstance(pnl_node.function, StatefulFunction):
        #     pnl_node.function.parameters.previous_value._set(torch_gru_output, context)

    # FIX: 2/18/25 REMOVE
    # def _node_values_hook(module, input, output):
    #     in_len = module.input_size
    #     hid_len = module.hidden_size
    #     z_idx = hid_len
    #     n_idx = 2 * hid_len
    #
    #     ih = module.weight_ih_l0
    #     hh = module.weight_hh_l0
    #     if module.bias:
    #         b_ih = module.bias_ih_l0
    #         b_hh = module.bias_hh_l0
    #     else:
    #         b_ih = torch.tensor(np.array([0] * 3 * hid_len))
    #         b_hh = torch.tensor(np.array([0] * 3 * hid_len))
    #
    #     w_ir = ih[:z_idx].T
    #     w_iz = ih[z_idx:n_idx].T
    #     w_in = ih[n_idx:].T
    #     w_hr = hh[:z_idx].T
    #     w_hz = hh[z_idx:n_idx].T
    #     w_hn = hh[n_idx:].T
    #
    #     b_ir = b_ih[:z_idx]
    #     b_iz = b_ih[z_idx:n_idx]
    #     b_in = b_ih[n_idx:]
    #     b_hr = b_hh[:z_idx]
    #     b_hz = b_hh[z_idx:n_idx]
    #     b_hn = b_hh[n_idx:]
    #
    #     assert len(input) > 1, f"PROGRAM ERROR: PytorchGRUCompositionWrapper hook received only one input: {input}"
    #     x = input[0]
    #     # h = input[1] if len(input) > 1 else torch.tensor([[0] * module.hidden_size], dtype=torch.float32)
    #     h = input[1]
    #
    #     # Reproduce GRU forward calculations
    #     r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
    #     z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
    #     n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
    #     h_t = (1 - z_t) * n_t + z_t * h
    #
    #     # Put internal calculations in dict with corresponding node names as keys
    #     node_values[RESET_NODE_NAME] = r_t.detach()
    #     node_values[UPDATE_NODE_NAME] = z_t.detach()
    #     node_values[NEW_NODE_NAME] = n_t.detach()
    #     node_values[HIDDEN_LAYER_NODE_NAME] = h_t.detach()

    def log_values(self):
        for node_wrapper in [n for n in self._wrapped_nodes if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()

    def copy_weights_to_psyneulink(self, context=None):
        for projection, proj_wrapper in self._projection_map.items():
            torch_parameter = proj_wrapper.torch_parameter
            torch_indices = proj_wrapper.matrix_indices
            matrix =  torch_parameter[torch_indices].detach().cpu().clone().numpy().T
            projection.parameters.matrix._set(matrix, context)
            projection.parameter_ports['matrix'].parameters.value._set(matrix, context)

    def copy_weights_to_torch_gru(self, context=None):
        for projection, proj_wrapper in self._projection_map.items():
            proj_wrapper.set_torch_gru_parameter(context)

    def get_weights_from_torch_gru(torch_gru)->tuple[torch.Tensor]:
        """Get parameters from PyTorch GRU module corresponding to GRUComposition's Projections.
        Format tensors:
          - transpose all weight and bias tensors;
          - reformat biases as 2d
        Return formatted tensors, which are used:
         - in set_weights_from_torch_gru(), where they are converted to numpy arrays
         - for or forward computation in pytorchGRUwrappers._copy_internal_nodes_values_to_pnl()
        """
        hid_len = torch_gru.hidden_size
        z_idx = hid_len
        n_idx = 2 * hid_len

        torch_gru_weights = torch_gru.state_dict()
        wts_ih = torch_gru_weights['weight_ih_l0']
        wts_ir = wts_ih[:z_idx].T.detach().cpu().numpy().copy()
        wts_iu = wts_ih[z_idx:n_idx].T.detach().cpu().numpy().copy()
        wts_in = wts_ih[n_idx:].T.detach().cpu().numpy().copy()
        wts_hh = torch_gru_weights['weight_hh_l0']
        wts_hr = wts_hh[:z_idx].T.detach().cpu().numpy().copy()
        wts_hu = wts_hh[z_idx:n_idx].T.detach().cpu().numpy().copy()
        wts_hn = wts_hh[n_idx:].T.detach().cpu().numpy().copy()
        weights = (wts_ir, wts_iu, wts_in, wts_hr, wts_hu, wts_hn)

        biases = None
        if torch_gru.bias:
            # Transpose 1d bias Tensors using permute instead of .T (per PyTorch warning)
            b_ih = torch_gru_weights['bias_ih_l0']
            b_ir = torch.atleast_2d(b_ih[:z_idx].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy(

            ).copy()
            b_iu = torch.atleast_2d(b_ih[z_idx:n_idx].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu(

            ).numpy().copy()
            b_in = torch.atleast_2d(b_ih[n_idx:].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy(

            ).copy()
            b_hh = torch_gru_weights['bias_hh_l0']
            b_hr = torch.atleast_2d(b_hh[:z_idx].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy(

            ).copy()
            b_hu = torch.atleast_2d(b_hh[z_idx:n_idx].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu(

            ).numpy().copy()
            b_hn = torch.atleast_2d(b_hh[n_idx:].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy(

            ).copy()
            biases = (b_ir, b_iu, b_in, b_hr, b_hu, b_hn)
        return weights, biases


class PytorchGRUMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for a GRU Node"""

    def _assign_pytorch_function(self, mechanism, device, context):
        # Assign PytorchGRUFunctionWrapper of Pytorch GRU module as function of GRU Node
        input_size = self._composition_wrapper_owner._composition.parameters.input_size.get(context)
        hidden_size = self._composition_wrapper_owner._composition.parameters.hidden_size.get(context)
        bias = self._composition_wrapper_owner._composition.parameters.bias.get(context)
        function_wrapper = PytorchGRUFunctionWrapper(torch.nn.GRU(input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    bias=bias), device, context)
        self.function = function_wrapper
        mechanism.function = function_wrapper.function

        # MODIFIED 2/16/25 NEW:
        # FIX: ARE WEIGHTS BEING COPIED FROM PNL TO TORCH GRU MODULE?  IF SO, WHERE? IF NOT, SHOULD DO IT HERE
        #      RELATIONSHIP TO _regenerate_paramlist

        # Assign input_port functions of GRU Node to PytorchGRUFunctionWrapper
        self.input_ports = [PytorchGRUFunctionWrapper(input_port.function, device, context)
                            for input_port in mechanism.input_ports]

    def execute(self, variable, context):
        """Execute GRU Node with input variable and return output value
        """
        self.input = variable
        self.output = self.function(*variable)
        return self.output

    def log_value(self):
        # FIX: LOG HIDDEN STATE OF COMPOSITION MECHANISM
        if self._mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self._mechanism.output_port.parameters.value._set(detached_value, self._context)
            self._mechanism.parameters.value._set(detached_value, self._context)

    def log_matrix(self):
        if self._projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self._projection.parameters.matrix._set(detached_matrix, context=self._context)
            self._projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)


class PytorchGRUProjectionWrapper(PytorchProjectionWrapper):
    """Wrapper for a Projection of the GRUComposition

    One is created for each Projection of the GRUComposition that is learnable.
    Sets of three of these correspond to the Parameters of the torch GRU module:

    PyTorch GRU parameter:  GRUComposition Projections:
         weight_ih_l0       wts_ir, wts_iu, wts_in
         weight_hh_l0       wts_hr, wts_hu, wts_hn
         bias_ih_l0         bias_ir, bias_iu, bias_in
         bias_hh_l0         bias_hr, bias_hu, bias_hn

    Attributes
    ----------
    _projection:  MappingProjection
        the Projection of the GRUComposition being wrapped

    torch_parameter: Pytorch parameter
        the tensor corresponding to the matrix of the Projection;

    matrix_indices: slice
        a slice specifying the part of the Pytorch parameter corresponding to the GRUCOmposition Projection's matrix.

    """
    def __init__(self, projection:MappingProjection, torch_parameter:tuple, device:str, context=None):
        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        # GRUComposition Projection being wrapped:
        self._projection = projection # PNL Projection being wrapped
        # Assign parameter and tensor indices of Pytorch GRU module parameter corresponding to the Projection's matrix:
        self.torch_parameter, self.matrix_indices = torch_parameter
        # Projections for GRUComposition are not included in autodiff; matrices are set directly in Pytorch GRU module:
        self._projection.exclude_in_autodiff = True

    def set_torch_gru_parameter(self, context):
        """Set relevant part of tensor for parameter of Pytorch GRU module from GRUComposition's Projections."""
        matrix = self._projection.parameters.matrix._get(context).T
        proj_matrix_as_tensor =  torch.tensor(matrix.squeeze(), dtype=torch.float32)
        self.torch_parameter[self.matrix_indices].data.copy_(proj_matrix_as_tensor)

    def log_matrix(self):
        if self._projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self._projection.parameters.matrix._set(detached_matrix, context=self._context)
            self._projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)


class PytorchGRUFunctionWrapper(PytorchFunctionWrapper):
    def __init__(self, function, device, context=None):
        self._pnl_function = function
        self.name = f"PytorchFunctionWrapper[GRU NODE]"
        self._context = context
        self.function = function

    def __repr__(self):
        return "PytorchWrapper for: " + self._pnl_function.__repr__()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
