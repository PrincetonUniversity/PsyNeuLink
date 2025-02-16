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

from psyneulink.core.components.functions.stateful.statefulfunction import StatefulFunction
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper, \
    PytorchProjectionWrapper, PytorchFunctionWrapper
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import ALL, CONTEXT, INPUTS, OUTPUTS

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
        self._torch_gru = self._composition.gru_mech.function
        # self._torch_gru.register_forward_hook(self._node_values_hook)
        # self._node_variables_hook_handle = None
        # self._node_values_hook_handle = None
        # # Set hooks here if they will always be in use
        # if self._composition.parameters.synch_node_variables_with_torch.get(kwargs[CONTEXT]) == ALL:
        #     self._node_variables_hook_handle = self._add_pytorch_hook(self.copy_node_variables_to_psyneulink)
        # if self._composition.parameters.synch_node_values_with_torch.get(kwargs[CONTEXT]) == ALL:
        #     self._node_values_hook_handle = self._add_pytorch_hook(self._copy_internal_nodes_values_to_pnl)

    def _instantiate_pytorch_mechanism_wrappers(self, composition, device, context):
        """Instantiate PytorchMechanismWrapper for GRU Node"""
        node = composition.gru_mech
        pytorch_node = PytorchGRUMechanismWrapper(node, self, 0, device, context)
        self._nodes_map[node] = pytorch_node
        self._wrapped_nodes.append(pytorch_node)
        if not composition.is_nested:
            node._is_input = True

    def _instantiate_pytorch_projection_wrappers(self, composition, device, context):
        """Assign PytorchProjectionWrapper's parameters to those of GRU Node"""
        if len(self._wrapped_nodes) == 1:
            self.parameters = self._wrapped_nodes[0].function.function.parameters
        else:
            if not len(self._wrapped_nodes):
                assert False, \
                    (f"PROGRAM ERROR: PytorchGRUCompositionWrapper has no wrapped nodes; should have one for "
                     f"'PYTORCH GRU NODE'.")
            else:
                extra_nodes = [node for node in self._wrapped_nodes
                               if node.name != 'PytorchMechanismWrapper[PYTORCH GRU NODE]']
                assert False, \
                    (f"PROGRAM ERROR: Somehow an extra node or more snuck into PytorchGRUCompositionWrapper; "
                     f"should only have one for 'PYTORCH GRU NODE', but also has: {extra_nodes}.")

    @handle_external_context()
    def forward(self, inputs, optimization_num, context=None)->dict:
        """Forward method of the model for PyTorch modes
        Returns a dictionary {output_node:value} with the output value for the module in case it is run as a
        standalone Composition; otherwise, those will be ignored and the outputs will be used by the aggregate_afferents
        method(s) of the other node(s) that receive Projections from the GRUComposition.
        """
        # Reshape iput for GRU module (from float64 to float32
        inputs = torch.tensor(np.array(inputs[self._composition.input_node]).astype(np.float32))
        hidden_state = self._composition.hidden_state
        output, self.hidden_state = self._wrapped_nodes[0].execute([inputs, hidden_state], context)
        # Assign output to the OUTPUT Node of the GRUComposition
        self._composition.output_node.parameters.value._set(output.detach().cpu().numpy(), context)
        self._composition.gru_mech.parameters.value._set(output.detach().cpu().numpy(), context)
        return {self._composition.output_node: output}

    def copy_weights_to_psyneulink(self, context=None):
        self._composition.set_weights_from_torch_gru(self._torch_gru, context)

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
        pnl_node = list(nodes)[0][0]
        pytorch_node = list(nodes)[0][1]

        assert len(nodes) == 1, \
            (f"PROGRAM ERROR: PytorchGRUCompositionWrapper should have only one node, "
             f"but has {len(nodes)}: {[node.name for node in nodes]}")
        assert pnl_node == self._composition.gru_mech, \
            f"PROGRAM ERROR: Bad mechanism passed ({pnl_node}); should be: {pnl_node.name}."
        assert pytorch_node == self._wrapped_nodes[0], \
            f"PROGRAM ERROR: Bad PyTorchMechanismWrapper passed ({pytorch_node}); should be: {pytorch_node.name}."

        # Update  node's value with the output of the corresponding wrapper in the PyTorch representation
        if pytorch_node.output is None:
            assert pytorch_node.exclude_from_gradient_calc, \
                (f"PROGRAM ERROR: Value of PyTorch wrapper for '{pnl_node.name}' is None during forward pass, "
                 f"but it is not excluded from gradient calculation.")
        torch_gru_output = pytorch_node.output[0].detach().cpu().numpy()
        h = pytorch_node.output[1][0].detach()

        # FIX: TEST WHICH IS FASTER:
        torch_gru_parameters = self._composition.get_weights_from_torch_gru(self._torch_gru)
        w_ir, w_iz, w_in, w_hr, w_hz, w_hn = torch_gru_parameters[0]
        if self._composition.bias:
            assert len(torch_gru_parameters) > 1, \
                (f"PROGRAM ERROR: '{self._composition.name}' has bias set to True, "
                 f"but no bias weights were returned for torch_gru_parameters.")
            b_ir, b_iz, b_in, b_hr, b_hz, b_hn = torch_gru_parameters[1]
        else:
            b_ir = b_iz = b_in = b_hr = b_hz = b_hn = 0.0

        x = self._wrapped_nodes[0].input[0][0]

        r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
        z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
        n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
        h_t = (1 - z_t) * n_t + z_t * h

        # KEEP FOR FUTURE DEBUGGING
        # result = self._composition(inputs={self._composition.input_node: x.detach().numpy()})

        # Set pnl_node's value to value
        for pnl_node in self._composition.nodes:
            pnl_node.parameters.value._set(torch_gru_output, context)

        # # KEEP THIS FOR REFERENCE IN CASE hidden_layer_node IS REPLACED WITH RecurrentTransferMechanism
        # # If pnl_node's function is Stateful, assign value to its previous_value parameter
        # #   so that if Python implementation is run it picks up where PyTorch execution left off
        # if isinstance(pnl_node.function, StatefulFunction):
        #     pnl_node.function.parameters.previous_value._set(torch_gru_output, context)



    def _node_values_hook(module, input, output):
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

        assert len(input) > 1, f"PROGRAM ERROR: PytorchGRUCompositionWrapper hook received only one input: {input}"
        x = input[0]
        # h = input[1] if len(input) > 1 else torch.tensor([[0] * module.hidden_size], dtype=torch.float32)
        h = input[1]

        # Reproduce GRU forward calculations
        r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
        z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
        n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
        h_t = (1 - z_t) * n_t + z_t * h

        # Put internal calculations in dict with corresponding node names as keys
        node_values[RESET_NODE_NAME] = r_t.detach()
        node_values[UPDATE_NODE_NAME] = z_t.detach()
        node_values[NEW_NODE_NAME] = n_t.detach()
        node_values[HIDDEN_LAYER_NODE_NAME] = h_t.detach()

    def log_values(self):
        for node_wrapper in [n for n in self._wrapped_nodes if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()


class PytorchGRUMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for a GRU Node"""

    def _assign_pytorch_function(self, mechanism, device, context):
        self.function = PytorchGRUFunctionWrapper(mechanism.function, device, context)

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
