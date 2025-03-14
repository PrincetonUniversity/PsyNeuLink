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
import graph_scheduler

from typing import Union, Optional, Literal

from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import DuplicateProjectionError
from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper, \
    PytorchProjectionWrapper, PytorchFunctionWrapper, ENTER_NESTED, EXIT_NESTED
from psyneulink.core.globals.context import handle_external_context, ContextFlags
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.globals.keywords import ALL, CONTEXT, INPUTS, LEARNING, SHOW_PYTORCH, SYNCH
from psyneulink.core.globals.log import LogCondition

__all__ = ['PytorchGRUCompositionWrapper', 'GRU_NODE_NAME', 'TARGET_NODE_NAME']

GRU_NODE_NAME = 'PYTORCH GRU NODE'
TARGET_NODE_NAME = 'GRU TARGET NODE'

class PytorchGRUCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for GRUComposition as a Pytorch Module
    Manage the exchange of the Composition's Projection `Matrices <MappingProjection_Matrix>`
    and the Pytorch GRU Module's parameters, and return its output value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._torch_gru = self._composition.gru_mech.function

        self.torch_dtype = kwargs.pop('dtype', torch.float64)
        self.numpy_dtype = torch.tensor([10], dtype=self.torch_dtype).numpy().dtype

    def _instantiate_pytorch_mechanism_wrappers(self, composition, device, context):
        """Instantiate PytorchMechanismWrapper for GRU Node"""
        node = composition.gru_mech
        pytorch_node = PytorchGRUMechanismWrapper(mechanism=node,
                                                  composition_wrapper=self,
                                                  component_idx=0,
                                                  use=[LEARNING, SHOW_PYTORCH],
                                                  dtype=self.torch_dtype,
                                                  device=device,
                                                  context=context)
        self.gru_pytorch_node = pytorch_node
        self.torch_gru = pytorch_node.function.function
        self._nodes_map[node] = pytorch_node
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
            self._projection_map[pnl_proj] = PytorchGRUProjectionWrapper(projection=pnl_proj,
                                                                         torch_parameter=torch_matrix,
                                                                         use=SYNCH,
                                                                         device=device)
        self._pnl_refs_to_torch_params_map = {'w_ih': w_ih, 'w_hh':  w_hh}

        if pnl.bias:
            assert torch_gru.bias, f"PROGRAM ERROR: '{pnl.name}' has bias=True but {GRU_NODE_NAME}.bias=False. "
            b_ih = torch_params['bias_ih_l0']
            b_hh = torch_params['bias_hh_l0']
            torch_gru_bias_indices = [(b_ih, slice(None, z_idx)), (b_ih, slice(z_idx, n_idx)),(b_ih, slice(n_idx, None)),
                                      (b_hh, slice(None, z_idx)), (b_hh, slice(z_idx, n_idx)), (b_hh, slice(n_idx, None))]
            pnl_biases = [pnl.bias_ir, pnl.bias_iu, pnl.bias_in, pnl.bias_hr, pnl.bias_hu, pnl.bias_hn]
            for pnl_bias_proj, torch_bias in zip(pnl_biases, torch_gru_bias_indices):
                self._projection_map[pnl_bias_proj] = PytorchGRUProjectionWrapper(projection=pnl_bias_proj,
                                                                                  torch_parameter=torch_bias,
                                                                                  use=SYNCH,
                                                                                  device=device)
            self._pnl_refs_to_torch_params_map.update({'b_ih': b_ih, 'b_hh':  b_hh})

        self.copy_weights_to_torch_gru(context)

    def _instantiate_execution_sets(self, composition, execution_context, base_context):
        self.execution_sets = [{self.gru_pytorch_node}]

    def _flatten_for_pytorch(self,
                             pnl_proj,
                             sndr_mech,
                             rcvr_mech,
                             nested_port,
                             nested_mech,
                             outer_comp,
                             outer_comp_pytorch_rep,
                             access,
                             context)->tuple:
        """Return PytorchProjectionWrappers for Projections to/from GRUComposition to nested Composition
        Replace GRUComposition's nodes with gru_mech and projections to and from it."""

        # sndr_mech_wrapper = None
        # rcvr_mech_wrapper = None
        direct_proj = None
        # use = [LEARNING, SYNCH, SHOW_PYTORCH]
        use = [LEARNING, SYNCH]

        if access == ENTER_NESTED:
            sndr_mech_wrapper = outer_comp_pytorch_rep._nodes_map[sndr_mech]
            rcvr_mech_wrapper = self._nodes_map[self._composition.gru_mech]
            try:
                direct_proj = MappingProjection(name="Projection to GRU COMP",
                                             sender=pnl_proj.sender,
                                             receiver=self._composition.gru_mech,
                                             learnable=pnl_proj.learnable)
            except DuplicateProjectionError:
                direct_proj = self._composition.gru_mech.afferents[0]
            # Index of input_CIM.output_ports for which pnl_proj is an efferent
            sender_port_idx = pnl_proj.sender.owner.output_ports.index(pnl_proj.sender)

        elif access == EXIT_NESTED:
            sndr_mech_wrapper = self._nodes_map[self._composition.gru_mech]
            rcvr_mech_wrapper = outer_comp_pytorch_rep._nodes_map[rcvr_mech]
            try:
                direct_proj = MappingProjection(name="Projection from GRU COMP",
                                                sender=self._composition.gru_mech,
                                                receiver=pnl_proj.receiver,
                                                learnable=pnl_proj.learnable)
            except DuplicateProjectionError:
                direct_proj = self._composition.gru_mech.efferents[0]
            # gru_mech has only one output_port
            sender_port_idx = 0

        else:
            assert False, f"PROGRAM ERROR: access must be ENTER_NESTED or EXIT_NESTED, not {access}"

        if direct_proj:
            component_idx = list(outer_comp._inner_projections).index(pnl_proj)
            proj_wrapper = PytorchProjectionWrapper(projection=direct_proj,
                                                    pnl_proj=pnl_proj,
                                                    component_idx=component_idx,
                                                    sender_port_idx=sender_port_idx,
                                                    use=[SHOW_PYTORCH],
                                                    device=self.device,
                                                    sender_wrapper=sndr_mech_wrapper,
                                                    receiver_wrapper=rcvr_mech_wrapper,
                                                    context=context)
            outer_comp_pytorch_rep._projection_wrappers.append(proj_wrapper)
            outer_comp_pytorch_rep._projection_map[direct_proj] = proj_wrapper
            outer_comp_pytorch_rep._composition._pytorch_projections.append(direct_proj)

        return pnl_proj, sndr_mech_wrapper, rcvr_mech_wrapper, use

    def _regenerate_paramlist(self):
        """Add Projection matrices to Pytorch Module's parameter list"""
        self.params = torch.nn.ParameterList()
        for proj_wrapper in [p for p in self._projection_wrappers if not p.projection.exclude_in_autodiff]:
            self.params.append(proj_wrapper.matrix)

        nested_node_params = [list(node.function.function.parameters())
                              for node in self._wrapped_nodes
                              if hasattr(node, 'function') and isinstance(node.function.function, torch.nn.Module)]
        for item in nested_node_params:
            for item_small in item:
                self.params.append(item_small)
        assert True

    @handle_external_context()
    def forward(self, inputs, optimization_num, context=None)->dict:
        """Forward method of the model for PyTorch modes
        Returns a dictionary {output_node:value} with the output value for the module in case it is run as a
        standalone Composition; otherwise, those will be ignored and the outputs will be used by the aggregate_afferents
        method(s) of the other node(s) that receive Projections from the GRUComposition.
        """
        # Reshape input for GRU module (to torch_dtype)
        # # MODIFIED 3/13/25 OLD:
        # inputs = torch.tensor(np.array(inputs[self._composition.input_node]).astype(self.numpy_dtype))
        # hidden_state = self._composition.hidden_state
        # output, self.hidden_state = self.gru_pytorch_node.execute([inputs, hidden_state], context)
        # MODIFIED 3/13/25 NEW:
        output, self.hidden_state = self.gru_pytorch_node.execute(inputs[self._composition.input_node],context)
        # MODIFIED 3/13/25 END
        # Assign output to the OUTPUT Node of the GRUComposition
        self._composition.output_node.parameters.value._set(output.detach().cpu().numpy(), context)
        self._composition.gru_mech.parameters.value._set(output.detach().cpu().numpy(), context)

        assert 'DEBUGGING BREAK POINT'

        return {self._composition.gru_mech: output}

    def log_weights(self):
        for proj_wrapper in self._projection_wrappers:
            proj_wrapper.log_matrix()

    # FIX: 3/9/25 - ALONG LINES OF _copy_pytorch_node_outputs_to_pnl_values
    def _copy_pytorch_node_inputs_to_pnl_variables(self,
                                                   nodes:Optional[Union[list,Literal[ALL, INPUTS]]]=ALL,
                                                   context=None):
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

    def _copy_pytorch_node_outputs_to_pnl_values(self, nodes, context):
        # FIX: 3/9/25 - FLESH THIS OUT TO BE SURE ONLY NODES ARE GRU, INPUT AND OUTPUT
        #               AND EXPLICITY ASSIGN pnl_node TO ONE IN MAP
        node_map = list(nodes)
        pnl_comp = self._composition
        pnl_node = list(nodes)[0][0]
        pytorch_node = list(nodes)[0][1]

        assert len(nodes) == 1, \
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
            torch_weights[i] = torch.tensor(weight, dtype=self.torch_dtype)
        w_ir, w_iz, w_in, w_hr, w_hz, w_hn = torch_weights
        if pnl_comp.bias:
            assert len(torch_gru_parameters) > 1, \
                (f"PROGRAM ERROR: '{pnl_comp.name}' has bias set to True, "
                 f"but no bias weights were returned for torch_gru_parameters.")
            b_ir, b_iz, b_in, b_hr, b_hz, b_hn = torch_gru_parameters[1]
        else:
            b_ir = b_iz = b_in = b_hr = b_hz = b_hn = 0.0

        x = self.gru_pytorch_node.input[0][0]

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

    def log_values(self):
        for node_wrapper in [n for n in self._wrapped_nodes if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()

    def copy_weights_to_psyneulink(self, context=None):
        for proj_wrapper in self._projection_map.values():
            if SYNCH in proj_wrapper._use:
                proj_wrapper._copy_params_to_pnl_proj(context)

    def copy_weights_to_torch_gru(self, context=None):
        for projection, proj_wrapper in self._projection_map.items():
            if SYNCH in proj_wrapper._use:
                proj_wrapper.set_torch_gru_parameter(context, self.torch_dtype)

    def get_weights_from_torch_gru(torch_gru)->tuple[torch.Tensor]:
        """Get parameters from PyTorch GRU module corresponding to GRUComposition's Projections.
        Format tensors:
          - transpose all weight and bias tensors;
          - reformat biases as 2d
        Return formatted tensors, which are used:
         - in set_weights_from_torch_gru(), where they are converted to numpy arrays
         - for forward computation in pytorchGRUwrappers._copy_pytorch_node_outputs_to_pnl_values()
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
            b_ir = torch.atleast_2d(b_ih[:z_idx].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_iu = torch.atleast_2d(b_ih[z_idx:n_idx].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_in = torch.atleast_2d(b_ih[n_idx:].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_hh = torch_gru_weights['bias_hh_l0']
            b_hr = torch.atleast_2d(b_hh[:z_idx].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_hu = torch.atleast_2d(b_hh[z_idx:n_idx].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_hn = torch.atleast_2d(b_hh[n_idx:].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            biases = (b_ir, b_iu, b_in, b_hr, b_hu, b_hn)
        return weights, biases


class PytorchGRUMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for a Pytorch GRU Node"""

    def __init__(self, mechanism, composition_wrapper, component_idx, use, dtype, device, context):
        self.torch_dtype = dtype
        super().__init__(mechanism, composition_wrapper, component_idx, use, device, context)

    def _assign_pytorch_function(self, mechanism, device, context):
        # Assign PytorchGRUFunctionWrapper of Pytorch GRU module as function of GRU Node
        input_size = self._composition_wrapper_owner._composition.parameters.input_size.get(context)
        hidden_size = self._composition_wrapper_owner._composition.parameters.hidden_size.get(context)
        bias = self._composition_wrapper_owner._composition.parameters.bias.get(context)
        # # MODIFIED 3/13/25 OLD:
        # function_wrapper = PytorchGRUFunctionWrapper(torch.nn.GRU(input_size=input_size,
        #                                                           hidden_size=hidden_size,
        #                                                           bias=bias,
        #                                                           dtype=self.torch_dtype),
        #                                              device, context)
        # MODIFIED 3/13/25 NEW:
        torch_GRU = torch.nn.GRU(input_size=input_size,
                                 hidden_size=hidden_size,
                                 bias=bias).to(dtype=self.torch_dtype)
        function_wrapper = PytorchGRUFunctionWrapper(torch_GRU, device, context)
        # MODIFIED 3/13/25 END
        self.function = function_wrapper
        mechanism.function = function_wrapper.function

        # Assign input_port functions of GRU Node to PytorchGRUFunctionWrapper
        self.input_ports = [PytorchGRUFunctionWrapper(input_port.function, device, context)
                            for input_port in mechanism.input_ports]

    # MODIFIED 3/13/25 OLD:
    # def execute(self, variable, context):
    #     """Execute GRU Node with input variable and return output value
    #     """
    #     # FIX: 3/9/25: THIS NEEDS TO BE CLEANED UP
    #
    #     # FIX:
    #     #  IF BEING CALLED AFTER CALL TO RUN OF GRUComposition, and only 1 item in variable shoud add...
    #     #  THEN NONE BELOW SHOUD BE REPLACED WITH hidden_node.value OF GRUComposition TO ESTABLISH STATE
    #     # hidden = self._composition.nodes["HIDDEN\nLAYER"].value
    #
    #     # MODIFIED 3/13/25 OLD:
    #     self.input = variable
    #     if isinstance(variable, list): # FIX NON-NESTED (CALLED FROM GRUComposition; NUMERICALLY VALIDATED)
    #         # # MODIFIED 3/13/25 OLD:
    #         # self.output = self.function(*variable)
    #         # MODIFIED 3/13/25 NEW:
    #         self.output = self.function(*[var.type(self.torch_dtype)
    #                                       if var is not None else None for var in variable])
    #         # MODIFIED 3/13/25 END
    #         return self.output
    #     else:  # FIX: NESTED (CALLED FROM AutodiffComposition); NOT NUMERICALLY VALIDATED, MAY NOT HHANDLE hidden
    #         variable = variable.type(self.torch_dtype)
    #         output, hidden = self.function(*variable)
    #         output = output.type(PytorchCompositionWrapper.torch_dtype)
    #         hidden = hidden.type(PytorchCompositionWrapper.torch_dtype)
    #         self.output = output
    #         return output, hidden
    # MODIFIED 3/13/25 NEW:
    def execute(self, inputs, context):
        """Execute GRU Node with input variable and return output value
        """
        composition = self._composition_wrapper_owner._composition
        # inputs = torch.tensor(np.array(inputs[composition.input_node]).astype(self.numpy_dtype))
        # FIX: NEED TO DEAL WITH EXTRA DIMENSIONAITY HERE
        inputs = torch.tensor(inputs[0]).to(self.torch_dtype)
        hidden_state = (torch.tensor(composition.hidden_state).to(self.torch_dtype)
                        if composition.hidden_state is not None else None)
        self.input = [inputs, hidden_state]
        output = self.function(*self.input)
        self.output = output[0].unsqueeze(1)
        composition.hidden_state = output[1].detach().numpy()
        return output
        # MODIFIED 3/13/25 END

    def collect_afferents(self, batch_size, port=None):
        """
        Return afferent projections for input_port(s) of the Mechanism
        If there is only one input_port, return the sum of its afferents (for those in Composition)
        If there are multiple input_ports, return a tensor (or list of tensors if input ports are ragged) of shape:

        (batch, input_port, projection, ...)

        Where the ellipsis represent 1 or more dimensions for the values of the projected afferent.

        FIX: AUGMENT THIS TO SUPPORT InputPort's function
        """
        assert self.afferents,\
            f"PROGRAM ERROR: No afferents found for '{self.mechanism.name}' in AutodiffComposition"

        proj_wrapper = self.afferents[0]
        curr_val = proj_wrapper.sender_wrapper.output
        if curr_val is not None:
            # proj_wrapper._curr_sender_value = proj_wrapper.sender_wrapper.output[proj_wrapper._value_idx]
            if type(curr_val) == torch.Tensor:
                proj_wrapper._curr_sender_value = curr_val[:, proj_wrapper._value_idx, ...]
            else:
                val = [batch_elem[proj_wrapper._value_idx] for batch_elem in curr_val]
                val = torch.stack(val)
                proj_wrapper._curr_sender_value = val
        else:
            val = torch.tensor(proj_wrapper.default_value)

            # We need to add the batch dimension to default values.
            val = val[None, ...].expand(batch_size, *val.shape)

            proj_wrapper._curr_sender_value = val

        proj_wrapper._curr_sender_value = torch.atleast_1d(proj_wrapper._curr_sender_value)

        res = []
        input_port = self.mechanism.input_port
        ip_res = [proj_wrapper.execute(proj_wrapper._curr_sender_value)]

        # Stack the results for this input port on the second dimension, we want to preserve
        # the first dimension as the batch
        ip_res = torch.stack(ip_res, dim=1)
        res.append(ip_res)

        try:
            # Now stack the results for all input ports on the second dimension again, this keeps batch
            # first again. We should now have a 4D tensor; (batch, input_port, projection, values)
            res = torch.stack(res, dim=1)
        except (RuntimeError, TypeError):
            # is ragged, will handle ports individually during execute
            # We still need to reshape things so batch size is first dimension.
            batch_size = res[0].shape[0]
            res = [[inp[b] for inp in res] for b in range(batch_size)]

        return res

    def execute_input_ports(self, variable):
        from psyneulink.core.components.functions.nonstateful.transformfunctions import TransformFunction

        assert type(variable) == torch.Tensor, (f"PROGRAM ERROR: Input to GRUComposition in ExecutionMode.Pytorch "
                                                f"should be a torch.Tensor, but is {type(variable)}.")

        # must iterate over at least 1d input per port
        variable = torch.atleast_2d(variable)

        res = [variable[:, 0, ...]] # Get the input for the port for all items in the batch

        try:
            res = torch.stack(res, dim=1) # Stack along the input port dimension, first dimension is batch
        except (RuntimeError, TypeError):
            # is ragged, need to reshape things so batch size is first dimension.
            batch_size = res[0].shape[0]
            res = [[inp[b] for inp in res] for b in range(batch_size)]
        return res

    def log_value(self):
        # FIX: LOG HIDDEN STATE OF COMPOSITION MECHANISM
        if self._mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self._mechanism.output_port.parameters.value._set(detached_value, self._context)
            self._mechanism.parameters.value._set(detached_value, self._context)

    def log_matrix(self):
        if self.projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self.projection.parameters.matrix._set(detached_matrix, context=self._context)
            self.projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)


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
    projection:  MappingProjection
        the Projection of the GRUComposition being wrapped

    torch_parameter: Pytorch parameter
        the tensor corresponding to the matrix of the Projection;

    matrix_indices: slice
        a slice specifying the part of the Pytorch parameter corresponding to the GRUCOmposition Projection's matrix.

    """
    def __init__(self,
                 projection:MappingProjection,
                 torch_parameter:tuple,
                 use:Union[list, Literal[LEARNING, SYNCH, SHOW_PYTORCH]],
                 device:str):
        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        # GRUComposition Projection being wrapped:
        self.projection = projection # PNL Projection being wrapped
        # Assign parameter and tensor indices of Pytorch GRU module parameter corresponding to the Projection's matrix:
        self.torch_parameter, self.matrix_indices = torch_parameter
        # Projections for GRUComposition are not included in autodiff; matrices are set directly in Pytorch GRU module:
        self.projection.exclude_in_autodiff = True
        self._use = convert_to_list(use)
        self.device = device

    def set_torch_gru_parameter(self, context, dtype):
        """Set relevant part of tensor for parameter of Pytorch GRU module from GRUComposition's Projections."""
        matrix = self.projection.parameters.matrix._get(context).T
        proj_matrix_as_tensor =  torch.tensor(matrix.squeeze(), dtype=dtype)
        self.torch_parameter[self.matrix_indices].data.copy_(proj_matrix_as_tensor)

    def _copy_params_to_pnl_proj(self, context):
        torch_parameter = self.torch_parameter
        torch_indices = self.matrix_indices
        matrix =  torch_parameter[torch_indices].detach().cpu().clone().numpy().T
        self.projection.parameters.matrix._set(matrix, context)
        self.projection.parameter_ports['matrix'].parameters.value._set(matrix, context)

    def log_matrix(self):
        if self.projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self.projection.parameters.matrix._set(detached_matrix, context=self._context)
            self.projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)


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

