# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for GRUComposition"""

import numpy as np
import graph_scheduler
import torch
from typing import Union, Optional, Literal, Tuple

from psyneulink.core.compositions.composition import NodeRole
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import DuplicateProjectionError
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper, \
    PytorchProjectionWrapper, PytorchFunctionWrapper, ENTER_NESTED, EXIT_NESTED, SUBCLASS_WRAPPERS
from psyneulink.core.globals.context import Context, handle_external_context
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.globals.keywords import (
    ALL, CONTEXT, INPUT, INPUTS, LEARNING, NODE_VALUES, RUN, SHOW_PYTORCH, SYNCH, SYNCH_WITH_PNL_OPTIONS)
from psyneulink.core.globals.log import LogCondition

__all__ = ['PytorchGRUCompositionWrapper']

class PytorchGRUCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for GRUComposition as a Pytorch Module
    Manage the exchange of the Composition's Projection `Matrices <MappingProjection_Matrix>`
    and the Pytorch GRU Module's parameters, and return its output value.
    """
    def __init__(self,
                 composition,
                 device,
                 outer_creator=None,
                 dtype=None,
                 subclass_components=None,
                 context=None,
                 base_context=Context(execution_id=None),
                 ):

        self._early_init(composition, device)

        _node_wrapper_pairs = self._instantiate_GRU_pytorch_mechanism_wrappers(composition, device, context)
        gru_pytorch_node = _node_wrapper_pairs[0][1]
        torch_gru = gru_pytorch_node.function.function
        _projection_wrapper_pairs = self._instantiate_GRU_pytorch_projection_wrappers(torch_gru, device, context)
        execution_sets = [{gru_pytorch_node}]

        super().__init__(composition=composition,
                         device=device,
                         outer_creator=outer_creator,
                         subclass_components=(_node_wrapper_pairs,
                                              _projection_wrapper_pairs,
                                              execution_sets,
                                              Context()),
                         context=context,
                         base_context=base_context,
                         )

        # The following have to be after super(), so that they can be assigned as attributes of torch.nn.module

        # IMPLEMENTATION NOTE:
        #    This is needed for access by subcomponents to PytorchGRUCompositionWrapper when GRUComposition is nested,
        #    and so _build_pytorch_representation is called on the outer Composition but not GRUComposition itelf;
        #    access must be provided via GRUComposition's pytorch_representation, rather than directly assigning
        #    PytorchGRUCompositionWrapper as an attribute on the subcomponents, since doing the latter introduces a
        #    recursion when torch.nn.module.state_dict() is called on any wrapper in the hiearchay.
        if self.composition.pytorch_representation is None:
            self.composition.pytorch_representation = self
        self.torch_gru = torch_gru
        self.gru_pytorch_node = gru_pytorch_node

        # Note: this has to be done after call to super, so that projections_map has been populated
        self.copy_weights_to_torch_gru(context)

        self.torch_dtype = dtype or torch.float64
        self.numpy_dtype = torch.tensor([10], dtype=self.torch_dtype).numpy().dtype

    def _instantiate_GRU_pytorch_mechanism_wrappers(self, gru_comp, device, context):
        """Instantiate PytorchMechanismWrapper for GRU Node"""
        gru_mech = gru_comp.gru_mech
        pytorch_node = PytorchGRUMechanismWrapper(mechanism=gru_mech,
                                                  composition=gru_comp,
                                                  component_idx=0,
                                                  use=[LEARNING, SHOW_PYTORCH],
                                                  dtype=self.torch_dtype,
                                                  device=device,
                                                  context=context)

        # Check if there is no source Node for the InputPort of the GRUComposition.input_CIM
        source = gru_comp.input_CIM._get_source_node_for_input_CIM(gru_comp.input_node.afferents[0].sender)
        if source is None or not gru_comp.is_nested:
            # If either the GRUComposition is not nested,
            # or it does not receive any Projections from the outer Composition,
            # then treat it as an INPUT Node (that receives inputs to the outer Composition in collect_afferents()
            gru_mech._is_input = True
            pytorch_node._is_input = True
            pytorch_node.afferents = INPUT
        destination = gru_comp.output_CIM._get_destination_info_for_output_CIM(gru_comp.output_node.efferents[
                                                                                   0].receiver)
        if destination is None or not gru_comp.is_nested:
            pytorch_node._is_output = True

        return [(gru_mech, pytorch_node)]

    def _instantiate_GRU_pytorch_projection_wrappers(self, torch_gru, device, context):
        """Create PytorchGRUProjectionWrappers for each learnable Projection of GRUComposition
        For each PytorchGRUProjectionWrapper, assign the current weight matrix of the PNL Projection
        to the corresponding part of the tensor in the parameter of the Pytorch GRU module.
        """

        pnl = self.composition
        self.torch_gru_parameters = torch_gru.parameters

        _projection_wrapper_pairs = []

        # Pytorch parameter info
        hid_len = pnl.hidden_size
        z_idx = hid_len
        n_idx = 2 * hid_len

        w_ih = torch_gru.state_dict()['weight_ih_l0']
        w_hh = torch_gru.state_dict()['weight_hh_l0']
        torch_gru_wts_indices = [(w_ih, slice(None, z_idx)), (w_ih, slice(z_idx, n_idx)),(w_ih, slice(n_idx, None)),
                                 (w_hh, slice(None, z_idx)), (w_hh, slice(z_idx, n_idx)), (w_hh, slice(n_idx, None))]
        pnl_proj_wts = [pnl.wts_ir, pnl.wts_iu, pnl.wts_in, pnl.wts_hr, pnl.wts_hu, pnl.wts_hn]
        for pnl_proj, torch_matrix in zip(pnl_proj_wts, torch_gru_wts_indices):
            _projection_wrapper_pairs.append((pnl_proj,
                                             PytorchGRUProjectionWrapper(projection=pnl_proj,
                                                                         torch_parameter=torch_matrix,
                                                                         use=SYNCH,
                                                                         composition=self.composition,
                                                                         device=device)))
        self._pnl_refs_to_torch_params_map = {'w_ih': w_ih, 'w_hh':  w_hh}

        if pnl.bias:
            from psyneulink.library.compositions.grucomposition.grucomposition import GRU_NODE
            assert torch_gru.bias, f"PROGRAM ERROR: '{pnl.name}' has bias=True but {GRU_NODE}.bias=False. "
            b_ih = torch_gru.state_dict()['bias_ih_l0']
            b_hh = torch_gru.state_dict()['bias_hh_l0']
            torch_gru_bias_indices = [(b_ih, slice(None, z_idx)), (b_ih, slice(z_idx, n_idx)),(b_ih, slice(n_idx, None)),
                                      (b_hh, slice(None, z_idx)), (b_hh, slice(z_idx, n_idx)), (b_hh, slice(n_idx, None))]
            pnl_biases = [pnl.bias_ir, pnl.bias_iu, pnl.bias_in, pnl.bias_hr, pnl.bias_hu, pnl.bias_hn]
            for pnl_bias_proj, torch_bias in zip(pnl_biases, torch_gru_bias_indices):
                _projection_wrapper_pairs.append((pnl_bias_proj,
                                                  PytorchGRUProjectionWrapper(projection=pnl_bias_proj,
                                                                              torch_parameter=torch_bias,
                                                                              use=SYNCH,
                                                                              composition=pnl,
                                                                              device=device)))
            self._pnl_refs_to_torch_params_map.update({'b_ih': b_ih, 'b_hh':  b_hh})

        return _projection_wrapper_pairs

    def _flatten_for_pytorch(self,
                             pnl_proj,
                             sndr_mech,
                             rcvr_mech,
                             nested_port,
                             nested_mech,
                             outer_comp,
                             outer_comp_pytorch_rep,
                             access,
                             context,
                             base_context=Context(execution_id=None),
                             ) -> Tuple:
        """Return PytorchProjectionWrappers for Projections to/from GRUComposition to nested Composition
        Replace GRUComposition's nodes with gru_mech and projections to and from it.
        """

        direct_proj = None
        use = [LEARNING, SYNCH]

        if access == ENTER_NESTED:
            sndr_mech_wrapper = outer_comp_pytorch_rep.nodes_map[sndr_mech]
            rcvr_mech_wrapper = self.nodes_map[self.composition.gru_mech]
            try:
                direct_proj = MappingProjection(name="Projection to GRU COMP",
                                             sender=pnl_proj.sender,
                                             receiver=self.composition.gru_mech,
                                             learnable=pnl_proj.learnable)
            except DuplicateProjectionError:
                direct_proj = self.composition.gru_mech.afferents[0]
            else:
                direct_proj._initialize_from_context(context, base_context)
            # Index of input_CIM.output_ports for which pnl_proj is an efferent
            sender_port_idx = pnl_proj.sender.owner.output_ports.index(pnl_proj.sender)

        elif access == EXIT_NESTED:
            sndr_mech_wrapper = self.nodes_map[self.composition.gru_mech]
            rcvr_mech_wrapper = outer_comp_pytorch_rep.nodes_map[rcvr_mech]
            try:
                direct_proj = MappingProjection(name="Projection from GRU COMP",
                                                sender=self.composition.gru_mech,
                                                receiver=pnl_proj.receiver,
                                                learnable=pnl_proj.learnable)
            except DuplicateProjectionError:
                direct_proj = self.composition.gru_mech.efferents[0]
            else:
                direct_proj._initialize_from_context(context, base_context)
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
            outer_comp_pytorch_rep.projection_wrappers.append(proj_wrapper)
            outer_comp_pytorch_rep.projections_map[direct_proj] = proj_wrapper
            outer_comp_pytorch_rep.composition._pytorch_projections.append(direct_proj)

        return pnl_proj, sndr_mech_wrapper, rcvr_mech_wrapper, use

    @handle_external_context()
    def forward(self, inputs, optimization_num, synch_with_pnl_options, context=None)->dict:
        """Forward method of the model for PyTorch modes

        This is called only when GRUComposition is run as a standalone Composition.
        Otherwise, the node.execute() method is called directly (i.e., it is treated as a single node).
        Returns a dictionary {output_node:value} with the output value for the torch GRU module (that is used
        by the collect_afferents method(s) of the other node(s) that receive Projections from the GRUComposition.

        """

        self._set_synch_with_pnl(synch_with_pnl_options)

        # Get input from GRUComposition's INPUT_NODE
        inputs = inputs[self.composition.input_node]

        # Execute GRU Node
        output = self.gru_pytorch_node.execute(inputs, optimization_num, synch_with_pnl_options, context)

        # Set GRUComposition's OUTPUT Node of output of GRU Node
        self.composition.output_node.parameters.value._set(output.detach().cpu().numpy(), context)
        self.composition.gru_mech.parameters.value._set(output.detach().cpu().numpy(), context)

        return {self.composition.gru_mech: output}

    def _set_synch_with_pnl(self, synch_with_pnl_options):
        if (NODE_VALUES in synch_with_pnl_options and synch_with_pnl_options[NODE_VALUES] == RUN):
            self.gru_pytorch_node.synch_with_pnl = True
        else:
            self.gru_pytorch_node.synch_with_pnl = False

    def copy_weights_to_torch_gru(self, context=None):
        for projection, proj_wrapper in self.projections_map.items():
            if SYNCH in proj_wrapper._use:
                proj_wrapper._copy_pnl_proj_to_torch_gru_parameter(context, self.torch_dtype)

    def get_parameters_from_torch_gru(torch_gru)->Tuple[torch.Tensor]:
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

        wts_ih = torch_gru.state_dict()['weight_ih_l0']
        wts_ir = wts_ih[:z_idx].T.detach().cpu().numpy().copy()
        wts_iu = wts_ih[z_idx:n_idx].T.detach().cpu().numpy().copy()
        wts_in = wts_ih[n_idx:].T.detach().cpu().numpy().copy()
        wts_hh = torch_gru.state_dict()['weight_hh_l0']
        wts_hr = wts_hh[:z_idx].T.detach().cpu().numpy().copy()
        wts_hu = wts_hh[z_idx:n_idx].T.detach().cpu().numpy().copy()
        wts_hn = wts_hh[n_idx:].T.detach().cpu().numpy().copy()
        weights = (wts_ir, wts_iu, wts_in, wts_hr, wts_hu, wts_hn)

        biases = None
        if torch_gru.bias:
            # Transpose 1d bias Tensors using permute instead of .T (per PyTorch warning)
            b_ih = torch_gru.state_dict()['bias_ih_l0']
            b_ir = torch.atleast_2d(b_ih[:z_idx].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_iu = torch.atleast_2d(b_ih[z_idx:n_idx].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_in = torch.atleast_2d(b_ih[n_idx:].permute(*torch.arange(b_ih.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_hh = torch_gru.state_dict()['bias_hh_l0']
            b_hr = torch.atleast_2d(b_hh[:z_idx].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_hu = torch.atleast_2d(b_hh[z_idx:n_idx].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            b_hn = torch.atleast_2d(b_hh[n_idx:].permute(*torch.arange(b_hh.ndim - 1, -1, -1))).detach().cpu().numpy().copy()
            biases = (b_ir, b_iu, b_in, b_hr, b_hu, b_hn)
        return weights, biases

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            proj_wrapper.log_matrix()

    def log_values(self):
        for node_wrapper in [n for n in self.node_wrappers if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()


class PytorchGRUMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for Pytorch GRU Node
    Handling of hidden_state: uses GRUComposition's HIDDEN_NODE.value to cache state of hidden layer:
    - gets input to function for hidden state from GRUComposition's HIDDEN_NODE.value
    - sets GRUComposition's HIDDEN_NODE.value to return value for hidden state
    """

    def __init__(self,
                 mechanism,
                 composition,
                 component_idx,
                 use,
                 dtype,
                 device,
                 context):

        super().__init__(mechanism=mechanism,
                         composition=composition,
                         component_idx=component_idx,
                         use=use,
                         dtype=dtype,
                         device=device,
                         subclass_specifies_function=True,
                         context=context)

        self._assign_GRU_pytorch_function(mechanism, device, context)

        self.synch_with_pnl = False

    def _assign_GRU_pytorch_function(self, mechanism, device, context):
        # Assign PytorchGRUFunctionWrapper of Pytorch GRU module as function of GRU Node
        input_size = self.composition.parameters.input_size.get(context)
        hidden_size = self.composition.parameters.hidden_size.get(context)
        bias = self.composition.parameters.bias.get(context)
        torch_GRU = torch.nn.GRU(input_size=input_size,
                                 hidden_size=hidden_size,
                                 bias=bias).to(dtype=self.torch_dtype)
        self.hidden_state = torch.zeros(1, 1, hidden_size, dtype=self.torch_dtype).to(device)

        function_wrapper = PytorchGRUFunctionWrapper(torch_GRU, device, context)
        self.function = function_wrapper
        mechanism.function = function_wrapper.function

        # Assign input_port functions of GRU Node to PytorchGRUFunctionWrapper
        self.input_ports = [PytorchFunctionWrapper(input_port.function, device, context)
                            for input_port in mechanism.input_ports]

    def execute(self, variable, optimization_num, synch_with_pnl_options, context=None)->torch.Tensor:
        """Execute GRU Node with input variable and return output value.
        Override to set GRU Node's synch_with_pnl option if GRUComposition is a nested Composition
        This is called directly if GRUComposition is in a nested Composition, rather than its forward method.
        Treats GRUComposition as a single node in the PytorchCompositionWrapper's graph, inputs
          received from other node(s) that project to the GRUComposition, and its outputs used by the
          collect_afferents method(s) of the other node(s) that receive Projections from the  GRUComposition.
        """
        # Get hidden state from GRUComposition's HIDDEN_NODE.value
        from psyneulink.library.compositions.grucomposition.grucomposition import HIDDEN_LAYER

        self.composition.pytorch_representation._set_synch_with_pnl(synch_with_pnl_options)

        self.input = variable

        hidden_state = self.composition.nodes[HIDDEN_LAYER].parameters.value.get(context)
        self.hidden_state = torch.tensor(hidden_state).unsqueeze(1)
        # Save starting hidden_state for re-computing current values in _copy_pytorch_node_outputs_to_pnl_values()
        self.previous_hidden_state = self.hidden_state.detach()

        if self.synch_with_pnl:
            self.torch_gru_internal_state_values = \
                self._calculate_torch_gru_internal_state_values(self.input[0][0], self.hidden_state.detach())

        # Execute torch GRU module with input (variable) and hidden state
        self.output, self.hidden_state = self.function(*[self.input, self.hidden_state])
        # self.output, self.hidden_state = self.function.function(*[input, self.hidden_state])

        # Set GRUComposition's HIDDEN_NODE.value to GRU Node's hidden state
        # Note: this must be done in case the GRUComposition is run after learning,
        self.composition.hidden_layer_node.output_port.parameters.value._set(
            self.hidden_state.detach().cpu().numpy().squeeze(), context)

        return self.output

    def collect_afferents(self, batch_size, port=None, inputs:dict=None)->torch.Tensor:
        """
        Return afferent projections for input_port(s) of the Mechanism
        If there is only one input_port, return the sum of its afferents (for those in Composition)
        If there are multiple input_ports, return a tensor (or list of tensors if input ports are ragged) of shape:

        (batch, input_port, projection, ...)

        Where the ellipsis represent 1 or more dimensions for the values of the projected afferent.

        """

        if self.afferents == INPUT:
            # GRUComposition is nested in an outer Composition, and GRU is INPUT Node of that Composition
            #  so get input specified for GRUComposition.input_node from the inputs dict provided in the learn() method
            assert self.mechanism._is_input, \
                f"PROGRAM ERROR: No afferents found for '{self.mechanism.name}' in AutodiffComposition"
            input_port = self.composition.input_node.input_port
            curr_val = inputs[input_port]
            if type(curr_val) == torch.Tensor:
                ip_res = [curr_val[:, 0, ...]]
            else:
                val = [batch_elem[0] for batch_elem in curr_val]
                val = torch.stack(val)
                ip_res = [val]
            res = []

        else:
            proj_wrapper = self.afferents[0]

            curr_val = proj_wrapper.sender_wrapper.output
            if curr_val is not None:
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

    def execute_input_ports(self, variable)->torch.Tensor:
        from psyneulink.core.components.functions.nonstateful.transformfunctions import TransformFunction
        assert type(variable) == torch.Tensor, (f"PROGRAM ERROR: Input to GRUComposition in ExecutionMode.Pytorch "
                                                f"should be a torch.Tensor, but is {type(variable)}.")
        # Return the input for the port for all items in the batch
        return variable[:, 0, ...]

    def _calculate_torch_gru_internal_state_values(self, input, hidden_state)->dict:
        """Manually calculate and store internal state values for torch GRU prior to backward pass
        These are needed for assigning to the corresponding nodes in the GRUComposition.
        Returns r_t, z_t, n_t, h_t current reset, update, new, hidden and state values, respectively
        """
        torch_gru_parameters = PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(self.function.function)

        # Get weights
        torch_weights = list(torch_gru_parameters[0])
        for i, weight in enumerate(torch_weights):
            torch_weights[i] = torch.tensor(weight, dtype=self.torch_dtype)
        w_ir, w_iz, w_in, w_hr, w_hz, w_hn = torch_weights

        # Get biases
        pnl_comp = self.composition
        if pnl_comp.bias:
            assert len(torch_gru_parameters) > 1, \
                (f"PROGRAM ERROR: '{pnl_comp.name}' has bias set to True, "
                 f"but no bias weights were returned for torch_gru_parameters.")
            b_ir, b_iz, b_in, b_hr, b_hz, b_hn = torch_gru_parameters[1]
        else:
            b_ir = b_iz = b_in = b_hr = b_hz = b_hn = 0.0

        # Do calculations for internal state values
        x = input.detach()
        h = hidden_state
        r_t = torch.sigmoid(torch.matmul(x, w_ir) + b_ir + torch.matmul(h, w_hr) + b_hr)
        z_t = torch.sigmoid(torch.matmul(x, w_iz) + b_iz + torch.matmul(h, w_hz) + b_hz)
        n_t = torch.tanh(torch.matmul(x, w_in) + b_in + r_t * (torch.matmul(h, w_hn) + b_hn))
        h_t = (1 - z_t) * n_t + z_t * h

        from psyneulink.library.compositions.grucomposition.grucomposition import GRU_INTERNAL_STATE_NAMES
        return {k:v for k,v in zip(GRU_INTERNAL_STATE_NAMES, [n_t, r_t, z_t, h_t])}

    def set_pnl_variable_and_values(self,
                                    set_variable:bool=False,
                                    set_value:bool=True,
                                    # FIX: 3/15/25 - ADD SUPPORT FOR THESE
                                    # set_output_values:bool=None,
                                    # execute_mech:bool=True,
                                    context=None):

        if set_variable:
            assert False, \
                f"PROGRAM ERROR: copying variables to GRUComposition from pytorch execution is not currently supported."

        if set_value:
            n_t, r_t, z_t, h_t = list(self.torch_gru_internal_state_values.values())
            try:
                # Ensure that result of manual-calculated state values matches output of actual call to PyTorch module
                np.testing.assert_allclose(h_t.detach().numpy(),
                                           self.output.detach().numpy(),
                                           atol=1e-8)
            except ValueError:
                assert False, "PROGRAM ERROR:  Problem with calculation of internal states of {pnl_comp.name} GRU Node."

            # Set values of nodes in pnl gru_comp to the result of the corresponding computations in the PyTorch module
            pnl_comp = self.composition
            pnl_comp.reset_node.output_port.parameters.value._set(r_t.detach().cpu().numpy().squeeze(), context)
            pnl_comp.update_node.output_ports[0].parameters.value._set(z_t.detach().cpu().numpy().squeeze(), context)
            pnl_comp.update_node.output_ports[1].parameters.value._set(z_t.detach().cpu().numpy().squeeze(), context)
            pnl_comp.new_node.output_port.parameters.value._set(n_t.detach().cpu().numpy().squeeze(), context)
            pnl_comp.output_node.output_port.parameters.value._set(h_t.detach().cpu().numpy().squeeze(), context)
            # Note: no need to set hidden_layer since it was already done when the GRU Node executed
            # pnl_comp.hidden_layer_node.output_port.parameters.value._set(h_t.detach().cpu().numpy().squeeze(), context)

            # # KEEP THIS FOR REFERENCE IN CASE hidden_layer_node IS REPLACED WITH RecurrentTransferMechanism
            # # If pnl_node's function is Stateful, assign value to its previous_value parameter
            # #   so that if Python implementation is run it picks up where PyTorch execution left off
            # if isinstance(pnl_node.function, StatefulFunction):
            #     pnl_node.function.parameters.previous_value._set(torch_gru_output, context)

    def log_value(self):
        # FIX: LOG HIDDEN STATE OF COMPOSITION MECHANISM
        if self.mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self.mechanism.output_port.parameters.value._set(detached_value, self._context)
            self.mechanism.parameters.value._set(detached_value, self._context)

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
        the `Projection` of the GRUComposition being wrapped

    composition : AutodiffComposition
        the `AutodiffComposition` to which the `Projection` being wrapped belongs
        (and for which the PytorchCompositionWrapper -- to which the PytorchProjectionWrapper
        belongs -- is the `pytorch_representation <AutodiffComposition.pytorch_representation>`).

    torch_parameter: Pytorch parameter
        the torch.nn.Parameter corresponding to the matrix of the Projection;

    matrix_indices: slice
        a slice specifying the part of the Pytorch parameter corresponding to the GRUCOmposition Projection's matrix.

    """
    def __init__(self,
                 projection:MappingProjection,
                 torch_parameter:Tuple,
                 use:Union[list, Literal[LEARNING, SYNCH, SHOW_PYTORCH]],
                 composition:AutodiffComposition,
                 device:str):
        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        # GRUComposition Projection being wrapped:
        self.projection = projection # PNL Projection being wrapped
        self._pnl_proj = projection
        # Assign parameter and tensor indices of Pytorch GRU module parameter corresponding to the Projection's matrix:
        self.torch_parameter, self.matrix_indices = torch_parameter
        # Projections for GRUComposition are not included in autodiff; matrices are set directly in Pytorch GRU module:
        self.projection.exclude_in_autodiff = True
        self._use = convert_to_list(use)
        self.composition = composition
        self.device = device

    def _copy_pnl_proj_to_torch_gru_parameter(self, context, dtype):
        """Set relevant part of tensor for parameter of Pytorch GRU module from GRUComposition's Projections."""
        matrix = self.projection.parameters.matrix._get(context).T
        torch_tensor = self.torch_parameter[self.matrix_indices]
        self.composition.copy_projection_matrix_to_torch_param(projection=self.projection,
                                                               torch_param=torch_tensor,
                                                               validate=False,
                                                               context=context)

    def _copy_torch_params_to_pnl_proj(self, context):
        """Override to deal with indexed tensor of Pytorch GRU module Parameter"""
        torch_parameter = self.torch_parameter
        torch_indices = self.matrix_indices
        matrix = torch_parameter[torch_indices].detach().cpu()
        self.composition.copy_torch_param_to_projection_matrix(torch_param=matrix,
                                                               projection=self.projection,
                                                               validate=False,
                                                               context=context)

    def log_matrix(self):
        if self.projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self.projection.parameters.matrix._set(detached_matrix, context=self._context)
            self.projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)


# class PytorchGRUFunctionWrapper(PytorchFunctionWrapper):
class PytorchGRUFunctionWrapper(torch.nn.Module):
    def __init__(self, function, device, context=None):
        super().__init__()
        self.name = f"PytorchFunctionWrapper[GRU NODE]"
        self._context = context
        self._pnl_function = function
        self.function = function

    def __repr__(self):
        return "PytorchWrapper for: " + self._pnl_function.__repr__()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
