# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for RNNComposition"""

import numpy as np
import graph_scheduler
import torch
from collections import defaultdict

from typing import Union, Optional, Literal, Tuple

from torch import nn

import psyneulink.core.scheduling.condition as conditions
from psyneulink.core.compositions.composition import LearningScale, NodeRole
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import Projection, DuplicateProjectionError
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
from psyneulink.library.compositions.pytorchwrappers import (
    PytorchCompositionWrapper, PytorchMechanismWrapper, PytorchProjectionWrapper, PytorchFunctionWrapper,
    PytorchLossMechanismWrapper, ENTER_NESTED, EXIT_NESTED, TorchParam, ParamNameCompositionTuple)

from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import convert_to_list, convert_to_np_array
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import (ALL, ANY, CONTEXT, DEFAULT, INPUT, INPUTS, LEARNING,
                                              NODE_VALUES, SAMPLE, SHOW_PYTORCH, SYNCH, SYNCH_WITH_PNL_OPTIONS)
from psyneulink.core.globals.log import LogCondition

__all__ = [
    'PytorchRNNCompositionWrapper',
    'BIAS_INPUT_TO_HIDDEN', 'BIAS_HIDDEN_TO_HIDDEN',
    'HIDDEN_TO_HIDDEN', 'INPUT_TO_HIDDEN',
    'W_IH_NAME', 'W_HH_NAME'
]

INPUT_TO_HIDDEN = 'INPUT TO HIDDEN'
HIDDEN_TO_HIDDEN = 'HIDDEN TO HIDDEN'
BIAS_INPUT_TO_HIDDEN = 'BIAS INPUT TO HIDDEN'
BIAS_HIDDEN_TO_HIDDEN = 'BIAS HIDDEN TO HIDDEN'
W_IH_NAME = 'weight_ih_l0'
W_HH_NAME = 'weight_hh_l0'
B_IH_NAME = 'bias_ih_l0'
B_HH_NAME = 'bias_hh_l0'


class PytorchRNNCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for RNNComposition as a Pytorch Module
    Manage the exchange of the Composition's Projection `Matrices <MappingProjection_Matrix>`
    and the Pytorch RNN Module's parameters, and return its output value.
    """

    def _pytorch_mechanism_wrapper_type(self, mech):
        return defaultdict(lambda: PytorchMechanismWrapper,
                           {LossMechanism: PytorchLossMechanismWrapper,
                            ProcessingMechanism: PytorchRNNMechanismWrapper}
                           )[mech.__class__]

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

        _node_wrapper_pairs = self._instantiate_RNN_pytorch_mechanism_wrappers(
            composition, outer_creator, device, context
        )
        rnn_pytorch_node = _node_wrapper_pairs[0][1]
        torch_rnn = rnn_pytorch_node.function.function
        _projection_wrapper_pairs = self._instantiate_RNN_pytorch_projection_wrappers(
            torch_rnn, device, context
        )
        execution_sets = [{rnn_pytorch_node}]

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
        #    This is needed for access by subcomponents to PytorchRNNCompositionWrapper when RNNComposition is nested,
        #    and so _build_pytorch_representation is called on the outer Composition but not RNNComposition itelf;
        #    access must be provided via RNNComposition's pytorch_representation, rather than directly assigning
        #    PytorchRNNCompositionWrapper as an attribute on the subcomponents, since doing the latter introduces a
        #    recursion when torch.nn.module.state_dict() is called on any wrapper in the hiearchay.
        if self.composition.pytorch_representation is None:
            self.composition.pytorch_representation = self
        self.torch_rnn = torch_rnn
        self.rnn_pytorch_node = rnn_pytorch_node

        # Note: this has to be done after call to super, so that projections_map has been populated
        if context.source != ContextFlags.SHOW_GRAPH:
            self.copy_weights_to_torch_rnn(context)

        self.torch_dtype = dtype or torch.float64
        self.numpy_dtype = torch.tensor([10], dtype=self.torch_dtype).numpy().dtype

        # since stateful objects are being created within a run call
        # (after context was initialized), need to initialize manually here
        for dummy_proj, _ in _projection_wrapper_pairs:
            dummy_proj._initialize_from_context(context, base_context)

    def _validate_optimizer_param_specs(self, optimizer_param_specs: dict, source: str, context, nested=False):
        """Override to filter and raise error for individual Projections (i.e., specifications of slices)"""
        from psyneulink.library.compositions.rnncomposition.rnncomposition import (
            RNNCompositionError, INPUT_TO_HIDDEN_WEIGHTS, HIDDEN_TO_HIDDEN_WEIGHTS)

        # Raise error for attempt to specify bias parameters when bias=False
        if not self.composition.bias:
            bias_specs = [spec for spec in optimizer_param_specs
                          if spec in {BIAS_INPUT_TO_HIDDEN, BIAS_HIDDEN_TO_HIDDEN}]
            if bias_specs:
                bias_specs = [spec.replace(BIAS_INPUT_TO_HIDDEN, 'BIAS_INPUT_TO_HIDDEN') for spec in bias_specs]
                bias_specs = [spec.replace(BIAS_HIDDEN_TO_HIDDEN, 'BIAS_HIDDEN_TO_HIDDEN') for spec in bias_specs]
                raise RNNCompositionError(
                    f"Attempt to set learning rate for bias(es) of RNN using '{' ,'.join(bias_specs)}' in the "
                    f"'learning_rate' arg of the {self.get_source_str(source)} for '{self.composition.name}' "
                    f"when its bias option is set to False; the spec(s) must be removed or bias set to True.")

        # Raise error for attempt to specify individual input_to_hidden or hidden_to_hidden Projections
        bad_ih_specs = [spec for spec in optimizer_param_specs if spec in INPUT_TO_HIDDEN_WEIGHTS]
        if bad_ih_specs:
            raise RNNCompositionError(f"RNNComposition does not support setting of learning rates "
                                      f"for individual input_to_hidden Projections ({' ,'.join(bad_ih_specs)}); "
                                      f"use 'INPUT_TO_HIDDEN' to set learning rate for all such weights.")
        bad_hh_specs = [spec for spec in optimizer_param_specs if spec in HIDDEN_TO_HIDDEN_WEIGHTS]
        if bad_hh_specs:
            raise RNNCompositionError(f"RNNComposition does not support setting of learning rates "
                                      f"for individual hidden_to_hidden Projections ({' ,'.join(bad_hh_specs)}); "
                                      f"use 'HIDDEN_TO_HIDDEN' to set learning rate for all such weights.")

    def _instantiate_RNN_pytorch_mechanism_wrappers(self, rnn_comp, outer_creator, device, context):
        """Instantiate PytorchMechanismWrapper for RNN Node"""
        rnn_mech = rnn_comp.rnn_mech
        pytorch_node = PytorchRNNMechanismWrapper(mechanism=rnn_mech,
                                                  rnn_composition=rnn_comp,
                                                  component_idx=0,
                                                  outer_creator=outer_creator,
                                                  use=[LEARNING, SHOW_PYTORCH],
                                                  dtype=self.torch_dtype,
                                                  device=device,
                                                  context=context)

        # Check if there is no source Node for the InputPort of the RNNComposition.input_CIM
        source = rnn_comp.input_CIM._get_source_node_for_input_CIM(rnn_comp.input_node.afferents[0].sender)
        if source is None or not rnn_comp.is_nested:
            # If either the RNNComposition is not nested,
            # or it does not receive any Projections from the outer Composition,
            # then treat it as an INPUT Node (that receives inputs to the outer Composition in collect_afferents()
            rnn_mech._is_input = True
            pytorch_node._is_input = True
            pytorch_node.afferents = INPUT
        destination = rnn_comp.output_CIM._get_destination_info_for_output_CIM(rnn_comp.output_node.efferents[
                                                                                   0].receiver)
        if destination is None or not rnn_comp.is_nested:
            pytorch_node._is_output = True

        return [(rnn_mech, pytorch_node)]

    def _instantiate_RNN_pytorch_projection_wrappers(self, torch_rnn, device, context):
        """Create PytorchRNNProjectionWrappers for each learnable Projection of RNNComposition."""
        pnl = self.composition
        self.torch_rnn_parameters = torch_rnn.parameters

        _projection_wrapper_pairs = []

        torch_param_specs = [
            TorchParam(W_IH_NAME, slice(None)),
            TorchParam(W_HH_NAME, slice(None)),
        ]
        pnl_projections = [pnl.wts_ih, pnl.wts_hh]

        for pnl_proj, torch_param_spec in zip(pnl_projections, torch_param_specs):
            torch_param_tuple = (torch_rnn.state_dict()[torch_param_spec[0]], torch_param_spec[1])
            pytorch_wrapper = PytorchRNNProjectionWrapper(
                projection=pnl_proj,
                torch_parameter=torch_param_tuple,
                use=SYNCH,
                composition=self.composition,
                device=device
            )
            _projection_wrapper_pairs.append((pnl_proj, pytorch_wrapper))

        # Construct DummyProjection for INPUT_TO_HIDDEN and HIDDEN_TO_HIDDEN that are used to access their learning_rate
        W_IH_projection = DummyProjection(INPUT_TO_HIDDEN)
        pytorch_wrapper = PytorchRNNProjectionWrapper(
            projection=W_IH_projection,
            torch_parameter=(W_IH_NAME, torch_rnn.state_dict()[W_IH_NAME]),
            use=LEARNING,
            composition=self.composition,
            device=device
        )
        _projection_wrapper_pairs.append((W_IH_projection, pytorch_wrapper))
        input_to_hidden_param_name_comp_tuple = ParamNameCompositionTuple(W_IH_projection, W_IH_NAME, self.composition)
        self._pnl_refs_to_torch_param_names.update({INPUT_TO_HIDDEN: input_to_hidden_param_name_comp_tuple})

        W_HH_projection = DummyProjection(HIDDEN_TO_HIDDEN)
        pytorch_wrapper = PytorchRNNProjectionWrapper(
            projection=W_HH_projection,
            torch_parameter=(W_HH_NAME, torch_rnn.state_dict()[W_HH_NAME]),
            use=LEARNING,
            composition=self.composition,
            device=device
        )
        _projection_wrapper_pairs.append((W_HH_projection, pytorch_wrapper))
        hidden_to_hidden_param_name_comp_tuple = ParamNameCompositionTuple(W_HH_projection, W_HH_NAME, self.composition)
        self._pnl_refs_to_torch_param_names.update({HIDDEN_TO_HIDDEN: hidden_to_hidden_param_name_comp_tuple})

        if pnl.bias:
            assert torch_rnn.bias, (
                f"PROGRAM ERROR: '{pnl.name}' has bias=True but the PyTorch RNN module has bias=False."
            )

            torch_bias_specs = [
                TorchParam(B_IH_NAME, slice(None)),
                TorchParam(B_HH_NAME, slice(None)),
            ]
            pnl_biases = [pnl.bias_ih, pnl.bias_hh]

            for pnl_bias_proj, torch_bias_spec in zip(pnl_biases, torch_bias_specs):
                torch_bias_tuple = (torch_rnn.state_dict()[torch_bias_spec[0]], torch_bias_spec[1])
                pytorch_wrapper = PytorchRNNProjectionWrapper(
                    projection=pnl_bias_proj,
                    torch_parameter=torch_bias_tuple,
                    use=SYNCH,
                    composition=pnl,
                    device=device
                )
                _projection_wrapper_pairs.append((pnl_bias_proj, pytorch_wrapper))

            B_IH_proj = DummyProjection(BIAS_INPUT_TO_HIDDEN)
            pytorch_wrapper = PytorchRNNProjectionWrapper(
                projection=B_IH_proj,
                torch_parameter=(B_IH_NAME, torch_rnn.state_dict()[B_IH_NAME]),
                use=LEARNING,
                composition=self.composition,
                device=device
            )
            _projection_wrapper_pairs.append((B_IH_proj, pytorch_wrapper))
            bias_in_to_hid_param_name_comp_tuple = ParamNameCompositionTuple(B_IH_proj, B_IH_NAME, self.composition)
            self._pnl_refs_to_torch_param_names.update({BIAS_INPUT_TO_HIDDEN: bias_in_to_hid_param_name_comp_tuple})

            B_HH_proj = DummyProjection(BIAS_HIDDEN_TO_HIDDEN)
            pytorch_wrapper = PytorchRNNProjectionWrapper(
                projection=B_HH_proj,
                torch_parameter=(B_HH_NAME, torch_rnn.state_dict()[B_HH_NAME]),
                use=LEARNING,
                composition=self.composition,
                device=device
            )
            _projection_wrapper_pairs.append((B_HH_proj, pytorch_wrapper))
            bias_hid_to_hid_param_name_comp_tuple = ParamNameCompositionTuple(B_HH_proj, B_HH_NAME, self.composition)
            self._pnl_refs_to_torch_param_names.update({BIAS_HIDDEN_TO_HIDDEN: bias_hid_to_hid_param_name_comp_tuple})

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
        """Return PytorchProjectionWrappers for Projections to/from RNNComposition to nested Composition
        Replace RNNComposition's nodes with rnn_mech and projections to and from it.
        """

        def _get_direct_proj(pnl_proj, direction: Literal['to', 'from']) -> Union[Projection, bool]:
            """Get direct Projection to/from RNNComposition's rnn_mech
            Checks for existing Projection and returns that if found; otherwise, constructs it.
            """
            sender = pnl_proj.sender if direction == 'to' else self.composition.rnn_mech
            receiver = self.composition.rnn_mech if direction == 'to' else pnl_proj.receiver
            dir_proj = outer_comp._check_for_existing_projections(sender=sender,
                                                                  receiver=receiver,
                                                                  in_composition=ANY)
            if dir_proj:
                assert len(dir_proj) == 1, (
                    f"PROGRAM ERROR: More than one ({len(direct_proj)} Projections found from "
                    f"{pnl_proj.sender.name} to {self.composition.rnn_mech.name} in {outer_comp.name}. ")
                dir_proj = dir_proj[0]
            else:
                dir_proj = MappingProjection(name="Projection to RNN COMP",
                                             sender=sender,
                                             receiver=receiver,
                                             learnable=pnl_proj.learnable,
                                             learning_rate=pnl_proj.learning_rate)
                dir_proj._initialize_from_context(context, base_context)
            return dir_proj

        direct_proj = None
        use = [LEARNING, SYNCH]

        if access == ENTER_NESTED:
            sndr_mech_wrapper = outer_comp_pytorch_rep.nodes_map[sndr_mech]
            rcvr_mech_wrapper = self.nodes_map[self.composition.rnn_mech]
            direct_proj = _get_direct_proj(pnl_proj, 'to')
            # Index of input_CIM.output_ports for which pnl_proj is an efferent
            sender_port_idx = pnl_proj.sender.owner.output_ports.index(pnl_proj.sender)

        elif access == EXIT_NESTED:
            sndr_mech_wrapper = self.nodes_map[self.composition.rnn_mech]
            rcvr_mech_wrapper = outer_comp_pytorch_rep.nodes_map[rcvr_mech]
            direct_proj = _get_direct_proj(pnl_proj, 'from')
            sender_port_idx = 0

        else:
            assert False, f"PROGRAM ERROR: access must be ENTER_NESTED or EXIT_NESTED, not {access}"

        if direct_proj:
            # component_idx = list(outer_comp._inner_projections).index(pnl_proj)
            component_idx = outer_comp_pytorch_rep._get_composition_projections(outer_comp).index(pnl_proj)

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

    def _get_processing_graph(self, context):
        """Override to use 'PYTORCH RNN NODE' instead of PNL nodes for PytorchShowGraph of standalone RNNComposition"""
        processing_graph = {self.composition.rnn_mech: set()}
        return processing_graph

    def _get_roles_by_node(self, node, context):
        """Override to return NodeRole for 'PYTORCH RNN NODE'"""
        return {self.composition.rnn_mech: [NodeRole.INTERNAL]}

    @handle_external_context()
    def forward(self, inputs, optimization_num, synch_with_pnl_options, retain_in_pnl_options,
                full_sequence_mode, sequence_lengths, context=None):
        """Forward method of the model for PyTorch modes

        This is called only when RNNComposition is run as a standalone Composition.
        Otherwise, the node.execute() method is called directly (i.e., it is treated as a single node).
        Returns a dictionary {output_node:value} with the output value for the torch RNN module (that is used
        by the collect_afferents method(s) of the other node(s) that receive Projections from the RNNComposition.
        """

        self._set_synch_with_pnl(self.rnn_pytorch_node, synch_with_pnl_options)

        # Get input from RNNComposition's INPUT_NODE
        inputs = inputs[self.composition.input_node]

        # Execute RNN Node
        output = self.rnn_pytorch_node.execute(variable=inputs,
                                               optimization_num=optimization_num,
                                               synch_with_pnl_options=synch_with_pnl_options,
                                               sequence_lengths=sequence_lengths,
                                               context=context)

        # Set RNNComposition's OUTPUT Node of output of RNN Node
        self.composition.output_node.parameters.value._set(output.detach().cpu().numpy(), context)
        self.composition.rnn_mech.parameters.value._set(output.detach().cpu().numpy(), context)

        # MODIFIED TEACHER_TARGET OLD:
        # return {self.composition.rnn_mech: output}
        # MODIFIED TEACHER_TARGET NEW:
        # BREADCRUMB: MAKE THIS A METHOD ON pytorchwrappers (AND DO SAME FOR pytorchwrappers.py)
        output = output.detach().cpu().numpy()
        output_values = convert_to_np_array(output)
        output_values = output_values.swapaxes(0, 1)
        self.all_output_values = output_values
        return output[0]
        # MODIFIED TEACHER_TARGET END

    def _set_synch_with_pnl(self, mech_wrapper, synch_with_pnl_options):
        if (NODE_VALUES in synch_with_pnl_options and synch_with_pnl_options[NODE_VALUES] == LearningScale.RUN):
            mech_wrapper.synch_with_pnl = True
        else:
            mech_wrapper.synch_with_pnl = False

    def copy_weights_to_torch_rnn(self, context=None):
        for projection, proj_wrapper in self.projections_map.items():
            if SYNCH in proj_wrapper._use:
                proj_wrapper._copy_pnl_proj_to_torch_rnn_parameter(context, self.torch_dtype)

    @staticmethod
    def get_parameters_from_torch_rnn(torch_rnn) -> Tuple[tuple, tuple | None]:
        """
        Get parameters from the PyTorch RNN module corresponding to RNNComposition's Projections.

        Returns
        -------
        weights : tuple
            (wts_ih, wts_hh), each transposed to match PNL projection matrix shape.

        biases : tuple | None
            (bias_ih, bias_hh), each formatted as 2d arrays, or None if bias=False.
        """
        wts_ih = torch_rnn.state_dict()[W_IH_NAME].T.detach().cpu().numpy().copy()
        wts_hh = torch_rnn.state_dict()[W_HH_NAME].T.detach().cpu().numpy().copy()
        weights = (wts_ih, wts_hh)

        biases = None
        if torch_rnn.bias:
            b_ih = torch_rnn.state_dict()[B_IH_NAME]
            b_hh = torch_rnn.state_dict()[B_HH_NAME]

            bias_ih = torch.atleast_2d(
                b_ih.permute(*torch.arange(b_ih.ndim - 1, -1, -1))
            ).detach().cpu().numpy().copy()

            bias_hh = torch.atleast_2d(
                b_hh.permute(*torch.arange(b_hh.ndim - 1, -1, -1))
            ).detach().cpu().numpy().copy()

            biases = (bias_ih, bias_hh)

        return weights, biases

    def _torch_params_to_projections(self, param_groups: list) -> dict:
        """Return dict of {torch parameter: Projection} for all wrapped Projections"""
        torch_params_to_projections = {}

        def get_dict_entries(names):
            for projection_name in names:
                torch_param_name = self._pnl_refs_to_torch_param_names[projection_name].param_name
                torch_param_long_name = self._torch_param_short_to_long_names_map[torch_param_name]
                torch_param = next((p[1] for p in self.named_parameters() if p[0] == torch_param_long_name), None)
                assert torch_param is not None, (f"PROGRAM ERROR: torch parameter for {projection_name} "
                                                 f"not found in named_parameters() of {self.name}")
                learning_rate = self.get_learning_rate_for_torch_param(torch_param, param_groups)
                projection = self._pnl_refs_to_torch_param_names[projection_name].projection
                torch_params_to_projections.update({torch_param: projection})

        get_dict_entries([INPUT_TO_HIDDEN, HIDDEN_TO_HIDDEN])
        if self.composition.bias:
            get_dict_entries([BIAS_INPUT_TO_HIDDEN, BIAS_HIDDEN_TO_HIDDEN])
        return torch_params_to_projections

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            proj_wrapper.log_matrix()

    def log_values(self):
        for node_wrapper in [n for n in self.node_wrappers if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()


class PytorchRNNMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for Pytorch RNN Node
    Handling of hidden_state: uses RNNComposition's HIDDEN_NODE.value to cache state of hidden layer:
    - gets input to function for hidden state from RNNComposition's HIDDEN_NODE.value
    - sets RNNComposition's HIDDEN_NODE.value to return value for hidden state
    """

    def __init__(self,
                 mechanism,
                 rnn_composition,
                 component_idx,
                 outer_creator=None,
                 use=None,
                 dtype=None,
                 device=None,
                 context=None):

        super().__init__(mechanism=mechanism,
                         composition=rnn_composition,
                         component_idx=component_idx,
                         outer_creator=outer_creator,
                         use=use,
                         dtype=dtype,
                         device=device,
                         subclass_specifies_function=True,
                         context=context)

        self._assign_RNN_pytorch_function(mechanism, device, context)

        if self.composition.is_nested:
            # Ensure that LossMechanism executes after 'PYTORCH RNN NODE'
            # IMPLEMENTATION NOTE:
            #     this is required because _get_execution_sets() calls Composition.scheduler to generate
            #     the order of execution of Pytorch nodes which uses the order the Composition's graph,
            #     and there LossMechanism is depdendent on the nested RNNComposition, and not 'PYTORCH RNN NODE'
            #     which is only used when executing the PyTorch graph
            outer_comp = context.composition
            for loss_mech in [mech for mech in outer_comp.nodes if isinstance(mech, LossMechanism)]:
                sample_mech = loss_mech.sample.owner
                if sample_mech is mechanism:
                    outer_comp.scheduler.add_condition(loss_mech, conditions.AfterNode(rnn_composition))

        self.synch_with_pnl = False

    def _assign_RNN_pytorch_function(self, mechanism, device, context):
        # Assign PytorchFunctionWrapper of Pytorch RNN module as function of RNN Node
        input_size = self.composition.parameters.input_size.get(context)
        hidden_size = self.composition.parameters.hidden_size.get(context)
        bias = self.composition.parameters.bias.get(context)
        torch_RNN = torch.nn.RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 bias=bias,
                                 nonlinearity='tanh',
                                 batch_first=True).to(dtype=self.torch_dtype)
        torch_RNN.name = f"PytorchFunctionWrapper[RNN NODE]"
        torch_RNN._gen_pytorch_fct = lambda x, y: torch_RNN
        self.hidden_state = torch.zeros(1, 1, hidden_size, dtype=self.torch_dtype).to(device)

        function_wrapper = PytorchFunctionWrapper(torch_RNN, device, context)
        self.function = function_wrapper
        mechanism.function = function_wrapper.function

        # Assign input_port functions of RNN Node to PytorchFunctionWrapper
        self.input_ports = [PytorchFunctionWrapper(input_port.function, device, context)
                            for input_port in mechanism.input_ports]

    def execute(self, variable, optimization_num, synch_with_pnl_options, sequence_lengths,
                context=None) -> torch.Tensor:
        """Execute RNN Node with input variable and return output value.
        Override to set RNN Node's synch_with_pnl option if RNNComposition is a nested Composition
        This is called directly if RNNComposition is in a nested Composition, rather than its forward method.
        Treats RNNComposition as a single node in the PytorchCompositionWrapper's graph, inputs
          received from other node(s) that project to the RNNComposition, and its outputs used by the
          collect_afferents method(s) of the other node(s) that receive Projections from the  RNNComposition.
        """
        # Get hidden state from RNNComposition's HIDDEN_NODE.value
        self.composition.pytorch_representation._set_synch_with_pnl(self, synch_with_pnl_options)

        self.input = variable

        hidden_state = self.composition.hidden_layer_node.output_port.parameters.value.get(context)
        self.hidden_state = torch.tensor(hidden_state).unsqueeze(1)
        # Save starting hidden_state for re-computing current values in _copy_pytorch_node_outputs_to_pnl_values()
        self.previous_hidden_state = self.hidden_state.detach()

        if self.synch_with_pnl:
            self.torch_rnn_internal_state_values = \
                self._calculate_torch_rnn_internal_state_values(self.input[-1], self.hidden_state.detach())

        # Execute torch RNN module with input (variable) and hidden state

        # Flatten the input ports into a 1D tensor because RNN can only take 3D inputs
        input_for_rnn = torch.flatten(self.input, start_dim=2)

        batched_hidden_state = self.hidden_state.expand(-1, input_for_rnn.shape[0], -1)

        if sequence_lengths is not None:
            input_for_rnn = nn.utils.rnn.pack_padded_sequence(input_for_rnn, sequence_lengths, batch_first=True,
                                                              enforce_sorted=False)

        # self.output, output_hidden_state = self.function(input_for_rnn, batched_hidden_state)
        _, output_hidden_state = self.function(input_for_rnn, batched_hidden_state)

        # Restore the input port dimension (flattened above) and the sequence dimension to the output
        self.output = output_hidden_state[-1][:, None, None, :]

        self.hidden_state = output_hidden_state

        # Set RNNComposition's HIDDEN_NODE.value to RNN Node's hidden state
        # Note: this must be done in case the RNNComposition is run after learning,
        self.composition.hidden_layer_node.output_port.parameters.value._set(
            self.hidden_state.detach().cpu().numpy().squeeze(), context)

        return self.output

    def collect_afferents(self, batch_size, port=None, inputs: dict = None) -> torch.Tensor:
        """
        Return afferent projections for input_port(s) of the Mechanism
        If there is only one input_port, return the sum of its afferents (for those in Composition)
        If there are multiple input_ports, return a tensor (or list of tensors if input ports are ragged) of shape:

        (batch, input_port, projection, ...)

        Where the ellipsis represent 1 or more dimensions for the values of the projected afferent.
        """

        if self.afferents == INPUT:
            # RNNComposition is nested in an outer Composition, and RNN is INPUT Node of that Composition
            #  so get input specified for RNNComposition.input_node from the inputs dict provided in the learn() method
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
                    proj_wrapper._curr_sender_value = curr_val[:, :, proj_wrapper._value_idx, ...]
                else:
                    proj_wrapper._curr_sender_value = torch.stack(
                        [torch.stack([s[proj_wrapper._value_idx] for s in b]) for b in curr_val])
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
        ip_res = torch.stack(ip_res, dim=2)
        res.append(ip_res)

        try:
            # Now stack the results for all input ports on the second dimension again, this keeps batch
            # first again. We should now have a 5D tensor; (batch, sequence, input_port, projection, values)
            res = torch.stack(res, dim=2)
        except (RuntimeError, TypeError):
            # res has a ragged structure, a list where each element corresponds to and input port. Each tensor
            # for an input port is 4D (batch, seq, projection, values). We need to reshape this so that list of lists
            # of lists where the dimensions are (batch, seq, input port, projection, values)
            batch_size = res[0].shape[0]
            seq_size = res[0].shape[1]
            res = [[[inp[b, s, ...] for inp in res] for s in range(seq_size)] for b in range(batch_size)]

        return res

    def _calculate_torch_rnn_internal_state_values(self, input, hidden_state) -> dict:
        """
        Manually calculate and store internal state values for torch RNN prior to backward pass.

        Returns
        -------
        dict
            Contains only the final hidden state.
        """
        torch_rnn = self.function.function

        w_ih = torch_rnn.state_dict()[W_IH_NAME].T.detach().to(self.torch_dtype)
        w_hh = torch_rnn.state_dict()[W_HH_NAME].T.detach().to(self.torch_dtype)

        if torch_rnn.bias:
            b_ih = torch.atleast_2d(torch_rnn.state_dict()[B_IH_NAME]).detach().to(self.torch_dtype)
            b_hh = torch.atleast_2d(torch_rnn.state_dict()[B_HH_NAME]).detach().to(self.torch_dtype)
        else:
            b_ih = 0.0
            b_hh = 0.0

        h = hidden_state
        for x in input:
            h = torch.tanh(torch.matmul(x, w_ih) + b_ih + torch.matmul(h, w_hh) + b_hh)

        from psyneulink.library.compositions.rnncomposition.rnncomposition import HIDDEN_LAYER
        return {HIDDEN_LAYER: h}

    def set_pnl_variable_and_values(self,
                                    set_variable: bool = False,
                                    set_value: bool = True,
                                    # FIX: 3/15/25 - ADD SUPPORT FOR THESE
                                    # set_output_values:bool=None,
                                    # execute_mech:bool=True,
                                    context=None):
        if set_variable:
            assert False, \
                f"PROGRAM ERROR: copying variables to RNNComposition from pytorch execution is not currently supported."

        if set_value:
            from psyneulink.library.compositions.rnncomposition.rnncomposition import HIDDEN_LAYER

            h_t = self.torch_rnn_internal_state_values[HIDDEN_LAYER]
            try:
                output = self.output.squeeze(2).detach().numpy()
                np.testing.assert_allclose(
                    h_t.detach().cpu().numpy()[0],
                    output[-1],
                    atol=1e-8
                )
            except ValueError:
                assert False, (
                    f"PROGRAM ERROR: Problem with calculation of internal states of "
                    f"{self.composition.name} RNN Node."
                )

            pnl_comp = self.composition
            pnl_comp.hidden_layer_node.output_port.parameters.value._set(
                h_t.detach().cpu().numpy().squeeze(), context
            )
            pnl_comp.output_node.output_port.parameters.value._set(
                h_t.detach().cpu().numpy().squeeze(), context
            )

    def log_value(self):
        # FIX: LOG HIDDEN STATE OF COMPOSITION MECHANISM
        if self.mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self.mechanism.output_port.parameters.value._set(detached_value, self._context)
            self.mechanism.parameters.value._set(detached_value, self._context)



class PytorchRNNProjectionWrapper(PytorchProjectionWrapper):
    """Wrapper for a Projection of the RNNComposition

    One is created for each Projection of the RNNComposition that is learnable.
    Sets of three of these correspond to the Parameters of the torch RNN module:

    PyTorch RNN parameter:  RNNComposition Projections:
         weight_ih_l0       wts_ir, wts_iu, wts_in
         weight_hh_l0       wts_hr, wts_hu, wts_hn
         bias_ih_l0         bias_ir, bias_iu, bias_in
         bias_hh_l0         bias_hr, bias_hu, bias_hn

    Attributes
    ----------
    projection:  MappingProjection
        the `Projection` of the RNNComposition being wrapped

    composition : AutodiffComposition
        the `AutodiffComposition` to which the `Projection` being wrapped belongs
        (and for which the PytorchCompositionWrapper -- to which the PytorchProjectionWrapper
        belongs -- is the `pytorch_representation <AutodiffComposition.pytorch_representation>`).

    torch_parameter: Pytorch parameter
        the torch.nn.Parameter corresponding to the matrix of the Projection;

    matrix_indices: slice
        a slice specifying the part of the Pytorch parameter corresponding to the RNNCOmposition Projection's matrix.
    """

    def __init__(self,
                 projection: MappingProjection,
                 torch_parameter: Tuple,
                 use: Union[list, Literal[LEARNING, SYNCH, SHOW_PYTORCH]],
                 composition: AutodiffComposition,
                 device: str):
        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        # RNNComposition Projection being wrapped:
        self.projection = projection  # PNL Projection being wrapped
        self._pnl_proj = projection
        # Assign parameter and tensor indices of Pytorch RNN module parameter corresponding to the Projection's matrix:
        self.torch_parameter, self.matrix_indices = torch_parameter
        # Projections for RNNComposition are not included in autodiff; matrices are set directly in Pytorch RNN module:
        self.projection.exclude_in_autodiff = True
        self._use = convert_to_list(use)
        self.composition = composition
        self.device = device

    def _copy_pnl_proj_to_torch_rnn_parameter(self, context, dtype):
        """Set relevant part of tensor for parameter of Pytorch RNN module from RNNComposition's Projections."""
        matrix = self.projection.parameters.matrix._get(context).T
        torch_tensor = self.torch_parameter[self.matrix_indices]
        self.composition.copy_projection_matrix_to_torch_param(projection=self.projection,
                                                               torch_param=torch_tensor,
                                                               validate=False,
                                                               context=context)

    def _copy_torch_params_to_pnl_proj(self, context):
        """Override to deal with indexed tensor of Pytorch RNN module Parameter"""
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


class DummyProjection(Projection):
    """Dummy Projection for access to the learning rate for the IH and WH torch parameter
    The IH and HH (and corresponding biases) torch parameters correspond to multiple PNL Projections,
    so DummyProjections are used to provide access to their learning_rates
    """
    name = ""

    class Parameters(Projection.Parameters):
        learning_rate = Parameter(None, stateful=True)

    @check_user_specified
    def __init__(self, name):
        self.name = name
        self._initialize_parameters(learning_rate=None, context=Context(execution_id=None))
        self.parameters.learning_rate.set(None, None)
        self.learnable = True

    def __getattr__(self, name):
        obj_name = f"{self.name} "
        if name not in {'learning_rate', 'name'}:
            raise AttributeError(f"This object is used to convey the learning rate for the torch parameters "
                                 f"corresponding to the set of {obj_name}Projections of a RNNComposition, "
                                 f"that cannot be set directly.  It has only 'name', 'learnable', and"
                                 f"'learning_rate' as attributes, and no others.")
