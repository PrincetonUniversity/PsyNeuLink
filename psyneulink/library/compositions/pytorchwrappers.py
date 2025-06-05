# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrappers for Composition, Mechanism, Projection, and Functions for use in AutodiffComposition"""
from psyneulink._typing import Optional, Literal, Union

import graph_scheduler
import numpy as np

# import torch
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None
else:
    import torch.nn as nn

from enum import Enum, auto

from psyneulink.core.components.functions.stateful import StatefulFunction
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.ports.port import Port
from psyneulink.core.components.projections.projection import Projection, DuplicateProjectionError
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition, CompositionInterfaceMechanism, NodeRole
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.library.compositions.compiledoptimizer import AdamOptimizer, SGDOptimizer
from psyneulink.library.compositions.compiledloss import MSELoss, CROSS_ENTROPYLoss
from psyneulink.core.globals.keywords import (AFTER, ALL, BEFORE, DEFAULT_VARIABLE, EPOCH, INPUTS,
                                              LEARNING, LEARNING_SCALE_LITERALS, Loss, MATRIX_WEIGHTS,
                                              NODE, NODE_VALUES, NODE_VARIABLES, OUTPUTS, RESULTS, RUN,
                                              SHOW_PYTORCH, SYNCH, TARGET_MECHANISM, )
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import convert_to_list, convert_to_np_array, get_deepcopy_with_shared
from psyneulink.core.globals.log import LogCondition
from psyneulink.core import llvm as pnlvm

__all__ = ['PytorchCompositionWrapper', 'PytorchMechanismWrapper', 'PytorchProjectionWrapper',
           'ENTER_NESTED', 'EXIT_NESTED', 'SUBCLASS_WRAPPERS']

SUBCLASS_WRAPPERS = 'subclass_wrappers'
ENTER_NESTED = 0
EXIT_NESTED = 1

class DataTypeEnum(Enum):

    TRAINED_OUTPUTS = 0
    TARGETS = auto()
    LOSSES = auto()


def _get_pytorch_function(obj, device, context):
    pytorch_fct = getattr(obj, '_gen_pytorch_fct', None)
    if pytorch_fct is None:
        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
        raise AutodiffCompositionError(
            f"Function {obj} is not currently supported by AutodiffComposition"
        )
    else:
        return pytorch_fct(device, context)


class PytorchCompositionWrapper(torch.nn.Module):
# NEEDED FOR torch MPS SUPPORT
# class PytorchCompositionWrapper(torch.jit.ScriptModule):
# END
    """Wrapper for a Composition as a Pytorch Module.

    Wraps an `AutodiffComposition` as a `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_, with each `Mechanism <Mechanism>` in
    the AutodiffComposition wrapped as a `PytorchMechanismWrapper`, each `Projection <Projection>` wrapped as
    a `PytorchProjectionWrapper`, and any nested Compositions wrapped as `PytorchCompositionWrapper`\\s. Each
    PytorchMechanismWrapper implements a Pytorch version of the `function(s) <Mechanism_Base.function>` of the wrapped
    `Mechanism`, which are executed in the PyTorchCompositionWrapper's `forward <PyTorchCompositionWrapper.forward>`
    method in the order specified by the AutodiffComposition's `scheduler <Composition.scheduler>`.  The `matrix
    <MappingProjection.matrix>` Parameters of each wrapped `Projection` are assigned as parameters of the
    `PytorchMechanismWrapper` Pytorch module and used, together with a Pytorch `matmul
    <https://pytorch.org/docs/main/generated/torch.matmul.html>`_ operation, to generate the input to each
    PyTorch function as specified by the `PytorchProjectionWrapper`\\'s `graph <Composition.graph>`.  The graph
    can be visualized using the AutodiffComposition's `show_graph <ShowGraph.show_graph>` method and setting its
    *show_pytorch* argument to True (see `PytorchShowGraph` for additional information).

    Two main responsibilities:

    1) Set up functions and parameters of PyTorch module required for its forward computation:
       - Handle nested compositions (flattened in infer_backpropagation_learning_pathways):
       - Deal with Projections into and/or out of a nested Composition as shown in figure below:
            (note: Projections in outer Composition to/from a nested Composition's CIMs are learnable,
                   and ones within a nested Composition from/to its CIMs are not)

         [      OUTER     ][                            NESTED                               ][     OUTER      ]
                \\learnable//   \\not learnable//                     \\not learnable//    \\learnable//
         ---> [Node] ----> [input_CIM] ~~~> [INPUT Node] ----> [OUTPUT Node] ~~~> [output_CIM] ----> [Node] --->
               sndr            rcvr          nested_rcvr         nested_sndr         sndr             rcvr
                ^--projection-->^                                                     ^---projection-->^
                ^----PytorchProjectionWrapper---->^                  ^----PytorchProjectionWrapper---->^
                          ENTER_NESTED                                            EXIT_NESTED

       .. _Mechanism_and_Projection_Uses:

       - The uses of Mechanisms and Projections in the pytorch_representation of an AutodiffComposition are
         determined, respecticely, by its PytorchMechanismWrapper's `use <PytorchMechanismWrapper.use>` and
         PytorchProjectionWrapper's `use <PytorchProjectionWrapper.use>`, as follows:

         * Mechanisms:
           - used in Python execution but not Pytorch execution: *SYNCH*
           - used in PyTorch execution but not Python execution: *LEARNING*, *SHOW_PYTORCH*
           - used for both Python and Pytorch execution: *LEARNING*, *SYNCH*, *SHOW_PYTORCH*

         * Projections:
           - among (non-CIM) Mechanisms within the same Composition: same as Mechanisms (see above)
           - to an input_CIM of a nested Composition:  *LEARNING*, *SYNCH*, *SHOW_PYTORCH*
           - from an input_CIM: None
           - to an output_CIM: None
           - from an output_CIM:  *LEARNING*, *SYNCH*
           - directly between (to/from) a nested and outer Composition: *SHOW_PYTORCH*

    2) Handle coordination of passing data and outcomes back to PsyNeuLink objects, handled by two main methods:

       - synch_with_psyneulink()
            Copies matrix weights, node variables, node values, and/or autoutdiff results
            at user-specified intervals (LearningScale:  OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH, RUN);
            these are specified by the user in the following arguments to run() or learn():
                synch_projection_matrices_with_torch=RUN,
                synch_node_variables_with_torch=None,
                synch_node_values_with_torch=RUN,
                synch_results_with_torch=RUN,
            and consolidated in the synch_with_pnl_options dict used by synch_with_psyneulink

       - retain_for_psyneulink()
            Retains learning-specific data used and outcomes generated during execution of PyTorch model
            (TRAINED_OUTPUT_VALUES, corresponding TARGETS and LOSSES), that are copied to PsyNeuLink
            at the end of a call to learn(); these are specified by the user in the following arguments
            to learn():
                retain_torch_trained_outputs=MINIBATCH,
                retain_torch_targets=MINIBATCH,
                retain_torch_losses=MINIBATCH,
            and consolidated in the retain_in_pnl_options dict used by retain_for_psyneulink

        - Note: RESULTS is handled in an idiosyncratic way: it is specified along with the synchronization
                parameters, since it is a value ordinarily generated in the execution of a Composition;
                however it's helper parallels the retain_for_psyneulink helper methods, and it is called
                from _update_results if TRIAL is specified, in order to integrate with the standard execution
                of a Composition.

    Arguments
    ---------


    Attributes
    ----------

    composition : AutodiffComposition
        The `AutodiffComposition` for which the PytorchCompositionWrapper is the `pytorch_representation
        <AutodiffComposition.pytorch_representation>`.

    node_wrappers : List[PytorchMechanismWrapper]
        list of nodes in the PytorchCompositionWrapper corresponding to the PyTorch functions that comprise the
        forward method of the Pytorch module implemented by the PytorchCompositionWrapper. Generally these are
        `Mechanisms <Mechanism>` wrapped in a `PytorchMechanismWrapper`, however, if the `AutodiffComposition` Node
        being wrapped is a nested Composition, then the wrapped node is itself a `PytorchCompositionWrapper` object.
        When the PyTorch model is executed, all of these are "flattened" into a single PyTorch module, corresponding
        to the outermost AutodiffComposition being wrapped, which can be visualized using that AutodiffComposition's
        `show_graph <ShowGraph.show_graph>` method and setting its *show_pytorch* argument to True (see
        `PytorchShowGraph` for additional information).

    nodes_map : Dict[Node: PytorchMechanismWrapper or PytorchCompositionWrapper]
        maps PsyNeuLink `Nodes <Composition_Nodes>` to PytorchCompositionWrapper nodes.

    projection_wrappers = List[PytorchProjectionWrapper]
        list of PytorchCompositionWrappers in the PytorchCompositionWrapper, each of which wraps a `Projection`
        in the AutodiffComposition being wrapped.

    projections_map : Dict[Projection: PytorchProjectionWrapper]
        maps `Projections <Projection>` in the AutodiffComposition being wrapped to `PytorchProjectionWrappers` in
        the PytorchCompositionWrapper.

    _nodes_to_execute_after_gradient_calc :  Dict[node : torch.Tensor]
        contains nodes specified as `exclude_from_gradient_calc` as keys, and their current variable as values

    optimizer : torch
        assigned by AutodffComposition after the wrapper is created, which passes the parameters to the optimizer

    device : torch.device
        device used to process torch Tensors in PyTorch functions

    params : nn.ParameterList()
        list of PyTorch parameters (connection weight matrices) in the PyTorch model.

    minibatch_loss : torch.Tensor
        accumulated loss over all trials (stimuli) within a batch.

    minibatch_loss_count : int
        count of losses (trials) within batch, used to calculate average loss per batch.

    retained_results : List[ndarray]
        list of the `output_values <Composition.output_values>` of the AutodiffComposition for ever trial executed
        in a call to `run <AutoDiffComposition.run>` or `learn <AutoDiffComposition.learn>`.

    retained_trained_outputs : List[ndarray]
        values of the trained `OUTPUT <NodeRole.OUTPUT>` Node (i.e., ones associated with `TARGET <NodeRole.TARGET`
        Node) for each trial executed in a call to `learn <AutoDiffComposition.learn>`.

    retained_targets : List[ndarray]
        values of the `TARGET <NodeRole.TARGET` Nodes for each trial executed in a call to `learn
        <AutoDiffComposition.learn>`.

    retained_losses : List[ndarray]
        losses per batch, epoch or run accumulated over a call to learn()
    """

    torch_dtype = torch.float64

    def __init__(self,
                 composition,
                 device,
                 outer_creator=None,
                 dtype=None,
                 subclass_components=None,
                 context=None,
                 base_context=Context(execution_id=None),
                 ):

        super(PytorchCompositionWrapper, self).__init__()

        if subclass_components is None:
            self._early_init(composition, device)
            # Instantiate standard PytorchWrappers for Mechanisms and Projections, and execution_sets used in forward()
            _node_wrapper_pairs = self._instantiate_pytorch_mechanism_wrappers(composition, device, context)
            self._construct_node_wrapper_maps(_node_wrapper_pairs)
            _projection_wrapper_pairs = self._instantiate_pytorch_projection_wrappers(composition, device, context, base_context)
            self._construct_projection_wrapper_maps(_projection_wrapper_pairs)
            self.execution_sets, execution_context = self._get_execution_sets(composition, context)

        else:
            # Construct node_wrappers, projection_wrappers, and execution_sets from subclass components passed in
            _node_wrapper_pairs, _projection_wrapper_pairs, _execution_sets, execution_context = subclass_components
            self._validate_subclass_components(_node_wrapper_pairs, _projection_wrapper_pairs, _execution_sets)
            self._construct_node_wrapper_maps(_node_wrapper_pairs)
            self._construct_projection_wrapper_maps(_projection_wrapper_pairs)
            self.execution_sets = _execution_sets

        # Assign INPUT Nodes for outermost Composition (including any that are nested within it at any level)
        # Note: Pytorch representation is "flattened" (i.e., any nested Compositions are replaced by their Nodes)
            #   so if any nested Compositions are INPUT Nodes of the outermost Composition,
            #   *their* INPUT Nodes are assigned as INPUT Nodes of the outermost Composition
        if not composition.is_nested:
            def _assign_input_nodes(nodes):
                for pytorch_node in nodes:
                    if isinstance(pytorch_node, PytorchMechanismWrapper):
                        pytorch_node._is_input = pytorch_node.mechanism in composition._get_input_receivers(type=NODE)
                    else:
                        _assign_input_nodes(pytorch_node.node_wrappers)
            _assign_input_nodes(self.node_wrappers)

        # Flatten maps
        for node_wrapper in self.node_wrappers:
            if isinstance(node_wrapper, PytorchCompositionWrapper):
                # For copying weights back to PNL in AutodiffComposition.do_gradient_optimization
                self.projections_map.update(node_wrapper.projections_map)
                for k, v in node_wrapper.nodes_map.items():
                    self._add_node_to_nodes_map(k, v)
        # Purge nodes_map of entries for nested Compositions (their nodes are now in self.nodes_map)
        nodes_to_remove = [k for k, v in self.nodes_map.items() if isinstance(v, PytorchCompositionWrapper)]
        for node in nodes_to_remove:
            self._remove_node_from_nodes_map(node)

        self.output_nodes = self.composition.get_nested_output_nodes_at_all_levels()

        self.composition.parameters.pytorch_representation._set(self, context, skip_history=True, skip_log=True)

        # Get projections from flattened set, so that they are all in the outer Composition
        #   and visible by _regenerate_torch_parameter_list;
        #   needed for call to backward() in AutodiffComposition.do_gradient_optimization
        self.projection_wrappers = list(self.projections_map.values())

        composition.scheduler._delete_counts(execution_context.execution_id)

        self._regenerate_torch_parameter_list()
        assert 'DEBUGGING BREAKPOINT'

    def _early_init(self, composition, device):
        """Early initialization of PytorchCompositionWrapper"""
                # Assign attributes
        self.name = f"PytorchCompositionWrapper[{composition.name}]"
        self.device = device
        self.optimizer = None # This gets assigned by self.composition after the wrapper is created,
                                # as the latter is needed to pass the parameters to the optimizer
        self._optimizer_param_groups = []

        self.composition = composition
        self.node_wrappers = []  # can be PytorchMechanismWrapper or PytorchCompositionWrapper
        self._nodes_to_execute_after_gradient_calc = {} # Nodes requiring execution after Pytorch forward/backward pass
        self._batch_size = 1 # Store the currently used batch size

        self.projection_wrappers = [] # PytorchProjectionWrappers
        self.projections_map = {}  # maps Projections -> PytorchProjectionWrappers
        self._pnl_refs_to_torch_params_map = {} # API for PNL refs to PyTorch params (used by _parse_optimizer_params)

        self.minibatch_loss = torch.zeros(1, device=self.device).double() # Accumulated losses within a batch
        self.minibatch_loss_count = 0  # Count of losses within batch

        # Data retained by the wrapper during execution and copied to pnl as specified by retain_for_psyneulink
        self.retained_results = []          # Values of all output NODES
        self.retained_trained_outputs = []  # Values of trained output NODES (i.e. associated with TARGETS)
        self.retained_targets = []  #       # Values of targets for all trials
        self.retained_losses = []           # Losses per trial or batch accumulated over a run

        # The following is a list of methods called in retain_for_psyneulink, indexed by keywords using DataTypeEnum
        # (this is constructed as a form of hash table for efficiency since that method can be called alot;
        #  it is constructed here to avoid doing so in the retain_for_psyneulink method itself)
        self.retain_method = [None] * len(DataTypeEnum)
        self.retain_method[DataTypeEnum.TRAINED_OUTPUTS.value] = self.retain_trained_outputs
        self.retain_method[DataTypeEnum.TARGETS.value] = self.retain_targets
        self.retain_method[DataTypeEnum.LOSSES.value] = self.retain_losses

    def _validate_subclass_components(self, _node_wrapper_pairs, _projection_wrapper_pairs, execution_sets):
        """Sublcass instantiated nodes_map, projections_map and execution_sets, so validate these."""
        assert all(isinstance(item[0], (Mechanism, Composition)) for item in _node_wrapper_pairs), \
            (f"PROGRAM ERROR: Constructor for {self} passed non-Mechanism or Composition object(s) "
             f"as node(s) from subclass.")
        assert all(isinstance(item[1], (PytorchMechanismWrapper, PytorchCompositionWrapper))
                   for item in _node_wrapper_pairs), \
            (f"PROGRAM ERROR: Constructor for {self} passed non-PytorchMechanismWrapper or PytorchCompositionWrapper "
             f"object(s) as node wrapper(s) from subclass.")
        assert all(isinstance(item[0], Projection) for item in _projection_wrapper_pairs), \
            (f"PROGRAM ERROR: Constructor for {self} passed non-Projection object(s) as Projection(s) from subclass.")
        assert all(isinstance(item[1], PytorchProjectionWrapper) for item in _projection_wrapper_pairs), \
            (f"PROGRAM ERROR: Constructor for {self} passed non-PytorchProjectionWrapper object(s) as "
             f"projection wrapper(s) from subclass.")
        for exec_set in execution_sets:
            assert isinstance(exec_set, set), \
                f"PROGRAM ERROR: {self}.execution_sets contains non-ExecutionSet object(s)."
            for item in exec_set:
                assert isinstance(item, (PytorchMechanismWrapper, PytorchCompositionWrapper)), \
                    (f"PROGRAM ERROR: {self}.execution_sets contains a set with non-PytorchMechanismWrapper "
                     f"or PytorchCompositionWrapper object).")

    def _construct_node_wrapper_maps(self, _node_wrapper_pairs):
        self.nodes_map = {} # maps Node(Mech | nested Comp) -> PytorchMechanismWrapper | PytorchCompositionWrapper
        self.node_wrappers = []
        self._modules_dict = torch.nn.ModuleDict()
        for node, pytorch_node_wrapper in _node_wrapper_pairs:
            self._add_node_to_nodes_map(node, pytorch_node_wrapper)

    def _construct_projection_wrapper_maps(self, _projection_wrapper_pairs):
        self.projections_map = {k:v for k,v in _projection_wrapper_pairs}
        self.projection_wrappers = list(self.projections_map.values())

    def _add_node_to_nodes_map(self, node, node_wrapper):
        """Keep nodes_map, node_wrappers and modules_dict in synch"""
        self.nodes_map[node] = node_wrapper
        if node not in self.node_wrappers:
            self.node_wrappers.append(node_wrapper)
        self._modules_dict[node.name] = node_wrapper
        self.state_dict()

    def _remove_node_from_nodes_map(self, node):
        """Keep nodes_map, node_wrappers and modules_dict in synch"""
        self.nodes_map.pop(node)
        if node in self.node_wrappers:
            self.node_wrappers.remove(node)
        self._modules_dict.pop(node.name)

    def _instantiate_pytorch_mechanism_wrappers(self, composition, device, context)->list:
        """Instantiate PytorchMechanismWrappers for Mechanisms in the Composition being wrapped"""
        from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition

        # Remove all learning-specific nodes
        nodes = list(set(composition.nodes) - set(composition.get_nodes_by_role(NodeRole.LEARNING)))

        # Remove nested nodes from nodes list (put there in flattening by infer_backpropagation_learning_pathways)
        #   so that they don't interfere with construction of execution_sets by scheduler
        # Will re-flatten execution sets below
        nodes = [n for n in nodes
                 # Leave nested Compositions
                 if (isinstance(n, AutodiffComposition)
                     # Needed since composition.nodes is flattened in infer_backpropagation_learning_pathways
                     or n not in [n[0] for n in self.composition._get_nested_nodes()])]

        _node_wrapper_pairs = []
        # Sort to be sure nested Compositions are processed last, as they need outer nodes that project in/out of them
        for node in sorted(nodes, key=lambda x: isinstance(x, AutodiffComposition)):
            # Wrap nested Composition
            if isinstance(node, AutodiffComposition):
                pytorch_node_wrapper = node.pytorch_composition_wrapper_type(composition=node,
                                                                             device=device,
                                                                             outer_creator=self,
                                                                             context=context)
            # Wrap Mechanism
            else:
                pytorch_node_wrapper = \
                    self.composition.pytorch_mechanism_wrapper_type(
                        mechanism=node,
                        composition=composition,
                        component_idx=self.composition._get_node_index(node),
                        use=[LEARNING, SYNCH, SHOW_PYTORCH],
                        dtype=self.torch_dtype,
                        device=device,
                        context=context)
                # pytorch_node._is_bias = all(input_port.default_input == DEFAULT_VARIABLE
                #                             for input_port in node.input_ports)
                pytorch_node_wrapper._is_bias = node in self.composition.get_nodes_by_role(NodeRole.BIAS)
            _node_wrapper_pairs.append((node, pytorch_node_wrapper))

        return _node_wrapper_pairs

    def _instantiate_pytorch_projection_wrappers(self, composition, device, context, base_context=Context(execution_id=None)) -> list:
        """Instantiate PytorchProjectionWrappers for Projections in the Composition being wrapped
        Assign Projections for outermost Composition (including any that are nested within it at any level)
        Note: Pytorch representation is "flattened" (i.e., any nested Compositions are replaced by their Nodes)
        so if any nested Compositions have Projections to/from them, they are assigned to the outermost Composition
        See figure in module docstring for explanation of how Projections to/from nested Compositions are handled.
        """

        proj_wrappers_pairs = []
        # Instantiate PyTorch ProjectionWrappers (ignoring any from/to CIMs in the same composition)
        for projection in composition._inner_projections:
            sndr_mech = projection.sender.owner
            rcvr_mech = projection.receiver.owner

            # Rule out that Composition has parameter_CIM,
            #    since autodiff does not (yet) support those and they are not (yet) handled by flattening below
            assert not hasattr(self, '_parameter_CIM'),\
                (f"PROGRAM ERROR: {self} has a parameter_CIM which is not should not currently be the case "
                 f"and is not handled by flatterning in {self.__class__.__name__}.")

            # Ignore input_CIM and output_CIM within the same Composition (they are not learnable)
            if sndr_mech is composition.input_CIM or rcvr_mech is composition.output_CIM:
                continue

            # Handle projection to or from a nested Composition
            elif (isinstance(sndr_mech, CompositionInterfaceMechanism) or
                  isinstance(rcvr_mech, CompositionInterfaceMechanism)):
                pnl_proj, proj_sndr, proj_rcvr, use = self._handle_nested_comp(projection, context, base_context)
                # # use = [LEARNING, SYNCH, SHOW_PYTORCH]
                # use = [LEARNING, SYNCH]

            # Projection within composition
            elif all(sndr_and_recvr in self.nodes_map for sndr_and_recvr in {sndr_mech, rcvr_mech}):
                proj_sndr = self.nodes_map[sndr_mech]
                proj_rcvr = self.nodes_map[rcvr_mech]
                pnl_proj = projection
                use = [LEARNING, SYNCH, SHOW_PYTORCH]

            else:
                continue

            component_idx = list(self.composition._inner_projections).index(projection)
            sender_port_idx = projection.sender.owner.output_ports.index(projection.sender)
            pytorch_proj_wrapper = PytorchProjectionWrapper(projection=projection,
                                                            pnl_proj=pnl_proj,
                                                            component_idx=component_idx,
                                                            sender_port_idx=sender_port_idx,
                                                            use=use,
                                                            device=device,
                                                            sender_wrapper=proj_sndr,
                                                            receiver_wrapper=proj_rcvr,
                                                            composition=composition,
                                                            context=context)
            proj_sndr.add_efferent(pytorch_proj_wrapper)
            proj_rcvr.add_afferent(pytorch_proj_wrapper)

            proj_wrappers_pairs.append((projection, pytorch_proj_wrapper))

        return proj_wrappers_pairs

    def _handle_nested_comp(
        self,
        projection: MappingProjection,
        context: Context,
        base_context: Context = Context(execution_id=None),
    ) -> tuple:
        """Flatten nested Composition and assign Projections to/from it to outermost Composition
        This method is called when a Projection is to/from a CIM in a nested Composition that is not in the current
        Composition, and is needed for learning.
        It may be overridden by a subclass (grucomposition) to handle flattening differently.
        See figure in module docstring for explanation of how Projections to/from nested Compositions are handled.
        """
        sndr_mech = projection.sender.owner
        rcvr_mech = projection.receiver.owner

        # ENTER_NESTED:
        # input_cim of nested Composition:
        #    - projection is to input_CIM that is not in current Composition so must be to a nested one;
        #    - needed for learning, so create map for Projection
        if (isinstance(rcvr_mech, CompositionInterfaceMechanism)
                and rcvr_mech.composition is not self
                and rcvr_mech is rcvr_mech.composition.input_CIM):
            # Replace rcvr_mech (input_CIM) with the node in the nested Composition that receives the projection
            nested_rcvr_port, nested_rcvr_mech, _ = \
                rcvr_mech._get_destination_info_from_input_CIM(projection.receiver)
            nested_pytorch_comp_wrapper = self.nodes_map[rcvr_mech.composition]
            proj, proj_sndr_wrapper, proj_rcvr_wrapper, use = (
                nested_pytorch_comp_wrapper._flatten_for_pytorch(projection,
                                                                 sndr_mech, rcvr_mech,
                                                                 nested_rcvr_port,
                                                                 nested_rcvr_mech,
                                                                 self.composition,
                                                                 self,
                                                                 ENTER_NESTED,
                                                                 context,
                                                                 base_context,
                                                                 )
            )
            if proj_sndr_wrapper is None:
                proj_sndr_wrapper = self.nodes_map[sndr_mech]

        # EXIT_NESTED
        # output_cim of nested Composition:
        #    - projection is from output_CIM that is not in current Composition so must be from a nested one;
        #    - needed for learning, so create map for Projection
        elif (isinstance(sndr_mech, CompositionInterfaceMechanism)
              and sndr_mech.composition is not self
              and sndr_mech is sndr_mech.composition.output_CIM):
            # Replace sndr_mech (output_CIM) with the node in the nested Composition that sends the projection
            nested_sndr_port, nested_sndr_mech, _ = \
                sndr_mech._get_source_info_from_output_CIM(projection.sender)
            nested_pytorch_comp_wrapper = self.nodes_map[sndr_mech.composition]
            proj, proj_sndr_wrapper, proj_rcvr_wrapper, use = (
                nested_pytorch_comp_wrapper._flatten_for_pytorch(projection,
                                                                 sndr_mech, rcvr_mech,
                                                                 nested_sndr_port,
                                                                 nested_sndr_mech,
                                                                 self.composition,
                                                                 self,
                                                                 EXIT_NESTED,
                                                                 context))
            if proj_rcvr_wrapper is None:
                proj_rcvr_wrapper = self.nodes_map[rcvr_mech]
        return proj, proj_sndr_wrapper, proj_rcvr_wrapper, use

    def _flatten_for_pytorch(self,
                             projection,
                             sndr_mech,
                             rcvr_mech,
                             nested_port,
                             nested_mech,
                             outer_comp,
                             outer_comp_pytorch_rep,
                             access,
                             context,
                             base_context=Context(execution_id=None),
                             ) -> tuple:
        proj_sndr_wrapper = None
        proj_rcvr_wrapper = None
        use = [LEARNING, SYNCH]

        if access == ENTER_NESTED:
            proj_rcvr_wrapper = self.nodes_map[nested_mech]
            # Assign Projection from input_CIM to nested_rcvr_port as pnl_proj (for use in forward())
            nested_comp = projection.receiver.owner.composition
            incoming_projections = [proj for proj in nested_comp.input_CIM.port_map[nested_port][1].efferents
                                    if proj in nested_comp.projections]
            assert len(incoming_projections) == 1, \
                (f"PROGRAM ERROR: There is more than one Projection registered in '{nested_comp.name}' "
                 f"from its input_CIM to '{nested_port.owner.name}'.")
            nested_port_afferents = [proj for proj in nested_port.path_afferents if proj in nested_comp.projections]
            pnl_proj = incoming_projections[0]
            if pnl_proj != nested_port.path_afferents[0]:
                from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
                raise AutodiffCompositionError(
                    f"First afferent Projection to '{nested_port.owner.name}' (which should be from "
                    f"'{nested_port.path_afferents[0].sender.owner.name}') is not the same as its "
                    f"Projection from the input_CIM of '{projection.receiver.owner.composition.name}'. "
                    f"One for this reason may be that these Components belong to different Compositions.")

            # Construct direct Projection from sender in outer Composition to receiver in nested Composition,
            #   and a PytorchCompositionWrapper for it that is assigned use=SHOW_PYTORCH,
            #   but don't add to either Composition as it is just used for show_graph(show_pytorch=True)
            destination_rcvr_port = rcvr_mech._get_destination_info_from_input_CIM(projection.receiver)[0]
            destination_rcvr_mech = rcvr_mech._get_destination_info_from_input_CIM(projection.receiver)[1]
            try:
                direct_proj = MappingProjection(name=f"Direct Projection from {projection.sender.owner.name} "
                                                     f"to {destination_rcvr_mech.name}",
                                                sender=projection.sender,
                                                receiver=destination_rcvr_port,
                                                learnable=projection.learnable)
            except DuplicateProjectionError:
                direct_proj = [proj for proj in projection.sender.efferents
                               if proj.receiver is destination_rcvr_port][0]
            else:
                direct_proj._initialize_from_context(context, base_context)

            if direct_proj not in self.projection_wrappers:
                proj_wrapper = PytorchProjectionWrapper(projection=direct_proj,
                                                        pnl_proj=pnl_proj,
                                                        component_idx=None,    # These are not needed since the wrapper
                                                        sender_port_idx=None,  # is only being used for SHOW_PYTORCH
                                                        use=[SHOW_PYTORCH],
                                                        device=self.device,
                                                        sender_wrapper=proj_sndr_wrapper,
                                                        receiver_wrapper=proj_rcvr_wrapper,
                                                        composition=self.composition,
                                                        context=context)
                outer_comp_pytorch_rep.projection_wrappers.append(proj_wrapper)
                outer_comp_pytorch_rep.projections_map[direct_proj] = proj_wrapper
                outer_comp_pytorch_rep.composition._pytorch_projections.append(direct_proj)

        elif access == EXIT_NESTED:
            proj_sndr_wrapper = self.nodes_map[nested_mech]

            # Assign Projection from nested_sndr_port to output_CIM as pnl_proj
            assert nested_port.efferents[0] == projection.sender.owner.port_map[nested_port][0].path_afferents[0], \
                (f"PROGRAM ERROR: First efferent Projection from '{nested_port.owner.name}' "
                 f"(to '{nested_port.efferents[0].receiver.owner.name}') is not the same as its "
                 f"Projection to '{projection.sender.owner.composition.name}.output_CIM'."
                 f"One for this reason may be that these Components belong to different Compositions.")
            pnl_proj = projection

            # Construct direct Projection from sender in nested Composition to receiver in outer Composition,
            #   and a PytorchCompositionWrapper for it that is assigned use=SHOW_PYTORCH,
            #   but don't add to either Composition as it is just used for show_graph(show_pytorch=True)
            source_sndr_port = sndr_mech._get_source_info_from_output_CIM(projection.sender)[0]
            source_sndr_mech = sndr_mech._get_source_info_from_output_CIM(projection.sender)[1]
            try:
                direct_proj = MappingProjection(name=f"Direct Projection from {source_sndr_mech.name} "
                                                     f"to {rcvr_mech.name}",
                                                sender=source_sndr_port,
                                                receiver=projection.receiver,
                                                learnable=projection.learnable)
            except DuplicateProjectionError:
                direct_proj = [proj for proj in projection.receiver.path_afferents
                               if proj.sender is source_sndr_port][0]
            else:
                direct_proj._initialize_from_context(context, base_context)

            if direct_proj not in self.projection_wrappers:
                proj_wrapper = PytorchProjectionWrapper(projection=direct_proj,
                                                        pnl_proj=pnl_proj,
                                                        component_idx=None,    # These are not needed since the wrapper
                                                        sender_port_idx=None,  # is only being used for SHOW_PYTORCH
                                                        use=[SHOW_PYTORCH],
                                                        device=self.device,
                                                        sender_wrapper=proj_sndr_wrapper,
                                                        receiver_wrapper=proj_rcvr_wrapper,
                                                        composition=self.composition,
                                                        context=context)
                outer_comp_pytorch_rep.projection_wrappers.append(proj_wrapper)
                outer_comp_pytorch_rep.projections_map[direct_proj] = proj_wrapper
                outer_comp_pytorch_rep.composition._pytorch_projections.append(direct_proj)

        else:
            assert False, f"PROGRAM ERROR: access must be ENTER_NESTED or EXIT_NESTED, not {access}"

        return pnl_proj, proj_sndr_wrapper, proj_rcvr_wrapper, use

    def _parse_optimizer_params(self, context):
        """Assign parameter-specific optimizer param groups for PyTorch GRU module"""
        composition = self.composition

        # Replace pnl names with actual torch params as keys in optimizer_params
        optimizer_params = self.composition._optimizer_params
        for param_name in optimizer_params.copy():
            param = self._pnl_refs_to_torch_params_map.get(param_name, None)
            if param:
                optimizer_params[param] = optimizer_params.pop(param_name)

        # FIX: NOT ALL PROJECTIONS FOR WHICH learning_rate COULD BE SET ARE IN
        #      _pnl_refs_to_torch_params_map (SEE ABOVE) AND THEREFORE FINDABLE BELOW (INCLUDING IN state_dict())
        # Parse learning rate specs in optimizer_params
        for param, learning_rate in optimizer_params.items():
            assert any(param is state_param for state_param in self.state_dict().values()), \
                f"PROGRAM ERROR: {param} not in state_dict for '{self.name}'"
            if composition.enable_learning is False:
                param.requires_grad = False
            else:
                if learning_rate is not False:
                    # If learning_rate is True, use composition.learning_rate, else specified value
                    lr = composition.learning_rate if isinstance(learning_rate, bool) else learning_rate
                    param.requires_grad = True
                    self._optimizer_param_groups.append({'params': param, 'lr': lr})

    def _get_execution_sets(self, composition, base_context)->list:
        """Return list of execution sets containing PytorchMechanismWrappers and/or PytorchCompositionWrappers"""
        execution_context = Context()
        try:
            composition.scheduler._init_counts(execution_id=execution_context.execution_id,
                                               base_execution_id=base_context.execution_id)
        except graph_scheduler.SchedulerError:
            # called from LLVM, no base context is provided
            composition.scheduler._init_counts(execution_id=execution_context.execution_id)

        # Setup execution sets
        # 1) Remove all learning-specific nodes
        execution_sets = [x - set(composition.get_nodes_by_role(NodeRole.LEARNING))
                               for x in composition.scheduler.run(context=execution_context)]
        # 2) Convert nodes to PytorchMechanismWrappers or PytorchCompositionWrappers
        execution_sets = [{self.nodes_map[comp] for comp in s if comp in self.nodes_map}
                               for s in execution_sets]
        # 3) Remove empty execution sets
        execution_sets = [x for x in execution_sets if len(x) > 0]

        # Flattening for forward() and AutodiffComposition.do_gradient_optimization

        # Flatten nested execution sets:
        nested_execution_sets = {}
        for exec_set in execution_sets:
            for node in exec_set:
                if isinstance(node, PytorchCompositionWrapper):
                    nested_execution_sets[node] = node.execution_sets
        for node, exec_sets in nested_execution_sets.items():
            index = execution_sets.index({node})
            # Remove nested Composition from execution sets
            execution_sets.remove({node})
            # Insert nested execution sets in place of nested Composition
            execution_sets[index:index] = exec_sets

        return execution_sets, execution_context

    __deepcopy__ = get_deepcopy_with_shared()

    def _regenerate_torch_parameter_list(self, base=None):
        """Add Projection matrices to Pytorch Module's parameter list"""

        # Register pytorch Parameters for ProjectionWrappers (since they are not already torch parameters
        for proj_wrapper in [p for p in self.projection_wrappers if not p.projection.exclude_in_autodiff]:
            self.register_parameter(proj_wrapper.name, proj_wrapper.matrix)

    # generates llvm function for self.forward
    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        args = [ctx.get_state_struct_type(self.composition).as_pointer(),
                ctx.get_param_struct_type(self.composition).as_pointer(),
                ctx.get_data_struct_type(self.composition).as_pointer()
                ]
        builder = ctx.create_llvm_function(args, self)

        state, params, data = builder.function.args
        if "learning" in tags:
            self._gen_llvm_training_function_body(ctx, builder, state, params, data)
        else:
            model_input = builder.gep(data,
                                      [ctx.int32_ty(0),
                                       ctx.int32_ty(0),
                                       ctx.int32_ty(self.composition._get_node_index(self.composition.input_CIM))])
            self._gen_llvm_forward_function_body(ctx, builder, state, params, model_input, data)

        builder.ret_void()
        return builder.function

    def _gen_llvm_forward_function_body(self, ctx, builder, state, params, arg_in, data):
        z_values = {}  # dict for storing values of terminal (output) nodes
        for current_exec_set in self.execution_sets:
            for component in current_exec_set:
                mech_input_ty = ctx.get_input_struct_type(component.mechanism)
                variable = builder.alloca(mech_input_ty)
                z_values[component] = builder.alloca(mech_input_ty.elements[0].elements[0])
                builder.store(z_values[component].type.pointee(None),z_values[component])

                if NodeRole.INPUT in self.composition.get_roles_by_node(component.mechanism):
                    input_ptr = builder.gep(
                        variable, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
                    input_id = component._idx
                    mech_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(input_id)])
                    builder.store(builder.load(mech_in), input_ptr)
                for (proj_idx, proj) in enumerate(component.afferents):
                    input_ptr = builder.gep(
                        variable, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(proj_idx)])
                    proj_output = proj._gen_llvm_execute(ctx, builder, state, params, data)
                    # store in input ports struct
                    builder.store(builder.load(proj_output), input_ptr)
                    # HACK: Add to z_values struct
                    gen_inject_vec_add(ctx, builder, proj_output, z_values[component], z_values[component])
                component._gen_llvm_execute(ctx, builder, state, params, variable, data)

        return z_values

    # generates a function responsible for a single epoch of the training
    def _gen_llvm_training_backprop(self, ctx, optimizer, loss):
        composition = self.composition
        args = [ctx.get_state_struct_type(composition).as_pointer(),
                ctx.get_param_struct_type(composition).as_pointer(),
                ctx.get_data_struct_type(composition).as_pointer(),
                optimizer._get_optimizer_struct_type(ctx).as_pointer(),
                ]
        name = self.composition.name + "_training_backprop"
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        for a in llvm_func.args:
            if isinstance(a.type, pnlvm.ir.PointerType):
                a.attributes.add('noalias')

        state, params, data, optim_struct = llvm_func.args
        model_input = builder.gep(data, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(self.composition._get_node_index(self.composition.input_CIM))])
        model_output = data
        # setup useful mappings
        input_nodes = set(self.composition.get_nodes_by_role(NodeRole.INPUT))

        # initialize optimizer params:
        delta_w = builder.gep(optim_struct, [ctx.int32_ty(0), ctx.int32_ty(optimizer._DELTA_W_NUM)])

        # 2) call forward computation
        z_values = self._gen_llvm_forward_function_body(
            ctx, builder, state, params, model_input, data)

        # 3) compute errors
        loss_fn = ctx.import_llvm_function(loss)
        total_loss = builder.alloca(ctx.float_ty)
        builder.store(total_loss.type.pointee(0), total_loss)

        error_dict = {}
        for exec_set in reversed(self.execution_sets):
            for node in exec_set:
                if node.mechanism in input_nodes:
                    continue

                node_z_value = z_values[node]
                activation_func_derivative = node._gen_llvm_execute_derivative_func(ctx, builder, state, params, node_z_value)
                error_val = builder.alloca(z_values[node].type.pointee)
                error_dict[node] = error_val

                if NodeRole.OUTPUT in self.composition.get_roles_by_node(node.mechanism):
                    # We handle output layer here
                    # compute  dC/da = a_l - y(x) (TODO: Allow other cost functions! This only applies to MSE)

                    # 1) Lookup desired target value
                    terminal_sequence = self.composition._terminal_backprop_sequences[node.mechanism]
                    target_idx = self.composition.get_nodes_by_role(NodeRole.INPUT).index(terminal_sequence[TARGET_MECHANISM])
                    node_target = builder.gep(model_input, [ctx.int32_ty(0), ctx.int32_ty(target_idx)])

                    # 2) Lookup desired output value
                    node_output = builder.gep(model_output,
                                              [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(node._idx), ctx.int32_ty(0)])

                    tmp_loss = loss.gen_inject_lossfunc_call(ctx, builder, loss_fn, node_output, node_target)

                    pnlvm.helpers.printf_float_array(ctx, builder, node_target, prefix=f"{node}\ttarget:\t", tags={"torch"})
                    pnlvm.helpers.printf_float_array(ctx, builder, node_output, prefix=f"{node}\tvalue:\t", tags={"torch"})

                    pnlvm.helpers.printf(ctx, builder, f"{node}\tloss:\t%f\n", tmp_loss, tags={"torch"})
                    builder.store(builder.fadd(builder.load(total_loss), tmp_loss), total_loss)
                    loss_derivative = loss._gen_inject_loss_differential(ctx, builder, node_output, node_target)

                    # compute δ_l = dσ/da ⊙ σ'(z)
                    gen_inject_vec_hadamard(ctx, builder, activation_func_derivative, loss_derivative, error_val)

                else:
                    # We propagate error backwards from next layer
                    for proj_idx, proj in enumerate(node.efferents):
                        efferent_node = proj.receiver_wrapper
                        efferent_node_error = error_dict[efferent_node]

                        weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, state, params)

                        if proj_idx == 0:
                            gen_inject_vxm_transposed(ctx, builder, efferent_node_error, weights_llvmlite, error_val)
                        else:
                            new_val = gen_inject_vxm_transposed(ctx, builder, efferent_node_error, weights_llvmlite)

                            gen_inject_vec_add(ctx, builder, new_val, error_val, error_val)

                    gen_inject_vec_hadamard(ctx, builder, activation_func_derivative, error_val, error_val)

                pnlvm.helpers.printf_float_array(ctx, builder, activation_func_derivative, prefix=f"{node}\tdSigma:\t", tags={"torch"})
                pnlvm.helpers.printf_float_array(ctx, builder, error_val, prefix=f"{node}\terror:\t", tags={"torch"})

        # 4) compute weight gradients
        for (node, err_val) in error_dict.items():
            if node in input_nodes:
                continue

            for proj in node.afferents:
                # get a_(l-1)
                afferent_node_activation = builder.gep(model_output, [ctx.int32_ty(0),
                                                                      ctx.int32_ty(0),
                                                                      ctx.int32_ty(proj.sender_wrapper._idx),
                                                                      ctx.int32_ty(0)])

                # get dimensions of weight matrix
                weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, state, params)
                pnlvm.helpers.printf_float_matrix(ctx,
                                                  builder,
                                                  weights_llvmlite,
                                                  prefix= f"{proj.sender_wrapper.mechanism} -> "
                                                          f"{proj.receiver_wrapper.mechanism}\n", tags={"torch"})
                # update delta_W
                node_delta_w = builder.gep(delta_w, [ctx.int32_ty(0), ctx.int32_ty(proj._idx)])

                dim_x, dim_y = proj.matrix.shape
                with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(dim_x),
                                                     "weight_update_loop_outer") as (b1, weight_row):
                    with pnlvm.helpers.for_loop_zero_inc(b1, ctx.int32_ty(dim_y),
                                                         "weight_update_loop_inner") as (b2, weight_column):
                        a_val = b2.load(b2.gep(afferent_node_activation,
                                               [ctx.int32_ty(0), weight_row]))
                        d_val = b2.load(b2.gep(err_val,
                                               [ctx.int32_ty(0), weight_column]))
                        old_val = b2.load(b2.gep(node_delta_w,
                                                 [ctx.int32_ty(0), weight_row, weight_column]))
                        new_val = b2.fadd(old_val, b2.fmul(a_val, d_val))
                        b2.store(new_val, b2.gep(node_delta_w,
                                                 [ctx.int32_ty(0), weight_row, weight_column]))

        pnlvm.helpers.printf(ctx, builder, "TOTAL LOSS:\t%.20f\n", builder.load(total_loss), tags={"torch"})
        builder.ret_void()

        return builder.function

    def _gen_llvm_training_function_body(self, ctx, builder, state, params, data):
        composition = self.composition

        optimizer = self._get_compiled_optimizer()
        # setup loss
        loss_type = self.composition.loss_spec
        if loss_type == Loss.MSE:
            loss = MSELoss()
        elif loss_type == Loss.CROSS_ENTROPY:
            loss = CROSS_ENTROPYLoss()
        else:
            raise Exception("LOSS TYPE", loss_type, "NOT SUPPORTED")

        optimizer_step_f = ctx.import_llvm_function(optimizer)
        optimizer_struct_idx = len(state.type.pointee.elements) - 1
        optimizer_struct = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(optimizer_struct_idx)])
        optimizer_zero_grad = ctx.import_llvm_function(optimizer.zero_grad(ctx).name)
        backprop = ctx.import_llvm_function(self._gen_llvm_training_backprop(ctx, optimizer, loss).name)

        # # FIXME: converting this call to inlined code results in
        # # significant longer compilation times
        builder.call(optimizer_zero_grad, [optimizer_struct])
        builder.call(backprop, [state, params, data,
                                optimizer_struct])
        builder.call(optimizer_step_f, [optimizer_struct, state, params])

    def _get_compiled_optimizer(self):
        # setup optimizer
        optimizer_type = self.composition.optimizer_type
        if optimizer_type == 'adam':
            optimizer = AdamOptimizer(self, lr=self.composition.learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGDOptimizer(self, lr=self.composition.learning_rate)
        else:
            raise Exception("OPTIMIZER TYPE", optimizer_type, "NOT SUPPORTED")
        return optimizer

    @handle_external_context()
    def forward(self, inputs, optimization_num, synch_with_pnl_options, context=None)->dict:
    # def forward(self, inputs, optimization_rep, context=None) -> dict:
        """Forward method of the model for PyTorch and LLVM modes
        Return a dictionary {output_node:value} of output values for the model
        """

        # Store the batch_size we are currently using
        inp = inputs[list(inputs.keys())[0]]
        if type(inp) is torch.Tensor:
            self._batch_size = inp.shape[0]
        elif type(inp) is list:
            self._batch_size = len(inp)
        else:
            raise ValueError("Inputs to PytorchCompositionWrapper.forward must be either torch.Tensors or lists of "
                             "torch.Tensors")

        outputs = {}  # dict for storing values of terminal (output) nodes
        for current_exec_set in self.execution_sets:
            for node in current_exec_set:

                # If node is nested Composition (wrapped in PytorchCompositionWrapper),
                #    call its forward method recursively; no need to manage outputs, as the Composition has been
                #    "flattened" (i.e., its nodes have been moved up into the outer Composition of the PyTorch
                #    representation) in _build_pytorch_representation), so its outputs will be "consumed" by the
                #    MechanismWrappers' `aggregate_afferents()` method to which it projects in the outer Composition.
                if isinstance(node, PytorchCompositionWrapper):
                    node.forward(inputs=None, optimization_num=optimization_num, context=context)
                    continue

                # Get input(s) to node
                elif node._is_input or node._is_bias:
                    # node is an INPUT to Composition
                    if node.mechanism in inputs:
                        # external input is specified for the Mechanism (i.e., Mechanism is a key in inputs dict)
                        if not node._is_bias:
                            # all input_ports receive external input, so use that
                            variable = inputs[node.mechanism]
                        else:
                            # node is also a BIAS node, so get input for each input_port individually
                            variable = []
                            for i, input_port in enumerate(node.mechanism.input_ports):
                                input = inputs[node.mechanism]
                                if not input_port.internal_only:
                                    # input_port receives external input, so get from inputs
                                    variable.append(input[i])
                                elif input_port.default_input == DEFAULT_VARIABLE:
                                    # input_port uses a bias, so get that
                                    val = input_port.defaults.variable

                                    # We need to add the batch dimension to default values.
                                    val = val[None, ...].expand(self._batch_size, *val.shape)

                                    variable.append(val)

                            # We now need to stack these so the batch dimension is first
                            try:
                                variable = torch.stack(variable, dim=1)
                            except (RuntimeError, TypeError):
                                # ragged, we need to reshape so batch dimension is first
                                # is ragged, need to reshape things so batch size is first dimension.
                                batch_size = variable[0].shape[0]
                                variable = [[inp[b] for inp in variable] for b in range(batch_size)]

                    # Input for the Mechanism is *not* explicitly specified, but its input_port(s) may have been
                    else:
                        # Get input for each input_port of the node
                        variable = []
                        for i, input_port in enumerate(node.mechanism.input_ports):
                            if input_port in inputs:
                                # input to input_port is specified in the inputs dict, so use that
                                variable.append(inputs[input_port])
                            elif input_port.default_input == DEFAULT_VARIABLE:
                                # input_port uses a bias, so get that
                                val = torch.from_numpy(input_port.defaults.variable)

                                # We need to add the batch dimension to default values.
                                val = val[None, ...].expand(self._batch_size, *val.shape)

                                variable.append(val)
                            elif not input_port.internal_only:
                                # otherwise, use the node's input_port's afferents
                                variable.append(node.collect_afferents(batch_size=self._batch_size,
                                                                       port=i,
                                                                       inputs=inputs))

                        # We now need to stack these so the batch dimension is first
                        try:
                            variable = torch.stack(variable, dim=1)
                        except (RuntimeError, TypeError):
                            # ragged, we need to reshape so batch dimension is first
                            # is ragged, need to reshape things so batch size is first dimension.
                            batch_size = variable[0].shape[0]
                            variable = [[inp[b] for inp in variable] for b in range(batch_size)]
                else:
                    # Node is not INPUT to Composition or BIAS, so get all input from its afferents
                    variable = node.collect_afferents(batch_size=self._batch_size, inputs=inputs)
                variable = node.execute_input_ports(variable)

                # Node is excluded from gradient calculations, so cache for later execution
                if node.exclude_from_gradient_calc:
                    if node.exclude_from_gradient_calc == AFTER:
                        # Cache variable for later exce execution
                        self._nodes_to_execute_after_gradient_calc[node] = variable
                        continue
                    elif node.exclude_from_gradient_calc == BEFORE:
                        assert False, 'PROGRAM ERROR: node.exclude_from_gradient_calc == BEFORE not yet implemented'
                    else:
                        assert False, \
                            (f'PROGRAM ERROR: Bad assignment to {node.name}.exclude_from_gradient_calc: '
                             f'{node.exclude_from_gradient_calc}; only {AFTER} is currently supported')

                # Execute the node (i.e., call its forward method) using composition_wrapper for Composition
                # to which it belongs; this is to support override of the execute_node method by subclasses of
                # PytorchCompositionWrapper (such as EMComposition and GRUComposition).
                node.execute(variable, optimization_num, synch_with_pnl_options, context)

                assert 'DEBUGGING BREAK POINT'

                # Add entry to outputs dict for OUTPUT Nodes of pytorch representation
                #  note: these may be different than for actual Composition, as they are flattened
                if node._is_output or node.mechanism in self.output_nodes:
                    outputs[node.mechanism] = node.output

        # NOTE: Context source needs to be set to COMMAND_LINE to force logs to update independently of timesteps
        # if not self.composition.is_nested:
        old_source = context.source
        context.source = ContextFlags.COMMAND_LINE
        self.log_values()
        self.log_weights()
        context.source = old_source

        # Return outputs of the outermost Composition
        return outputs

    def synch_with_psyneulink(self,
                              synch_with_pnl_options:dict,
                              current_condition:LEARNING_SCALE_LITERALS,
                              context:Context,
                              params:Optional[list]=None):
        """Copy weights, variables, values, and/or results from Pytorch to PsyNeuLink at specified junctures
        params can be used to restrict copy to a specific (set of) param(s). If params is not specified, all are copied;
        """
        all = [MATRIX_WEIGHTS, NODE_VARIABLES, NODE_VALUES,
               # 3/15/25 FIX: ADD SUPPORT FOR THESE IN AutodiffComposition AND BELOW
               # NODE_OUTPUT_VALUES, EXECUTE_NODES,
               RESULTS]
        params = convert_to_list(params) or all
        illegal_params = [param for param in params if param not in all]
        assert not illegal_params, \
            f"PROGRAM ERROR: Illegal attributes ({' ,'.join(illegal_params)}) specified in call to synch_with_psyneulink"

        if MATRIX_WEIGHTS in params and synch_with_pnl_options[MATRIX_WEIGHTS] == current_condition:
            self._copy_weights_to_psyneulink(context)

        # If either NODE_VARIABLES or NODE_VALUES is specified, and current condition is met, do relevant copies
        if ((NODE_VARIABLES in params and synch_with_pnl_options[NODE_VARIABLES] == current_condition)
                or (NODE_VALUES in params and synch_with_pnl_options[NODE_VALUES] == current_condition)):
            self.copy_node_variables_and_values_to_psyneulink({k:v for k,v in synch_with_pnl_options.items()
                                                               if (k in {NODE_VARIABLES, NODE_VALUES} and
                                                                   v == current_condition)},
                                                              context)

        if RESULTS in params and synch_with_pnl_options[RESULTS] == current_condition:
            self.copy_results_to_psyneulink(current_condition, context)

    def _copy_weights_to_psyneulink(self, context=None):
        for proj_wrapper in self.projections_map.values():
            if SYNCH in proj_wrapper._use:
                proj_wrapper._copy_torch_params_to_pnl_proj(context)

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            proj_wrapper.log_matrix()

    def copy_node_variables_and_values_to_psyneulink(self, options:dict, context=None):
        for pytorch_node in self.nodes_map.values():
            pytorch_node.set_pnl_variable_and_values(set_variable=True if NODE_VARIABLES in options else False,
                                                     set_value=True if NODE_VALUES in options else False,
                                                     # FIX: 3/15/25 - ADD SUPPORT FOR THESE
                                                     # set_output_values=True if OUTPUT_VALUES in options else False,
                                                     # execute_mech=True if EXECUTE_NODES in options else False,
                                                     context=context)

        # Update output_values of autodiff Composition by executing its output_CIM with pytorch_rep all_output_values
        if self.all_output_values is not None:
            # Execute the output_CIM on the last element of the batch to update the output ports
            self.composition.output_CIM.execute(self.all_output_values[-1, ...], context=context)

    def log_values(self):
        for node_wrapper in [n for n in self.node_wrappers if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()

    def copy_results_to_psyneulink(self, current_condition, context=None):
        """Append outputs of Pytorch forward() to AutodiffComposition.results attribute."""
        # IMPLEMENTATION NOTE: no need to do anything for TRIAL or MINIBATCH,
        #  as Composition's _update_results() method is getting called to do that locally
        if current_condition in {EPOCH, RUN}:
            results_param = self.composition.parameters.results
            prev_results = results_param._get(context)
            curr_results = convert_to_np_array(self.retained_results)
            if len(prev_results):
                new_results = np.append(prev_results, curr_results, 0)
            else:
                new_results = curr_results
            self.retained_results = []
            results_param._set(new_results, context)



    def retain_for_psyneulink(self,
                              data:dict,
                              retain_in_pnl_options:dict,
                              context):
        """Store outputs, targets, and losses from Pytorch execution for copying to PsyNeuLink at end of learn().
        Arguments
        ---------
        data : dict
            specifies local data available to retain (for copying to pnl at end of run;
            keys must be one or more of the keywords OUTPUTS, TARGETS, or LOSSES; values must be a torch.Tensor
        retain_in_pnl_options : dict
            specifies which data the user has requested be retained (and copied to pnl at end of run)
            keys must be OUTPUTS, TARGETS, or LOSSES; value must be a LearningScale.name or None (which suppresses copy)
        Note:  does not actually copy data to pnl; that is done by _getter methods for the relevant autodiff Parameters
        """
        try:
            for data_type, data_val in data.items():
                try:
                    if retain_in_pnl_options[data_type]:
                        retain_method_idx = DataTypeEnum._member_map_[data_type.upper()].value
                        self.retain_method[retain_method_idx](data_val)
                except KeyError:
                    assert False, \
                        (f"PROGRAM ERROR: No entry for {data_type} found in retain_in_pnl_options "
                         f"in call to retain_for_psyneulink()")
        except KeyError:
            assert False, \
                (f"PROGRAM ERROR: Invalid key(s) specified in call to retain_for_psyneulink: {list(data.keys())}")

    def retain_results(self, results:list):
        """Track outputs and copy to AutodiffComposition.pytorch_outputs at end of learn()."""
        if len(results):
            self.retained_results.append(results)

    def retain_trained_outputs(self, trained_outputs:list):
        """Track outputs and copy to AutodiffComposition.pytorch_outputs at end of learn()."""
        self.retained_trained_outputs.append(trained_outputs)

    def retain_targets(self, targets:list):
        """Track targets and copy to AutodiffComposition.pytorch_targets at end of learn()."""
        self.retained_targets.append(targets)

    def retain_losses(self, loss:torch.Tensor):
        """Track losses and copy to AutodiffComposition.pytorch_targets at end of learn()."""
        self.retained_losses.append(loss.detach().cpu().numpy().copy().tolist())

    def detach_all(self):
        for projection in self.projections_map.values():
            projection.matrix.detach()


class PytorchMechanismWrapper(torch.nn.Module):
    """Wrapper for a Mechanism in a PytorchCompositionWrapper
    These comprise nodes of the PytorchCompositionWrapper, and generally correspond to functions in a Pytorch model.

    Attributes
    ----------

    mechanism : Mechanism
        the PsyNeuLink `Mechanism` being wrapped.

    composition : AutodiffComposition
        the `AutodiffComposition` to which the `Mechanism` being wrapped belongs
        (and for which the PytorchCompositionWrapper -- to which the PytorchMechanismWrapper
        belongs -- is the pytorch_representation).

    afferents : List[PytorchProjectionWrapper]
        list of `PytorchProjectionWrapper` objects that project to the PytorchMechanismWrapper.

    input : torch.Tensor
        most recent input to the PytorchMechanismWrapper.

    function : _gen_pytorch_fct
        Pytorch version of the Mechanism's function assigned in its __init__.

    integrator_function : _gen_pytorch_fct
        Pytorch version of the Mechanism's integrator_function assigned in its __init__ if Mechanism
        has an integrator_function;  this assumes the Mechanism also has an integrator_mode attribute
        that is used to determine whether to execute the integrator_function first, and use its result
        as the input to its function.

    output : torch.Tensor
        most recent output of the PytorchMechanismWrapper.

    efferents : List[PytorchProjectionWrapper]
        list of `PytorchProjectionWrapper` objects that project from the PytorchMechanismWrapper.

    exclude_from_gradient_calc : bool or str[BEFORE | AFTER]: False
        used to prevent a node from being included in the Pytorch gradient calculation by excluding it in calls to
        the forward() and backward().  If AFTER is specified, the node is executed after at the end of the
        `update_learning_parameters` method.  BEFORE is not currently supported

    _use : list[LEARNING, SYNCH]
        designates the uses of the Mechanism, specified by the following keywords (see
        PytorchCompositionWrapper `docstring <Mechanism_and_Projection_Uses>` for additional details):

        * *LEARNING*: inputs and `function <Mechanism_Base.function>` Parameters) are used
          for actual execution of the corresponding Pytorch Module;

        * *SYNCH*: used to store results of executing a Pytorch module that are then transferred to
          the `value <Mechanism_Base.value>` Parameter of the PytorchMechanismWrapper\\s `mechanism
          <PytorchMechanismWrapper.mechanism>`;

        * *SHOW_PYTORCH*:  `Mechanism <PytorchProjectionWrapper.projection>` is included when the
          `AutoDiffComposition`\\s `show_graph <AutoDiffComposition.show_graph>` method to used with the
          ``show_pytorch`` option to display its `pytorch_representation <AutodiffComposition.pytorch_representation>`;
          if it is not specified, the `Mechanism <PytorchProjectionWrapper.projection>` is not displayed when the
          `AutoDiffComposition`\\s `show_graph <AutoDiffComposition.show_graph>` method is called, even if the
          ``show_pytorch`` option is specified.
    """

    def __init__(self,
                 mechanism:ProcessingMechanism,                 # Mechanism to be wrapped
                 composition,                                   # one to which mech belongs (for nested executions)
                 component_idx:Optional[int],                   # index of the Mechanism in the Composition
                 use:Union[list, Literal[LEARNING, SYNCH, SHOW_PYTORCH]], # learning, synching of values and/or display
                 dtype:torch.dtype,                             # needed for Pytorch
                 device:str,                                    # needed for Pytorch
                 subclass_specifies_function:bool=False,        # used to determine whether to assign function here
                 context=None):
        # # MODIFIED 7/10/24 NEW: NEEDED FOR torch MPS SUPPORT
        # super().__init__()
        # MODIFIED 7/10/24 END
        super().__init__()
        self.name = f"PytorchMechanismWrapper[{mechanism.name}]"
        self.mechanism = mechanism
        self._idx = component_idx
        self._context = context
        self._is_input = False
        self._is_bias = False
        self._is_output = False
        self._use = use or [LEARNING, SYNCH, SHOW_PYTORCH]
        self._curr_sender_value = None # Used to assign initializer or default if value == None (i.e., not yet executed)
        self.exclude_from_gradient_calc = False # Used to execute node before or after forward/backward pass methods

        from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
        assert isinstance(composition, AutodiffComposition), \
            f"PROGRAM ERROR: {composition} must be an AutodiffComposition."
        self.composition = composition
        self.torch_dtype = dtype

        self.input = None
        self.output = None

        if mechanism.parameters.has_initializers._get(context) and mechanism.parameters.value.initializer:
            self.default_output = mechanism.parameters.value.initializer.get(context)
        else:
            self.default_output = mechanism.defaults.value
        self.afferents = []
        self.efferents = []

        if subclass_specifies_function is False:
            self._assign_pytorch_function(mechanism, device, context)

    def _assign_pytorch_function(self, mechanism, device, context):
        self.function = PytorchFunctionWrapper(mechanism.function, device, context)

        if hasattr(mechanism, 'integrator_function'):
            self.integrator_function = PytorchFunctionWrapper(mechanism.integrator_function, device, context)
            self.integrator_previous_value = mechanism.integrator_function._get_pytorch_fct_param_value('initializer', device, context)

        self.input_ports = [PytorchFunctionWrapper(input_port.function, device, context)
                            for input_port in mechanism.input_ports]

    def add_afferent(self, afferent):
        """Add ProjectionWrapper for afferent to MechanismWrapper.
        For use in call to collect_afferents
        """
        assert afferent not in self.afferents
        self.afferents.append(afferent)

    def add_efferent(self, efferent):
        """Add ProjectionWrapper for efferent from MechanismWrapper.
        Implemented for completeness;  not currently used
        """
        assert efferent not in self.efferents
        self.efferents.append(efferent)

    def execute(self, variable, optimization_num, synch_with_pnl_options, context=None)->torch.Tensor:
        """Execute Mechanism's _gen_pytorch version of function on variable.
        Enforce result to be 2d, and assign to self.output
        """
        def execute_function(function, variable, fct_has_mult_args=False):
            """Execute _gen_pytorch_fct on variable, enforce result to be 2d, and return it
            If fct_has_mult_args is True, treat each item in variable as an arg to the function
            If False, compute function for each item in variable and return results in a list
            """
            from psyneulink.core.components.functions.nonstateful.transformfunctions import TransformFunction
            if fct_has_mult_args:
                res = function(*variable)
            # variable is ragged
            elif isinstance(variable, list):
                # res = [function(variable[i]) for i in range(len(variable))]
                res = [function(torch.stack([batch_elem[i] for batch_elem in variable])) for i in range(len(variable[0]))]

                # Reshape to batch dimension first
                batch_size = res[0].shape[0]
                res = [[inp[b] for inp in res] for b in range(batch_size)]

            else:
                # Functions handle batch dimensions, just run the
                # function with the variable and get back a tensor.
                res = function(variable)
            # TransformFunction can reduce output to single item from
            # multi-item input
            if isinstance(function._pnl_function, TransformFunction):
                res = res.unsqueeze(1)
            return res

        # If mechanism has an integrator_function and integrator_mode is True,
        #   execute it first and use result as input to the main function;
        #   assumes that if PyTorch node has been assigned an integrator_function then mechanism has an integrator_mode
        if hasattr(self, 'integrator_function') and self.mechanism.parameters.integrator_mode._get(context):
            variable = execute_function(self.integrator_function,
                                        [self.integrator_previous_value, variable],
                                        fct_has_mult_args=True)
            # Keep track of previous value in Pytorch node for use in next forward pass
            self.integrator_previous_value = variable

        self.input = variable

        # Compute main function of mechanism and return result
        self.output = execute_function(self.function, variable)
        return self.output

    def collect_afferents(self, batch_size:int, port:Optional[Port]=None, inputs:Optional[dict]=None):
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

        for proj_wrapper in self.afferents:
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

        # Specific port is specified
        if port is not None:
            res = [
                proj_wrapper.execute(proj_wrapper._curr_sender_value)
                for proj_wrapper in self.afferents
                if proj_wrapper._pnl_proj in self.mechanism.input_ports[port].path_afferents
            ]
        else:
            res = []
            for input_port in self.mechanism.input_ports:
                ip_res = []
                for proj_wrapper in self.afferents:
                    if proj_wrapper._pnl_proj in input_port.path_afferents:
                        ip_res.append(proj_wrapper.execute(proj_wrapper._curr_sender_value))

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

        if not isinstance(variable, torch.Tensor):
            try:
                variable = torch.stack(variable)
            except (RuntimeError, TypeError):
                # is ragged, need to reshape things so batch size is first dimension.
                pass

        # must iterate over at least 1d input per port
        if type(variable) == torch.Tensor:
            variable = torch.atleast_2d(variable)

        res = []
        for i in range(len(self.input_ports)):
            if type(variable) == torch.Tensor:
                v = variable[:, i, ...] # Get the input for the port for all items in the batch
            else:
                v = [batch_elem[i] for batch_elem in variable]

                # We should be able to stack now, since the ragged structure is only on input ports
                v = torch.stack(v)

            if isinstance(self.input_ports[i]._pnl_function, TransformFunction):
                # Add input port dimension back to account for input port dimension reduction, we should have shape
                # (batch, input_port, ... variable dimensions ) or
                # (batch, input_port, projection, ... variable dimensions ...) if execute_input_ports is invoked
                # after collect_afferents.
                if len(v.shape) == 2:
                    v = v[:, None, ...]

            res.append(self.input_ports[i].function(v))

        try:
            res = torch.stack(res, dim=1) # Stack along the input port dimension, first dimension is batch
        except (RuntimeError, TypeError):
            # is ragged, need to reshape things so batch size is first dimension.
            batch_size = res[0].shape[0]
            res = [[inp[b] for inp in res] for b in range(batch_size)]

        return res

    def set_pnl_variable_and_values(self,
                                    set_variable:bool=False,
                                    set_value:bool=True,
                                    # FIX: 3/15/25 - ADD SUPPORT FOR THESE
                                    # set_output_values:bool=None,
                                    # execute_mech:bool=True,
                                    context=None):
        """Set the state of the PytorchMechanismWrapper's Mechanism
        Note: if execute_mech=True requires that variable=True
        """
        if SYNCH not in self._use:
            return

        pnl_mech = self.mechanism

        if set_variable:
            # First get variable in numpy format
            if isinstance(self.input, list):
                variable = np.array([val.detach().cpu().numpy() for val in self.input], dtype=object)
            else:
                variable = self.input.detach().cpu().numpy()
            # Set pnl_mech's variable
            pnl_mech.parameters.variable._set(variable, context)

        if set_value:
            # self.mechanism.parameters.value._set(value.detach().cpu().numpy().squeeze(1), context)
            if self.output is None:
                assert self.exclude_from_gradient_calc, \
                    (f"PROGRAM ERROR: Value of PyTorch wrapper for {self.name} is None during forward pass, "
                     f"but it is not excluded from gradient calculation.")
                return

            # First get value in numpy format
            if isinstance(self.output, list):
                batch_size = len(self.output)
                num_outputs = len(self.output[0])
                value = np.empty((batch_size, num_outputs), dtype=object)
                for bi in range(batch_size):
                    for i in range(num_outputs):
                        value[bi, i] = self.output[bi][i].detach().cpu().numpy()

            else:
                value = self.output.detach().cpu().numpy()

            # Set pnl_mech's value
            pnl_mech.parameters.value._set(value, context)

            # If pnl_mech's function is Stateful, assign value to its previous_value parameter
            #   so that if Python implementation is run it picks up where PyTorch execution left off
            if isinstance(pnl_mech.function, StatefulFunction):
                pnl_mech.function.parameters.previous_value._set(value, context)
            # Do same for integrator_function of TransferMechanism if it is in integrator_mode
            if isinstance(pnl_mech, TransferMechanism) and pnl_mech.integrator_mode:
                pnl_mech.integrator_function.parameters.previous_value._set(self.integrator_previous_value,
                                                                            context)

        # FIX: 3/15/25 - ADD SUPPORT FOR THESE
        # if output_values:
        #     for value, port in zip(output_values, self.mechanism.output_ports):
        #         port.parameters.value._set(value.detach().cpu().numpy().squeeze(), context)
        # if execute:
        #     if variable:
        #         self.execute(variable)
        else:
            assert False, "PROGRAM ERROR: set_state called but neither set_variable nor set_value is specified"

    def _gen_llvm_execute(self, ctx, builder, state, params, mech_input, data):
        mech_func = ctx.import_llvm_function(self.mechanism)

        mech_param = builder.gep(params, [ctx.int32_ty(0),
                                          ctx.int32_ty(0),
                                          ctx.int32_ty(self._idx)])

        mech_state = builder.gep(state, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(self._idx)])

        mech_output = builder.gep(data, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(self._idx)])

        builder.call(mech_func, [mech_param,
                                 mech_state,
                                 mech_input,
                                 mech_output])

        pnlvm.helpers.printf_float_array(ctx,
                                         builder,
                                         builder.gep(mech_output, [ctx.int32_ty(0), ctx.int32_ty(0)]),
                                         prefix=f"{self} output:\n",
                                         tags={"torch"})

        return mech_output

    def log_value(self):
        if self.mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self.mechanism.output_port.parameters.value._set(detached_value, self._context)
            self.mechanism.parameters.value._set(detached_value, self._context)

    def _gen_llvm_execute_derivative_func(self, ctx, builder, state, params, arg_in):
        # psyneulink functions expect a 2d input, where index 0 is the vector
        fun = ctx.import_llvm_function(self.mechanism.function, tags=frozenset({"derivative"}))
        fun_input_ty = fun.args[2].type.pointee

        mech_input = builder.alloca(fun_input_ty)
        mech_input_ptr = builder.gep(mech_input, [ctx.int32_ty(0),
                                                  ctx.int32_ty(0)])
        builder.store(builder.load(arg_in), mech_input_ptr)

        mech_params = builder.gep(params, [ctx.int32_ty(0),
                                           ctx.int32_ty(0),
                                           ctx.int32_ty(self._idx)])

        mech_state = builder.gep(state, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(self._idx)])

        f_params, f_state = ctx.get_param_or_state_ptr(builder,
                                                       self.mechanism,
                                                       "function",
                                                       param_struct_ptr=mech_params,
                                                       state_struct_ptr=mech_state)

        f_params, builder = self.mechanism._gen_llvm_param_ports_for_obj(
                self.mechanism.function, f_params, ctx, builder, mech_params, mech_state, mech_input)

        output, _ = self.mechanism._gen_llvm_invoke_function(ctx, builder, self.mechanism.function,
                                                              f_params, f_state, mech_input, None,
                                                              tags=frozenset({"derivative"}))
        return builder.gep(output, [ctx.int32_ty(0),
                                    ctx.int32_ty(0)])

    def __repr__(self):
        return "PytorchWrapper for: " +self.mechanism.__repr__()


class PytorchProjectionWrapper():
    """Wrapper for Projection in a PytorchCompositionWrapper

    The matrix of the wrapped `projection <PytorchProjectionWrapper.projection>` is assigned as a parameter of
    (set of connection weights in ) the PyTorch Module that, coupled with a corresponding input and `torch.matmul
    <https://pytorch.org/docs/main/generated/torch.matmul.html>`_ operation, provide the input to the Pytorch
    function associated with the `Node <Composition_Node>` of the AutdiffComposition that is the `receiver
    <Projection_Base.receiver>` of the wrapped Projection.

    .. note::
       In the case of a nested Composition, the sender and/or receiver attributes may be mapped to different Node(s)
       than the Mechanism(s) of the Projection's actual sender and/or receiver. This is because the sender and/or
       receiver of the Projection may be a nested Composition, in which case the actual sender and/or receiver of the
       Projection will be a `CompositionInterfaceMechanism` (CIM) for the nested Composition.  In that case, the sender
       and/or receiver of the PytorchProjectionWrapper will be assigned to the PytorchMechanismWrapper for the Node in
       the outer Composition that Projects to/from the CIM, and that is the source/destination of the Projection
       actually being learned, and that projection will be referenced in the `PytorchCompositionWrapper.projections_map`
       (see `PytorchCompositionWrapper` for descriptive figure and additional details);  the actual projection is stored
       in pnl_proj.

    Attributes
    ----------

    projection : Projection
        PsyNeuLink `Projection` being wrapped.

    composition : AutodiffComposition
        the `AutodiffComposition` to which the `Projection` being wrapped belongs
        (and for which the PytorchCompositionWrapper -- to which the PytorchProjectionWrapper
        belongs -- is the `pytorch_representation <AutodiffComposition.pytorch_representation>`).

    matrix : torch.nn.Parameter
        Pytorch parameter for the matrix of the Projection.

    sender : PytorchMechanismWrapper
        the PytorchMechanismWrapper node from which the PytorchProjectionWrapper receives its variable.

    receiver : PytorchMechanismWrapper
        the PytorchMechanismWrapper node from which the PytorchProjectionWrapper sends it value.

    function : _gen_pytorch_fct
        Pytorch version of the Projection's function assigned in its __init__.

    .. technical_note::
        _use : list[LEARNING, SYNCH, SHOW_PYTORCH]
            designates the uses of the Projection, specified by the following keywords see PytorchCompositionWrapper
            `docstring <Mechanism_and_Projection_Uses>` for additional details):

            * *LEARNING*: inputs and `function <MappingProjection.function>` Parameters) are used for actual execution
              of the corresponding Pytorch Module;

            * *SYNCH*: store connection weights, for synching them between the `matrix
              <MappingProjection.matrix>` Parameter of its PsyNeuLink `projection <PytorchProjectionWrapper.projection>`
              and the corresponding parameters of a Pytorch module being used for learning;

            * *SHOW_PYTORCH*:  `projection <PytorchProjectionWrapper.projection>` is included when the
              `AutoDiffComposition`\\s `show_graph <AutoDiffComposition.show_graph>` method to used with
              the ``show_pytorch`` option to display its `pytorch_representation
              <AutodiffComposition.pytorch_representation>`; if it is not specified, the `Projection
              <PytorchProjectionWrapper.projection>` is not displayed when the `AutoDiffComposition`\\s
              `show_graph <AutoDiffComposition.show_graph>` method is called, even if the ``show_pytorch``
              option is specified.
    """

    def __init__(self,
                 projection:Projection,                      # Projection to be wrapped
                 pnl_proj:Projection,                        # one that directly projects to/from sender/receiver
                 component_idx:Optional[int],                   # index of the Projection in the Composition
                 sender_port_idx:Optional[int],                 # index in the sender's Mechanism.output_ports
                 use:Union[list, Literal[LEARNING, SYNCH, SHOW_PYTORCH]],
                 device:str,
                 sender_wrapper:PytorchMechanismWrapper=None,
                 receiver_wrapper:PytorchMechanismWrapper=None,
                 composition:Composition=None,
                 context=None):

        self.projection = projection  # Projection being wrapped (may *not* be the one being learned; see note above)
        self._pnl_proj = pnl_proj     # Projection to/from CIM that actually projects to/from sender/receiver
        self._use = convert_to_list(use) or [LEARNING, SYNCH, SHOW_PYTORCH]  # learn, synch, and/or display connection
        # weights
        self._idx = component_idx     # Index of Projection in Composition's list of projections
        self._sender_port_idx = sender_port_idx  # Index of sender output_ports for which Projection is an efferent
        self._value_idx = 0           # Index of value in sender's value (used in collect_afferents)
        self._curr_sender_value = None

        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        self.composition = composition            # Composition to which CompositionWrapper belongs
        self.sender_wrapper = sender_wrapper      # PytorchMechanismWrapper to which Projection's sender is mapped
        self.receiver_wrapper = receiver_wrapper  # PytorchMechanismWrapper to which Projection's receiver is mapped
        self._context = context

        if (
            projection.parameters.has_initializers._get(context)
            and projection.parameters.value.initializer
        ):
            self.default_value = projection.parameters.value.initializer.get(context)
        else:
            self.default_value = projection.defaults.value

        # Get item of value corresponding to OutputPort that is Projection's sender
        # Note: this may not be the same as _sender_port_idx if the sender Mechanism has OutputPorts for Projections
        #       that are not in the current Composition
        if context.composition and LEARNING in self._use:
            for i, output_port in enumerate(self.sender_wrapper.mechanism.output_ports):
                if all(p in context.composition.projections for p in output_port.efferents):
                    if self._pnl_proj in output_port.efferents:
                        self._value_idx = i
                        break
                    i += 1

        matrix = projection.parameters.matrix.get(context=context)
        if matrix is None:
            matrix = projection.parameters.matrix.get(context=None)
        # Create a Pytorch Parameter for the matrix
        self.matrix = torch.nn.Parameter(torch.tensor(matrix.copy(),
                                         device=device,
                                         dtype=torch.double))
        # Use Projection's name as key to align with name of torch Parameter
        self._pnl_refs_to_torch_params_map = {pnl_proj.name: self.matrix}
        # 2/16/25 - FIX: RECONCILE THIS WITH ANY SPECS FOR PROJECTION IN optimizer_params
        #           cf _parse_optimizer_params():
        if projection.learnable is False:
            self.matrix.requires_grad = False

        self.function = projection.function._gen_pytorch_fct(device, context)

    def execute(self, variable):
        # return torch.matmul(variable, self.matrix)
        return self.function(variable, self.matrix)

    def _copy_torch_params_to_pnl_proj(self, context):
        composition = self.composition
        composition.copy_torch_param_to_projection_matrix(torch_param=self.matrix.detach().cpu().T,
                                                          projection=self.projection,
                                                          validate=False,
                                                          context=context)

    def log_matrix(self):
        if self.projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self.projection.parameters.matrix._set(detached_matrix, context=self._context)
            self.projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)

    def _extract_llvm_matrix(self, ctx, builder, state, params):
        proj_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(self._idx)])
        proj_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(self._idx)])

        dim_x, dim_y = self.matrix.detach().numpy().shape

        func_p, func_s = ctx.get_param_or_state_ptr(builder,
                                                    self.projection,
                                                    self.projection.parameters.function,
                                                    param_struct_ptr=proj_params,
                                                    state_struct_ptr=proj_state)

        proj_matrix = ctx.get_param_or_state_ptr(builder,
                                                 self.projection.function,
                                                 self.projection.function.parameters.matrix,
                                                 param_struct_ptr=func_p,
                                                 state_struct_ptr=func_s)

        proj_matrix = builder.bitcast(proj_matrix, pnlvm.ir.types.ArrayType(
            pnlvm.ir.types.ArrayType(ctx.float_ty, dim_y), dim_x).as_pointer())

        return proj_matrix

    def _gen_llvm_execute(self, ctx, builder, state, params, data):
        proj_matrix = self._extract_llvm_matrix(ctx, builder, state, params)

        input_vec = builder.gep(data, [ctx.int32_ty(0),
                                       ctx.int32_ty(0),
                                       ctx.int32_ty(self.sender_wrapper._idx),
                                       ctx.int32_ty(self._sender_port_idx)])

        output_vec = gen_inject_vxm(ctx, builder, input_vec, proj_matrix)

        pnlvm.helpers.printf_float_array(ctx,
                                         builder,
                                         input_vec,
                                         prefix=f"{self.sender_wrapper.mechanism} "
                                                f"-> {self.receiver_wrapper.mechanism} input:\n",
                                         tags={"torch"})
        pnlvm.helpers.printf_float_matrix(ctx,
                                          builder,
                                          proj_matrix,
                                          prefix=f"{self.sender_wrapper.mechanism} "
                                                 f"-> {self.receiver_wrapper.mechanism} mat:\n",
                                          tags={"torch"})
        pnlvm.helpers.printf_float_array(ctx,
                                         builder,
                                         output_vec,
                                         prefix=f"{self.sender_wrapper.mechanism} "
                                                f"-> {self.receiver_wrapper.mechanism} output:\n",
                                         tags={"torch"})

        return output_vec

    def __repr__(self):
        return "PytorchWrapper for: " +self.projection.__repr__()


class PytorchFunctionWrapper(torch.nn.Module):
    def __init__(self, function, device, context=None):
        super().__init__()
        self.name = f"PytorchFunctionWrapper[{function.name}]"
        self._context = context
        self._pnl_function = function
        self.function = _get_pytorch_function(function, device, context)

    def __repr__(self):
        return "PytorchWrapper for: " + self._pnl_function.__repr__()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
