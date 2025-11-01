# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrappers for Composition, Mechanism, Projection, and Functions for use in AutodiffComposition"""
from h5py.h5f import namedtuple

from psyneulink._typing import Iterable, Literal, Optional, Union

import graph_scheduler
import numpy as np

from psyneulink.core.globals.parameters import ParameterNoValueError

# import torch
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None
else:
    import torch.nn as nn

import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING

from psyneulink.core.components.functions.stateful import StatefulFunction
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.ports.port import Port
from psyneulink.core.components.projections.projection import Projection, DuplicateProjectionError
from psyneulink.core.components.projections.pathway.mappingprojection import (MappingProjection, PROXY_FOR, PROXY_FOR_ATTRIB)
from psyneulink.core.compositions.composition import Composition, CompositionError, CompositionInterfaceMechanism, LearningScale, NodeRole
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.library.compositions.compiledoptimizer import AdamOptimizer, SGDOptimizer
from psyneulink.library.compositions.compiledloss import MSELoss, CROSS_ENTROPYLoss
from psyneulink.core.globals.keywords import (
    AFTER,
    ALL,
    BEFORE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SUFFIX,
    DEFAULT_VARIABLE,
    EXCLUDE,
    FIRST,
    LAST,
    LEARNING,
    MATRIX_WEIGHTS,
    NODE,
    NODE_VALUES,
    NODE_VARIABLES,
    RESULTS,
    SHOW_PYTORCH,
    SYNCH,
    TARGET_MECHANISM,
    Loss,
)
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import (
    convert_to_list, convert_to_np_array, get_deepcopy_with_shared, is_numeric_scalar, is_iterable)
from psyneulink.core.globals.log import LogCondition
from psyneulink.core import llvm as pnlvm


if TYPE_CHECKING:
    from psyneulink.library.compositions.autodiffcomposition import SynchRetainArg


__all__ = ['PytorchCompositionWrapper', 'PytorchMechanismWrapper', 'PytorchProjectionWrapper',
           'ENTER_NESTED', 'EXIT_NESTED', 'ParamNameCompositionTuple']

ENTER_NESTED = 0
EXIT_NESTED = 1

LEARN_CONSTRUCTION = 'learn_construction'
LEARN_OVERRIDE = 'learn_override'
LEARN_METHOD = 'learn() method'
CONSTRUCTOR = 'constructor'

NUM_OPTIMIZATIONS = 'num_optimizations'

TorchParam = namedtuple("TorchParam", "name slice")
TorchParamTuple = namedtuple('ParamTuple', "orig_spec, value")
ParamNameCompositionTuple = namedtuple('ParamNameCompositionTuple',
                                       "projection, param_name composition")

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


# NEEDED FOR torch MPS SUPPORT
# class PytorchCompositionWrapper(torch.jit.ScriptModule)
class PytorchCompositionWrapper(torch.nn.Module):
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
                self._pnl_refs_to_torch_param_names.update(node_wrapper._pnl_refs_to_torch_param_names)
                # Get any optimizer_constructor_params from nested Composition, but give precedence to any in outer comp
                for k, v in node_wrapper.composition._optimizer_constructor_params.items():
                    if k not in self.composition._optimizer_constructor_params:
                        self.composition._optimizer_constructor_params[k] = v

        # Purge nodes_map of entries for nested Compositions (their nodes are now in self.nodes_map)
        nodes_to_remove = [k for k, v in self.nodes_map.items() if isinstance(v, PytorchCompositionWrapper)]
        for node in nodes_to_remove:
            self._remove_node_from_nodes_map(node)

        self.output_nodes = self.composition.get_nested_output_nodes_at_all_levels()

        self.composition.parameters.pytorch_representation._set(self, context, skip_history=True, skip_log=True)
        self.projection_wrappers = list(self.projections_map.values())

        composition.scheduler._delete_counts(execution_context.execution_id)

        self._execute_in_additional_optimizations = (
            self._validate_and_parse_additional_optimizations(
                self.composition.execute_in_additional_optimizations,
                self.composition.optimizations_per_minibatch,
                source=CONSTRUCTOR,
            )
        )

        self._regenerate_torch_parameter_list()
        assert 'DEBUGGING BREAKPOINT'

    def _early_init(self, composition, device):
        """Early initialization of PytorchCompositionWrapper"""
        # Assign attributes
        self.name = f"PytorchCompositionWrapper[{composition.name}]"
        self.device = device
        self.optimizer = None # This gets assigned by self.composition after the wrapper is created,
        # as the latter is needed to pass the parameters to the optimizer
        self.composition = composition
        self.node_wrappers = []  # can be PytorchMechanismWrapper or PytorchCompositionWrapper
        self._nodes_to_execute_after_gradient_calc = {} # Nodes requiring execution after Pytorch forward/backward pass
        self._batch_size = 1 # Store the currently used batch size

        self.projection_wrappers = [] # PytorchProjectionWrappers
        self.projections_map = {}  # {Projection: PytorchProjectionWrappers}
        self._pnl_refs_to_torch_param_names = {} # {Projection.name: ParamNameCompositionTuple}

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
        self._store_learn_params = None

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

    def _add_node_to_nodes_map(self, node, node_wrapper):
        """Keep nodes_map, node_wrappers and modules_dict in synch"""
        self.nodes_map[node] = node_wrapper
        if node not in self.node_wrappers:
            self.node_wrappers.append(node_wrapper)
        self._modules_dict[node.name] = node_wrapper

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
                pnl_proj, pnl_proj_name, proj_sndr, proj_rcvr, use = (
                    self._handle_nested_comp(projection, context, base_context))
                # # use = [LEARNING, SYNCH, SHOW_PYTORCH]
                # use = [LEARNING, SYNCH]

            # Projection within composition
            elif all(sndr_and_recvr in self.nodes_map for sndr_and_recvr in {sndr_mech, rcvr_mech}):
                proj_sndr = self.nodes_map[sndr_mech]
                proj_rcvr = self.nodes_map[rcvr_mech]
                pnl_proj = projection
                pnl_proj_name = pnl_proj.name
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
            # Use PsyNeuLink Projection's name as key to align with name of torch Parameter
            # Add entries for both pnl_proj (user-specifie one) and projection (to input_CIM / from output_CIM)
            #   so both can be used to reference PytorchProjectionWrapper name (e.g., in _update_optimizer_params())
            pnl_proj_param_name_comp_tuple = ParamNameCompositionTuple(projection,
                                                                       pytorch_proj_wrapper.name,
                                                                       composition)
            projection_param_name_comp_tuple = ParamNameCompositionTuple(projection,
                                                                         pytorch_proj_wrapper.name,
                                                                         composition)
            self._pnl_refs_to_torch_param_names.update({pnl_proj_name: pnl_proj_param_name_comp_tuple,
                                                        projection.name: projection_param_name_comp_tuple})

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

            # Get name of the direct afferent to the nested INPUT Node,
            # to use as key in _pnl_refs_to_torch_param_names, as that is the name the User will use
            source_port = projection.sender
            destination_port = rcvr_mech._get_destination_info_from_input_CIM(projection.receiver)[0]
            direct_proj = [proj for proj in destination_port.path_afferents if proj in source_port.efferents]
            assert len(direct_proj) <= 1, (f"PROGRAM ERROR: {source_port.owner.name} has more than one efferent"
                                           f" that projects to {destination_port.owner.name}")
            proj_name = direct_proj[0].name if direct_proj else projection.name

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

            # Get name of the direct efferent from the nested OUTPUT Node
            # to use as key in _pnl_refs_to_torch_param_names, as that is the name the User will use
            destination_port = projection.receiver
            source_port = sndr_mech._get_source_info_from_output_CIM(projection.sender)[0]
            direct_proj = [proj for proj in source_port.efferents if proj in destination_port.path_afferents]
            assert len(direct_proj) <= 1, (f"PROGRAM ERROR: {destination_port.owner.name} has more than one afferent"
                                           f" that projects to it from {source_port.owner.name}")
            proj_name = direct_proj[0].name if direct_proj else projection.name

        return proj, proj_name, proj_sndr_wrapper, proj_rcvr_wrapper, use

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
                                                learnable=projection.learnable,
                                                learning_rate=projection.parameters.learning_rate.get(context))
            except DuplicateProjectionError:
                direct_proj = [proj for proj in projection.sender.efferents
                               if proj.receiver is destination_rcvr_port][0]
                pnl_proj.learnable = direct_proj.learnable
                pnl_proj.parameters.learning_rate.set(direct_proj.parameters.learning_rate.get(context), context)
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
                                                learnable=projection.learnable,
                                                learning_rate=projection.parameters.learning_rate.get(context))

            except DuplicateProjectionError:
                direct_proj = [proj for proj in projection.receiver.path_afferents
                               if proj.sender is source_sndr_port][0]
                pnl_proj.learnable = direct_proj.learnable
                pnl_proj.parameters.learning_rate.set(direct_proj.parameters.learning_rate.get(context), context)
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

    def get_all_projection_wrappers(self, start_wrapper=None)->dict:
        """Return dict of {PytorchProjectionWrapper: PytorchCompositionWrapper} in start_comp and any nested in it."""
        comp_wrapper = start_wrapper or self
        proj_wrappers = {proj_wrapper: comp_wrapper for proj_wrapper in comp_wrapper.projection_wrappers}
        for wrapper in [w for w in comp_wrapper.node_wrappers if isinstance(w, PytorchCompositionWrapper)]:
            proj_wrappers.update(wrapper.get_all_projection_wrappers())
        return proj_wrappers

    def get_all_learnable_projection_wrappers(self, start_wrapper=None, context=None)->list:
        """Return list of all learnable Projections in the Composition and any nested with it"""
        # Get ProjectionWrappers for all learnable Projections this Composition and any nested within it
        start_wrapper = start_wrapper or self
        return [proj_wrapper for proj_wrapper, comp_wrapper in self.get_all_projection_wrappers(start_wrapper).items()
                if (proj_wrapper.projection.learnable
                    and proj_wrapper.projection.parameters.learning_rate.get(context) is not False)]

    @property
    def wrapped_projections(self):
        return [wrapper.projection for wrapper in self.projection_wrappers]

    @property
    def projection_wrappers_map(self):
        return {wrapper: wrapper.projection for wrapper in self.projection_wrappers}

    def get_all_nested_composition_wrappers(self, start_wrapper=None)->dict:
        """Return list of all PytorchCompositionWrappers nested in start_wrapprer and within those."""
        comp_wrapper = start_wrapper or self
        comp_wrappers = [w for w in comp_wrapper.node_wrappers if isinstance(w, PytorchCompositionWrapper)]
        for wrapper in comp_wrappers:
            nested_wrappers = wrapper.get_all_nested_composition_wrappers()
            if nested_wrappers:
                comp_wrappers.extend(nested_wrappers)
        return comp_wrappers

    def _update_optimizer_params(self, optimizer, optimizer_params_user_specs:dict, context):
        """Assign or update parameter-specific optimizer param groups for PyTorch GRU module
        Relevant data structures:
            self.composition._optimizer_params: {Projection or Projection.name: lr}
            optimizer_params_user_specs: {Projection or Projection.name: lr}
            optimizer_params_user_parsed: {Projection.name: TorchParamTuple(Projection or Projection.name, lr)}
            optimizer_torch_params_full_with_specified:  {torch param: lr}
            self._pnl_refs_to_torch_param_names: {Projection.name: ParamNameCompositionTuple(proj, wrapper name, comp)}
            self._torch_param_short_to_long_names_map: {local name of torch param: full hierarchical name}
            self.named_parameters(): (full hierarchical name of torch param, torch param)
            self.state_dict(): (local name of torch param, Tensor)
        """
        # These are used both for error messages (hence strings) as we well determining how to update param_groups

        if context.runmode == ContextFlags.LEARNING_MODE:
            if context.source == ContextFlags.COMPOSITION:
                source = LEARN_CONSTRUCTION
                assert not self.optimizer, (f"PROGRAM ERROR: '{self.name}' is being constructed in learning mode "
                                            f"but optimizer is already specified ")
                self._store_constructor_proj_learning_rates_and_torch_params(optimizer, context)

            elif context.source == ContextFlags.METHOD:
                source = LEARN_OVERRIDE
                assert self.optimizer, (f"PROGRAM ERROR: 'no optimizer in call to learn() for '{self.name}' ")
                # revert to learning_rate assignments made in constructor before assigning any new values
                self._restore_constructor_proj_learning_rates_and_torch_params(self.optimizer, context)

            else:
                assert False, f"PROGRAM ERROR: {self.name} called in learning mode with an unexpected context.source"

        else:
            source = CONSTRUCTOR
            if self.optimizer and not optimizer_params_user_specs:
                # No need to construct (optimizer exists) or to update_optimizer (no new params)
                return

            if not optimizer_params_user_specs and not self.get_all_learnable_projection_wrappers(context=context):
                # If user didn't provide any specs, and there are no learnable Projections, not much to do;
                # just assign default optimizer.param_groups and store learning_rates for Projections
                return

        # Proceed to either construct new optimizer.param_groups (if called from constructor)
        #   or update existing ones (if called from learn() method)

        run_time_default_learning_rate = optimizer_params_user_specs.pop(DEFAULT_LEARNING_RATE, None)

        optimizer_params_user_parsed, optimizer_torch_params_full_with_specified = (
            self._parse_learning_rate_specs(optimizer,
                                            optimizer_params_user_specs,
                                            run_time_default_learning_rate,
                                            source,
                                            context))

        if source == CONSTRUCTOR and self.optimizer:
            # If user has specified dict with learning_rates in call to _build_pytorch_representation,
            #    need to update the construct_param_groups with specififed values
            self._update_constructor_param_groups(self.composition, optimizer_params_user_specs)

        self._assign_learning_rates(optimizer,
                                    optimizer_params_user_parsed,
                                    optimizer_torch_params_full_with_specified,
                                    run_time_default_learning_rate,
                                    source,
                                    context)

    def _parse_learning_rate_specs(self,
                                   optimizer:torch.optim.Optimizer,
                                   optimizer_params_user_specs:dict,
                                   run_time_default_learning_rate: Union[float, bool, None],
                                   source:str,
                                   context:Context)-> (dict, dict):
        """Parse user-specified learning_rates.
        User specifications can be in:
        - constructor for of outermost and/or nested AutodiffComposition;
        - learn() method of outermost AutodiffComposition.
        """

        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError

        # Replace any Projections in optimizer_params_user_specs with their names -> optimizer_params_user_parsed
        optimizer_params_user_parsed = {}
        if optimizer_params_user_specs:
            optimizer_params_user_parsed = {(k.name if isinstance(k, Projection) else k): TorchParamTuple(k, v)
                                       for k, v in optimizer_params_user_specs.items()}

        if optimizer_params_user_parsed:
            self._validate_optimizer_param_specs(set(optimizer_params_user_parsed.keys()), source, context)

        # Assign any user-specified Projection-specific learning_rates to the corresponding Projection.learning_rate
        #    in prep for construction optimer_params_full_with_specs
        #    (these will be overridden below if numerically specified)
        for proj_name in optimizer_params_user_parsed:
            if proj_name == DEFAULT_LEARNING_RATE:
                continue
            if proj_name in self._pnl_refs_to_torch_param_names:
                proj = self._pnl_refs_to_torch_param_names[proj_name].projection
                proj_lr = optimizer_params_user_parsed[proj_name].value
                if proj.learnable is False and proj_lr is None:
                    proj_lr = False
                if source == CONSTRUCTOR:
                    # comp_name = self.composition.name
                    comp_name = self._pnl_refs_to_torch_param_names[proj.name].composition.name
                    proj.parameters.learning_rate.set(proj_lr, comp_name + DEFAULT_SUFFIX)
                else:
                    proj.parameters.learning_rate.set(proj_lr, context)
            else:
                assert False, (f"PROGRAM ERROR: Projection '{proj_name}', for which a learning_rate has been specified "
                               f"in '{self.composition.name}, was not found in self._pnl_refs_to_torch_param_names.")

        # Gather all numerically-specified Projection.learning_rates in same format as optimizer_params_user_parsed:
        #     {Projection.name: (Projection or Projection.name, learning_rate)}
        projection_lr_specs = {proj.name:TorchParamTuple(proj, proj.parameters.learning_rate.get(context))
                               for proj in [p.projection for p in self.projection_wrappers
                                            if LEARNING in p._use and p.projection.learnable
                                            and is_numeric_scalar(p.projection.parameters.learning_rate.get(context))]}
        # Integrate optimizer_params_user_parsed, giving precedence to any learning_rates specified in learn() method()
        projection_lr_specs.update(optimizer_params_user_parsed)

        if not projection_lr_specs and not run_time_default_learning_rate:
            if source == CONSTRUCTOR:
                self._store_constructor_proj_learning_rates_and_torch_params(optimizer, context)
            return {}, {}

        # Replace Projection names (in projection_lr_specs) for torch params (used in state_dict())
        # with their actual torch parameter objects to genernate optimizer_torch_params_full_with_specified:
        #     {torch param: learning_rate}
        optimizer_torch_params_full_with_specified = {}
        for pnl_param_name, param_tuple in projection_lr_specs.items():
            param_val = param_tuple.value
            param_orig_spec = param_tuple.orig_spec
            # Get torch parameter specification for Projection names specified in projection_lr_specs
            try:
                param = self._pnl_refs_to_torch_param_names[pnl_param_name].param_name
            except KeyError:
                raise AutodiffCompositionError(
                    f"Projection specified in 'learning_rate' arg of the {self.get_source_str(source)} for "
                    f"'{self.composition.name}' ('{pnl_param_name}') is not associated with "
                    f"a learnable Pytorch parameter.")

            # IMPLEMENTATION NOTE:
            #    The following allows torch Parameters to be specified using tuples of their name and a slice
            #    (currently this is not supported by Pytorch, as learning_rates can only be specified
            #     for torch.nn.Parameter objects and not Tensors)
            # # If param spec is tuple, param name from first item (second is slice)
            # torch_param_name = param.name if isinstance(param, TorchParam) else param
            # torch_param_slice = param.slice if isinstance(param, TorchParam) else None
            # if torch_param_name not in _torch_param_short_to_long_names_map:
            #     raise AutodiffCompositionError(
            #         f"Projection specified in 'learning_rate' arg of constructor for '{self.composition.name}' "
            #         f"('{pnl_param_name}') is not associated with the name of one of its learnable Projections.")
            # # Get torch parameter for specified param_ref in named_parameters()
            # param = next((p[1] for p in self.named_parameters()
            #               if p[0] == _torch_param_short_to_long_names_map[torch_param_name]), None)

            # # Get torch parameter for specified param_ref in named_parameters()
            torch_param_name = param
            param = next((p[1] for p in self.named_parameters()
                          if p[0] == self._torch_param_short_to_long_names_map[torch_param_name]), None)
            assert param is not None, (f"PROGRAM ERROR: '{torch_param_name}' not found in "
                                       f"{self.name}.named_parameters() even though it was found in its state_dict().")

            if not param.requires_grad and param_val is not False:
                # If param was set to False in previous call to learn() but was not False at construction
                if (source == LEARN_OVERRIDE
                        and param_orig_spec in self.composition._optimizer_constructor_params
                        and self.composition._optimizer_constructor_params[param_orig_spec] is not False):
                    # Turn gradient back on
                    param.requires_grad = True
                proj_wrapper_name = self._pnl_refs_to_torch_param_names[pnl_param_name].param_name
                proj_wrapper = [wrapper for wrapper in self.projection_wrappers if wrapper.name is proj_wrapper_name][0]
                proj_lr = proj_wrapper.projection.parameters.learning_rate.get(context)
                if not proj_wrapper.projection.learnable and proj_lr is not False:
                    raise AutodiffCompositionError(
                        f"Projection ('{pnl_param_name}') specified in the dict for the 'learning_rate' arg of "
                        f"the {self.get_source_str(source)} for '{self.composition.name}' is not learnable; check that "
                        f"its 'learnable' attribute is set to 'True' and its learning_rate is not 'False', "
                        f"or remove it from the dict.")

            # IMPLEMENTATION NOTE: see above
            # if torch_param_slice:
            #     # If param spec is tuple, use param name (from above) to get param from state_dict() & apply slice
            #     param = torch.nn.Parameter(param[torch_param_slice])

            optimizer_torch_params_full_with_specified[param] = projection_lr_specs[pnl_param_name].value

        return optimizer_params_user_parsed, optimizer_torch_params_full_with_specified

    def _validate_optimizer_param_specs(self, specs_to_validate:set, source:str, context, nested=False):
        """Allow override by subclasses for custom handling of optimizer_params_user_specs (e.g., pytorchGRUWrappers)"""

        for proj_spec in specs_to_validate.copy():
            if proj_spec in self._pnl_refs_to_torch_param_names:
                specs_to_validate.remove(proj_spec)

        if specs_to_validate:
            # Give subclasses a chance to identify specs by calling _validate_optimizer_param_specs()
            #   recursively on any nested PytorchCompositionWrappers
            from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
            for nested_composition_wrapper in [node_wrapper for node_wrapper in self.node_wrappers
                                               if (isinstance(node_wrapper, PytorchCompositionWrapper))]:
                specs_to_validate = nested_composition_wrapper._validate_optimizer_param_specs(specs_to_validate,
                                                                                               source,
                                                                                               context,
                                                                                               nested=True)
        if nested:
            return specs_to_validate
        if specs_to_validate:
            if len(specs_to_validate) == 1:
                err_msg = (f"The following Projection specified in the 'learning_rate' arg of the "
                           f"{self.get_source_str(source)} for '{self.composition.name}' is not in that Composition "
                           f"or any nested within it: '{list(specs_to_validate)[0]}'.")
            else:
                err_msg = (f"The following Projections specified in the 'learning_rate' arg of the "
                           f"{self.get_source_str(source)} for '{self.composition.name}' are not in that Composition "
                           f"or any nested within it: '{', '.join(list(specs_to_validate))}'.")
            raise AutodiffCompositionError(err_msg)

    def _assign_learning_rates(self,
                               optimizer:torch.optim.Optimizer,
                               optimizer_params_user_parsed:dict,
                               optimizer_torch_params_full_with_specified:dict,
                               run_time_default_learning_rate:Union[float, bool, None],
                               source:str,
                               context:Context):
        """Assign parsed learning_rate specifications."""

        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError

        # BREADCRUMB: ?IS THIS STILL NEEDED:
        # Set Composition default learning rate for the current context
        comp_lr = run_time_default_learning_rate or self.composition.parameters.learning_rate.get(context)
        self.composition.parameters.learning_rate.set(comp_lr, context)

        # Process *every* parameter in the optimizer's param_groups
        # Get fresh copy of param_groups and assign to optimizer_params
        old_param_groups = optimizer.param_groups
        new_param_groups = self._copy_torch_param_groups(optimizer.param_groups)
        optimizer.param_groups = new_param_groups
        all_requires_grads_false = True

        # Use old_param_groups for reference, and modify new_param_groups (now assigned to optimizer)
        for old_param_group, new_param_group in zip(old_param_groups, new_param_groups):
            # Get each param in the param_groups
            default_learning_rate = old_param_group['lr']
            for param in old_param_group['params']:
                projection = self._torch_params_to_projections(old_param_groups)[param]
                specified_learning_rate = (
                    self._get_specified_learning_rate_for_param(param, projection,
                                                                optimizer_params_user_parsed,
                                                                run_time_default_learning_rate,
                                                                source, context))
                if specified_learning_rate is not False:
                    all_requires_grads_false = False
                self._update_torch_param_group(specified_learning_rate, param,
                                               old_param_group, new_param_group, new_param_groups, optimizer)

        if all_requires_grads_false:
            err_msg_start = f"There are no learnable Projections in '{self.composition.name}' nor any nested under it; "
            if (self.composition._is_learning(context) is False
                    or any(comp._is_learning(context) is False
                           for comp in self.composition._get_nested_compositions())):
                # enable_learning might be False at construction
                insert = (f"this may be because the learning_rates for all of the Projections and/or "
                          f"'enable_learning' Parameters for the Composition(s) are all set to 'False'. ")
            else:
                # must be that enable_learning is True but no Projections are learnable
                insert = f"this is because the learning_rates for all of the Projections are set to 'False'. "
            err_msg_end = (f"The learning_rate for at least one Projection must be a non-False value within a "
                           f"Composition with 'enable_learning' set to 'True' in order to execute the learn() "
                           f"method for {self.composition.name}.")
            raise AutodiffCompositionError(err_msg_start + insert + err_msg_end)

        # Remove any remaining empty param_groups
        for param_group in new_param_groups.copy():
            if not param_group['params']:
                optimizer.param_groups.remove(param_group)

        if source in {CONSTRUCTOR, LEARN_CONSTRUCTION}:
            # Store constructor-specified learning_rates (for reversion after learn())
            self._store_constructor_proj_learning_rates_and_torch_params(optimizer, context)

    def _get_specified_learning_rate_for_param(self,
                                               param,
                                               projection,
                                               optimizer_params_user_parsed,
                                               run_time_default_learning_rate,
                                               source,
                                               context):
        """Get learning_rate specified for Projection by the user or appropriate default
        learning_rate assignment is made with the following precedence, highest first
        (also see _Composition_Learning_Rate_Precedence_Hierarchy):
            Projection-specific in call to learn() [optimizer_params_user_specs]
            Projection-specific in composition.learning_rates_dict(context) [_optimizer_constructor_params]
            run_time_default_learning_rate
            Composition.learning_rate(context) for the Composition to which the Projection belongs
        """
        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError

        proj_composition = self._pnl_refs_to_torch_param_names[projection.name].composition
        proj_comp_lr = self._get_default_composition_learning_rate(proj_composition, self.composition, context)

        # Get default learning_rate for Projection for current Composition
        specified_learning_rate = \
            projection.parameters.learning_rate.get(proj_composition.name + DEFAULT_SUFFIX)

        if optimizer_params_user_parsed:
            # Get Projection-specific learning_rate if specified in call to constructor or in learn()
            if projection.name in optimizer_params_user_parsed:
                specified_learning_rate = optimizer_params_user_parsed[projection.name].value
            elif hasattr(projection, PROXY_FOR_ATTRIB) and projection._proxy_for.name in optimizer_params_user_parsed:
                specified_learning_rate = optimizer_params_user_parsed[projection._proxy_for.name].value

        if specified_learning_rate in {None, True}:
            # No Projection-specific learning_rate specified, so get default one from a Composition in the hierarchy
            if (specified_learning_rate is None and proj_comp_lr is False):
                # If Projection's learning_rate is None, then assign runtime value if specified, otherwise False
                specified_learning_rate = run_time_default_learning_rate or False
            else:
                # If Projection's learning_rate is:
                #   - None and the Composition's is *not* False or
                #   - True, irrespective of whether Composition's *is* False,
                # Then assign, in order of precedence:
                #   - run_time_default_learning_rate if that is specified,
                #   - default learning_rate for Composition to which Projection belongs if that is explicitly specified,
                #   - search up the nesting hierarchy for the first default learning_rate that is explicity specified
                #   - default learning_rate for outermost Composition
                # MODIFIED 8/10/25 NEW:
                specified_learning_rate = (run_time_default_learning_rate if run_time_default_learning_rate is not None
                                           else proj_comp_lr if proj_comp_lr
                                           else self._get_default_composition_learning_rate(proj_composition,
                                                                                            self.composition,
                                                                                            context,
                                                                                            ignore_false=True))

        if not isinstance(specified_learning_rate, (int, float, bool)):
            # Check for bad value
            raise AutodiffCompositionError(
                f"A value ('{specified_learning_rate}') specified in the 'learning_rate' arg of the "
                f"{self.get_source_str(source)} for '{self.composition.name}' is not valid; "
                f"it must be an int, float, bool or None.")

        if proj_composition.enable_learning is False or projection.learnable is False:
            # If learning is not enabled for the Projection or Composition, set learning_rate to False
            specified_learning_rate = False
            projection.parameters.learning_rate._set(False, context)
            param.requires_grad = False
        else:
            # Learning is enabled for the Projection
            # If learning_rate = True or None, use composition.learning_rate, else use specified value
            # Otherwise, use learning_rate specified at run time or in constructor for Composition
            param.requires_grad = False if specified_learning_rate is False else True

            assert specified_learning_rate is not None, \
                (f"PROGRAM ERROR: Unable to determine learning_rate ({specified_learning_rate}) "
                 f"for '{projection.name}'")

        return specified_learning_rate

    def _update_torch_param_group(self,
                                  specified_learning_rate,
                                  param,
                                  old_param_group,
                                  new_param_group,
                                  new_param_groups,
                                  optimizer):
        """Update torch param_group for specified learning_rate
        Create new group if one does not already exist for the specified learning_rate,
        """
        # Move param from existing param_group to one for the specified learning_rate
        if specified_learning_rate != old_param_group['lr']:
            # NOTE: removal of param from param_group must be done by identity using del, since
            #       param_group.remove() is content-based and some parameters may have identical values
            del new_param_group['params'][next(i for i, p in enumerate(new_param_group['params'])
                                               if p is param)]
            # Check if a param_group already exists for the specified learning_rate
            exisiting_param_group_with_specified_lr = next((param_group for param_group in new_param_groups
                                                            if param_group['lr'] == specified_learning_rate),
                                                           None)
            if exisiting_param_group_with_specified_lr:
                # Move to exisiting param_group with specified learning_rate
                exisiting_param_group_with_specified_lr['params'].append(param)
            else:
                # Create new param_group for the specified learning_rate
                optimizer.add_param_group({'params': [param], 'lr': specified_learning_rate})

    def _update_constructor_param_groups(cls, composition, optimizer_params_user_specs:dict):
        for pytorch_rep in composition.parameters.pytorch_representation.values.values():
            for proj, lr in optimizer_params_user_specs.items():
                torch_param = pytorch_rep.get_torch_param_for_projection(proj)
                param_groups = pytorch_rep._constructor_param_groups
                for i, param_group in enumerate(param_groups.copy()):
                    # if id(torch_param) in [id(p) for p in param_group['params']]:
                    # id's for all params in param_groups: [id(p) for pg in param_groups for p in pg['params']]
                    param_idx = next((j for j, p_id in enumerate([id(p) for p in param_group['params']])
                                      if p_id == id(torch_param)),None)
                    if not param_idx:
                        continue
                    # Found torch_param in param_group i
                    if lr == param_group['lr']:
                        # Already has specified value
                        warnings.warn(f"'{proj.name}' already has learning_rate of {lr} being assigned"
                                      f"in constructor for '{self.composition.name}'.")
                        return
                    del param_group['params'][param_idx]

                    # Check if a param_group already exists for the specified learning_rate
                    exisiting_param_group_with_specified_lr = next((param_group for param_group in param_groups
                                                                   if param_group['lr'] == lr),None)
                    if exisiting_param_group_with_specified_lr:
                        # Move to exisiting param_group with specified learning_rate
                        exisiting_param_group_with_specified_lr['params'].append(torch_param)
                    else:
                        # Create new param_group for the specified learning_rate
                        new_param_group = param_group.copy()
                        new_param_group['params'] = [torch_param]
                        new_param_group['lr'] = lr
                        param_groups.append(new_param_group)

    def _store_constructor_proj_learning_rates_and_torch_params(self, optimizer:torch.optim.Optimizer, context):
        """Store Composition constructor-specified learning_rates and torch parameters for Projections"""
        self._constructor_param_groups = self._copy_torch_param_groups(optimizer.param_groups)
        self._constructor_proj_learning_rates = {proj: proj.parameters.learning_rate.get(context)
                                                 for proj in self.wrapped_projections}

    def _restore_constructor_proj_learning_rates_and_torch_params(self, optimizer:torch.optim.Optimizer, context):
        """Restore Composition constructor-specified learning_rates and torch parameters for Projections"""
        try:
            self.optimizer.param_groups = self._copy_torch_param_groups(self._constructor_param_groups)
            for proj in self.wrapped_projections:
                proj.parameters.learning_rate.set(self._constructor_proj_learning_rates[proj], context)
            comp_constructor_learning_rate = self.composition.parameters.learning_rate.get(None)
            self.composition.parameters.learning_rate.set(comp_constructor_learning_rate, context)
        except AttributeError:
            assert self.optimizer, (
                f"PROGRAM ERROR: _restore_constructor_proj_learning_rates_and_torch_params() called for "
                f"'{self.composition.name}' but it does not (yet) have an optimizer."
                f"for its pytorch_representation have not been constructed.")
            assert self._constructor_param_groups, (
                f"PROGRAM ERROR: learn() called for '{self.composition.name} but the _constructor_param_groups "
                f"for its pytorch_representation have not been constructed.")
            assert self._constructor_param_groups, (
                f"PROGRAM ERROR: learn() called for '{self.composition.name} but the _constructor_proj_learning_rates "
                f"for its pytorch_representation have not been constructed.")

    def _copy_torch_param_groups(self, param_groups:list)->list:
        """Return copy of param_groups with copies of the lists of parameters in the 'params' entry
        NOTE: can't use deepcopy, as want to isolate the actual torch parameters in the 'params' lists;
              assertion checks that 'params" is the only interable attribute in each param_group
        """
        new_params_group = []
        for param_group in param_groups:
            # Future proof by ensuring that there are no other iterable attributes in param_group other than 'params'
            iterables = [param for param in param_group if (param != 'params' and not is_iterable(param))]
            assert not iterables, (f"PyTorch seems to have introduced one or moreIterable params ('{iterables}') "
                                   f"to its param_groups attribute (other than 'params'), found in optmizer for "
                                   f"pytorch_representation of '{composition.name}'")
            new_params_group.append(param_group.copy())
            new_params_group[-1]['params'] = param_group['params'].copy()
        return new_params_group

    def get_torch_param_for_projection(self, projection:Union[str, MappingProjection])->(int, torch.nn.Parameter):
        """Return torch Parameter for specified Projection"""
        projection_name = projection.name if isinstance(projection, Projection) else projection
        param_name = self._pnl_refs_to_torch_param_names[projection_name].param_name
        torch_long_param_name = self._torch_param_short_to_long_names_map[param_name]
        for param_tuple in self.named_parameters():
            # param_tuple is a tuple of (name, torch.nn.Parameter)
            if torch_long_param_name == param_tuple[0]:
                return param_tuple[1]

    def get_learning_rate_for_torch_param(self, param:torch.nn.Parameter, param_groups:list)->float:
        """Get learning_rate for torch parameter"""
        # First get parameter's python id
        for param_group in param_groups:
            for p in param_group['params']:
                if param is p:
                    return param_group['lr']

    def get_torch_learning_rate_for_projection(self,
                                               projection:Union[str, MappingProjection],
                                               optimizer:torch.optim=None)->float:
        """Get torch learning_rate for a Projection"""
        optimizer = optimizer or self.optimizer or self._optimizer_error('get_torch_learning_rate_for_projection')
        param = self.get_torch_param_for_projection(projection)
        return self.get_learning_rate_for_torch_param(param, optimizer.param_groups)

    def torch_params_to_projections(self, optimizer:torch.optim=None)->dict:
        """Return dict of {torch parameter: Projection} for all wrapped Projections, including nested ones"""
        # Get optimizer if not provided
        optimizer = optimizer or self.optimizer or self._optimizer_error('torch_params_to_projections')
        param_groups = optimizer.param_groups
        # Return dict of {torch parameter: Projection}
        return self._torch_params_to_projections(param_groups)

    def projections_to_torch_params(self)->dict:
        """Return dict of {Projection: torch parameter} for all wrapped Projections, including nested ones"""
        projections_to_torch_params = {}
        for projection in self.wrapped_projections:
            projections_to_torch_params.update({projection: self.get_torch_param_for_projection(projection)})
        # Call recursively on any nested PytorchCompositionWrappers,
        #    also giving subclasses a chance for custom handling of torch_param -> projection mapping
        for comp_wrapper in self.get_all_nested_composition_wrappers():
            projections_to_torch_params.update(comp_wrapper.projections_to_torch_params())
        return projections_to_torch_params

    @property
    def projections_with_learnable_torch_params(self)->list:
        """Return list of Projections for which torch parameters are learnable"""
        return [pw.projection for pw in self.projection_wrappers if pw.matrix.requires_grad]

    def _torch_params_to_projections(self, param_groups:list)->dict:
        """Return dict of {torch parameter: Projection} for all wrapped Projections, including nested ones"""
        torch_params_to_projections = {}
        for proj in self.wrapped_projections:
            if proj.name in self._pnl_refs_to_torch_param_names:
                torch_params_to_projections.update({self.get_torch_param_for_projection(proj): proj})
        # Give subclasses a chance for custom handling of param->projection mapping
        for comp_wrapper in self.get_all_nested_composition_wrappers():
            torch_params_to_projections.update(comp_wrapper._torch_params_to_projections(param_groups))
        return torch_params_to_projections

    @property
    def _torch_param_short_to_long_names_map(self)->dict:
        """Return map of short torch Parameter names to their full (hierarchical) names in named_parameters()
        The "full" names should include prefixes for parameters in nested PytorchCompositionWrappers.
        """
        return {k.split('.')[-1]:k for k in [p[0] for p in self.named_parameters()]}

    @property
    # Return execution-specific learning_rates
    def _torch_params_for_execution(self)->dict:
        return {proj: self.get_torch_learning_rate_for_projection(proj) for proj in self.wrapped_projections}

    def _get_default_composition_learning_rate(self, nested_comp, outer_comp, context, ignore_false=False):
        """Get learning_rate for first Composition for which a learning_rate has been explicitly specified
        Search recursively through Composition hierarchy, from nested_comp to outer_comp, for first Composition
        that has a learning_rate that is explicitly specified with a numeric value, returning default_learning_rate
        for outer_comp if not is found;
        If **ignore_false** is True, then search continues if the learning_rate for a Composition is False;
        this is to accomodate assigning  a Projection's learning_rate as``True``, which "protects" if from False
        and uses the first learning_rate found above its Composition in the hierarchy.
         """
        comp_nesting_hierarchy = nested_comp._get_outer_compositions(outer_comp)
        for comp in comp_nesting_hierarchy:
            if comp.parameters.learning_rate._user_specified:
                comp_lr = comp.parameters.learning_rate.get(context)
                if comp_lr is False and ignore_false:
                    continue
                return comp_lr
        return comp.parameters.learning_rate.get(context)

    def _optimizer_error(self, method:str):
        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
        raise AutodiffCompositionError(
            f"{self.composition.name} has not yet been assigned an optimizer; try calling"
            f"_build_pytorch_representation() before {method}.")

    __deepcopy__ = get_deepcopy_with_shared()

    def _regenerate_torch_parameter_list(self, base=None):
        """Add Projection matrices to Pytorch Module's parameter list"""

        # Register pytorch Parameters for ProjectionWrappers (since they are not already torch parameters
        for proj_wrapper in [p for p in self.projection_wrappers if not p.projection.exclude_in_autodiff]:
            self.register_parameter(proj_wrapper.name, proj_wrapper.matrix)

    def _validate_and_parse_additional_optimizations(
        self, user_specs: dict, num_optimizations, source: str
    ) -> dict:
        """Validate entries of dict specified for execute_in_additional_optimizations in Composition or learn()
        Keys should be nodes in self or a Composition nested within it, and values a list of tuples containing Parameter
        values to use during multiple optimizations, or None if no Parameters should be modified for that node.
        """
        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError

        if not isinstance(user_specs, dict):
            raise AutodiffCompositionError(f"Specification for 'execute_in_additional_optimizations' arg "
                                           f"in {source} of '{self.name}' must be a dict: {user_specs}.")
        execute_in_additional_optimizations = {}
        nodes_to_exclude = set()
        bad_nodes = []
        bad_opt_specs = []

        for nested_comp in self.composition._get_nested_compositions():
            user_specs.update(nested_comp.execute_in_additional_optimizations)

        for node, opt_spec in user_specs.items():
            if node not in self.composition._get_all_nodes() + self.composition._get_nested_compositions():
                # Node is not in the Composition or any nested within it
                bad_nodes.append(node)
                continue
            if opt_spec in {False, EXCLUDE, None}:
                # Deal with exclusion outside for loop (see comment below)
                nodes_to_exclude.add(node)
                continue
            if isinstance(node, Composition):
                # BREADCRUMB: 8/21/25 - TEST SPECIFICATION OF NESTED COMP HERE AND BELOW
                # # BREADCRUMB: ??IS THIS STILL NEEDED (SINCE NESTED NODES ARE FLATTENED INTO self.nodes[map]
                # # Add Composition (needed for forward() method
                # execute_in_additional_optimizations[self.nodes_map[node]] = opt_spec
                # Add all nodes within that Composition and any nested within in
                execute_in_additional_optimizations.update({self.nodes_map[nested_node]: opt_spec
                                                            for nested_node in node._get_all_nodes()})
            else:
                execute_in_additional_optimizations[self.nodes_map[node]] = opt_spec

                # # BREADCRUMB: 8/21/25 - ??IS THIS STILL NEEDED (SINCE NESTED NODES ARE FLATTENED INTO self.nodes[map]
                # if node not in self.composition.nodes:
                #     # For a nested node, add its Composition to nodes_to_execute since that will be needed by forward()
                #     try:
                #         comp = next(item[1] for item in self.composition._get_nested_nodes() if node is item[0])
                #     except StopIteration:
                #         assert False, f"PROGRAM ERROR: Can't find nested Composition to which '{node.name}' belongs."
                #     execute_in_additional_optimizations[self.nodes_map[comp]] = opt_spec

            # Validate opt_spec
            if not (opt_spec in {FIRST, LAST, ALL, True}
                    or isinstance(opt_spec, (int, range) and opt_spec.stop < num_optimizations)
                    or (isinstance(opt_spec, list)
                        and all(isinstance(spec, int) for spec in opt_spec)
                        and (max < num_optimizations))):
                bad_opt_specs.append(opt_spec)
                continue

            # parse opt_spec
            if opt_spec == FIRST:
                opt_spec = [0]
            elif opt_spec == LAST:
                opt_spec = [num_optimizations - 1]
            elif opt_spec in {True, ALL}:
                opt_spec = range(0, num_optimizations)

            execute_in_additional_optimizations[self.nodes_map[node]] = opt_spec

        if bad_nodes:
            raise CompositionError(
                f"The following nodes were specified in the 'execute_in_additional_optimizations' arg of learn() method"
                f"for '{self.name}' but were either not found in the Composition (or any nested within it) or "
                f"associated with a badly formatted Parameter specification: {', '.join(bad_nodes)}.")
        if bad_opt_specs:
            raise CompositionError(
                f"The following entries in 'execute_in_additional_optimizations' arg for {source} of "
                f"'{self.composition.name}' have a bad value, which must be an appropriate keyword "
                f"('FIRST', 'LAST', 'EXCLUDE', or 'ALL'), a bool, or a numeric value: {', '.join(bad_opt_specs)}.")

        # Remove any nodes specified for exclusion
        # Note: do this after the for loop above since
        #       some nodes might have been added from a nested Composition before being identified for exclusion
        (execute_in_additional_optimizations.pop(self.nodes_map[node]) for node in nodes_to_exclude)
        if execute_in_additional_optimizations:
            # If any valid nodes have been specified,
            #   add num_optimizations to dict for reference in learn() in case optimizations_for_minibatch is changed
            execute_in_additional_optimizations[NUM_OPTIMIZATIONS] = num_optimizations
        return execute_in_additional_optimizations

    # # MODIFIED 8/20/25 OLD:
    # def _call_before_additional_optimizations(self, context):
    #     """Assign specified Parameters values used for additional optimizations
    #     Get values to assigne from self._params_to_modify_in_additional_optimizations,
    #     and then use that to cache old values until _call_after_additional_optimizations is called.
    #     """
    #     # BREADCRUMB: MAY NEED TO UPDATE PYTORCH WRAPPER FOR MECHANISM FOR CHANGES IN PARAMETER VALUES TO TAKE EFFECT
    #     for node, params_list in self._params_to_modify_in_additional_optimizations.items():
    #         for i, item in enumerate(params_list.copy()):
    #             param, mod_value = item
    #             orig_value = param.get(context)
    #             param._set(mod_value, context)
    #             # Cache old value in place of new one in params list
    #             params_list[i] = (param, orig_value)
    #
    # def _call_after_additional_optimizations(self, context):
    #     """Restore Parameter values that were modified during additional optimizations to their original values
    #     Get values to restore from ones cached in self._params_to_modify_in_additional_optimizations
    #     """
    #     # BREADCRUMB: MAY NEED TO UPDATE PYTORCH WRAPPER FOR MECHANISM FOR CHANGES IN PARAMETER VALUES TO TAKE EFFECT
    #     for node, params_list in self._params_to_modify_in_additional_optimizations.items():
    #         for i, item in enumerate(params_list.copy()):
    #             param, orig_value = item
    #             mod_value = param.get(context)
    #             # Resore original value of Parameter
    #             param._set(orig_value, context)
    #             # Restore values used for additional optimizations in self._params_to_modify_in_additional_optimizations
    #             #   for use in subsequent trials
    #             params_list[i] = (param, mod_value)
    # # MODIFIED 8/20/25 END

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
        # BREADCRUMB: 7/1/25 - THIS NEEDS TO BE MODIFIED TO USE CONTEXT-SPECIFIC LEARNING RATES
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
    def forward(self, inputs, optimization_num, synch_with_pnl_options, full_sequence_mode, context=None)->dict:
    # def forward(self, inputs, optimization_rep, context=None) -> dict:
        """Forward method of the model for PyTorch and LLVM modes
        Return a dictionary {output_node:value} of output values for the model
        """

        def get_nodes_to_execute_for_optimization(opt_num: int, exec_set: set) -> set:
            # Return exec_set filtered for nodes specified in _execute_in_additional_optimizations
            addition_opts_dict = self._execute_in_additional_optimizations
            # nodes_to_execute = {node for node in addition_opts_dict
            #                     # BREADCRUMB:
            #                     # if (node is not NUM_OPTIMIZATIONS and opt_num in addition_opts_dict[node])}
            #                     if (isinstance(addition_opts_dict[node],list) and opt_num in addition_opts_dict[node])}
            # for node in exec_set:
            #     if (node in nodes_to_execute):
            #         exec_set -= {torch_node}
            nodes_to_exclude = {
                node
                for node in addition_opts_dict
                if (
                    isinstance(addition_opts_dict[node], list)
                    and opt_num not in addition_opts_dict[node]
                )
            }
            exec_set -= nodes_to_exclude

            return exec_set

        # Store the batch_size we are currently using
        inp = inputs[list(inputs.keys())[0]]
        if type(inp) is torch.Tensor:
            self._batch_size = inp.shape[0]
        elif type(inp) is list:
            self._batch_size = len(inp)
        else:
            raise ValueError("Inputs to PytorchCompositionWrapper.forward must be either torch.Tensors or lists of "
                             "torch.Tensors")

        self._sequence_lens = [len(i) for i in inp]

        # If we are in sequence mode, individual sequence elements are processed by the model one-by-one.
        if full_sequence_mode:
            seq_indices = range(max(self._sequence_lens))
        else:
            seq_indices = [0]

        # Process each sequence element, if not in sequence mode, we will process everythin at once and this loop will
        # only run once.
        for seq_index in seq_indices:

            # If we are in sequence mode, individual sequence elements are processed by the model one-by-one, so get
            # the correct one.
            if full_sequence_mode:
                # Get inputs for the current sequence index, maintain the sequence dimension even though its singleton.
                if type(inp) is torch.Tensor:
                    inputs_to_run = {k: v[:, seq_index:seq_index + 1, ...] for k, v in inputs.items()}
                elif type(inp) is list:
                    inputs_to_run = {k: [v[seq_index:seq_index + 1] for v in b_v] for k, b_v in inputs.items()}
            else:
                inputs_to_run = inputs

            # Execute nodes
            outputs = {}  # dict for storing values of terminal (output) nodes
            for current_exec_set in self.execution_sets:

                if self._execute_in_additional_optimizations:
                    # If _execute_in_additional_optimizations is specified,
                    #   only execute nodes specified for curent optmization_num
                    current_exec_set = get_nodes_to_execute_for_optimization(optimization_num, current_exec_set.copy())

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
                        if node.mechanism in inputs_to_run:
                            # external input is specified for the Mechanism (i.e., Mechanism is a key in inputs dict)
                            if not node._is_bias:
                                # all input_ports receive external input, so use that
                                variable = inputs_to_run[node.mechanism]
                            else:
                                # node is also a BIAS node, so get input for each input_port individually
                                variable = []
                                for i, input_port in enumerate(node.mechanism.input_ports):
                                    input = inputs_to_run[node.mechanism]
                                    if not input_port.internal_only:
                                        # input_port receives external input, so get from inputs
                                        variable.append(input[i])
                                    elif input_port.default_input == DEFAULT_VARIABLE:
                                        # input_port uses a bias, so get that
                                        val = input_port.defaults.variable

                                        # We need to add the batch dimension to default values.
                                        val = val[None, ...].expand(self._batch_size, *val.shape)

                                        # We also need to add a sequence dimension if it doesn't exist
                                        if val.ndim == 2:
                                            val = val[:, None, ...]

                                        variable.append(val)

                                # We now need to stack these so the batch dimension is first
                                try:
                                    variable = torch.stack(variable, dim=2)
                                except (RuntimeError, TypeError):
                                    # ragged, we need to reshape so batch dimension is first
                                    # is ragged, need to reshape things so batch size is first dimension.
                                    batch_size = variable[0].shape[0]
                                    variable = [[inp[:, b, ...] for inp in variable] for b in range(batch_size)]

                        # Input for the Mechanism is *not* explicitly specified, but its input_port(s) may have been
                        else:
                            # Get input for each input_port of the node
                            variable = []
                            for i, input_port in enumerate(node.mechanism.input_ports):
                                if input_port in inputs_to_run:
                                    # input to input_port is specified in the inputs dict, so use that
                                    variable.append(inputs_to_run[input_port])
                                elif input_port.default_input == DEFAULT_VARIABLE:
                                    # input_port uses a bias, so get that
                                    val = torch.from_numpy(input_port.defaults.variable)

                                    val = torch.atleast_2d(val)

                                    # We need to add the batch dimension to default values.
                                    val = val[None, ...].expand(self._batch_size, *val.shape)

                                    # We also need to add a sequence dimension if it doesn't exist
                                    if val.ndim == 3:
                                        val = val[:, None, ...]

                                    variable.append(val)
                                elif not input_port.internal_only:
                                    # otherwise, use the node's input_port's afferents
                                    variable.append(node.collect_afferents(batch_size=self._batch_size,
                                                                           port=i,
                                                                           inputs=inputs_to_run))

                            # We now need to stack these so the batch dimension is first
                            try:
                                variable = torch.stack(variable, dim=2)
                            except (RuntimeError, TypeError):
                                # ragged, we need to reshape so batch dimension is first
                                # is ragged, need to reshape things so batch size is first dimension.
                                batch_size = variable[0].shape[0]
                                variable = [[inp[:, b, ...] for inp in variable] for b in range(batch_size)]
                    else:
                        # Node is not INPUT to Composition or BIAS, so get all input from its afferents
                        variable = node.collect_afferents(batch_size=self._batch_size, inputs=inputs_to_run)
                    variable = node.execute_input_ports(variable)

                    # Node is excluded from gradient calculations, so cache for later execution
                    if node.exclude_from_gradient_calc:
                        # if node.exclude_from_gradient_calc in {AFTER, LAST}:
                        if node.exclude_from_gradient_calc == AFTER:
                            # Store variable for execution after gradient calculations are complete
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
                              current_condition: Union['SynchRetainArg', Iterable['SynchRetainArg']],
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

        if current_condition is None:
            current_condition = set()
        elif isinstance(current_condition, str):
            # allow ValueError to raise if current_condition is not valid
            current_condition = {LearningScale(current_condition)}
        else:
            try:
                current_condition = {current_condition}
            except TypeError:
                current_condition = set(current_condition)

        if MATRIX_WEIGHTS in params and synch_with_pnl_options[MATRIX_WEIGHTS] in current_condition:
            self._copy_weights_to_psyneulink(context)

        # If either NODE_VARIABLES or NODE_VALUES is specified, and current condition is met, do relevant copies
        if (
            (NODE_VARIABLES in params and synch_with_pnl_options[NODE_VARIABLES] in current_condition)
            or (NODE_VALUES in params and synch_with_pnl_options[NODE_VALUES] in current_condition)
        ):
            self.copy_node_variables_and_values_to_psyneulink({k:v for k,v in synch_with_pnl_options.items()
                                                               if (k in {NODE_VARIABLES, NODE_VALUES} and
                                                                   v in current_condition)},
                                                              context)

        if RESULTS in params and synch_with_pnl_options[RESULTS] in current_condition:
            self.copy_results_to_psyneulink(current_condition, context)

    def _copy_weights_to_psyneulink(self, context=None):
        for proj_wrapper in self.projections_map.values():
            if SYNCH in proj_wrapper._use:
                proj_wrapper._copy_torch_params_to_pnl_proj(context)

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            if isinstance(proj_wrapper.projection, MappingProjection):
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
        if current_condition.intersection({LearningScale.EPOCH, LearningScale.RUN}):
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

    def get_source_str(self, source):
        return LEARN_METHOD if 'learn' in source else CONSTRUCTOR


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

    exclude_from_gradient_calc : bool or str[BEFORE | AFTER | LAST]: False
        prevents a node from being included in the Pytorch gradient calculation by execluding it in calls to
        Autodiff.autodiff_backward(); entered in PytorchCompositionWrapper._nodes_to_execute_after_gradient_calc
        as a key, and the current variable that it uses for execution at the end of CompositionRuner._batch_input().

        * *AFTER*: the node is executed on every optimization step, after all gradient updates have been done;

        * *LAST*: if `Compositon.optimizations_per_minibatch` is greater than 1, the node is executed only
          after the last optimization step

        * *BEFORE*: not currently supported

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
        from psyneulink.library.compositions.autodiffcomposition import EXCLUDE_FROM_GRADIENT_CALC
        self.exclude_from_gradient_calc = mechanism.exclude_from_gradient_calc \
            if hasattr(mechanism, EXCLUDE_FROM_GRADIENT_CALC) else False
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
                    proj_wrapper._curr_sender_value = curr_val[:, :, proj_wrapper._value_idx, ...]
                else:
                    proj_wrapper._curr_sender_value = torch.stack([torch.stack([s[proj_wrapper._value_idx] for s in b]) for b in curr_val])

            else:
                val = torch.tensor(proj_wrapper.default_value)

                val = torch.atleast_2d(val)

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

                # Stack the results for this input port on the third dimension, we want to preserve
                # the first dimension as the batch, the second dimension as the sequence
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
                v = variable[:, :, i, ...] # Get the input for the port for all sequences in the batch
            else:
                v = [[s[i] for s in b] for b in variable]

                # We should be able to stack now, since the ragged structure is only on input ports
                v = torch.stack([torch.stack(b) for b in v])

            if isinstance(self.input_ports[i]._pnl_function, TransformFunction):
                # Add input port dimension back to account for input port dimension reduction, we should have shape
                # (batch, sequence, input_port, ... variable dimensions ) or
                # (batch, sequence, input_port, projection, ... variable dimensions ...) if execute_input_ports is invoked
                # after collect_afferents.
                if len(v.shape) == 3:
                    v = v[:, :, None, ...]

            res.append(self.input_ports[i].function(v))

        try:
            res = torch.stack(res, dim=2) # Stack along the input port dimension, first dimension is batch. second is sequence
        except (RuntimeError, TypeError):
            # res has a ragged structure, a list where each element corresponds to and input port. Each tensor
            # for an input port is 4D (batch, seq, projection, values). We need to reshape this so that list of lists
            # of lists where the dimensions are (batch, seq, input port, projection, values)
            batch_size = res[0].shape[0]
            seq_size = res[0].shape[1]
            res = [[[inp[b, s, ...] for inp in res] for s in range(seq_size)] for b in range(batch_size)]

        return res

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
                res = []
                for inp_i in range(len(variable[0][0])):
                    inp_t = torch.stack([torch.stack([s[inp_i] for s in b]) for b in variable])
                    inp_res = function(inp_t)
                    res.append(inp_res)

                # Reshape to batch dimension first
                batch_size = res[0].shape[0]
                seq_size = res[0].shape[1]
                res = [[[inp[b, s, ...] for inp in res] for s in range(seq_size)] for b in range(batch_size)]

            else:
                # Functions handle batch dimensions, just run the
                # function with the variable and get back a tensor.
                res = function(variable)

            # TransformFunction can reduce output to single item from
            # multi-item input
            if isinstance(function._pnl_function, TransformFunction):
                res = res.unsqueeze(2)

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
                seq_size = len(self.output[0])
                num_outputs = len(self.output[0][0])
                value = np.empty((batch_size, seq_size, num_outputs), dtype=object)
                for bi in range(batch_size):
                    for si in range(seq_size):
                        for i in range(num_outputs):
                            value[bi, si, i] = self.output[bi][si][i].detach().cpu().numpy()

                # If the sequence size is 1, squeeze it off
                if seq_size == 1:
                    value = np.squeeze(value, axis=1)

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
                integrator_prev_val = self.integrator_previous_value.detach().cpu().numpy()
                pnl_mech.integrator_function.parameters.previous_value._set(integrator_prev_val, context)

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

        if (projection.parameters.has_initializers.get(context)
                and projection.parameters.value.initializer):
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

        # Create a Pytorch Parameter for the matrix
        matrix = projection.parameters.matrix.get(context=context)
        self.matrix = torch.nn.Parameter(torch.tensor(matrix.copy(),
                                         device=device,
                                         dtype=torch.double))
        # MODIFIED 8/18/25 OLD:
        # BREADCRUMB: DELETE THIS IF DOING SO PASSES pytrests()
        # Use Projection's name as key to align with name of torch Parameter
        self._pnl_refs_to_torch_params_map = {pnl_proj.name: self.matrix}
        # 2/16/25 - FIX: RECONCILE THIS WITH ANY SPECS FOR PROJECTION IN optimizer_params
        #           cf _parse_optimizer_params():
        # MODIFIED 8/18/25 END
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
    def __init__(self, pnl_function, device, context=None):
        super().__init__()
        self.name = f"PytorchFunctionWrapper[{pnl_function.name}]"
        self._context = context
        self.function = _get_pytorch_function(pnl_function, device, context)
        if pnl_function is not self.function:
            self._pnl_function = pnl_function

    def __repr__(self):
        return "PytorchWrapper for: " + self._pnl_function.__repr__()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
