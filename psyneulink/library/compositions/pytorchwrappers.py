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
import torch
import torch.nn as nn
import numpy as np

from enum import Enum, auto

from psyneulink.core.components.functions.nonstateful.transformfunctions import LinearCombination, PRODUCT, SUM
from psyneulink.core.components.functions.stateful.integratorfunctions import IntegratorFunction
from psyneulink.core.components.functions.stateful import StatefulFunction
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import NodeRole, CompositionInterfaceMechanism
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.library.compositions.compiledoptimizer import AdamOptimizer, SGDOptimizer
from psyneulink.library.compositions.compiledloss import MSELoss, CROSS_ENTROPYLoss
from psyneulink.core.globals.keywords import (ADD, AFTER, ALL, BEFORE, DEFAULT_VARIABLE, EPOCH, INPUTS,
                                              LEARNING_SCALE_LITERALS, Loss, LOSSES, MATRIX_WEIGHTS,
                                              NODE, NODE_VALUES, NODE_VARIABLES, OUTPUTS, RESULTS, RUN,
                                              TARGETS, TARGET_MECHANISM, )
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import convert_to_np_array, get_deepcopy_with_shared, convert_to_list
from psyneulink.core.globals.log import LogCondition
from psyneulink.core import llvm as pnlvm

__all__ = ['PytorchCompositionWrapper', 'PytorchMechanismWrapper', 'PytorchProjectionWrapper']

class DataTypeEnum(Enum):

    TRAINED_OUTPUTS = 0
    TARGETS = auto()
    LOSSES = auto()

# # MODIFIED 7/29/24 OLD:
class PytorchCompositionWrapper(torch.nn.Module):
# # MODIFIED 7/29/24 NEW: NEEDED FOR torch MPS SUPPORT
# class PytorchCompositionWrapper(torch.jit.ScriptModule):
# MODIFIED 7/29/24 END
    """Wrapper for a Composition as a Pytorch Module
    Class that wraps a `Composition <Composition>` as a PyTorch module.

    Two main responsibilities:

    1) Set up parameters of PyTorch model & information required for forward computation:
        Handle nested compositions (flattened in infer_backpropagation_learning_pathways):
        Deal with Projections into and/or out of a nested Composition as shown in figure below:
            (note: Projections in outer Composition to/from a nested Composition's CIMs are learnable,
                   and ones in a nested Composition from/to its CIMs are not)
         [      OUTER     ][                            NESTED                               ][     OUTER      ]
                \\learnable//   \\not learnable//                     \\not learnable//    \\learnable//
         ---> [Node] ----> [input_CIM] ~~~> [INPUT Node] ----> [OUTPUT Node] ~~~> [output_CIM] ----> [Node] --->
               sndr            rcvr          nested_rcvr         nested_sndr         sndr             rcvr
                ^--projection-->^                                                     ^---projection-->^
                ^----PytorchProjectionWrapper---->^                  ^----PytorchProjectionWrapper---->^
                         ENTRY                                                       EXIT

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

    _composition: Composition
        `AutodiffComposition` being wrapped.

    wrapped_nodes : List[PytorchMechanismWrapper]
        list of nodes in the PytorchCompositionWrapper corresponding to PyTorch modules. Generally these are
        `Mechanisms <Mechanism>` wrapped in a `PytorchMechanismWrapper`, however, if the `AutodiffComposition`
        being wrapped is itself a nested Composition, then the wrapped nodes are `PytorchCompositionWrapper` objects.
        When the PyTorch model is executed these are "flattened" into a single PyTorch module, which can be visualized
        using the AutodiffComposition's `show_graph <ShowGraph.show_graph>` method and setting its *show_pytorch*
        argument to True (see `PytorchShowGraph` for additional information).

    nodes_map : Dict[Node: PytorchMechanismWrapper or PytorchCompositionWrapper]
        maps psyneulink `Nodes <Composition_Nodes>` to PytorchCompositionWrapper nodes.

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
        device used to process torch Tensors in PyTorch modules

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

    def __init__(self,
                 composition,
                 device,
                 outer_creator=None,
                 context=None):

        super(PytorchCompositionWrapper, self).__init__()

        from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition

        # Assign attributes
        self.name = f"PytorchCompositionWrapper[{composition.name}]"
        self._composition = composition
        self.device = device
        self.optimizer = None # This gets assigned by self._composition after the wrapper is created,
                                # as the latter is needed to pass the parameters to the optimizer

        self.wrapped_nodes = []  # can be PytorchMechanismWrapper or PytorchCompositionWrapper
        self.nodes_map = {}    # maps Node (Mech or nested Comp) -> PytorchMechanismWrapper or PytorchCompositionWrapper
        self._nodes_to_execute_after_gradient_calc = {} # Nodes requiring execution after Pytorch forward/backward pass

        self.projection_wrappers = [] # PytorchProjectionWrappers
        self.projections_map = {}  # maps Projections -> PytorchProjectionWrappers

        self.params = nn.ParameterList()

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

        # Instantiate pytorch Mechanisms
        nodes = list(set(composition.nodes) - set(composition.get_nodes_by_role(NodeRole.LEARNING)))
        # Remove nested nodes from nodes list (put there in flattening by infer_backpropagation_learning_pathways)
        #   so that they don't interfere with construction of execution_sets by scheduler
        # Will re-flatten execution sets below
        nodes = [n for n in nodes
                 # Leave nested Compositions
                 if (isinstance(n, AutodiffComposition)
                     # Needed since composition.nodes is flattened in infer_backpropagation_learning_pathways
                     or n not in [n[0] for n in self._composition._get_nested_nodes()])]
        # Sort to be sure nested Compositions are processed last, as they need outer nodes that project in/out of them
        for node in sorted(nodes, key=lambda x: isinstance(x, AutodiffComposition)):
            # Wrap nested Composition
            if isinstance(node, AutodiffComposition):
                pytorch_node = node.pytorch_composition_wrapper_type(node, device, outer_creator=self, context=context)
            # Wrap Mechanism
            else:
                pytorch_node = PytorchMechanismWrapper(node,
                                                       self,
                                                       self._composition._get_node_index(node),
                                                       device,
                                                       context=context)
                pytorch_node._is_bias = any(input_port.default_input == DEFAULT_VARIABLE
                                            for input_port in node.input_ports)
            self.nodes_map[node] = pytorch_node
            self.wrapped_nodes.append(pytorch_node)

        # Assign INPUT Nodes for outermost Composition (including any that are nested within it at any level)
        # Note: Pytorch representation is "flattened" (i.e., any nested Compositions are replaced by their Nodes)
            #   so if any nested Compositions are INPUT Nodes of the outermost Composition,
            #   *their* INPUT Nodes are assigned as INPUT Nodes of the outermost Composition
        if not composition.is_nested:
            def _assign_input_nodes(nodes):
                for pytorch_node in nodes:
                    if isinstance(pytorch_node, PytorchMechanismWrapper):
                        pytorch_node._is_input = pytorch_node._mechanism in composition._get_input_receivers(type=NODE)
                    else:
                        _assign_input_nodes(pytorch_node.wrapped_nodes)
            _assign_input_nodes(self.wrapped_nodes)

        # Instantiate PyTorch ProjectionWrappers (ignoring any from/to CIMs in the same composition)
        for projection in composition._inner_projections:
            sndr_mech = projection.sender.owner
            rcvr_mech = projection.receiver.owner

            # Projection within composition
            if all(sndr_and_recvr in self.nodes_map for sndr_and_recvr in {sndr_mech, rcvr_mech}):
                proj_sndr = self.nodes_map[sndr_mech]
                proj_rcvr = self.nodes_map[rcvr_mech]
                pnl_proj = projection

            # Ignore CIMs within the same Composition (they are not learnable; see figure in docstring)
            elif sndr_mech is composition.input_CIM or rcvr_mech is composition.output_CIM:
                continue

            # See figure in docstring above for explanation of the following:

            # ENTRY:
            # Projection to input_CIM of a nested Composition: needed for learning, so create map for Projection
            elif (isinstance(rcvr_mech, CompositionInterfaceMechanism)
                  and rcvr_mech is not self._composition.output_CIM):
                proj_sndr = self.nodes_map[sndr_mech]
                # Replace rcvr_mech (input_CIM) with the node in the nested Composition that receives the projection
                nested_rcvr_port, nested_rcvr_mech, _ = \
                    rcvr_mech._get_destination_info_from_input_CIM(projection.receiver)
                nested_pytorch_comp = self.nodes_map[rcvr_mech.composition]
                proj_rcvr = nested_pytorch_comp.nodes_map[nested_rcvr_mech]
                # Assign Projection from input_CIM to nested_rcvr_port as pnl_proj (for use in forward())
                pnl_proj = projection.receiver.owner.port_map[nested_rcvr_port][1].efferents[0]
                assert pnl_proj == nested_rcvr_port.path_afferents[0], \
                    (f"PROGRAM ERROR: First afferent Projection to '{nested_rcvr_port.owner.name}' "
                     f"(which should be from '{nested_rcvr_port.path_afferents[0].sender.owner.name}') is "
                     f"not the same as its Projection from '{projection.receiver.owner.composition.name}.input_CIM'")

            # EXIT
            # Projection from output_CIM of a nested Composition: needed for learning, so create map for Projection
            elif (isinstance(sndr_mech, CompositionInterfaceMechanism)
                  and sndr_mech is not self._composition.input_CIM):
                proj_rcvr = self.nodes_map[rcvr_mech]
                # Replace sndr_mech (output_CIM) with the node in the nested Composition that sends the projection
                nested_sndr_port, nested_sndr_mech, _ = \
                    sndr_mech._get_source_info_from_output_CIM(projection.sender)
                nested_pytorch_comp = self.nodes_map[sndr_mech.composition]
                proj_sndr = nested_pytorch_comp.nodes_map[nested_sndr_mech]

                # Assign Projection from nested_sndr_port to output_CIM as pnl_proj
                pnl_proj = projection.sender.owner.port_map[nested_sndr_port][0].path_afferents[0]
                assert pnl_proj == nested_sndr_port.efferents[0], \
                    (f"PROGRAM ERROR: First efferent Projection from '{nested_sndr_port.owner.name}' "
                     f"(to '{nested_sndr_port.efferents[0].receiver.owner.name}') is not the same as its "
                     f"Projection to '{projection.sender.owner.composition.name}.output_CIM'")
                pnl_proj = projection

            else:
                continue

            port_idx = projection.sender.owner.output_ports.index(projection.sender)
            pytorch_proj_wrapper = PytorchProjectionWrapper(
                projection,
                pnl_proj,
                list(self._composition._inner_projections).index(projection),
                port_idx,
                device,
                sender=proj_sndr,
                receiver=proj_rcvr,
                context=context)
            proj_sndr.add_efferent(pytorch_proj_wrapper)
            proj_rcvr.add_afferent(pytorch_proj_wrapper)
            self.projections_map[projection] = pytorch_proj_wrapper
            self.projection_wrappers.append(pytorch_proj_wrapper)

        c = Context()
        try:
            composition.scheduler._init_counts(execution_id=c.execution_id, base_execution_id=context.execution_id)
        except graph_scheduler.SchedulerError:
            # called from LLVM, no base context is provided
            composition.scheduler._init_counts(execution_id=c.execution_id)

        # Setup execution sets
        # 1) Remove all learning-specific nodes
        self.execution_sets = [x - set(composition.get_nodes_by_role(NodeRole.LEARNING))
                               for x in composition.scheduler.run(context=c)]
        # 2) Convert to pytorchcomponent representation
        self.execution_sets = [{self.nodes_map[comp] for comp in s if comp in self.nodes_map}
                               for s in self.execution_sets]
        # 3) Remove empty execution sets
        self.execution_sets = [x for x in self.execution_sets if len(x) > 0]


        # Flattening for forward() and AutodiffComposition.do_gradient_optimization

        # Flatten nested execution sets:
        nested_execution_sets = {}
        for exec_set in self.execution_sets:
            for node in exec_set:
                if isinstance(node, PytorchCompositionWrapper):
                    nested_execution_sets[node] = node.execution_sets
        for node, exec_sets in nested_execution_sets.items():
            index = self.execution_sets.index({node})
            # Remove nested Composition from execution sets
            self.execution_sets.remove({node})
            # Insert nested execution sets in place of nested Composition
            self.execution_sets[index:index] = exec_sets

        # Flatten maps
        for node_wrapper in self.wrapped_nodes:
            if isinstance(node_wrapper, PytorchCompositionWrapper):
                # For copying weights back to PNL in AutodiffComposition.do_gradient_optimization
                self.projections_map.update(node_wrapper.projections_map)
                # Not sure if this is needed, but just to be safe
                self.nodes_map.update(node_wrapper.nodes_map)
        # Purge nodes_map of entries for nested Compositions (their nodes are now in self.nodes_map)
        self.nodes_map = {k: v for k, v in self.nodes_map.items() if not isinstance(v, PytorchCompositionWrapper)}

        # Flatten projections so that they are all in the outer Composition and visible by _regenerate_paramlist
        #     needed for call to backward() in AutodiffComposition.do_gradient_optimization
        # FIX: MAYBE SHOULD DO THIS AS LIST IS CREATED ABOVE?
        self.projection_wrappers = list(self.projections_map.values())

        composition.scheduler._delete_counts(c.execution_id)

        self._regenerate_paramlist()

    __deepcopy__ = get_deepcopy_with_shared()

    def _regenerate_paramlist(self):
        """Add Projection matrices to Pytorch Module's parameter list"""
        self.params = nn.ParameterList()
        for proj_wrapper in [p for p in self.projection_wrappers if not p._projection.exclude_in_autodiff]:
            self.params.append(proj_wrapper.matrix)

    # generates llvm function for self.forward
    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        args = [ctx.get_state_struct_type(self._composition).as_pointer(),
                ctx.get_param_struct_type(self._composition).as_pointer(),
                ctx.get_data_struct_type(self._composition).as_pointer()
                ]
        builder = ctx.create_llvm_function(args, self)

        state, params, data = builder.function.args
        if "learning" in tags:
            self._gen_llvm_training_function_body(ctx, builder, state, params, data)
        else:
            model_input = builder.gep(data,
                                      [ctx.int32_ty(0),
                                       ctx.int32_ty(0),
                                       ctx.int32_ty(self._composition._get_node_index(self._composition.input_CIM))])
            self._gen_llvm_forward_function_body(ctx, builder, state, params, model_input, data)

        builder.ret_void()
        return builder.function

    def _gen_llvm_forward_function_body(self, ctx, builder, state, params, arg_in, data):
        z_values = {}  # dict for storing values of terminal (output) nodes
        for current_exec_set in self.execution_sets:
            for component in current_exec_set:
                mech_input_ty = ctx.get_input_struct_type(component._mechanism)
                variable = builder.alloca(mech_input_ty)
                z_values[component] = builder.alloca(mech_input_ty.elements[0].elements[0])
                builder.store(z_values[component].type.pointee(None),z_values[component])

                if NodeRole.INPUT in self._composition.get_roles_by_node(component._mechanism):
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
        composition = self._composition
        args = [ctx.get_state_struct_type(composition).as_pointer(),
                ctx.get_param_struct_type(composition).as_pointer(),
                ctx.get_data_struct_type(composition).as_pointer(),
                optimizer._get_optimizer_struct_type(ctx).as_pointer(),
                ]
        name = self._composition.name + "_training_backprop"
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        for a in llvm_func.args:
            if isinstance(a.type, pnlvm.ir.PointerType):
                a.attributes.add('noalias')

        state, params, data, optim_struct = llvm_func.args
        model_input = builder.gep(data, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(self._composition._get_node_index(self._composition.input_CIM))])
        model_output = data
        # setup useful mappings
        input_nodes = set(self._composition.get_nodes_by_role(NodeRole.INPUT))

        # initialize optimizer params:
        delta_w = builder.gep(optim_struct, [ctx.int32_ty(0), ctx.int32_ty(optimizer._DELTA_W_NUM)])

        # 2) call forward computation
        z_values = self._gen_llvm_forward_function_body(
            ctx, builder, state, params, model_input, data)

        # 3) compute errors
        loss_fn = ctx.import_llvm_function(loss)
        total_loss = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(0), total_loss)

        error_dict = {}
        for exec_set in reversed(self.execution_sets):
            for node in exec_set:
                if node._mechanism in input_nodes:
                    continue

                node_z_value = z_values[node]
                activation_func_derivative = node._gen_llvm_execute_derivative_func(ctx, builder, state, params, node_z_value)
                error_val = builder.alloca(z_values[node].type.pointee)
                error_dict[node] = error_val

                if NodeRole.OUTPUT in self._composition.get_roles_by_node(node._mechanism):
                    # We handle output layer here
                    # compute  dC/da = a_l - y(x) (TODO: Allow other cost functions! This only applies to MSE)

                    # 1) Lookup desired target value
                    terminal_sequence = self._composition._terminal_backprop_sequences[node._mechanism]
                    target_idx = self._composition.get_nodes_by_role(NodeRole.INPUT).index(terminal_sequence[TARGET_MECHANISM])
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
                        efferent_node = proj.receiver
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
                                                                      ctx.int32_ty(proj.sender._idx),
                                                                      ctx.int32_ty(0)])

                # get dimensions of weight matrix
                weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, state, params)
                pnlvm.helpers.printf_float_matrix(ctx,
                                                  builder,
                                                  weights_llvmlite,
                                                  prefix= f"{proj.sender._mechanism} -> {proj.receiver._mechanism}\n",
                                                  tags={"torch"})
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
        composition = self._composition

        optimizer = self._get_compiled_optimizer()
        # setup loss
        loss_type = self._composition.loss_spec
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
        optimizer_type = self._composition.optimizer_type
        if optimizer_type == 'adam':
            optimizer = AdamOptimizer(self, lr=self._composition.learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGDOptimizer(self, lr=self._composition.learning_rate)
        else:
            raise Exception("OPTIMIZER TYPE", optimizer_type, "NOT SUPPORTED")
        return optimizer

    @handle_external_context()
    def forward(self, inputs, optimization_rep, context=None)->dict:
        """Forward method of the model for PyTorch and LLVM modes
        Returns a dictionary {output_node:value} of output values for the model
        """
        outputs = {}  # dict for storing values of terminal (output) nodes
        for current_exec_set in self.execution_sets:
            for node in current_exec_set:

                # If node is nested Composition (wrapped in PytorchCompositionWrapper),
                #    calls its forward method recursively
                if isinstance(node, PytorchCompositionWrapper):
                    node.forward(inputs=None, optimization_rep=optimization_rep, context=context)
                    continue

                elif node._is_input or node._is_bias:
                    # node is an INPUT to Composition
                    if node._mechanism in inputs:
                        # external input is specified for the Mechanism (i.e., Mechanism is a key in inputs dict)
                        if not node._is_bias:
                            # all input_ports receive external input, so use that
                            variable = inputs[node._mechanism]
                        else:
                            # node is also a BIAS node, so get input for each input_port individually
                            for i, input_port in enumerate(node._mechanism.input_ports):
                                input = inputs[node._mechanism]
                                variable = []
                                if not input_port.internal_only:
                                    # input_port receives external input, so get from inputs
                                    variable.append(input[i])
                                elif input_port.default_input == DEFAULT_VARIABLE:
                                    # input_port uses a bias, so get that
                                    variable.append(input_port.defaults.variable)

                    # Input for the Mechanism is *not* explicitly specified, but its input_port(s) may have been
                    else:
                        # Get input for each input_port of the node
                        variable = []
                        for i, input_port in enumerate(node._mechanism.input_ports):
                            if input_port in inputs:
                                # input to input_port is specified in the inputs dict, so use that
                                variable.append(inputs[input_port])
                            elif input_port.default_input == DEFAULT_VARIABLE:
                                # input_port uses a bias, so get that
                                variable.append(input_port.defaults.variable)
                            elif not input_port.internal_only:
                                # otherwise, use the node's input_port's afferents
                                variable.append(node.aggregate_afferents(i).squeeze(0))
                        if len(variable) == 1:
                            variable = variable[0]
                else:
                    # Node is not INPUT to Composition or BIAS, so get all input from its afferents
                    variable = node.aggregate_afferents()

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

                # Execute the node using composition_wrapper_owner for Composition wrapper to which it belongs
                # Note: this is to support overrides of execute_node method by subclasses (such as in EMComposition)
                node._composition_wrapper_owner.execute_node(node, variable, optimization_rep, context)

                # 7/20/24 FIX: CACHE get_nested_output_nodes_at_all_levels() IN composition
                # Add entry to outputs dict for OUTPUT Nodes of pytorch representation
                #  note: these may be different than for actual Composition, as they are flattened
                if (node._mechanism in self._composition.get_nested_output_nodes_at_all_levels()):
                    outputs[node._mechanism] = node.output

        # NOTE: Context source needs to be set to COMMAND_LINE to force logs to update independently of timesteps
        # if not self._composition.is_nested:
        old_source = context.source
        context.source = ContextFlags.COMMAND_LINE
        self.log_values()
        self.log_weights()
        context.source = old_source

        return outputs

    def execute_node(self, node, variable, optimization_num, context=None):
        """Execute node and store the result in the node's value attribute
        Implemented as method (and includes optimization_rep and context as args)
          so that it can be overridden by subclasses of PytorchCompositionWrapper
        """
        value = node.execute(variable, context)

    def synch_with_psyneulink(self,
                              synch_with_pnl_options:dict,
                              current_condition:LEARNING_SCALE_LITERALS,
                              context:Context,
                              params:Optional[list]=None):
        """Copy weights, values, and/or results from Pytorch to PsyNeuLink at specified junctures
        params can be used to restrict copy to a specific (set of) param(s). If params is not specified, all are copied;
        """
        # 8/7/24: FIX - THIS COULD BE MADE TO BE MORE EFFICIENT ALONG THE LINES OF retain_for_psyneulink()
        #               AND REFACTORED TO USE DICT WITH DATATYPES AS KEYS AND PARAMS AS VALUES;
        all = [MATRIX_WEIGHTS, NODE_VARIABLES, NODE_VALUES, RESULTS]
        params = convert_to_list(params) or all
        illegal_params = [param for param in params if param not in all]
        assert not illegal_params, \
            f"PROGRAM ERROR: Illegal attributes ({' ,'.join(illegal_params)}) specified in call to synch_with_psyneulink"

        if MATRIX_WEIGHTS in params and synch_with_pnl_options[MATRIX_WEIGHTS] == current_condition:
            self.copy_weights_to_psyneulink(context)

        if NODE_VARIABLES in params and synch_with_pnl_options[NODE_VARIABLES] == current_condition:
            self.copy_node_variables_to_psyneulink(ALL, context)

        if NODE_VALUES in params and synch_with_pnl_options[NODE_VALUES] == current_condition:
            self.copy_node_values_to_psyneulink(ALL, context)

        if RESULTS in params and synch_with_pnl_options[RESULTS] == current_condition:
            self.copy_results_to_psyneulink(current_condition, context)

    def copy_weights_to_psyneulink(self, context=None):
        for projection, pytorch_rep in self.projections_map.items():
            matrix = pytorch_rep.matrix.detach().cpu().numpy()
            projection.parameters.matrix._set(matrix, context)
            projection.parameters.matrix._set(matrix, context)
            projection.parameter_ports['matrix'].parameters.value._set(matrix, context)

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            proj_wrapper.log_matrix()

    def copy_node_variables_to_psyneulink(self, nodes:Optional[Union[list,Literal[ALL, INPUTS]]]=ALL, context=None):
        """Copy input to Pytorch nodes to variable of AutodiffComposition nodes.
        IMPLEMENTATION NOTE:  list included in nodes arg to allow for future specification of specific nodes to copy
        """
        if nodes == ALL:
            nodes = self.nodes_map.items()
        for pnl_node, pytorch_node in nodes:
            # First get variable in numpy format
            if isinstance(pytorch_node.input, list):
                variable = np.array([val.detach().cpu().numpy() for val in pytorch_node.input], dtype=object)
            else:
                variable = pytorch_node.input.detach().cpu().numpy()
            # Set pnl_node's value to value
            pnl_node.parameters.variable._set(variable, context)

    def copy_node_values_to_psyneulink(self, nodes:Optional[Union[list,Literal[ALL, OUTPUTS]]]=ALL, context=None):
        """Copy output of Pytorch nodes to value of AutodiffComposition nodes.
        IMPLEMENTATION NOTE:  list included in nodes arg to allow for future specification of specific nodes to copy
        """
        if nodes == ALL:
            nodes = self.nodes_map.items()
        # elif nodes == OUTPUTS:
        #     nodes = [(node, self.nodes_map[node]) for node in self._composition.get_output_nodes()]

        def update_autodiff_all_output_values():
            """Update autodiff's output_values by executing its output_CIM's with pytorch_rep all_output_values"""
            if self.all_output_values:
                self._composition.output_CIM.execute(self.all_output_values, context=context)

        # Allow selective updating of just autodiff.output_values if specified
        if nodes == OUTPUTS:
            update_autodiff_all_output_values()
            return

        for pnl_node, pytorch_node in nodes:
            # Update each node's value with the output of the corresponding wrappter in the PyTorch representation
            if pytorch_node.output is None:
                assert pytorch_node.exclude_from_gradient_calc, \
                    (f"PROGRAM ERROR: Value of PyTorch wrapper for {pnl_node.name} is None during forward pass, "
                     f"but it is not excluded from gradient calculation.")
                continue
            # First get value in numpy format
            if isinstance(pytorch_node.output, list):
                value = np.array([val.detach().cpu().numpy() for val in pytorch_node.output], dtype=object)
            else:
                value = pytorch_node.output.detach().cpu().numpy()

            # Set pnl_node's value to value
            pnl_node.parameters.value._set(value, context)

            # If pnl_node's function is Stateful, assign value to its previous_value parameter
            #   so that if Python implementation is run it picks up where PyTorch execution left off
            if isinstance(pnl_node.function, StatefulFunction):
                pnl_node.function.parameters.previous_value._set(value, context)
            # Do same for integrator_function of TransferMechanism if it is in integrator_mode
            if isinstance(pnl_node, TransferMechanism) and pnl_node.integrator_mode:
                pnl_node.integrator_function.parameters.previous_value._set(pytorch_node.integrator_previous_value,
                                                                            context)
        # Finally, update the output_values of the autodiff Composition by executing its output_CIM
        update_autodiff_all_output_values()

    def log_values(self):
        for node_wrapper in [n for n in self.wrapped_nodes if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()

    def copy_results_to_psyneulink(self, current_condition, context=None):
        """Copy outputs of Pytorch forward() to AutodiffComposition.results attribute."""
        # IMPLEMENTATION NOTE: no need to do amything for TRIAL or MINIBATCH,
        #  as Composition's _update_results() method is getting called to do that locally
        if current_condition in {EPOCH, RUN}:
            self._composition.parameters.results._set(convert_to_np_array(self.retained_results), context)

    def retain_for_psyneulink(self,
                              data:dict,
                              retain_in_pnl_options:dict,
                              context):
        """Store outputs, targets, and losses from Pytorch execution for copying to PsyNeuLink at end of learn().
        Arguments
        ---------
        data : dict
            specifies local data available to retain (for copying to pnl at end of run;
            keys must be one or more of the keywords OUTPUTS, TARGETS, or LOSSES; value must be a torch.Tensor
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


class PytorchMechanismWrapper():
    """Wrapper for a Mechanism in a PytorchCompositionWrapper
    These comprise nodes of the PytorchCompositionWrapper, and generally correspond to modules of a Pytorch model.

    Attributes
    ----------

    _mechanism : Mechanism
        the PsyNeuLink `Mechanism` being wrapped.

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
    """
    def __init__(self,
                 mechanism,            # Mechanism to be wrapped
                 composition_wrapper,  # Composition wrapper to which node belongs (for executing nested Compositions)
                 component_idx,        # index of the Mechanism in the Composition
                 device,               # needed for Pytorch
                 context=None):
        # # MODIFIED 7/10/24 NEW: NEEDED FOR torch MPS SUPPORT
        # super().__init__()
        # MODIFIED 7/10/24 END
        self.name = f"PytorchMechanismWrapper[{mechanism.name}]"
        self._mechanism = mechanism
        self._idx = component_idx
        self._context = context
        self._is_input = False
        self._is_bias = False
        self._curr_sender_value = None # Used to assign initializer or default if value == None (i.e., not yet executed)
        self.exclude_from_gradient_calc = False # Used to execute node before or after forward/backward pass methods
        self._composition_wrapper_owner = composition_wrapper

        self.input = None
        self.output = None

        if mechanism.parameters.has_initializers._get(context) and mechanism.parameters.value.initializer:
            self.default_output = mechanism.parameters.value.initializer.get(context)
        else:
            self.default_output = mechanism.defaults.value
        self.afferents = []
        self.efferents = []

        from psyneulink.core.components.functions.function import FunctionError
        from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
        try:
            pnl_fct = mechanism.function
            self.function = pnl_fct._gen_pytorch_fct(device, context)
            if hasattr(mechanism, 'integrator_function'):
                pnl_fct = mechanism.integrator_function
                self.integrator_function = pnl_fct._gen_pytorch_fct(device, context)
                self.integrator_previous_value = pnl_fct._get_pytorch_fct_param_value('initializer', device, context)
        except FunctionError as error:
            from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
            raise AutodiffCompositionError(error.args[0])
        except:
            raise AutodiffCompositionError(f"Function {pnl_fct} is not currently supported by AutodiffComposition")


    def add_efferent(self, efferent):
        """Add ProjectionWrapper for efferent from MechanismWrapper.
        Implemented for completeness;  not currently used
        """
        assert efferent not in self.efferents
        self.efferents.append(efferent)

    def add_afferent(self, afferent):
        """Add ProjectionWrapper for afferent to MechanismWrapper.
        For use in call to aggregate_afferents
        """
        assert afferent not in self.afferents
        self.afferents.append(afferent)

    def aggregate_afferents(self, port=None):
        """Return weight-multiplied sum of afferent projections for input_port(s) of the Mechanism
        If there is only one input_port, return the sum of its afferents (for those in Composition)
        If there are multiple input_ports, return an array with the sum for each input_port
        FIX: AUGMENT THIS TO SUPPORT InputPort's function
        """
        assert self.afferents,\
            f"PROGRAM ERROR: No afferents found for '{self._mechanism.name}' in AutodiffComposition"

        for proj_wrapper in self.afferents:
            curr_val = proj_wrapper.sender.output
            if curr_val is not None:
                proj_wrapper._curr_sender_value = proj_wrapper.sender.output[proj_wrapper._value_idx]
            else:
                proj_wrapper._curr_sender_value = torch.tensor(proj_wrapper.default_value)

        # Specific port is specified
        # FIX: USING _port_idx TO INDEX INTO sender.value GETS IT WRONG IF THE MECHANISM HAS AN OUTPUT PORT
        #      USED BY A PROJECTION NOT IN THE CURRENT COMPOSITION
        if port is not None:
            return sum(proj_wrapper.execute(proj_wrapper._curr_sender_value).unsqueeze(0)
                                            for proj_wrapper in self.afferents
                                            if proj_wrapper._pnl_proj
                                            in self._mechanism.input_ports[port].path_afferents)
        # Has only one input_port
        elif len(self._mechanism.input_ports) == 1:
            # Get value corresponding to port from which each afferent projects
            return sum((proj_wrapper.execute(proj_wrapper._curr_sender_value).unsqueeze(0)
                        for proj_wrapper in self.afferents))
        # Has multiple input_ports
        else:
            return [sum(proj_wrapper.execute(proj_wrapper._curr_sender_value).unsqueeze(0)
                         for proj_wrapper in self.afferents
                         if proj_wrapper._pnl_proj in input_port.path_afferents)
                     for input_port in self._mechanism.input_ports]

    def execute(self, variable, context):
        """Execute Mechanism's _gen_pytorch version of function on variable.
        Enforce result to be 2d, and assign to self.output
        """
        def execute_function(function, variable, fct_has_mult_args=False, is_combination_fct=False):
            """Execute _gen_pytorch_fct on variable, enforce result to be 2d, and return it
            If fct_has_mult_args is True, treat each item in variable as an arg to the function
            If False, compute function for each item in variable and return results in a list
            """
            if ((isinstance(variable, list) and len(variable) == 1)
                or (isinstance(variable, torch.Tensor) and len(variable.squeeze(0).shape) == 1)
                    or isinstance(self._mechanism.function, LinearCombination)):
                # Enforce 2d on value of MechanismWrapper (using unsqueeze) for single InputPort
                # or if TransformFunction (which reduces output to single item from multi-item input)
                if isinstance(variable, torch.Tensor):
                    variable = variable.squeeze(0)
                return function(variable).unsqueeze(0)
            elif is_combination_fct:
                # Function combines the elements
                return function(variable)
            elif fct_has_mult_args:
                # Assign each element of variable as an arg to the function
                return function(*variable)
            else:
                # Treat each item in variable as a separate input to the function and get result for each in a list:
                # make return value 2d by creating list of the results of function returned for each item in variable
                return [function(variable[i].squeeze(0)) for i in range(len(variable))]

        # If mechanism has an integrator_function and integrator_mode is True,
        #   execute it first and use result as input to the main function;
        #   assumes that if PyTorch node has been assigned an integrator_function then _mechanism has an integrator_mode
        if hasattr(self, 'integrator_function') and self._mechanism.parameters.integrator_mode._get(context):
            variable = execute_function(self.integrator_function,
                                        [self.integrator_previous_value, variable],
                                        fct_has_mult_args=True)
            # Keep track of previous value in Pytorch node for use in next forward pass
            self.integrator_previous_value = variable

        self.input = variable

        # Compute main function of mechanism and return result
        from psyneulink.core.components.functions.nonstateful.transformfunctions import TransformFunction
        self.output = execute_function(self.function, variable,
                                       is_combination_fct=isinstance(self._mechanism.function, TransformFunction))
        return self.output

    def _gen_llvm_execute(self, ctx, builder, state, params, mech_input, data):
        mech_func = ctx.import_llvm_function(self._mechanism)

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
        if self._mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self._mechanism.output_port.parameters.value._set(detached_value, self._context)
            self._mechanism.parameters.value._set(detached_value, self._context)

    def _gen_llvm_execute_derivative_func(self, ctx, builder, state, params, arg_in):
        # psyneulink functions expect a 2d input, where index 0 is the vector
        fun = ctx.import_llvm_function(self._mechanism.function, tags=frozenset({"derivative"}))
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
                                                       self._mechanism,
                                                       "function",
                                                       param_struct_ptr=mech_params,
                                                       state_struct_ptr=mech_state)

        f_params, builder = self._mechanism._gen_llvm_param_ports_for_obj(
                self._mechanism.function, f_params, ctx, builder, mech_params, mech_state, mech_input)

        output, _ = self._mechanism._gen_llvm_invoke_function(ctx, builder, self._mechanism.function,
                                                              f_params, f_state, mech_input, None,
                                                              tags=frozenset({"derivative"}))
        return builder.gep(output, [ctx.int32_ty(0),
                                    ctx.int32_ty(0)])

    def __repr__(self):
        return "PytorchWrapper for: " +self._mechanism.__repr__()


class PytorchProjectionWrapper():
    """Wrapper for Projection in a PytorchCompositionWrapper

    The matrix of the wrapped `_projection <PytorchProjectionWrapper._projection>` corresponds to the parameters
    (connection weights) of the PyTorch Module that is the `function <Mechanism_Base.function>` of the
    `receiver <Projection_Base.receiver>` of the wrapped Projection.

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

    _projection : Projection
        PsyNeuLink `Projection` being wrapped.

    sender : PytorchMechanismWrapper
        the PytorchMechanismWrapper node from which the PytorchProjectionWrapper receives its variable.

    receiver : PytorchMechanismWrapper
        the PytorchMechanismWrapper node from which the PytorchProjectionWrapper sends it value.

    function : _gen_pytorch_fct
        Pytorch version of the Projection's function assigned in its __init__.

    """

    def __init__(self,
                 projection,
                 pnl_proj,
                 component_idx,
                 port_idx, device,
                 sender=None,
                 receiver=None,
                 context=None):
        self._projection = projection # Projection being wrapped (may *not* be the one being learned; see note above)
        self._pnl_proj = pnl_proj     # Projection that directly projects to/from sender/receiver (see above)
        self._idx = component_idx     # Index of Projection in Composition's list of projections
        self._port_idx = port_idx     # Index of sender's port (used by LLVM)
        self._value_idx = 0           # Index of value in sender's value (used in aggregate_afferents)
        self._curr_sender_value = None

        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        self.sender = sender          # PytorchMechanismWrapper to which Projection's sender is mapped
        self.receiver = receiver      # PytorchMechanismWrapper to which Projection's receiver is mapped
        self._context = context

        if projection.parameters.has_initializers._get(context) and projection.parameters.value.initializer:
            self.default_value = projection.parameters.value.initializer.get(context)
        else:
            self.default_value = projection.defaults.value


        # Get item of value corresponding to OutputPort that is Projection's sender
        # Note: this may not be the same as _port_idx if the sender Mechanism has OutputPorts for Projections
        #       that are not in the current Composition
        if context._composition:
            for i, output_port in enumerate(self.sender._mechanism.output_ports):
                if all(p in context._composition.projections for p in output_port.efferents):
                    if self._pnl_proj in output_port.efferents:
                        self._value_idx = i
                        break
                    i += 1

        matrix = projection.parameters.matrix.get(context=context)
        if matrix is None:
            matrix = projection.parameters.matrix.get(context=None)
        self.matrix = torch.nn.Parameter(torch.tensor(matrix.copy(),
                                         device=device,
                                         dtype=torch.double))
        if projection.learnable is False:
            self.matrix.requires_grad = False

        self.function = projection.function._gen_pytorch_fct(device, context)


    def execute(self, variable):
        # return torch.matmul(variable, self.matrix)
        return self.function(variable, self.matrix)

    def log_matrix(self):
        if self._projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self._projection.parameters.matrix._set(detached_matrix, context=self._context)
            self._projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)

    def _extract_llvm_matrix(self, ctx, builder, state, params):
        proj_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(self._idx)])
        proj_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(self._idx)])

        dim_x, dim_y = self.matrix.detach().numpy().shape

        func_p, func_s = ctx.get_param_or_state_ptr(builder,
                                                    self._projection,
                                                    self._projection.parameters.function,
                                                    param_struct_ptr=proj_params,
                                                    state_struct_ptr=proj_state)

        proj_matrix = ctx.get_param_or_state_ptr(builder,
                                                 self._projection.function,
                                                 self._projection.function.parameters.matrix,
                                                 param_struct_ptr=func_p,
                                                 state_struct_ptr=func_s)

        proj_matrix = builder.bitcast(proj_matrix, pnlvm.ir.types.ArrayType(
            pnlvm.ir.types.ArrayType(ctx.float_ty, dim_y), dim_x).as_pointer())

        return proj_matrix

    def _gen_llvm_execute(self, ctx, builder, state, params, data):
        proj_matrix = self._extract_llvm_matrix(ctx, builder, state, params)

        input_vec = builder.gep(data, [ctx.int32_ty(0),
                                       ctx.int32_ty(0),
                                       ctx.int32_ty(self.sender._idx),
                                       ctx.int32_ty(self._port_idx)])

        output_vec = gen_inject_vxm(ctx, builder, input_vec, proj_matrix)

        pnlvm.helpers.printf_float_array(ctx,
                                         builder,
                                         input_vec,
                                         prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} input:\n",
                                         tags={"torch"})
        pnlvm.helpers.printf_float_matrix(ctx,
                                          builder,
                                          proj_matrix,
                                          prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} mat:\n",
                                          tags={"torch"})
        pnlvm.helpers.printf_float_array(ctx,
                                         builder,
                                         output_vec,
                                         prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} output:\n",
                                         tags={"torch"})

        return output_vec

    def __repr__(self):
        return "PytorchWrapper for: " +self._projection.__repr__()
