# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrappers for Composition, Mechanism, Projection, and Functions for use in AutodiffComposition"""

import graph_scheduler
import torch
import torch.nn as nn

from psyneulink.core.components.functions.nonstateful.combinationfunctions import LinearCombination, PRODUCT, SUM
from psyneulink.core.compositions.composition import NodeRole, CompositionInterfaceMechanism
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.library.compositions.compiledoptimizer import AdamOptimizer, SGDOptimizer
from psyneulink.library.compositions.compiledloss import MSELoss, CROSS_ENTROPYLoss
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE, Loss, NODE, TARGET_MECHANISM
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import get_deepcopy_with_shared
from psyneulink.core.globals.log import LogCondition
from psyneulink.core import llvm as pnlvm

__all__ = ['PytorchCompositionWrapper', 'PytorchMechanismWrapper', 'PytorchProjectionWrapper']


class PytorchCompositionWrapper(torch.nn.Module):
    """Wrapper for a Composition as a Pytorch Module
    Set up parameters of PyTorch model & information required for forward computation

    Handle nested compositions (flattened in infer_backpropagation_learning_pathways):
    Deal with Projections into or out of a nested Composition as follows:

     [      OUTER     ][                            NESTED                               ][     OUTER      ]
            \\learnable//   \\not learnable//                     \\not learnable//    \\learnable//
     ---> [Node] ----> [input_CIM] ~~~> [INPUT Node] ----> [OUTPUT Node] ~~~> [output_CIM] ----> [Node] --->
           sndr            rcvr          nested_rcvr         nested_sndr         sndr             rcvr
            ^--projection-->^                                                     ^---projection-->^
            ^----PytorchProjectionWrapper---->^                  ^----PytorchProjectionWrapper---->^
                     ENTRY                                                       EXIT

    Attributes
    ----------

    nodes : List[PytorchMechanismWrapper]

    projections_map : Dict[Projection, PytorchProjectionWrapper]
        keys are Projections in the Composition being wrapped, and keys are the ProjectionWrappers to which they
        are mapped (see above).

    """
    def __init__(self,
                 composition,
                 device,
                 outer_creator=None,
                 context=None):

        super(PytorchCompositionWrapper, self).__init__()

        from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition

        self.name = f"PytorchCompositionWrapper[{composition.name}]"

        self.node_wrappers = []  # can be PytorchMechanismWrapper or PytorchCompositionWrapper
        self.nodes_map = {} # maps Node (Mech or nested Comp) -> PytorchMechanismWrapper or PytorchCompositionWrapper

        self.projection_wrappers = [] # PytorchProjectionWrappers
        self.projections_map = {}  # maps Projections -> PytorchProjectionWrappers

        self.params = nn.ParameterList()
        self.device = device

        self._composition = composition

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
                pytorch_node = PytorchCompositionWrapper(node, device, outer_creator=self, context=context)
            # Wrap Mechanism
            else:
                pytorch_node = PytorchMechanismWrapper(node,
                                                       self._composition._get_node_index(node),
                                                       device,
                                                       context=context)
                pytorch_node._is_bias = any(input_port.default_input == DEFAULT_VARIABLE
                                            for input_port in node.input_ports)
            self.nodes_map[node] = pytorch_node
            self.node_wrappers.append(pytorch_node)

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
                        _assign_input_nodes(pytorch_node.node_wrappers)
            _assign_input_nodes(self.node_wrappers)

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


        # Flattening for forward() and AutodiffComposition._update_learning_parameters

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
        for node_wrapper in self.node_wrappers:
            if isinstance(node_wrapper, PytorchCompositionWrapper):
                # For copying weights back to PNL in AutodiffComposition._update_learning_parameters
                self.projections_map.update(node_wrapper.projections_map)
                # Not sure if this is needed, but just to be safe
                self.nodes_map.update(node_wrapper.nodes_map)
        # Purge nodes_map of entries for nested Compositions (their nodes are now in self.nodes_map)
        self.nodes_map = {k: v for k, v in self.nodes_map.items() if not isinstance(v, PytorchCompositionWrapper)}

        # Flatten projections so that they are all in the outer Composition and visible by _regenerate_paramlist
        #     needed for call to backward() in AutodiffComposition._update_learning_parameters
        # FIX: MAYBE SHOULD DO THIS AS LIST IS CREATED ABOVE?
        self.projection_wrappers = list(self.projections_map.values())

        composition.scheduler._delete_counts(c.execution_id)

        self._regenerate_paramlist()

    __deepcopy__ = get_deepcopy_with_shared()

    def _regenerate_paramlist(self):
        """Add Projection matrices to Pytorch Module's parameter list"""
        self.params = nn.ParameterList()
        for proj_wrapper in [p for p in self.projection_wrappers if not p._projection._exclude_from_autodiff]:
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
                    target_idx = self._composition.get_nodes_by_role(
                        NodeRole.INPUT).index(terminal_sequence[TARGET_MECHANISM])
                    node_target = builder.gep(model_input, [ctx.int32_ty(0), ctx.int32_ty(target_idx)])

                    # 2) Lookup desired output value
                    node_output = builder.gep(model_output, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(node._idx), ctx.int32_ty(0)])

                    tmp_loss = loss.gen_inject_lossfunc_call(
                        ctx, builder, loss_fn, node_output, node_target)

                    pnlvm.helpers.printf_float_array(
                        builder, node_target, prefix=f"{node}\ttarget:\t")
                    pnlvm.helpers.printf_float_array(
                        builder, node_output, prefix=f"{node}\tvalue:\t")

                    pnlvm.helpers.printf(
                        builder, f"{node}\tloss:\t%f\n", tmp_loss, override_debug=False)
                    builder.store(builder.fadd(builder.load(
                        total_loss), tmp_loss), total_loss)
                    loss_derivative = loss._gen_inject_loss_differential(
                        ctx, builder, node_output, node_target)
                    # compute δ_l = dσ/da ⊙ σ'(z)

                    gen_inject_vec_hadamard(
                        ctx, builder, activation_func_derivative, loss_derivative, error_val)

                else:
                    # We propagate error backwards from next layer
                    for proj_idx, proj in enumerate(node.efferents):
                        efferent_node = proj.receiver
                        efferent_node_error = error_dict[efferent_node]

                        weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, state, params)

                        if proj_idx == 0:
                            gen_inject_vxm_transposed(
                                ctx, builder, efferent_node_error, weights_llvmlite, error_val)
                        else:
                            new_val = gen_inject_vxm_transposed(
                                ctx, builder, efferent_node_error, weights_llvmlite)

                            gen_inject_vec_add(
                                ctx, builder, new_val, error_val, error_val)

                    gen_inject_vec_hadamard(
                        ctx, builder, activation_func_derivative, error_val, error_val)

                pnlvm.helpers.printf_float_array(
                    builder, activation_func_derivative, prefix=f"{node}\tdSigma:\t")
                pnlvm.helpers.printf_float_array(
                    builder, error_val, prefix=f"{node}\terror:\t")

        # 4) compute weight gradients
        for (node, err_val) in error_dict.items():
            if node in input_nodes:
                continue
            for proj in node.afferents:
                # get a_(l-1)
                afferent_node_activation = builder.gep(model_output, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(proj.sender._idx), ctx.int32_ty(0)])

                # get dimensions of weight matrix
                weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, state, params)
                pnlvm.helpers.printf_float_matrix(builder, weights_llvmlite, prefix= f"{proj.sender._mechanism} -> {proj.receiver._mechanism}\n", override_debug=False)
                # update delta_W
                node_delta_w = builder.gep(delta_w, [ctx.int32_ty(0), ctx.int32_ty(proj._idx)])

                dim_x, dim_y = proj.matrix.shape
                with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(dim_x), "weight_update_loop_outer") as (b1, weight_row):
                    with pnlvm.helpers.for_loop_zero_inc(b1, ctx.int32_ty(dim_y), "weight_update_loop_inner") as (b2, weight_column):
                        a_val = b2.load(b2.gep(afferent_node_activation,
                                               [ctx.int32_ty(0), weight_row]))
                        d_val = b2.load(b2.gep(err_val,
                                               [ctx.int32_ty(0), weight_column]))
                        old_val = b2.load(b2.gep(node_delta_w,
                                                 [ctx.int32_ty(0), weight_row, weight_column]))
                        new_val = b2.fadd(old_val, b2.fmul(a_val, d_val))
                        b2.store(new_val, b2.gep(node_delta_w,
                                                 [ctx.int32_ty(0), weight_row, weight_column]))

        pnlvm.helpers.printf(builder, "TOTAL LOSS:\t%.20f\n",
                             builder.load(total_loss), override_debug=False)
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
    def forward(self, inputs, context=None)->dict:
        """Forward method of the model for PyTorch and LLVM modes
        Returns a dictionary {output_node:value} of output values for the model
        """
        outputs = {}  # dict for storing values of terminal (output) nodes
        for current_exec_set in self.execution_sets:
            for node in current_exec_set:

                # If node is nested Composition (wrapped in PytorchCompositionWrapper),
                #    calls its forward method recursively
                if isinstance(node, PytorchCompositionWrapper):
                    node.forward(inputs=None)
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
                                variable.append(node.collate_afferents(i).squeeze(0))
                        if len(variable) == 1:
                            variable = variable[0]
                else:
                    # Node is not INPUT to Composition or BIAS, so get all input from its afferents
                    variable = node.collate_afferents()

                self.execute_node(node, variable, context)

                # Add entry to outputs dict for OUTPUT Nodes of pytorch representation
                #  note: these may be different than for actual Composition, as they are flattened
                if (node._mechanism in self._composition.get_nested_nodes_output_nodes_at_levels()):
                    outputs[node._mechanism] = node.value

        # NOTE: Context source needs to be set to COMMAND_LINE to force logs to update independently of timesteps
        # if not self._composition.is_nested:
        old_source = context.source
        context.source = ContextFlags.COMMAND_LINE
        self.log_values()
        self.log_weights()
        context.source = old_source

        return outputs

    def execute_node(self, node, variable, context=None):
        """Execute node and store the result in the node's value attribute
        Implemented as method (and includes context as arg) so that it can be overridden
        by subclasses of PytorchCompositionWrapper
        """
        node.execute(variable)

    def detach_all(self):
        for projection in self.projections_map.values():
            projection.matrix.detach()

    def copy_weights_to_psyneulink(self, context=None):
        for projection, pytorch_rep in self.projections_map.items():
            projection.parameters.matrix._set(
                pytorch_rep.matrix.detach().cpu().numpy(), context)
            projection.parameters.matrix._set(
                pytorch_rep.matrix.detach().cpu().numpy(), context)
            projection.parameter_ports['matrix'].parameters.value._set(
                pytorch_rep.matrix.detach().cpu().numpy(), context)
            assert True

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            proj_wrapper.log_matrix()

    def log_values(self):
        for node_wrapper in [n for n in self.node_wrappers if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()


class PytorchMechanismWrapper():
    """Wrapper for a Mechanism in a PytorchCompositionWrapper"""
    def __init__(self, mechanism, component_idx, device, context=None):
        self._mechanism = mechanism
        self.name = f"PytorchMechanismWrapper[{mechanism.name}]"
        self._idx = component_idx
        self._context = context

        self._is_input = False
        self._is_bias = False
        self.afferents = []
        self.efferents = []
        try:
            self.function = mechanism.function._gen_pytorch_fct(device, context)
        except:
            from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
            raise AutodiffCompositionError(
                f"Function {mechanism.function} is not currently supported by AutodiffComposition")

        self.value = None
        self._target_mechanism = None

    def add_efferent(self, efferent):
        assert efferent not in self.efferents
        self.efferents.append(efferent)

    def add_afferent(self, afferent):
        assert afferent not in self.afferents
        self.afferents.append(afferent)


    def collate_afferents(self, port=None):
        """Return weight-multiplied sum of afferent projections for input_port(s) of the Mechanism
        If there is only one input_port, return the sum of its afferents (for those in Composition)
        If there are multiple input_ports, return an array with the sum for each input_port
        # FIX: AUGMENT THIS TO SUPPORT InputPort's function
        """
        assert self.afferents,\
            f"PROGRAM ERROR: No afferents found for '{self._mechanism.name}' in AutodiffComposition"
        # Specific port is specified
        # FIX: USING _port_idx TO INDEX INTO sender.value GETS IT WRONG IF THE MECHANISM HAS AN OUTPUT PORT
        #      USED BY A PROJECTION NOT IN THE CURRENT COMPOSITION
        if port is not None:
            return sum(proj_wrapper.execute(proj_wrapper.sender.value[proj_wrapper._value_idx]).unsqueeze(0)
                                            for proj_wrapper in self.afferents
                                            if proj_wrapper._pnl_proj
                                            in self._mechanism.input_ports[port].path_afferents)
        # Has only one input_port
        elif len(self._mechanism.input_ports) == 1:
            # Get value corresponding to port from which each afferent projects
            return sum((proj_wrapper.execute(proj_wrapper.sender.value[proj_wrapper._value_idx]).unsqueeze(0)
                        for proj_wrapper in self.afferents))
        # Has multiple input_ports
        else:
            return [sum(proj_wrapper.execute(proj_wrapper.sender.value[proj_wrapper._value_idx]).unsqueeze(0)
                         for proj_wrapper in self.afferents
                         if proj_wrapper._pnl_proj in input_port.path_afferents)
                     for input_port in self._mechanism.input_ports]

    def execute(self, variable):
        """Execute Mechanism's function on variable, enforce result to be 2d, and assign to self.value"""
        if ((isinstance(variable, list) and len(variable) == 1)
            or (isinstance(variable, torch.Tensor) and len(variable.squeeze(0).shape) == 1)
                or isinstance(self._mechanism.function, LinearCombination)):
            # Enforce 2d on value of MechanismWrapper (using unsqueeze)
            # for single InputPort or if CombinationFunction (which reduces output to single item from multi-item input)
            if isinstance(variable, torch.Tensor):
                variable = variable.squeeze(0)
            self.value = self.function(variable).unsqueeze(0)
        else:
            # Make value 2d by creating list of values returned by function for each item in variable
            self.value = [self.function(variable[i].squeeze(0)) for i in range(len(variable))]
        return self.value

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

        pnlvm.helpers.printf_float_array(builder, builder.gep(mech_output, [ctx.int32_ty(0), ctx.int32_ty(0)]), prefix=f"{self} output:\n", override_debug=False)

        return mech_output

    def log_value(self):
        if self._mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.value.detach().cpu().numpy()
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
    """

    def __init__(self, projection,
                 pnl_proj,
                 component_idx,
                 port_idx, device,
                 sender=None,
                 receiver=None,
                 context=None):
        self.name = f"PytorchProjectionWrapper[{projection.name}]"
        self._projection = projection # Projection being wrapped (may *not* be the one being learned; see note above)
        self._pnl_proj = pnl_proj # Projection that directly projects to/from sender/receiver (see above)
        self._idx = component_idx     # Index of Projection in Composition's list of projections
        self._port_idx = port_idx     # Index of sender's port (used by LLVM)
        self._value_idx = 0           # Index of value in sender's value (used in collate_afferents)
        self.sender = sender          # PytorchMechanismWrapper to which Projection's sender is mapped
        self.receiver = receiver      # PytorchMechanismWrapper to which Projection's receiver is mapped
        self._context = context

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

    def execute(self, variable):
        return torch.matmul(variable, self.matrix)

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

        pnlvm.helpers.printf_float_array(builder, input_vec, prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} input:\n", override_debug=False)
        pnlvm.helpers.printf_float_matrix(builder, proj_matrix, prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} mat:\n", override_debug=False)
        pnlvm.helpers.printf_float_array(builder, output_vec, prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} output:\n", override_debug=False)

        return output_vec

    def __repr__(self):
        return "PytorchWrapper for: " +self._projection.__repr__()
