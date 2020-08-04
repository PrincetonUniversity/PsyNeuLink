from psyneulink.core.compositions.composition import NodeRole
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core import llvm as pnlvm
from psyneulink.library.compositions.compiledoptimizer import AdamOptimizer, SGDOptimizer
from psyneulink.library.compositions.compiledloss import MSELoss
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.core.globals.keywords import TARGET_MECHANISM
from .pytorchcomponents import *

try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False

__all__ = ['PytorchModelCreator']

class PytorchModelCreator(torch.nn.Module):
    # sets up parameters of model & the information required for forward computation
    def __init__(self, composition, device, context=None):

        if not torch_available:
            raise Exception('Pytorch python module (torch) is not installed. Please install it with '
                            '`pip install torch` or `pip3 install torch`')

        super(PytorchModelCreator, self).__init__()

        # Maps Mechanism -> PytorchMechanismWrapper
        self.nodes = []
        self.component_map = {}

        # Maps Projections -> PytorchProjectionWrappers
        self.projections = []
        self.projection_map = {}

        self.params = nn.ParameterList()
        self.device = device
        self._composition = composition

        # Instantiate pytorch mechanisms
        for node in set(composition.nodes) - set(composition.get_nodes_by_role(NodeRole.LEARNING)):
            pytorch_node = PytorchMechanismWrapper(node, self._composition._get_node_index(node), device, context=context)
            self.component_map[node] = pytorch_node
            self.nodes.append(pytorch_node)

        # Instantiate pytorch projections
        for projection in composition.projections:
            if projection.sender.owner in self.component_map and projection.receiver.owner in self.component_map:
                proj_send = self.component_map[projection.sender.owner]
                proj_recv = self.component_map[projection.receiver.owner]

                port_idx = projection.sender.owner.output_ports.index(projection.sender)
                new_proj = PytorchProjectionWrapper(projection, list(self._composition._inner_projections).index(projection), port_idx, device, sender=proj_send, receiver=proj_recv, context=context)
                proj_send.add_efferent(new_proj)
                proj_recv.add_afferent(new_proj)
                self.projection_map[projection] = new_proj
                self.projections.append(new_proj)
                self.params.append(new_proj.matrix)

        # Setup execution sets
        # 1) Remove all learning-specific nodes
        self.execution_sets = [x - set(composition.get_nodes_by_role(NodeRole.LEARNING)) for x in composition.scheduler.run(context=context)]
        # 2) Convert to pytorchcomponent representation
        self.execution_sets = [{self.component_map[comp] for comp in s if comp in self.component_map} for s in self.execution_sets]
        # 3) Remove empty execution sets
        self.execution_sets = [x for x in self.execution_sets if len(x) > 0]

    # gets the index of 'afferent_node' in the forward info weights list
    def _get_afferent_node_index(self, node, afferent_node):
        return [proj.receiver for proj in node.afferents].index(self.component_map[afferent_node])

    def _get_afferent_nodes(self, node):
        forward_info_weights = self.component_map[node].afferents
        return [(vertex.component, weights) for (vertex, weights) in forward_info_weights.items()]

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
            model_input = builder.gep(data, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(self._composition._get_node_index(self._composition.input_CIM))])
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

                        weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, params)

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
                weights_llvmlite = proj._extract_llvm_matrix(ctx, builder, params)
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
        if loss_type == 'mse':
            loss = MSELoss()
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
        builder.call(optimizer_step_f, [optimizer_struct, params])

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

    # performs forward computation for the model
    @handle_external_context()
    def forward(self, inputs, context=None):
        outputs = {}  # dict for storing values of terminal (output) nodes
        for current_exec_set in self.execution_sets:
            for component in current_exec_set:
                if NodeRole.INPUT in self._composition.get_roles_by_node(component._mechanism):
                    component.execute(inputs[component._mechanism])
                else:
                    variable = component.collate_afferents()
                    component.execute(variable)

                # save value in output list if we're at a node in the last execution set
                if NodeRole.OUTPUT in self._composition.get_roles_by_node(component._mechanism):
                    outputs[component._mechanism] = component.value

        # NOTE: Context source needs to be set to COMMAND_LINE to force logs to update independantly of timesteps
        old_source = context.source
        context.source = ContextFlags.COMMAND_LINE
        self.log_values()
        self.log_weights()
        context.source = old_source

        return outputs

    def detach_all(self):
        for projection in self.projection_map.values():
            projection.matrix.detach()

    def copy_weights_to_psyneulink(self, context=None):
        for projection, pytorch_rep in self.projection_map.items():
            projection.parameters.matrix._set(
                pytorch_rep.matrix.detach().cpu().numpy(), context)
            projection.parameter_ports['matrix'].parameters.value._set(
                pytorch_rep.matrix.detach().cpu().numpy(), context)

    def copy_outputs_to_psyneulink(self, outputs, context=None):
        for component, value in outputs.items():
            detached_value = value.detach().cpu().numpy()
            component.parameters.value._set(
                detached_value, context, skip_history=True, skip_log=True)
            component.output_port.parameters.value._set(
                detached_value, context, skip_history=True, skip_log=True)

    def log_weights(self):
        for proj in self.projections:
            proj.log_matrix()

    def log_values(self):
        for node in self.nodes:
            node.log_value()
