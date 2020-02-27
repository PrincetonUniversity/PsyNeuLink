import numpy as np
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.globals.utilities import NodeRole
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core import llvm as pnlvm
from psyneulink.library.compositions.compiledoptimizer import AdamOptimizer,SGDOptimizer
from psyneulink.library.compositions.compiledloss import MSELoss
import ctypes
from collections import deque
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.core.globals.keywords import TARGET_MECHANISM
from psyneulink.core.globals.log import LogCondition
debug_env = pnlvm.debug_env

try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False


__all__ = ['PytorchModelCreator']
# Class that is called to create pytorch representations of autodiff compositions based on their processing graphs.
# Called to do so when the composition is run for the first time.

# Note on notation: the "nodes" that are constantly referred to are vertices of the composition's processing
# graph. For general compositions, the component a node represents can be a mechanism or a nested composition,
# but for autodiff compositions, nodes always represent mechanisms. "Nodes" can be thought of as
# (but are not literally) mechanisms.


class PytorchModelCreator(torch.nn.Module):

    # sets up parameters of model & the information required for forward computation
    def __init__(self, processing_graph, param_init_from_pnl, execution_sets, device, context=None, composition=None):

        if not torch_available:
            raise Exception('Pytorch python module (torch) is not installed. Please install it with '
                            '`pip install torch` or `pip3 install torch`')

        super(PytorchModelCreator, self).__init__()

        self.execution_sets = execution_sets  # saved for use in the forward method
        # dict mapping PNL nodes to their forward computation information
        self.component_to_forward_info = {}
        # dict mapping PNL projections to Pytorch weights
        self.projections_to_pytorch_weights = {}
        # list that Pytorch optimizers will use to keep track of parameters
        self.params = nn.ParameterList()
        self.device = device

        self._composition = composition

        for i, current_exec_set in enumerate(self.execution_sets):
            self.execution_sets[i] = current_exec_set - set(composition.get_nodes_by_role(NodeRole.LEARNING))
            for component in current_exec_set:
                value = None  # the node's (its mechanism's) value
                function = self.function_creator(
                    component, context)  # the node's function
                afferents = {}  # dict for keeping track of afferent nodes and their connecting weights
                if param_init_from_pnl:
                    if component.parameters.value._get(context) is None:
                        value = torch.tensor(component.parameters.value.get(None)[0], device=self.device)
                    else:
                        value = torch.tensor(component.parameters.value._get(context)[0], device=self.device)
                else:
                    input_length = len(
                        component.input_ports[0].parameters.value.get(None))
                    value = torch.zeros(
                        input_length, device=self.device).double()

                # if `node` is not an origin node (origin nodes don't have biases or afferent connections)
                if i != 0:
                    # iterate over incoming projections and set up pytorch weights for them
                    for mapping_proj in component.path_afferents:

                        # get projection, sender node--pdb for projection
                        input_component = mapping_proj.sender.owner
                        input_node = processing_graph.comp_to_vertex[input_component]

                        # CW 12/3/18: Check this logic later
                        proj_matrix = mapping_proj.parameters.matrix._get(
                            context)
                        if proj_matrix is None:
                            proj_matrix = mapping_proj.parameters.matrix.get(
                                None)
                        # set up pytorch weights that correspond to projection. If copying params from psyneulink,
                        # copy weight values from projection. Otherwise, use random values.
                        if param_init_from_pnl:
                            weights = nn.Parameter(
                                    torch.tensor(
                                            proj_matrix.copy(),
                                            device=self.device).double(),
                                    requires_grad=mapping_proj.learnable)
                        else:
                            weights = nn.Parameter(torch.rand(
                                np.shape(proj_matrix), device=self.device).double())
                        afferents[input_node] = weights
                        self.params.append(weights)
                        self.projections_to_pytorch_weights[mapping_proj] = weights
                        
                node_forward_info = {
                    'value':value,
                    'function':function,
                    'afferents':afferents,
                    'component':component}

                self.component_to_forward_info[component] = node_forward_info

        # CW 12/3/18: this copies by reference so in theory it only needs to be called during init
        # but we call copy_weights_to_psyneulink after every run in order to make Autodiff less stateful
        self.copy_weights_to_psyneulink(context)


    # gets the index of 'afferent_node' in the forward info weights list
    def _get_afferent_node_index(self,node,afferent_node):
        forward_info_weights = self.component_to_forward_info[node]['afferents']
        for (idx,vertex) in enumerate(forward_info_weights):
            if vertex.component == afferent_node:
                return idx

    # returns a list of all efferent nodes and weights stored in component_to_forward_info
    def _get_afferent_nodes(self,node):
        forward_info_weights = self.component_to_forward_info[node]['afferents']
        return [(vertex.component,weights) for (vertex,weights) in forward_info_weights.items()]

    # generates llvm function for self.forward
    def _gen_llvm_function(self, *, tags:frozenset):
        llvm_func = None
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            args = [ctx.get_state_struct_type(self._composition).as_pointer(),
                    ctx.get_param_struct_type(self._composition).as_pointer(),
                    ctx.get_data_struct_type(self._composition).as_pointer()
                    ]
            builder = ctx.create_llvm_function(args, self)
            llvm_func = builder.function

            state, params, data = llvm_func.args
            if "learning" in tags:
                self._gen_llvm_training_function_body(ctx, builder, state, params, data)
            else:
                model_input = builder.gep(data, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(self._composition._get_node_index(self._composition.input_CIM))])
                self._gen_llvm_forward_function_body(ctx, builder, state, params, model_input, data)
            builder.ret_void()
        return llvm_func

    # gets a pointer for the weights matrix between node and afferent_node
    def _gen_get_node_weight_ptr(self, ctx, builder, params, node, afferent_node):
        node_idx = self._composition._get_node_index(node)
        forward_info_weights = self.component_to_forward_info[node]['afferents']
        afferent_node_index = self._get_afferent_node_index(node,afferent_node)
        projection = [i for i in afferent_node.efferents if i.receiver.owner == node][0]
        inner_projections = list(self._composition._inner_projections)
        projection_idx = inner_projections.index(projection)
        projection_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(projection_idx)])

        for (vertex,matrix) in forward_info_weights.items():
            if vertex.component == afferent_node:
                weight_matrix = matrix
                break
        dim_x,dim_y = weight_matrix.detach().numpy().shape
        node_weights = ctx.get_param_ptr(projection, builder, projection_params, "matrix")
        node_weights = builder.bitcast(node_weights, pnlvm.ir.types.ArrayType(
                 pnlvm.ir.types.ArrayType(ctx.float_ty, dim_y), dim_x).as_pointer())

        return node_weights,dim_x,dim_y

    def _gen_llvm_forward_function_body(self, ctx, builder, state, params, arg_in, arg_out, store_z_values=False):
        out_t = arg_out.type.pointee
        if isinstance(out_t, pnlvm.ir.ArrayType) and isinstance(out_t.element, pnlvm.ir.ArrayType):
            assert len(out_t) == 1
        z_values = {}
        for i, current_exec_set in enumerate(self.execution_sets):
            for component in current_exec_set:
                component_id = self._composition._get_node_index(component)
                value = self._get_output_value_ptr(ctx, builder, arg_out, component_id)
                afferents = self.component_to_forward_info[component]['afferents']

                mech_input_ty = ctx.get_input_struct_type(component)
                mech_input = builder.alloca(mech_input_ty)

                if i == 0:
                    # input struct provides data for input nodes
                    input_id = self._composition.get_nodes_by_role(NodeRole.INPUT).index(component)
                    cmp_arg = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(input_id)])
                    # node inputs are 2d arrays in a struct
                    input_ptr = builder.gep(mech_input, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
                    builder.store(builder.load(cmp_arg), input_ptr)
                else:
                    # is_set keeps track of if we already have valid (i.e. non-garbage) values inside the alloc'd value
                    is_set = False
                    for j, (input_vertex, weights) in enumerate(afferents.items()):
                        source_node = input_vertex.component
                        source_node_idx = self._composition._get_node_index(source_node)
                        input_value = self._get_output_value_ptr(ctx, builder, arg_out, source_node_idx)

                        # We cast the ctype weights array to llvmlite pointer
                        weights_llvmlite, _, _ = self._gen_get_node_weight_ptr(ctx, builder, params, component, source_node)
                        pnlvm.helpers.printf_float_matrix(builder, weights_llvmlite, prefix=f"{source_node} -> {component}\tweight:\n")
                        # node inputs are 2d arrays in a struct
                        input_ptr = builder.gep(mech_input, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(j)])
                        gen_inject_vxm(ctx, builder, input_value, weights_llvmlite, input_ptr)
                        if store_z_values:
                            if is_set == False:
                                # copy weighted_inp to value
                                gen_inject_vec_copy(ctx, builder, input_ptr, value)
                                is_set = True
                            else:
                                # add to value
                                gen_inject_vec_add(ctx, builder, input_ptr, value, value)

                    cmp_arg = value
                # Apply Activation Func to values
                if store_z_values is True:
                    z_values[component] = gen_inject_vec_copy(ctx, builder, cmp_arg)

                mech_func = ctx.import_llvm_function(component)
                mech_param = builder.gep(params, [ctx.int32_ty(0),
                                                  ctx.int32_ty(0),
                                                  ctx.int32_ty(component_id)])
                mech_state = builder.gep(state, [ctx.int32_ty(0),
                                                 ctx.int32_ty(0),
                                                 ctx.int32_ty(component_id)])
                mech_output = builder.gep(arg_out, [ctx.int32_ty(0),
                                                    ctx.int32_ty(0),
                                                    ctx.int32_ty(component_id)])
                builder.call(mech_func, [mech_param, mech_state,
                                         mech_input, mech_output])

                if store_z_values is True:
                    pnlvm.helpers.printf_float_array(builder, z_values[component], prefix=f"{component}\tforward input:\t")
                pnlvm.helpers.printf_float_array(builder, value, prefix=f"{component}\tforward output:\t", suffix="\t")
                pnlvm.helpers.printf(builder, "\n")
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

        context, params, model_output, optim_struct = llvm_func.args
        model_input = builder.gep(model_output, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(self._composition._get_node_index(self._composition.input_CIM))])
        # setup useful mappings
        input_nodes = composition.get_nodes_by_role(NodeRole.INPUT)
        output_nodes = composition.get_nodes_by_role(NodeRole.OUTPUT)

        # initialize optimizer params:
        delta_w = builder.gep(optim_struct, [ctx.int32_ty(0), ctx.int32_ty(optimizer._DELTA_W_NUM)])


        # 2) call forward computation
        z_values = self._gen_llvm_forward_function_body(
            ctx, builder, context, params, model_input, model_output, store_z_values=True)
        # 3) compute errors

        error_dict = {}
        backprop_queue = deque()
        for node in set(output_nodes) - set(self._composition.get_nodes_by_role(NodeRole.LEARNING)):
            backprop_queue.append(node)

        loss_fn = ctx.import_llvm_function(loss)
        total_loss = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(0),total_loss)

        while(len(backprop_queue) > 0):
            node = backprop_queue.popleft()
            if node in error_dict or not hasattr(node, "afferents") or node == composition.input_CIM or node in input_nodes:
                continue

            for (afferent_node,weights) in self._get_afferent_nodes(node):
                backprop_queue.append(afferent_node)

            node_idx = composition._get_node_index(node)

            activation_func_derivative_bin_func = ctx.import_llvm_function(self.bin_function_derivative_creator(ctx,node).name)
            activation_func_derivative = gen_inject_unary_function_call(ctx, builder, activation_func_derivative_bin_func, z_values[node])

            error_val = builder.alloca(z_values[node].type.pointee)

            error_dict[node] = error_val

            if node in set(output_nodes) - set(self._composition.get_nodes_by_role(NodeRole.LEARNING)):
                # We handle output layer here
                # compute  dC/da = a_l - y(x) (TODO: Allow other cost functions! This only applies to MSE)
                out_node_idx = output_nodes.index(node)
                node_target = self._get_target_value_ptr(ctx, builder, model_input, node )
                node_output = self._get_output_value_ptr(ctx, builder, model_output, node_idx)

                tmp_loss = loss.gen_inject_lossfunc_call(ctx, builder, loss_fn, node_output, node_target)

                pnlvm.helpers.printf_float_array(builder, node_target, prefix=f"{node}\ttarget:\t")
                pnlvm.helpers.printf_float_array(builder, node_output, prefix=f"{node}\tvalue:\t")

                pnlvm.helpers.printf(builder,f"{node}\tloss:\t%f\n",tmp_loss)
                builder.store(builder.fadd(builder.load(total_loss),tmp_loss),total_loss)
                loss_derivative = loss._gen_inject_loss_differential(ctx, builder, node_output, node_target)
                # compute δ_l = dσ/da ⊙ σ'(z)

                gen_inject_vec_hadamard(ctx, builder, activation_func_derivative, loss_derivative, error_val)

            else:
                # We propagate error backwards from next layer

                is_set = False

                # We calculate δ_(l-1) = sum (a_(l-1) W^T) ⊙ δ_l, where (l-1) is the current layer, l is layer of efferents, summed over all efferents
                efferents = [
                    proj.receiver._owner for proj in node.efferents]
                for efferent_node in set(efferents) - set(self._composition.get_nodes_by_role(NodeRole.LEARNING)):
                    efferent_node_error = error_dict[efferent_node]

                    weights_llvmlite, _, _ = self._gen_get_node_weight_ptr(ctx, builder, params, efferent_node, node)

                    if is_set is False:
                        gen_inject_vxm_transposed(ctx, builder, efferent_node_error, weights_llvmlite, error_val)
                        is_set = True
                    else:
                        new_val = gen_inject_vxm_transposed(ctx, builder, efferent_node_error, weights_llvmlite)

                        gen_inject_vec_add(ctx, builder, new_val, error_val, error_val)

                gen_inject_vec_hadamard(ctx, builder, activation_func_derivative, error_val, error_val)

            pnlvm.helpers.printf_float_array(builder, activation_func_derivative, prefix=f"{node}\tdSigma:\t")
            pnlvm.helpers.printf_float_array(builder, error_val, prefix=f"{node}\terror:\t")

        # 4) compute weight gradients
        for (node, err_val) in error_dict.items():
            if node in input_nodes:
                continue
            node_idx = self._composition._get_node_index(node)
            for (afferent_node,weight) in self._get_afferent_nodes(node):
                # get a_(l-1)
                afferent_node_idx = self._get_afferent_node_index(node,afferent_node)

                afferent_node_activation = self._get_output_value_ptr(ctx,builder,model_output,self._composition._get_node_index(afferent_node))

                # get dimensions of weight matrix
                weights_llvmlite,weights_dim_x,weights_dim_y = self._gen_get_node_weight_ptr(ctx, builder, params, node, afferent_node)
                # update delta_W
                node_delta_w = builder.gep(delta_w,[ctx.int32_ty(0),ctx.int32_ty(node_idx), ctx.int32_ty(afferent_node_idx)])

                with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_x), "weight_update_loop_outer") as (b1, weight_row):
                    with pnlvm.helpers.for_loop_zero_inc(b1, ctx.int32_ty(weights_dim_y), "weight_update_loop_inner") as (b2, weight_column):
                        a_val = b2.load(b2.gep(afferent_node_activation,
                                               [ctx.int32_ty(0), weight_row]))
                        d_val = b2.load(b2.gep(err_val,
                                               [ctx.int32_ty(0), weight_column]))
                        old_val = b2.load(b2.gep(node_delta_w,
                                                 [ctx.int32_ty(0), weight_row, weight_column]))
                        new_val = b2.fadd(old_val, b2.fmul(a_val, d_val))
                        b2.store(new_val, b2.gep(node_delta_w,
                                                 [ctx.int32_ty(0), weight_row, weight_column]))
                    
        builder.store(builder.fmul(ctx.float_ty(.5),builder.load(total_loss)),total_loss)
        pnlvm.helpers.printf(builder,"TOTAL LOSS:\t%f\n",builder.load(total_loss))
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
            raise Exception("LOSS TYPE",loss_type,"NOT SUPPORTED")

        optimizer_step_f = ctx.import_llvm_function(optimizer)
        optimizer_struct = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(2)])
        optimizer_zero_grad = ctx.import_llvm_function(optimizer.zero_grad(ctx).name)
        backprop = ctx.import_llvm_function(self._gen_llvm_training_backprop(ctx, optimizer, loss).name)
        
        # # FIXME: converting this call to inlined code results in
        # # significant longer compilation times
        builder.call(optimizer_zero_grad, [optimizer_struct])
        builder.call(backprop, [state, params, data,
                            optimizer_struct])
        builder.call(optimizer_step_f, [optimizer_struct, params])


    def _get_output_value_ptr(self, ctx, builder, arg_out, index):
        return builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(index), ctx.int32_ty(0)])
    
    def _get_target_value_ptr(self, ctx, builder, arg_in, output_node):
        terminal_sequence = self._composition._terminal_backprop_sequences[output_node]
        idx = self._composition.get_nodes_by_role(NodeRole.INPUT).index(terminal_sequence[TARGET_MECHANISM])
        return builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(idx)])

    def _get_compiled_optimizer(self):
        # setup optimizer
        optimizer_type = self._composition.optimizer_type
        if optimizer_type == 'adam':
            optimizer = AdamOptimizer(self,lr = self._composition.learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGDOptimizer(self,lr = self._composition.learning_rate)
        else:
            raise Exception("OPTIMIZER TYPE",optimizer_type,"NOT SUPPORTED")
        return optimizer
    # performs forward computation for the model
    @handle_external_context()
    def forward(self, inputs, context=None, do_logging=True, scheduler=None):
        outputs = {}  # dict for storing values of terminal (output) nodes

        for i, current_exec_set in enumerate(self.execution_sets):
            frozen_values = {}
            for component in current_exec_set:
                if NodeRole.LEARNING in self._composition.get_roles_by_node(component) or NodeRole.TARGET in self._composition.get_roles_by_node(component):
                    continue
                frozen_values[component] = self.component_to_forward_info[component]['value']
            for component in current_exec_set:
                if NodeRole.LEARNING in self._composition.get_roles_by_node(component) or NodeRole.TARGET in self._composition.get_roles_by_node(component):
                    continue
                # get forward computation info for current component
                function = self.component_to_forward_info[component]['function']
                afferents = self.component_to_forward_info[component]['afferents']
                # forward computation if we have origin node
                if i == 0:
                    value = function(inputs[component])
                # forward computation if we do not have origin node
                else:
                    value = torch.zeros(
                        len(component.input_ports[0].defaults.value), device=self.device).double()
                    for input_node, weights in afferents.items():
                        if input_node.component in current_exec_set:
                            input_value = frozen_values[input_node.component]
                        else:
                            input_value = self.component_to_forward_info[input_node.component]['value']
                        value += torch.matmul(input_value, weights)
                    value = function(value)
                # store the current value of the node
                self.component_to_forward_info[component]['value'] = value
                old_source = context.source
                context.source = ContextFlags.COMMAND_LINE
                detached_value = value.detach().cpu().numpy()
                component.parameters.value._set(detached_value, context)
                context.source = old_source

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    outputs[component] = value

        # Maybe need to comment this out!
        # self.copy_outputs_to_psyneulink(outputs, context)

        old_source = context.source
        context.source = ContextFlags.COMMAND_LINE
        self.log_weights(context)
        self.copy_outputs_to_psyneulink(outputs, context)
        context.source = old_source

        return outputs

    def detach_all(self):
        for component, info in self.component_to_forward_info.items():
            info['value'].detach_()

    def copy_weights_to_psyneulink(self, context=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._set(
                weights.detach().cpu().numpy(), context)

    def copy_outputs_to_psyneulink(self, outputs, context=None):
        for component, value in outputs.items():
            detached_value = value.detach().cpu().numpy()
            component.parameters.value._set(
                detached_value, context, skip_history=True, skip_log=True)
            component.output_port.parameters.value._set(
                detached_value, context, skip_history=True, skip_log=True)

    @handle_external_context()
    def log_weights(self, context=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            if projection.parameters.matrix.log_condition != LogCondition.OFF:
                projection.parameters.matrix._set(
                    weights.detach().cpu().numpy(), context)

    # Helper method that creates a bin func that returns the derivative of the function into the builder
    # FIXME: Add compiled derivative functions, and move these calls there
    @handle_external_context()
    def bin_function_derivative_creator(self, ctx, node, context=None):
        # first try to get cached func
        name = node.name + "_" + node.function.name + "_derivative"
        try:
            llvm_func = ctx.import_llvm_function(name)
            return llvm_func
        except Exception as e:
            pass


        # args: 1) ptr to input vector
        #       2) sizeof vector
        #       3) ptr to output vector
        float_ptr_ty = ctx.float_ty.as_pointer()
        args = [float_ptr_ty, ctx.int32_ty, float_ptr_ty]
        builder = ctx.create_llvm_function(args, self,name )
        llvm_func = builder.function
        llvm_func.attributes.add('alwaysinline')

        input_vector, dim, output_vector = llvm_func.args
        def get_fct_param_value(param_name):
            val = node.function._get_current_function_param(
                param_name, context)
            if val is None:
                val = node.function._get_current_function_param(
                    param_name, None)
            return ctx.float_ty(val[0])

        if isinstance(node.function, Linear): # f(x) = mx + b, f'(x) = m
            slope = get_fct_param_value('slope')
            def modify_value(x):
                return slope

        elif isinstance(node.function, Logistic):# f'(x) = f(x)(1-f(x))

            neg_one = ctx.float_ty(-1)
            gain = builder.fmul(neg_one, get_fct_param_value('gain'))
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            one = ctx.float_ty(1)
            exp = ctx.import_llvm_function("__pnl_builtin_exp")

            def modify_value(x):
                arg = builder.fadd(x, bias)
                arg = builder.fmul(gain, arg)
                arg = builder.fadd(arg, offset)

                f_x = builder.call(exp, [arg])
                f_x = builder.fadd(one, f_x)
                f_x = builder.fdiv(one, f_x)

                ret = builder.fsub(one ,f_x)
                ret = builder.fmul(f_x, ret)
                return ret

        else:
            raise Exception(f"Function type {node.function} is currently unsupported by compiled execution!")

        # do computations
        with pnlvm.helpers.for_loop_zero_inc(builder, dim, "derivative_loop") as (builder, iterator):
            val_ptr = builder.gep(input_vector,[iterator])
            val = builder.load(val_ptr)
            val = modify_value(val)
            output_location = builder.gep(output_vector,[iterator])
            builder.store(val,output_location)

        builder.ret_void()

        return llvm_func

    # helper method that identifies the type of function used by a node, gets the function
    # parameters and uses them to create a function object representing the function, then returns it
    def function_creator(self, node, context=None):
        def get_fct_param_value(param_name):
            val = node.function._get_current_function_param(
                param_name, context)
            if val is None:
                val = node.function._get_current_function_param(
                    param_name, Context(execution_id=None))
            return float(val)

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            return lambda x: x * slope + intercept

        elif isinstance(node.function, Logistic):
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            return lambda x: 1 / (1 + torch.exp(-gain * (x + bias) + offset))

        # if we have relu function (the only other kind of function allowed by the autodiff composition)
        else:
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            leak = get_fct_param_value('leak')
            return lambda x: (torch.max(input=(x - bias), other=torch.tensor([0], device=self.device).double()) * gain +
                              torch.min(input=(x - bias), other=torch.tensor([0], device=self.device).double()) * leak)

    # returns dict mapping psyneulink projections to corresponding pytorch weights. Pytorch weights are copied
    # over from tensors inside Pytorch's Parameter data type to numpy arrays (and thus copied to a different
    # memory location). This keeps the weights - and Pytorch in general - away from the user
    def get_weights_for_projections(self):
        weights_in_numpy = {}
        for projection, weights in self.projections_to_pytorch_weights.items():
            weights_in_numpy[projection] = weights.detach().cpu().numpy().copy()
        return weights_in_numpy
