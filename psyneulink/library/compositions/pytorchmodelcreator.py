import numpy as np
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.globals.utilities import NodeRole
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core import llvm as pnlvm
from llvmlite import ir
import numpy
import ctypes
import functools
import timeit
import pprint
from collections import deque

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
    def __init__(self, processing_graph, param_init_from_pnl, execution_sets, device, execution_id=None, composition=None):

        if not torch_available:
            raise Exception('Pytorch python module (torch) is not installed. Please install it with '
                            '`pip install torch` or `pip3 install torch`')

        super(PytorchModelCreator, self).__init__()

        self.execution_sets = execution_sets  # saved for use in the forward method
        # dict mapping PNL nodes to their forward computation information
        self.component_to_forward_info = {}
        # dict mapping PNL projections to Pytorch weights
        self.projections_to_pytorch_weights = {}
        # dict mapping PNL mechanisms to Pytorch biases
        self.mechanisms_to_pytorch_biases = {}
        # list that Pytorch optimizers will use to keep track of parameters
        self.params = nn.ParameterList()
        self.device = device
        self.__bin_exec_func = None
        self._forward_llvm_func = None
        self._cached_param_list = None
        self._cached_tupleized_param_list = None

        self._composition = composition

        for i in range(len(self.execution_sets)):
            # SKG: We have to add a counter to map components to an internal int id (for bin execute purposes, since there is no concept of a 'dict' in llvm)
            id_map_ct = 0
            for component in self.execution_sets[i]:
                id_map_ct += 1

                value = None  # the node's (its mechanism's) value
                biases = None  # the node's bias parameters
                function = self.function_creator(
                    component, execution_id)  # the node's function
                afferents = {}  # dict for keeping track of afferent nodes and their connecting weights
                if param_init_from_pnl:
                    if component.parameters.value._get(execution_id) is None:
                        value = torch.tensor(
                            component.parameters.value._get(None)[0])
                    else:
                        value = torch.tensor(
                            component.parameters.value._get(execution_id)[0])
                else:
                    input_length = len(
                        component.input_states[0].parameters.value._get(None))
                    value = torch.zeros(
                        input_length, device=self.device).double()

                # if `node` is not an origin node (origin nodes don't have biases or afferent connections)
                if i != 0:
                    # if not copying parameters from psyneulink, set up pytorch biases for node
                    if not param_init_from_pnl:
                        input_length = len(
                            component.input_states[0].parameters.value._get(None))
                        biases = nn.Parameter(torch.zeros(
                            input_length, device=self.device).double())
                        self.params.append(biases)
                        self.mechanisms_to_pytorch_biases[component] = biases
                    # iterate over incoming projections and set up pytorch weights for them
                    for k in range(len(component.path_afferents)):

                        # get projection, sender node--pdb for projection
                        mapping_proj = component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = processing_graph.comp_to_vertex[input_component]

                        # CW 12/3/18: Check this logic later
                        proj_matrix = mapping_proj.parameters.matrix._get(
                            execution_id)
                        if proj_matrix is None:
                            proj_matrix = mapping_proj.parameters.matrix._get(
                                None)

                        # set up pytorch weights that correspond to projection. If copying params from psyneulink,
                        # copy weight values from projection. Otherwise, use random values.
                        if param_init_from_pnl:
                            weights = nn.Parameter(torch.tensor(
                                proj_matrix.copy(), device=self.device).double())
                        else:
                            weights = nn.Parameter(torch.rand(
                                np.shape(proj_matrix), device=self.device).double())
                        afferents[input_node] = weights
                        self.params.append(weights)
                        self.projections_to_pytorch_weights[mapping_proj] = weights
                node_forward_info = [value, biases,
                                     function, afferents, component]
                # node_forward_info = [value, biases, function, afferents, value]

                self.component_to_forward_info[component] = node_forward_info

        # CW 12/3/18: this copies by reference so it only needs to be called during init, rather than
        # every time the weights are updated
        self.copy_weights_to_psyneulink(execution_id)


    # gets the index of 'afferent_node' in the forward info weights list
    def _get_afferent_node_index(self,node,afferent_node):
        forward_info_weights = self.component_to_forward_info[node][3]
        for (idx,vertex) in enumerate(forward_info_weights):
            if vertex.component == afferent_node:
                return idx
    
    # returns a list of all efferent nodes and weights stored in component_to_forward_info
    def _get_afferent_nodes(self,node):
        forward_info_weights = self.component_to_forward_info[node][3]
        return [(vertex.component,weights) for (vertex,weights) in forward_info_weights.items()]

    # defines input type
    def _get_input_struct_type(self, ctx):  # Test case: {[1 x [2 x double]]}
        input_ty = [None]*len(self.execution_sets[0])
        for component in self.execution_sets[0]:
            component_id = self._composition._get_node_index(component)
            input_ty[component_id] = ctx.convert_python_struct_to_llvm_ir(
                component.defaults.variable[0])
        struct_ty = ir.types.LiteralStructType(input_ty)
        return struct_ty

    # Converts tensor input to ctype
    def _get_input_struct(self, input):
        bin_func = self._bin_exec_func
        inp_cty = bin_func.byref_arg_types[2]
        vals = [None]*len(self.execution_sets[0])
        for component, value in input.items():
            component_id = self._id_map[0][component]
            if "ref_pass" in debug_env:
                vals[component_id] = value.numpy().ctypes.data_as(
                    ctypes.c_void_p).value
            else:
                vals[component_id] = value.numpy()
        vals = pnlvm.execution._tupleize(vals)
        return inp_cty(*vals)

    def _get_data_struct_type(self, ctx):
        # Ensures that data struct is the same as the autodiffcomp
        return self._composition._get_data_struct_type(ctx)

    # gets the params (i.e. weights and biases) structure for llvm
    # param struct shape:
    #   For weights:    [component_id][0][efferent_id_map_id]
    #   For bias:       [component_id][1]
    # The efferent id map is needed for weights because we must start from index 0, so each node keeps track of its efferents by iterating them
    def _get_param_struct_type(self, ctx):

        param_list = [None]*len(self._composition.nodes)
        for (node, forward_info) in self.component_to_forward_info.items():

            node_idx = self._composition._get_node_index(node)
            node_params = [None]

            # 1) setup weights
            afferents = forward_info[3]
            weight_array = [None]*len(afferents)
            for (afferent_vertex, weights) in afferents.items():
                afferent_node = afferent_vertex.component
                afferent_index = self._get_afferent_node_index(node,afferent_node)
                if "ref_pass" in debug_env:
                    afferent_weight = ir.types.IntType(64)
                else:
                    afferent_weight = ctx.convert_python_struct_to_llvm_ir(
                        weights.detach().numpy())
                weight_array[afferent_index] = afferent_weight
            node_params[0] = ir.types.LiteralStructType(weight_array)

            # 2) setup bias
            bias = forward_info[1]
            if bias is not None:
                if "ref_pass" in debug_env:
                    node_params += [ir.types.IntType(64)]
                else:
                    node_params += [ctx.convert_python_struct_to_llvm_ir(
                        bias.detach().numpy())]

            param_list[node_idx] = ir.types.LiteralStructType(node_params)
        struct_ty = ir.types.LiteralStructType(
            param_list)
        return struct_ty

    def _get_param_initializer(self):
        if self._cached_param_list is None:
            param_list = [None]*len(self._composition.nodes)
            for (node, forward_info) in self.component_to_forward_info.items():

                node_idx = self._composition._get_node_index(node)
                node_params = [None]

                # 1) initialize weights
                afferents = forward_info[3]
                weight_array = [None]*len(afferents)
                for (afferent_vertex, weights) in afferents.items():
                    afferent_node = afferent_vertex.component
                    afferent_index = self._get_afferent_node_index(node,afferent_node)
                    if "ref_pass" in debug_env: # this gets the actual memory address of the weights - is static (according to https://github.com/numpy/numpy/issues/13906)
                        afferent_weight = weights.detach().numpy().ctypes.data_as(ctypes.c_void_p).value
                    else:
                        afferent_weight = weights.detach().numpy()
                    weight_array[afferent_index] = afferent_weight
                node_params[0] = weight_array

                # 2) initialize bias
                bias = forward_info[1]
                if bias is not None:
                    if "ref_pass" in debug_env:
                        node_params += [bias.detach().numpy().ctypes.data_as(ctypes.c_void_p).value]
                    else:
                        node_params += [ctx.convert_python_struct_to_llvm_ir(
                            bias.detach().numpy())]
                
                param_list[node_idx] = node_params
            if "ref_pass" in debug_env:
                self._cached_param_list = pnlvm.execution._tupleize(param_list)
            else:
                self._cached_param_list = param_list
        if "ref_pass" in debug_env:
            return self._cached_param_list
        return pnlvm.execution._tupleize(self._cached_param_list)

    # Gets param struct for pytorch model (i.e. weights)
    def _get_param_struct(self):
        bin_func = self._bin_exec_func
        param_cty = bin_func.byref_arg_types[1]
        params = self._get_param_initializer()

        return params

    def _get_state_struct(self):
        bin_func = self._bin_exec_func
        context_cty = bin_func.byref_arg_types[0]
        return context_cty(*(1,))

    def _get_state_struct_type(self, ctx):
        return self._composition._get_state_struct_type(ctx)

    # generates llvm function for self.forward
    def _gen_llvm_function(self, extra_args=[], name=None):
        llvm_func = None
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            args = [ctx.get_state_struct_type(self).as_pointer(),
                    ctx.get_param_struct_type(self).as_pointer(),
                    ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_data_struct_type(self).as_pointer()
                    ]
            builder = ctx.create_llvm_function(args+extra_args, self, name)
            llvm_func = builder.function

            context, params, arg_in, arg_out = llvm_func.args[:len(args)]
            self._gen_llvm_forward_function_body(
                ctx, builder, context, params, arg_in, arg_out)
            builder.ret_void()
            llvm_func = builder.function
        self._forward_llvm_func = llvm_func
        return llvm_func

    def _gen_inject_vxm(self, ctx, builder, m1, m2, y, z, output_vec=None):
        # create output vec
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ir.types.DoubleType(), z))
        builtin = ctx.get_llvm_function("__pnl_builtin_vxm")
        builder.call(builtin, [builder.bitcast(m1, ctx.float_ty.as_pointer()), builder.bitcast(m2, ctx.float_ty.as_pointer(
        )), ctx.int32_ty(y), ctx.int32_ty(z), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec

    def _gen_inject_vec_copy(self,ctx,builder,vector,dim,output_vec = None):
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ir.types.DoubleType(), dim))
        builtin = ctx.get_llvm_function("__pnl_builtin_vec_copy")
        builder.call(builtin, [builder.bitcast(vector, ctx.float_ty.as_pointer()), ctx.int32_ty(dim), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec
    
    def _gen_inject_vec_add(self,ctx,builder,u,v,dim,output_vec = None):
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ir.types.DoubleType(), dim))
        builtin = ctx.get_llvm_function("__pnl_builtin_vec_add")
        builder.call(builtin, [builder.bitcast(u, ctx.float_ty.as_pointer()), builder.bitcast(v, ctx.float_ty.as_pointer()),ctx.int32_ty(dim), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec
    
    def _gen_inject_vec_sub(self,ctx,builder,u,v,dim,output_vec = None):
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ir.types.DoubleType(), dim))
        builtin = ctx.get_llvm_function("__pnl_builtin_vec_sub")
        builder.call(builtin, [builder.bitcast(u, ctx.float_ty.as_pointer()), builder.bitcast(v, ctx.float_ty.as_pointer()),ctx.int32_ty(dim), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec
    
    def _gen_inject_vec_hadamard(self,ctx,builder,u,v,dim,output_vec = None):
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ir.types.DoubleType(), dim))
        builtin = ctx.get_llvm_function("__pnl_builtin_vec_hadamard")
        builder.call(builtin, [builder.bitcast(u, ctx.float_ty.as_pointer()), builder.bitcast(v, ctx.float_ty.as_pointer()),ctx.int32_ty(dim), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec

    def _gen_inject_vxm_transposed(self, ctx, builder, m1, m2, y, z, output_vec=None):
        # create output vec
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ctx.float_ty, y))
        builtin = ctx.get_llvm_function("__pnl_builtin_vxm_transposed")
        builder.call(builtin, [builder.bitcast(m1, ctx.float_ty.as_pointer()), builder.bitcast(m2, ctx.float_ty.as_pointer(
        )), ctx.int32_ty(y), ctx.int32_ty(z), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec

    def _gen_inject_bin_function_call(self,ctx,builder,bin_func,vector,dim,output_vec=None):
        if output_vec is None:
            output_vec = builder.alloca(
                ir.types.ArrayType(ctx.float_ty, y))
        builder.call(bin_func, [builder.bitcast(vector, ctx.float_ty.as_pointer()), ctx.int32_ty(dim), builder.bitcast(output_vec, ctx.float_ty.as_pointer())])
        return output_vec
    # gets a pointer for the weights matrix between node and afferent_node
    def _gen_get_node_weight_pointer(self, ctx, builder,model_params,node,afferent_node):
        node_idx = self._composition._get_node_index(node)
        forward_info_weights = self.component_to_forward_info[node][3]
        afferent_node_index = self._get_afferent_node_index(node,afferent_node)
        for (vertex,matrix) in forward_info_weights.items():
            if vertex.component == afferent_node:
                weight_matrix = matrix
                break
        dim_x,dim_y = weight_matrix.detach().numpy().shape
        node_weights = builder.gep(model_params,[ctx.int32_ty(0),
                                                ctx.int32_ty(node_idx),
                                                ctx.int32_ty(0),
                                                ctx.int32_ty(afferent_node_index)])
        if "ref_pass" in debug_env:
            mem_addr = builder.load(node_weights)
            ctx.inject_printf(builder,"GOT WEIGHT MATRIX WITH ADDRESS: %ld (dimensionality: %d x %d )\n",mem_addr,ctx.int32_ty(dim_x),ctx.int32_ty(dim_y))
            node_weights = builder.inttoptr(mem_addr, ir.types.ArrayType(
                ir.types.ArrayType(ir.types.DoubleType(), dim_y), dim_x).as_pointer())
    
        return node_weights,dim_x,dim_y

    # gets a pointer for the bias vector for a node
    def _gen_get_node_bias_pointer(self, ctx, builder,model_params,node):
        node_idx = self._composition._get_node_index(node)
        forward_info_bias = self.component_to_forward_info[node][1]
        dim = forward_info_bias.detach().numpy().shape
        node_bias = builder.gep(model_params,[ctx.int32_ty(0),
                                                ctx.int32_ty(node_idx),
                                                ctx.int32_ty(1)])
        if "ref_pass" in debug_env:
            mem_addr = builder.load(node_bias)
            node_bias = builder.inttoptr(mem_addr,
                ir.types.ArrayType(ir.types.DoubleType(), dim).as_pointer())
    
        return node_bias,dim

    def _gen_llvm_forward_function_body(self, ctx, builder, _, params, arg_in, arg_out, store_z_values=False):
        out_t = arg_out.type.pointee
        if isinstance(out_t, pnlvm.ir.ArrayType) and isinstance(out_t.element, pnlvm.ir.ArrayType):
            assert len(out_t) == 1
        arg_out = builder.gep(arg_out, [ctx.int32_ty(0),
                                        ctx.int32_ty(0)])
        if store_z_values is True:
            z_values = {}
        for i in range(len(self.execution_sets)):
            current_exec_set = self.execution_sets[i]

            for component in current_exec_set:
                component_id = self._composition._get_node_index(component)
                biases = self.component_to_forward_info[component][1]
                value = self._get_output_index(
                    ctx, builder, arg_out, component_id)
                afferents = self.component_to_forward_info[component][3]
                dim_x, dim_y = component.defaults.variable.shape
                
                if i == 0:
                    input_slot = self._composition._get_node_index(component)
                    cmp_arg = builder.gep(
                            arg_in, [ctx.int32_ty(0), ctx.int32_ty(input_slot)])
                else:
                    # is_set keeps track of if we already have valid (i.e. non-garbage) values inside the alloc'd value
                    is_set = False
                    for input_vertex, weights in afferents.items():
                        input_node = input_vertex.component
                        input_node_idx = self._composition._get_node_index(
                            input_node)
                        # frozen_values[input_node.component]
                        input_value = self._get_output_index(
                            ctx, builder, arg_out, input_node_idx)

                        # We cast the ctype weights array to llvmlite pointer
                        weights_llvmlite, weights_dim_x, weights_dim_y = self._gen_get_node_weight_pointer(ctx,builder,params,component,input_node)
                        weighted_inp = self._gen_inject_vxm(
                            ctx, builder, input_value, weights_llvmlite, weights_dim_x, weights_dim_y)
                        if is_set == False:
                            # copy weighted_inp to value
                            self._gen_inject_vec_copy(ctx,builder,weighted_inp,weights_dim_y,value)
                            is_set = True
                        else:
                            # add to value
                            self._gen_inject_vec_add(ctx,builder,weighted_inp,value,weights_dim_y,value)

                    cmp_arg = value
                    
                # Apply Activation Func to values
                if store_z_values is True:
                    z_values[component] = self._gen_inject_vec_copy(ctx,builder,cmp_arg,dim_y)
                bin_func = ctx.get_llvm_function(self.bin_function_creator(ctx,component).name)
                self._gen_inject_bin_function_call(ctx,builder,bin_func,cmp_arg,dim_y,value)

                # TODO: Add bias to value
                # if biases is not None:
                #   value = value + biases
                if store_z_values is True:
                    ctx.inject_printf_float_array(
                        builder, z_values[component], dim_y, prefix=f"Z VALUE FOR {component} :\t")
                ctx.inject_printf_float_array(
                    builder, value, dim_y, prefix=f"FORWARD VALUE FOR {component} :\t")
        if store_z_values is True:
            return z_values

    # generates a function responsible for a single epoch of the training
    def _gen_llvm_training_epoch_function(self, ctx, composition, extra_args=[], name="autodiff_training_epoch"):
        learning_targets = pnlvm.ir.LiteralStructType([
            pnlvm.ir.IntType(32),  # idx of the node
            pnlvm.ir.IntType(32),  # idx of the node
            pnlvm.ir.IntType(64)
        ])
        args = [ctx.get_state_struct_type(self).as_pointer(),
                ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                ctx.get_data_struct_type(self).as_pointer(),
                learning_targets.as_pointer(),  # inputs
                learning_targets.as_pointer(),  # targets
                ctx.int32_ty.as_pointer()  # num_inputs
                ]
        builder = ctx.create_llvm_function(args+extra_args, self, name)
        llvm_func = builder.function
        for a in llvm_func.args:
            a.attributes.add('noalias')

        model_context, model_params, model_input, model_output, input_struct_ptr, target_struct_ptr, num_inputs = llvm_func.args[:len(
            args)]
        num_inputs = builder.load(num_inputs)
        # setup builtins
        vec_sub = ctx.get_llvm_function(
            "__pnl_builtin_vec_sub")
        vec_hadamard = ctx.get_llvm_function(
            "__pnl_builtin_vec_hadamard")
        vxm_transposed = ctx.get_llvm_function(
            "__pnl_builtin_vxm_transposed")

        # setup useful mappings
        input_nodes = composition.get_nodes_by_role(NodeRole.INPUT)
        output_nodes = composition.get_nodes_by_role(NodeRole.OUTPUT)

        node_value_ir_types = dict([(node, pnlvm.ir.ArrayType(
            pnlvm.ir.ArrayType(
                ctx.float_ty,
                len(node.defaults.value[0])
            ),
            1
        )) for node in composition.nodes])

        def _get_node_array_ptr(node, struct_ptr):
            node_idx = composition._get_node_index(node)
            array_ptr = builder.gep(
                struct_ptr, [ctx.int32_ty(node_idx), ctx.int32_ty(2)])
            array_ptr = builder.load(array_ptr)
            return builder.inttoptr(
                array_ptr, node_value_ir_types[node].as_pointer())

        node_input_arrays = dict([
            (node, _get_node_array_ptr(node, input_struct_ptr))
            for node in input_nodes
        ])

        node_target_arrays = dict([
            (node, _get_node_array_ptr(node, target_struct_ptr))
            for node in output_nodes
        ])
        
        # initialize delta_w matrices
        delta_w = {}
        for node in composition.nodes:
            delta_w[node] = {}
            
            for (afferent_node,matrix) in self._get_afferent_nodes(node):
                afferent_node_index = self._get_afferent_node_index(node,afferent_node)
                weights_llvmlite, weights_dim_x, weights_dim_y = self._gen_get_node_weight_pointer(ctx,builder,model_params,node,afferent_node)
                
                delta_w_array = builder.alloca(pnlvm.ir.ArrayType(
                    pnlvm.ir.ArrayType(
                        ctx.float_ty,
                        weights_dim_y
                    ),
                    weights_dim_x
                ))
                delta_w[node][afferent_node] = delta_w_array


                # zero weight array
                weight_row = None
                with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_x), "weight_zero_loop_outer") as (builder, weight_row):
                    weight_column = None
                    with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_y), "weight_zero_loop_inner") as (builder, weight_column):
                        builder.store(ctx.float_ty(0), builder.gep(delta_w_array, [
                                        ctx.int32_ty(0), weight_row, weight_column]))

        input_idx = None
        with pnlvm.helpers.for_loop_zero_inc(builder, num_inputs, "input_loop") as (builder, input_idx):
            ctx.inject_printf(builder, "INPUT %d\n", input_idx)
            # first we copy input values to data struct of input_CIM
            for node in input_nodes:
                node_idx = composition._get_node_index(node)
                _, node_dim = node.defaults.variable.shape
                node_input_array_ptr = builder.gep(node_input_arrays[node], [
                                                    input_idx, ctx.int32_ty(0)])
                self._gen_inject_vec_copy(ctx,builder,node_input_array_ptr,node_dim,model_input)
                
                ctx.inject_printf_float_array(
                    builder, node_input_array_ptr, node_dim, prefix=f"\tNODE {node_idx} INPUT: ")

            # 2) call forward computation
            z_values = self._gen_llvm_forward_function_body(
                ctx, builder, model_context, model_params, model_input, model_output, store_z_values=True)
            # 3) compute errors

            ctx.inject_printf(builder, "\tCOMPUTE ERR FOR INPUT %d\n", input_idx)

            error_dict = {}
            backprop_queue = deque()
            for node in output_nodes:
                backprop_queue.append(node)

            while(len(backprop_queue) > 0):
                node = backprop_queue.popleft()
                if node in error_dict or not hasattr(node, "afferents") or node == composition.input_CIM or node in input_nodes:
                    continue
                forward_info_weights = self.component_to_forward_info[node][3]

                for (afferent_node,weights) in self._get_afferent_nodes(node):
                    backprop_queue.append(afferent_node)

                node_idx = composition._get_node_index(node)
                _, node_dim = node.defaults.value.shape
                node_dim_ir = ctx.int32_ty(node_dim)

                # compute da/dz = dσ/dz 
                activation_func_derivative = builder.alloca(
                    pnlvm.ir.ArrayType(ctx.float_ty, node_dim))

                activation_func_derivative_bin_func = ctx.get_llvm_function(self.bin_function_derivative_creator(ctx,node).name)
                self._gen_inject_bin_function_call(ctx,builder,activation_func_derivative_bin_func,z_values[node],node_dim,activation_func_derivative)
                
                error_val = builder.alloca(
                        pnlvm.ir.ArrayType(ctx.float_ty, node_dim))
                
                error_dict[node] = error_val
                
                if node in output_nodes:
                    # We handle output layer here
                    # compute  dC/da = a_l - y(x) (TODO: Allow other cost functions! This only applies to MSE)
                    node_target = builder.gep(node_target_arrays[node], [
                                                input_idx, ctx.int32_ty(0)])
                    node_output = builder.gep(model_output, [ctx.int32_ty(
                        0), ctx.int32_ty(0), ctx.int32_ty(node_idx)])
                    node_output_target_diff = builder.alloca(
                        pnlvm.ir.ArrayType(ctx.float_ty, node_dim))

                    self._gen_inject_vec_sub(ctx,builder,node_output,node_target,node_dim,node_output_target_diff)

                    # compute δ_l = dσ/da ⊙ σ'(z)
                    self._gen_inject_vec_hadamard(ctx,builder,activation_func_derivative,node_output_target_diff,node_dim,error_val)

                else:
                    # We propagate error backwards from next layer
                    
                    is_set = False
                    
                    # We calculate δ_(l-1) = sum (a_(l-1) W^T) ⊙ δ_l, where (l-1) is the current layer, l is layer of efferents, summed over all efferents
                    efferents = [
                        proj.receiver._owner for proj in node.efferents]
                    for efferent_node in efferents:
                        efferent_node_error = error_dict[efferent_node]
                        
                        weights_llvmlite, weights_dim_x, weights_dim_y = self._gen_get_node_weight_pointer(ctx,builder,model_params,efferent_node,node)
                        
                        if is_set is False:
                            self._gen_inject_vxm_transposed(
                                ctx, builder, efferent_node_error, weights_llvmlite, weights_dim_x, weights_dim_y, error_val)
                            is_set = True
                        else:
                            new_val = self._gen_inject_vxm_transposed(
                                ctx, builder, efferent_node_error, weights_llvmlite, weights_dim_x, weights_dim_y)
                            
                            self._gen_inject_vec_add(ctx,builder,new_val,error_val,node_dim,error_val)

                    self._gen_inject_vec_hadamard(ctx,builder,activation_func_derivative,error_val,node_dim,error_val)
                
                ctx.inject_printf_float_array(builder,activation_func_derivative,node_dim,prefix=f"dSIGMA VALUE FOR {node}:\t")
                ctx.inject_printf_float_array(builder,error_val,node_dim,prefix=f"ERROR VALUE FOR {node}:\t")
            
            # 4) compute weight errors
            for (node, err_val) in error_dict.items():
                if node in input_nodes:
                    continue

                for (afferent_node,weight) in self._get_afferent_nodes(node):
                    _, afferent_node_dim = afferent_node.defaults.variable.shape
                    # get a_(l-1)
                    afferent_node_idx = composition._get_node_index(
                        afferent_node)
                    afferent_node_activation = builder.gep(model_output, [ctx.int32_ty(
                        0), ctx.int32_ty(0), ctx.int32_ty(afferent_node_idx)])

                    # get dimensions of weight matrix
                    _,weights_dim_x,weights_dim_y = self._gen_get_node_weight_pointer(ctx,builder,model_params,node,afferent_node)
                    # update delta_W
                    node_delta_w = delta_w[node][afferent_node]
                    weight_row = None
                    with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_x), "weight_update_loop_outer") as (builder, weight_row):
                        weight_column = None
                        with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_y), "weight_update_loop_inner") as (builder, weight_column):
                            a_val = builder.load(builder.gep(afferent_node_activation, [
                                                    ctx.int32_ty(0), ctx.int32_ty(0), weight_row]))
                            d_val = builder.load(builder.gep(
                                err_val, [ctx.int32_ty(0), weight_column]))
                            old_val = builder.load(builder.gep(node_delta_w, [
                                                    ctx.int32_ty(0), weight_row, weight_column]))
                            new_val = builder.fadd(
                                old_val, builder.fmul(a_val, d_val))
                            builder.store(new_val, builder.gep(node_delta_w, [
                                            ctx.int32_ty(0), weight_row, weight_column]))
        # now we update the weights
        ctx.inject_printf(builder,"\tUPDATING WEIGHTS\n")
        for node in delta_w:
            _, node_dim = node.defaults.value.shape

            for (afferent_node, delta_w_matrix) in delta_w[node].items():
                weights_llvmlite, weights_dim_x, weights_dim_y = self._gen_get_node_weight_pointer(ctx,builder,model_params,node,afferent_node)
                
                weight_row = None
                with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_x), "delta_w_loop_outer") as (builder, weight_row):
                    weight_column = None
                    with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_y), "delta_w_loop_inner") as (builder, weight_column):
                        old_val = builder.load(builder.gep(
                            weights_llvmlite, [ctx.int32_ty(0), weight_row, weight_column]))
                        new_val = builder.load(builder.gep(
                            delta_w_matrix, [ctx.int32_ty(0), weight_row, weight_column]))
                        new_val = builder.fmul(ctx.float_ty(self._composition.learning_rate), new_val)
                        new_val = builder.fsub(old_val, new_val)
                        builder.store(new_val, builder.gep(weights_llvmlite, [
                                        ctx.int32_ty(0), weight_row, weight_column]))
        builder.ret_void()
        llvm_func = builder.function
        return llvm_func

    def _gen_llvm_training_function_body(self, ctx, composition, builder, context, params, comp_in, data_arg, cond):
        # 1) Setup autodiff learning stuff
        # if "ref_pass" not in debug_env:
        #    raise Exception("ref_pass must be enabled in debug!")
        # gets a reference to the autodiff_stimuli_struct from params
        autodiff_stimuli_struct = builder.gep(
            params, [ctx.int32_ty(0), ctx.int32_ty(3)])

        learning_targets = pnlvm.ir.LiteralStructType([
            pnlvm.ir.IntType(32),  # idx of the node
            pnlvm.ir.IntType(32),  # dimensionality
            pnlvm.ir.IntType(64),  # array of input/output values
        ])
        learning_params = pnlvm.ir.LiteralStructType([
            pnlvm.ir.IntType(32),  # epochs
            pnlvm.ir.IntType(32),  # number of targets/inputs to train with
            pnlvm.ir.IntType(32),  # number target nodes
            pnlvm.ir.IntType(64),  # addr of beginning of target struct arr
            pnlvm.ir.IntType(32),  # number input nodes
            pnlvm.ir.IntType(64),  # addr of beginning of input struct arr
        ])

        epochs = builder.load(builder.gep(autodiff_stimuli_struct, [
                              ctx.int32_ty(0), ctx.int32_ty(0)]))
        num_inputs = builder.gep(autodiff_stimuli_struct, [
            ctx.int32_ty(0), ctx.int32_ty(1)])

        num_target_structs = builder.load(builder.gep(
            autodiff_stimuli_struct, [ctx.int32_ty(0), ctx.int32_ty(2)]))
        target_struct_ptr = builder.load(builder.gep(
            autodiff_stimuli_struct, [ctx.int32_ty(0), ctx.int32_ty(3)]))
        target_struct_ptr = builder.inttoptr(
            target_struct_ptr, learning_targets.as_pointer())

        num_input_structs = builder.load(builder.gep(
            autodiff_stimuli_struct, [ctx.int32_ty(0), ctx.int32_ty(4)]))
        input_struct_ptr = builder.load(builder.gep(
            autodiff_stimuli_struct, [ctx.int32_ty(0), ctx.int32_ty(5)]))
        input_struct_ptr = builder.inttoptr(
            input_struct_ptr, learning_targets.as_pointer())

        if "const_params" in debug_env:
            const_params = params.type.pointee(
                composition._get_param_initializer(None))
            params = builder.alloca(const_params.type, name="const_params_loc")
            builder.store(const_params, params)

        if "alloca_data" in debug_env:
            data = builder.alloca(data_arg.type.pointee)
            data_vals = builder.load(data_arg)
            builder.store(data_vals, data)
        else:
            data = data_arg

        input_cim_idx = composition._get_node_index(composition.input_CIM)
        model_context = context
        model_params = builder.gep(params, [ctx.int32_ty(0),
                                            ctx.int32_ty(2)])

        # Extract the input that should be inserted into the model
        model_input = builder.gep(data, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(input_cim_idx)])
        model_output = builder.gep(data, [ctx.int32_ty(0),
                                          ])
        epoch_idx = None

        epoch_func = self._gen_llvm_training_epoch_function(ctx,composition)
        epoch_func = ctx.get_llvm_function(epoch_func.name)
        with pnlvm.helpers.for_loop_zero_inc(builder, epochs, "epoch_loop") as (builder, epoch_idx):
            ctx.inject_printf(builder, "EPOCH %d\n", epoch_idx)
            builder.call(epoch_func, [model_context, model_params, model_input,
                                      model_output, input_struct_ptr, target_struct_ptr, num_inputs])

    # inserts a value into the forward computation output array struct
    def _output_forward_computation(self, ctx, builder, arg_out, index, y, value):
        loc = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(
            index), ctx.int32_ty(0), ctx.int32_ty(y)])
        builder.store(value, loc)

    def _get_output_index(self, ctx, builder, arg_out, index):
        return builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(index), ctx.int32_ty(0)])

    @property
    def _bin_exec_func(self):
        if self.__bin_exec_func is None:
            self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(self)
        return self.__bin_exec_func

    # performs forward computation for the model
    def forward(self, inputs, execution_id=None, do_logging=True, scheduler=None):
        outputs = {}  # dict for storing values of terminal (output) nodes

        for i in range(len(self.execution_sets)):
            current_exec_set = self.execution_sets[i]
            frozen_values = {}
            for component in current_exec_set:
                frozen_values[component] = self.component_to_forward_info[component][0]
            for component in current_exec_set:

                # get forward computation info for current component
                biases = self.component_to_forward_info[component][1]
                function = self.component_to_forward_info[component][2]
                afferents = self.component_to_forward_info[component][3]

                # forward computation if we have origin node
                if i == 0:
                    value = function(inputs[component])
                # forward computation if we do not have origin node
                else:
                    value = torch.zeros(
                        len(component.input_states[0].defaults.value), device=self.device).double()
                    for input_node, weights in afferents.items():
                        if input_node.component in current_exec_set:
                            input_value = frozen_values[input_node.component]
                        else:
                            input_value = self.component_to_forward_info[input_node.component][0]
                        value += torch.matmul(input_value, weights)
                    if biases is not None:
                        value = value + biases
                    value = function(value)

                # store the current value of the node
                self.component_to_forward_info[component][0] = value
                if do_logging:
                    detached_value = value.detach().numpy()
                    component.parameters.value._log_value(
                        detached_value, execution_id, ContextFlags.COMMAND_LINE)

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    outputs[component] = value

            if scheduler is not None:
                scheduler.get_clock(execution_id)._increment_time(
                    TimeScale.TIME_STEP)

        self.copy_outputs_to_psyneulink(outputs, execution_id)
        if do_logging:
            self.log_weights(execution_id)
        return outputs

    def detach_all(self):
        for component, info in self.component_to_forward_info.items():
            info[0].detach_()
            if info[1] is not None:
                info[1].detach_()

    def copy_weights_to_psyneulink(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._set(
                weights.detach().numpy(), execution_id)

    def copy_outputs_to_psyneulink(self, outputs, execution_id=None):
        for component, value in outputs.items():
            detached_value = value.detach().numpy()
            component.parameters.value._set(
                detached_value, execution_id, skip_history=True, skip_log=True)
            component.output_state.parameters.value._set(
                detached_value, execution_id, skip_history=True, skip_log=True)

    def log_weights(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._log_value(
                weights.detach().numpy(), execution_id, ContextFlags.COMMAND_LINE)

    # Helper method that functions the same as function_creator, but instead injects the computation to the builder
    def bin_function_creator(self, ctx, node, execution_id=None):
        # first try to get cached func
        name = node.name+"_"+node.function.name
        try:
            llvm_func = ctx.get_llvm_function(name)
            return llvm_func
        except Exception as e:
            pass
        float_ty = ctx.float_ty
        float_ptr_ty = float_ty.as_pointer()
        int32_ty = ctx.int32_ty

        # args: 1) ptr to input vector
        #       2) sizeof vector
        #       3) ptr to output vector
        
        args = [float_ptr_ty, ctx.int32_ty, float_ptr_ty]
        
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        llvm_func.attributes.add('alwaysinline')
        input_vector, dim, output_vector = llvm_func.args

        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(
                param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(
                    param_name, None)
            return float_ty(val[0])

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            def modify_value(x):
                ret = builder.fadd(builder.fmul(x, slope), intercept)
                return ret

        elif isinstance(node.function, Logistic):
            neg_one = float_ty(-1)
            gain = builder.fmul(neg_one, get_fct_param_value('gain'))
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            one = float_ty(1)
            exp = ctx.get_llvm_function("__pnl_builtin_exp")
            def modify_value(x):
                arg = builder.fadd(
                    builder.fmul(
                        gain, builder.fsub(x, bias)
                    ), offset)

                ret = builder.fdiv(one, builder.fadd(
                    one, builder.call(exp, [arg])))
                return ret

        # if we have relu function (the only other kind of function allowed by the autodiff composition)
        else:
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            leak = get_fct_param_value('leak')
            zero = float_ty(0)
            def modify_value(x):
                val = builder.fsub(x, bias)
                pred = builder.fcmp_ordered(">", val, zero)
                then = None
                otherwise = None
                with builder.if_else(pred) as (then, otherwise):
                    with then:
                        max = val
                        min = zero
                    with otherwise:
                        max = zero
                        min = val

                ret = builder.fadd(
                    builder.fmul(max, gain),
                    builder.fmul(min, leak))
                return ret
        
        #do computations
        iterator = None
        with pnlvm.helpers.for_loop_zero_inc(builder, dim, "function_loop") as (builder, iterator):
            val_ptr = builder.gep(input_vector,[iterator])
            val = builder.load(val_ptr)
            val = modify_value(val)
            output_location = builder.gep(output_vector,[iterator])
            builder.store(val,output_location)
        
        builder.ret_void()

        return llvm_func
    # Helper method that creates a bin func that returns the derivative of the function into the builder
    def bin_function_derivative_creator(self, ctx, node, execution_id=None):
        # first try to get cached func
        name = node.name+"_"+node.function.name+"_derivative"
        try:
            llvm_func = ctx.get_llvm_function(name)
            return llvm_func
        except Exception as e:
            pass

        float_ty = ctx.float_ty
        float_ptr_ty = float_ty.as_pointer()
        int32_ty = ctx.int32_ty

        # args: 1) ptr to input vector
        #       2) sizeof vector
        #       3) ptr to output vector
        args = [float_ptr_ty, ctx.int32_ty, float_ptr_ty]
        builder = ctx.create_llvm_function(args, self,name )
        llvm_func = builder.function
        llvm_func.attributes.add('alwaysinline')

        input_vector, dim, output_vector = llvm_func.args
        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(
                param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(
                    param_name, None)
            return float_ty(val[0])

        if isinstance(node.function, Linear): # f(x) = mx + b, f'(x) = m
            slope = get_fct_param_value('slope')
            def modify_value(x):
                return slope

        elif isinstance(node.function, Logistic):# f'(x) = f(x)(1-f(x))

            neg_one = float_ty(-1)
            gain = builder.fmul(neg_one, get_fct_param_value('gain'))
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            one = float_ty(1)
            exp = ctx.get_llvm_function("__pnl_builtin_exp")
            
            def modify_value(x):
                arg = builder.fadd(
                    builder.fmul(
                        gain, builder.fsub(x, bias)
                    ), offset)

                ret = builder.fdiv(one, builder.fadd(
                    one, builder.call(exp, [arg])))
                ret = builder.fmul(ret,builder.fsub(float_ty(1),ret))
                return ret

        # if we have relu function (the only other kind of function allowed by the autodiff composition)
        else:
            raise Exception(f"Function type {node.function} is currently unsupported by compiled execution!")
        
        # do computations
        iterator = None
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
    def function_creator(self, node, execution_id=None):
        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(
                param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(
                    param_name, None)
            return float(val)

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            return lambda x: x * slope + intercept

        elif isinstance(node.function, Logistic):
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))

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
            weights_in_numpy[projection] = weights.detach().numpy().copy()
        return weights_in_numpy

    # returns dict mapping psyneulink mechanisms to corresponding pytorch biases, the same way as the above function.
    # If composition is initialized with "param_init_from_PNL" set to true, then no biases are created in Pytorch,
    # and when called, this function returns an empty list.
    def get_biases_for_mechanisms(self):
        biases_in_numpy = {}
        for mechanism, biases in self.mechanisms_to_pytorch_biases.items():
            biases_in_numpy[mechanism] = biases.detach().numpy().copy()
        return biases_in_numpy
