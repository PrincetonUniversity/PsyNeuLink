from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core import llvm as pnlvm
from llvmlite import ir
import numpy
import ctypes
import functools
import timeit
import pprint

debug_env = pnlvm.debug_env
from psyneulink.core.scheduling.time import TimeScale

try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False

import numpy as np

totaltime = 0
total_nonbin_time = 0
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
        
        self._id_map = {}
        self._afferent_id_map = {}
        for i in range(len(self.execution_sets)):
            # SKG: We have to add a counter to map components to an internal int id (for bin execute purposes, since there is no concept of a 'dict' in llvm)
            id_map_ct = 0
            self._id_map[i] = {}
            for component in self.execution_sets[i]:
                self._id_map[i][component] = id_map_ct
                id_map_ct += 1

                value = None  # the node's (its mechanism's) value
                biases = None  # the node's bias parameters
                function = self.function_creator(
                    component, execution_id)  # the node's function
                afferents = {}  # dict for keeping track of afferent nodes and their connecting weights
                self._afferent_id_map[component] = {}
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
                    afferent_id_map_ct = 0
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
                        self._afferent_id_map[component][input_node.component] = afferent_id_map_ct
                        afferent_id_map_ct += 1
                node_forward_info = [value, biases,
                                     function, afferents, component]
                # node_forward_info = [value, biases, function, afferents, value]

                self.component_to_forward_info[component] = node_forward_info

        # CW 12/3/18: this copies by reference so it only needs to be called during init, rather than
        # every time the weights are updated
        self.copy_weights_to_psyneulink(execution_id)

    # defines input type
    def _get_input_struct_type(self, ctx):  # Test case: {[1 x [2 x double]]}
        input_ty = [None]*len(self.execution_sets[0])
        for component in self.execution_sets[0]:
            component_id = self._id_map[0][component]
            input_ty[component_id] = ctx.convert_python_struct_to_llvm_ir(component.defaults.variable[0])
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
                vals[component_id] = value.numpy().ctypes.data_as(ctypes.c_void_p).value
            else:
                vals[component_id] = value.numpy()
        vals = pnlvm.execution._tupleize(vals)
        return inp_cty(*vals)

    
    def _get_data_struct_type(self, ctx):
        # Ensures that data struct is the same as the autodiffcomp
        return self._composition._get_data_struct_type(ctx)

    def _get_param_struct_type(self, ctx):
        param_list = [None]*(len(self.execution_sets)-1)
        for i in range(1,len(self.execution_sets)):
            param_list[i-1] = [None]*len(self._id_map[i])
            for component in self.execution_sets[i]:
                component_id = self._id_map[i][component]
                param_list[i-1][component_id] = [None]*len(self._afferent_id_map[component])
                afferents = self.component_to_forward_info[component][3]
                for node,weights in afferents.items():
                    input_component = node.component
                    input_component_id = self._afferent_id_map[component][input_component]
                    if "ref_pass" in debug_env:
                        param_list[i-1][component_id][input_component_id] = ir.types.IntType(64)
                    else:
                        param_list[i-1][component_id][input_component_id] = ctx.convert_python_struct_to_llvm_ir(weights.detach().numpy())
                param_list[i-1][component_id] = ir.types.LiteralStructType(param_list[i-1][component_id])
            param_list[i-1] = ir.types.LiteralStructType(param_list[i-1])
        struct_ty = ir.types.LiteralStructType(
            param_list)
        return struct_ty


    def _get_param_initializer(self):
        if self._cached_param_list is None:
            param_list = [None]*(len(self.execution_sets)-1)
            for i in range(1,len(self.execution_sets)):
                param_list[i-1] = [None]*len(self._id_map[i])
                for component in self.execution_sets[i]:
                    component_id = self._id_map[i][component]
                    param_list[i-1][component_id] = [None]*len(self._afferent_id_map[component])
                    afferents = self.component_to_forward_info[component][3]
                    for node,weights in afferents.items():
                        input_component = node.component
                        input_component_id = self._afferent_id_map[component][input_component]
                        if "ref_pass" in debug_env:
                            param_list[i-1][component_id][input_component_id] = weights.detach().numpy().ctypes.data_as(ctypes.c_void_p).value
                        else:
                            param_list[i-1][component_id][input_component_id] = weights.detach().numpy()
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
        

    def _get_context_struct(self):
        bin_func = self._bin_exec_func
        context_cty = bin_func.byref_arg_types[0]
        return context_cty(*(1,))

    def _get_context_struct_type(self, ctx):
        struct_ty = ir.types.LiteralStructType([
            ctx.int32_ty
        ])
        return struct_ty

    # generates llvm function for self.forward
    def _gen_llvm_function(self,extra_args=[],name=None):
        llvm_func = None
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            args = [ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_param_struct_type(self).as_pointer(),
                    ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_data_struct_type(self).as_pointer()
                    ]
            builder = ctx.create_llvm_function(args+extra_args, self,name)
            llvm_func = builder.function

            context, params, arg_in, arg_out = llvm_func.args[:len(args)]
            self._gen_llvm_forward_function_body(
                ctx, builder, params, context, arg_in, arg_out)
            builder.ret_void()
            llvm_func = builder.function
        self._forward_llvm_func = llvm_func
        return llvm_func

    def _gen_inject_vxm(self, ctx, builder, m1, m2, y, z):
        # create output vec
        output_vec = builder.alloca(ir.types.ArrayType(ir.types.DoubleType(), z))
        builtin = ctx.get_llvm_function("__pnl_builtin_vxm")
        builder.call(builtin, [builder.bitcast(m1,ctx.float_ty.as_pointer()), builder.bitcast(m2,ctx.float_ty.as_pointer()), ctx.int32_ty(y), ctx.int32_ty(z), builder.bitcast(output_vec,ctx.float_ty.as_pointer())])
        return output_vec

    def _gen_llvm_forward_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        out_t = arg_out.type.pointee
        if isinstance(out_t, pnlvm.ir.ArrayType) and isinstance(out_t.element, pnlvm.ir.ArrayType):
            assert len(out_t) == 1
        arg_out = builder.gep(arg_out,[ctx.int32_ty(0),
                                        ctx.int32_ty(0)])
        frozen_values = {}
        for i in range(len(self.execution_sets)):
            current_exec_set = self.execution_sets[i]

            for component in current_exec_set:
                frozen_values[component] = builder.alloca(
                    ctx.convert_python_struct_to_llvm_ir(component.defaults.variable[0]))
            for component in current_exec_set:
                component_id = self._id_map[i][component]
                biases = self.component_to_forward_info[component][1]
                value = frozen_values[component]
                afferents = self.component_to_forward_info[component][3]
                dim_x, dim_y = component.defaults.variable.shape
                if i == 0:
                    input_slot = self._composition._get_node_index(component)
                    #_y = None
                    #with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(dim_y), "zero") as (builder, _y):
                    for _y in range(0,dim_y):
                        _y = ctx.int32_ty(_y) 
                        cmp_arg = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(input_slot)])
                        
                        cmp_arg = builder.gep(cmp_arg, [ctx.int32_ty(0),_y])  # get input
                        res = self.bin_function_creator(
                            ctx, builder, component, builder.load(cmp_arg), execution_id=None)
                        builder.store(res, builder.gep(
                            value, [ctx.int32_ty(0), _y]))
                else:
                    # is_set keeps track of if we already have valid (i.e. non-garbage) values inside the alloc'd value
                    is_set = False
                    for input_node, weights in afferents.items():
                        input_value = frozen_values[input_node.component]

                        weights_np = weights.detach().numpy()
                        x, y = weights_np.shape

                        # We cast the ctype weights array to llvmlite pointer
                        afferent_node_id = self._afferent_id_map[component][input_node.component]
                        weights_llvmlite = builder.gep(params,[ctx.int32_ty(0), ctx.int32_ty(i-1),ctx.int32_ty(component_id),ctx.int32_ty(afferent_node_id)])
                        if "ref_pass" in debug_env:
                            mem_addr = builder.load(weights_llvmlite)
                            weights_llvmlite = builder.inttoptr(mem_addr,ir.types.ArrayType(ir.types.ArrayType(ir.types.DoubleType(), y),x).as_pointer())
                        weighted_inp = self._gen_inject_vxm(
                            ctx, builder, input_value, weights_llvmlite, x, y)
                        if is_set == False:
                            # copy weighted_inp to value
                            for _y in range(0, y):
                                loc = builder.gep(value, [ctx.int32_ty(
                                    0), ctx.int32_ty(_y)])
                                val_ptr = builder.gep(weighted_inp, [ctx.int32_ty(
                                    0), ctx.int32_ty(_y)])
                                builder.store(builder.load(val_ptr), loc)
                            is_set = True
                        else:
                            # add to value
                            for _y in range(0, y):
                                loc = builder.gep(value, [ctx.int32_ty(
                                    0), ctx.int32_ty(_y)])
                                val_ptr = builder.gep(weighted_inp, [ctx.int32_ty(
                                    0), ctx.int32_ty(_y)])
                                builder.store(builder.fadd(builder.load(
                                    loc), builder.load(val_ptr)), loc)

                    # Apply Activation Func to values
                    for _y in range(0, dim_y):
                        cmp_arg = builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(_y)])
                        res = self.bin_function_creator(
                            ctx, builder, component, builder.load(cmp_arg), execution_id=None)
                        builder.store(res, cmp_arg)
                    # TODO: Add bias to value
                    # if biases is not None:
                    #   value = value + biases

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    # We first grab which index we should insert into:
                    # Here, arr should be an array of arrays; each index correlates to self._id_map[i][component]
                    output_index = self._composition._get_node_index(component)
                    output_size = len(component.defaults.value[0])
                    # get ptr to first thing in struct (should be arr)
                    outer_arr_ptr = builder.gep(
                        arg_out, [ctx.int32_ty(0)])
                    for _y in range(0, output_size):
                        val_ptr = builder.gep(value, [ctx.int32_ty(
                            0), ctx.int32_ty(_y)])
                        self._output_forward_computation(
                            ctx, builder, arg_out, output_index, _y, builder.load(val_ptr))

    # inserts a value into the forward computation output array struct
    def _output_forward_computation(self, ctx, builder, arg_out, index, y, value):
        loc = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(
            index), ctx.int32_ty(0),ctx.int32_ty(y)])
        builder.store(value, loc)

    @property
    def _bin_exec_func(self):
        if self.__bin_exec_func is None:
            self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(self)
        return self.__bin_exec_func

    # performs forward computation for the model
    def forward(self, inputs, execution_id=None, do_logging=True,scheduler=None):
        global total_nonbin_time
        start_time = timeit.default_timer()
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
                scheduler.get_clock(execution_id)._increment_time(TimeScale.TIME_STEP)

        self.copy_outputs_to_psyneulink(outputs, execution_id)
        if do_logging:
            self.log_weights(execution_id)
        #print("EXPECTED:",outputs)
        total_nonbin_time += timeit.default_timer() - start_time
        #print("NOBIN:",total_nonbin_time)
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
            component.parameters.value._set(detached_value, execution_id, skip_history=True, skip_log=True)
            component.output_state.parameters.value._set(detached_value, execution_id, skip_history=True, skip_log=True)

    def log_weights(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._log_value(
                weights.detach().numpy(), execution_id, ContextFlags.COMMAND_LINE)

    # Helper method that functions the same as function_creator, but instead injects the computation to the builder
    def bin_function_creator(self, ctx, builder, node, x, execution_id=None):
        ir_dbl = ir.types.DoubleType()

        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(
                param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(
                    param_name, None)
            return ir_dbl(val[0])

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')

            ret = builder.fadd(builder.fmul(x, slope), intercept)
            return ret

        elif isinstance(node.function, Logistic):
            neg_one = ir_dbl(-1)
            gain = builder.fmul(neg_one, get_fct_param_value('gain'))
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            one = ir_dbl(1)
            exp = ctx.get_llvm_function("__pnl_builtin_exp")
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
            zero = ir_dbl(0)
            val = builder.fsub(x, bias)
            pred = builder.fcmp_ordered(">", val, zero)

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
