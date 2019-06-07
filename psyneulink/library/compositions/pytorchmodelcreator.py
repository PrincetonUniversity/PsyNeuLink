from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core import llvm as pnlvm
from llvmlite import ir
import functools
try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False

import numpy as np

__all__ = ['PytorchModelCreator']
# Class that is called to create pytorch representations of autodiff compositions based on their processing graphs.
# Called to do so when the composition is run for the first time.

# Note on notation: the "nodes" that are constantly referred to are vertices of the composition's processing
# graph. For general compositions, the component a node represents can be a mechanism or a nested composition,
# but for autodiff compositions, nodes always represent mechanisms. "Nodes" can be thought of as
# (but are not literally) mechanisms.


class PytorchModelCreator(torch.nn.Module):

    # sets up parameters of model & the information required for forward computation
    def __init__(self, processing_graph, param_init_from_pnl, execution_sets, device, execution_id=None):

        if not torch_available:
            raise Exception('Pytorch python module (torch) is not installed. Please install it with '
                            '`pip install torch` or `pip3 install torch`')

        super(PytorchModelCreator, self).__init__()

        self.execution_sets = execution_sets  # saved for use in the forward method
        self.component_to_forward_info = {}  # dict mapping PNL nodes to their forward computation information
        self.projections_to_pytorch_weights = {}  # dict mapping PNL projections to Pytorch weights
        self.mechanisms_to_pytorch_biases = {}  # dict mapping PNL mechanisms to Pytorch biases
        self.params = nn.ParameterList()  # list that Pytorch optimizers will use to keep track of parameters
        self.device = device


        # Get first component in network
        inp_comp = self.execution_sets[0]
        # Get any mechanism
        inp_mech = min(inp_comp)

        
        # Get last component in network
        out_comp = self.execution_sets[len(self.execution_sets)-1]
        # Get any mechanism
        out_mech = min(out_comp)

        self.defaults = { # each of these should be of type array([[ ]]) for some reason
            # Assign input shape
            'variable':inp_mech.variable,
            # Assign output shape
            'value':out_mech.value
        }
        for i in range(len(self.execution_sets)):
            for component in self.execution_sets[i]:

                value = None  # the node's (its mechanism's) value
                biases = None  # the node's bias parameters
                function = self.function_creator(component, execution_id)  # the node's function
                afferents = {}  # dict for keeping track of afferent nodes and their connecting weights
                with pnlvm.LLVMBuilderContext.get_global() as ctx:
                    bin_func = self.bin_function_creator(ctx,component, execution_id)

                if param_init_from_pnl:
                    if component.parameters.value.get(execution_id) is None:
                        value = torch.tensor(component.parameters.value.get(None)[0])
                    else:
                        value = torch.tensor(component.parameters.value.get(execution_id)[0])
                else:
                    input_length = len(component.input_states[0].parameters.value.get(None))
                    value = torch.zeros(input_length, device=self.device).double()

                # if `node` is not an origin node (origin nodes don't have biases or afferent connections)
                if i != 0:
                    # if not copying parameters from psyneulink, set up pytorch biases for node
                    if not param_init_from_pnl:
                        input_length = len(component.input_states[0].parameters.value.get(None))
                        biases = nn.Parameter(torch.zeros(input_length, device=self.device).double())
                        self.params.append(biases)
                        self.mechanisms_to_pytorch_biases[component] = biases

                    # iterate over incoming projections and set up pytorch weights for them
                    for k in range(len(component.path_afferents)):

                        # get projection, sender node for projection
                        mapping_proj = component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = processing_graph.comp_to_vertex[input_component]

                        # CW 12/3/18: Check this logic later
                        proj_matrix = mapping_proj.parameters.matrix.get(execution_id)
                        if proj_matrix is None:
                            proj_matrix = mapping_proj.parameters.matrix.get(None)

                        # set up pytorch weights that correspond to projection. If copying params from psyneulink,
                        # copy weight values from projection. Otherwise, use random values.
                        if param_init_from_pnl:
                            weights = nn.Parameter(torch.tensor(proj_matrix.copy(), device=self.device).double())
                        else:
                            weights = nn.Parameter(torch.rand(np.shape(proj_matrix), device=self.device).double())
                        afferents[input_node] = weights
                        self.params.append(weights)
                        self.projections_to_pytorch_weights[mapping_proj] = weights

                node_forward_info = [value, biases, function, afferents,bin_func]
                # node_forward_info = [value, biases, function, afferents, value]

                self.component_to_forward_info[component] = node_forward_info

        # CW 12/3/18: this copies by reference so it only needs to be called during init, rather than
        # every time the weights are updated
        self.copy_weights_to_psyneulink(execution_id)


        # TEST METHOD:
        self._gen_llvm_forward_function()
    # defines input type
    def _get_input_struct_type(self,ctx):
         return ir.types.LiteralStructType([
             ctx.convert_python_struct_to_llvm_ir(self.defaults['variable'])
         ])
    
    def _get_output_struct_type(self,ctx):
        return ir.types.LiteralStructType([
            ctx.convert_python_struct_to_llvm_ir(self.defaults['value'])
        ])

    def _get_param_struct_type(self,ctx):
        return ir.types.DoubleType()

    def _get_context_struct_type(self,ctx):
        return ir.types.DoubleType()

    # generates llvm function for self.forward (#TODO: allow forward func llvm to take weights as input, removing rebuild step)
    def _gen_llvm_forward_function(self):
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            args = [ctx.get_param_struct_type(self).as_pointer(),
                    ctx.get_context_struct_type(self).as_pointer(),
                    ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_output_struct_type(self).as_pointer()]
            print(args)
            builder = ctx.create_llvm_function(args,self) #NEED TO GET INPUT VECTOR SIZE (maybe just pass as ir.VectorType?)
            llvm_func = builder.function
            params, context, arg_in, arg_out = llvm_func.args[:len(args)]
            llvm_func.attributes.add('alwaysinline')

            params, context, arg_in, arg_out = llvm_func.args[:len(args)]
            self._gen_llvm_forward_function_body(ctx,builder,params,context,arg_in,arg_out)

        return llvm_func

    # figure out type of arg_in
    def _gen_llvm_forward_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        # Sometimes we arg_out to 2d array
        out_t = arg_out.type.pointee
        if isinstance(out_t, pnlvm.ir.ArrayType) and isinstance(out_t.element, pnlvm.ir.ArrayType):
            assert len(out_t) == 1
            # arg_out is a pointer to the beginning of the output array (TODO: add support for > 1 state)
            arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
        
        frozen_values = builder.alloca(ctx.int32_ty,ctx.int32_ty(i)) # this should be static because i is known before compilation
        for i in range(len(self.execution_sets)):
            current_exec_set = self.execution_sets[i]
            for component in current_exec_set:
                biases = self.component_to_forward_info[component][1]
                bin_func = self.component_to_forward_info[component][4]
                afferents = self.component_to_forward_info[component][3] #TODO: Pass in afferents as input to func
                value = None
                cmp_arg = builder.gep(arg_in,[ctx.int32_ty(0),ctx.int32_ty(i)])
                if i == 0:
                    value = builder.call(bin_func,[cmp_arg])
                # forward computation if we do not have origin node
                else:
                    value = np.zeros(len(component.input_states[0].defaults.value)).ctypes.data_as() #convert inp to ctype
                    for input_node, weights in afferents.items():
                        if input_node.component in current_exec_set:
                            j = current_exec_set.index()
                            input_value = builder.load(frozen_values,builder.gep(frozen_values,[ctx.int32_ty(j)])) # load the value from prev node
                        else:
                            input_value = self.component_to_forward_info[input_node.component][0].ctypes.data_as()
                        value += torch.matmul(input_value, weights)
                    if biases is not None:
                        value = value + biases
                    value = function(value)

                # store the current value of the node
                self.component_to_forward_info[component][0] = value
                if do_logging:
                    detached_value = value.detach().numpy()
                    component.parameters.value._log_value(detached_value, execution_id, ContextFlags.COMMAND_LINE)

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    outputs[component] = value

    # performs forward computation for the model
    def forward(self, inputs, execution_id=None, do_logging=True):

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
                    value = torch.zeros(len(component.input_states[0].defaults.value), device=self.device).double()
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
                    component.parameters.value._log_value(detached_value, execution_id, ContextFlags.COMMAND_LINE)

                # save value in output list if we're at a node in the last execution set
                if i == len(self.execution_sets) - 1:
                    outputs[component] = value

        self.copy_outputs_to_psyneulink(outputs, execution_id)
        if do_logging:
            self.log_weights(execution_id)
        return outputs

    def detach_all(self):
        for component, info in self.component_to_forward_info.items():
            info[0].detach_()
            if info[1] is not None:
                info[1].detach_()

    # def reset_all(self):
    #     for component, info in self.component_to_forward_info.items():
    #         info[0] = info[4]

    def copy_weights_to_psyneulink(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix.set(weights.detach().numpy(), execution_id)

    def copy_outputs_to_psyneulink(self, outputs, execution_id=None):
        for component, value in outputs.items():
            detached_value = value.detach().numpy()
            component.parameters.value.set(detached_value, execution_id, override=True, skip_history=True, skip_log=True)
            component.output_state.parameters.value.set(detached_value, execution_id, override=True, skip_history=True, skip_log=True)

    def log_weights(self, execution_id=None):
        for projection, weights in self.projections_to_pytorch_weights.items():
            projection.parameters.matrix._log_value(weights.detach().numpy(), execution_id, ContextFlags.COMMAND_LINE)

    # helper method that functions the same as function_creator, but returns a llvm function instead
    def bin_function_creator(self,ctx,node,execution_id=None):

        builder = ctx.create_llvm_function([ir.types.DoubleType()],self,return_type=ir.types.DoubleType()) #NEED TO GET INPUT VECTOR SIZE (maybe just pass as ir.VectorType?)
        llvm_func = builder.function
        
        llvm_func.attributes.add('alwaysinline')
        
        x = llvm_func.args[0]
        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(param_name, None)
            return ir.Constant(ir.types.DoubleType(),val)

        if isinstance(node.function, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            
            ret = builder.fadd(builder.fmul(x,slope),intercept)
            builder.ret(ret)

        elif isinstance(node.function, Logistic):
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')

            exp = ctx.get_llvm_function("__pnl_builtin_exp")
            arg = builder.fadd(
                builder.fmul(
                    builder.neg(gain), builder.fsub(x,bias)
                ),offset)
            one = ir.Constant(ir.types.DoubleType(),1)
            ret = builder.fdiv(one,builder.fadd(one,builder.call(exp,[arg])))
            builder.ret(ret)
            #return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))

        else:  # if we have relu function (the only other kind of function allowed by the autodiff composition)
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            leak = get_fct_param_value('leak')
            zero = ir.Constant(ir.types.DoubleType(),0)
            val = builder.fsub(x,bias)
            pred = builder.fcmp_ordered(">",val,zero)

            with builder.if_else(pred) as (then, otherwise):
                with then:
                    max = val
                    min = zero
                with otherwise:
                    max = zero
                    min = val
            
            ret = builder.fadd(
                builder.fmul(max,gain),
                builder.fmul(min,leak))
            builder.ret(ret)
        return builder.function

    # helper method that identifies the type of function used by a node, gets the function
    # parameters and uses them to create a function object representing the function, then returns it
    def function_creator(self, node, execution_id=None):
        def get_fct_param_value(param_name):
            val = node.function.get_current_function_param(param_name, execution_id)
            if val is None:
                val = node.function.get_current_function_param(param_name, None)
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

        else:  # if we have relu function (the only other kind of function allowed by the autodiff composition)
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