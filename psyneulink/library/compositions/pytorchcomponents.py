from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.core.globals.log import LogCondition
from psyneulink.core import llvm as pnlvm

import torch

__all__ = ['PytorchMechanismWrapper', 'PytorchProjectionWrapper']

def pytorch_function_creator(function, device, context=None):
    """
    Converts a PsyNeuLink function into an equivalent PyTorch lambda function.
    NOTE: This is needed due to PyTorch limitations (see: https://github.com/PrincetonUniversity/PsyNeuLink/pull/1657#discussion_r437489990)
    """
    def get_fct_param_value(param_name):
        val = function._get_current_function_param(
            param_name, context=context)
        if val is None:
            val = getattr(function.defaults, param_name)

        return float(val)

    if isinstance(function, Linear):
        slope = get_fct_param_value('slope')
        intercept = get_fct_param_value('intercept')
        return lambda x: x * slope + intercept

    elif isinstance(function, Logistic):
        gain = get_fct_param_value('gain')
        bias = get_fct_param_value('bias')
        offset = get_fct_param_value('offset')
        return lambda x: 1 / (1 + torch.exp(-gain * (x + bias) + offset))

    elif isinstance(function, ReLU):
        gain = get_fct_param_value('gain')
        bias = get_fct_param_value('bias')
        leak = get_fct_param_value('leak')
        return lambda x: (torch.max(input=(x - bias), other=torch.tensor([0], device=device).double()) * gain +
                            torch.min(input=(x - bias), other=torch.tensor([0], device=device).double()) * leak)

    else:
        raise Exception(f"Function {function} is not currently supported in AutodiffCompositions!")

def bin_function_derivative_creator(ctx, node, context=None):
    """
    Returns the compiled derivative version of a PsyNeuLink node
    TODO: Add functionality for derivitives into base PsyNeuLink Functions, and move this functionality there
    """
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
    builder = ctx.create_llvm_function(args, node, name)
    llvm_func = builder.function

    input_vector, dim, output_vector = llvm_func.args

    def get_fct_param_value(param_name):
        val = node.function._get_current_function_param(
            param_name, context)
        if val is None:
            val = node.function._get_current_function_param(
                param_name, None)
        return ctx.float_ty(val[0])

    if isinstance(node.function, Linear):  # f(x) = mx + b, f'(x) = m
        slope = get_fct_param_value('slope')

        def modify_value(x):
            return slope

    elif isinstance(node.function, Logistic):  # f'(x) = f(x)(1-f(x))
        gain = pnlvm.helpers.fneg(builder, get_fct_param_value('gain'))
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

            ret = builder.fsub(one, f_x)
            ret = builder.fmul(f_x, ret)
            return ret

    else:
        raise Exception(
            f"Function type {node.function} is currently unsupported by compiled execution!")

    # do computations
    with pnlvm.helpers.for_loop_zero_inc(builder, dim, "derivative_loop") as (builder, iterator):
        val_ptr = builder.gep(input_vector, [iterator])
        val = builder.load(val_ptr)
        val = modify_value(val)
        output_location = builder.gep(output_vector, [iterator])
        builder.store(val, output_location)

    builder.ret_void()

    return llvm_func


class PytorchMechanismWrapper():
    """
    An interpretation of a mechanism as an equivalent pytorch object
    """
    def __init__(self, mechanism, component_idx, device, context=None):
        self._mechanism = mechanism
        self._idx = component_idx
        self._context = context

        self.function = pytorch_function_creator(mechanism.function, device, context)
        self._context = context
        self.value = None
        self.afferents = []
        self.efferents = []

        self._target_mechanism = None

    def add_efferent(self, efferent):
        assert efferent not in self.efferents
        self.efferents.append(efferent)

    def add_afferent(self, afferent):
        assert afferent not in self.afferents
        self.afferents.append(afferent)


    def collate_afferents(self):
        """
        Returns weight-multiplied sum of all afferent projections
        """
        return sum((proj.execute(proj.sender.value) for proj in self.afferents))

    def execute(self, variable):
        self.value = self.function(variable)

        return self.value

    def _gen_execute_llvm(self, ctx, builder, state, params, mech_input, data):
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

    def _gen_execute_derivative_func_llvm(self, ctx, builder, mech_input):
        derivative_func = ctx.import_llvm_function(
                    bin_function_derivative_creator(ctx, self._mechanism, context=self._context).name)
        return gen_inject_unary_function_call(ctx, builder, derivative_func, mech_input)

    def __repr__(self):
        return "PytorchWrapper for: " +self._mechanism.__repr__()

class PytorchProjectionWrapper():
    """
    An interpretation of a projection as an equivalent pytorch object
    """
    def __init__(self, projection, component_idx, port_idx, device, sender=None, receiver=None, context=None):
        self._projection = projection
        self._idx = component_idx
        self._context = context

        self.sender = sender
        self.receiver = receiver
        self._port_idx = port_idx

        matrix = projection.parameters.matrix.get(
                            context=context)
        if matrix is None:
            matrix = projection.parameters.matrix.get(
                context=None
            )
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

    def _extract_llvm_matrix(self, ctx, builder, params):
        proj_params = builder.gep(params, [ctx.int32_ty(0),
                                           ctx.int32_ty(1),
                                           ctx.int32_ty(self._idx)])

        dim_x, dim_y = self.matrix.detach().numpy().shape
        proj_matrix = pnlvm.helpers.get_param_ptr(builder, self._projection, proj_params, "matrix")
        proj_matrix = builder.bitcast(proj_matrix, pnlvm.ir.types.ArrayType(
            pnlvm.ir.types.ArrayType(ctx.float_ty, dim_y), dim_x).as_pointer())

        return proj_matrix

    def _gen_execute_llvm(self, ctx, builder, state, params, data):
        proj_matrix = self._extract_llvm_matrix(ctx, builder, params)

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
