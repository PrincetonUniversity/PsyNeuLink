import torch
import numpy as np

from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.library.compositions.pytorchllvmhelper import *
from psyneulink.library.components.mechanisms.processing.transfer.lstmmechanism import LSTMMechanism

__all__ = ['PytorchMechanismWrapper', 'PytorchProjectionWrapper', 'PytorchLSTMMechanismWrapper', 'wrap_mechanism']

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

def wrap_mechanism(mechanism, index, device, context=None):
    if isinstance(mechanism, LSTMMechanism):
        return PytorchLSTMMechanismWrapper(mechanism, index, device, context=context)
    else:
        return PytorchMechanismWrapper(mechanism, index, device, context=context)

class PytorchWrapper():
    def _get_learnable_param_ids(self):
        return []

    def _get_learnable_param_struct_type(self, ctx):
        structs = []
        for param_id in self._get_learnable_param_ids():
            param = getattr(self._object.parameters, param_id)
            assert param is not None, f"Failed to get param {param_id} from {self}"
            structs.append(ctx.convert_python_struct_to_llvm_ir(param._get(context=self._context)))
        return pnlvm.ir.LiteralStructType(structs)

    @property
    def _object(self):
        raise Exception("Unimplemented method!")

    def _extract_llvm_param_ptr(self, ctx, builder, params, param_id):
        raise Exception("Unimplemented method!")

    def _copy_params_to_psyneulink(self):
        for param_id in self._get_learnable_param_ids():
            pytorch_param = getattr(self, param_id)
            param = getattr(self._object.parameters, param_id)

            pytorch_value = pytorch_param.detach().cpu().numpy()
            param._set(pytorch_value, self._context)

            param_port = self._object.parameter_ports[param_id]
            param_port.parameters.value._set(pytorch_value, self._context)

    def _update_llvm_param_gradients(self, ctx, builder, state, params, delta_w, node_input, node_output):
        pass

    def _get_pytorch_params(self):
        return []

class PytorchMechanismWrapper(PytorchWrapper):
    """
    An interpretation of a mechanism as an equivalent pytorch object
    """
    def __init__(self, mechanism, component_idx, device, context=None):
        self._mechanism = mechanism
        self._idx = component_idx
        self._context = context

        self.function = pytorch_function_creator(mechanism.function, device, context)
        self.value = torch.tensor(mechanism.defaults.value.copy(), device=device, requires_grad=True)
        self.afferents = []
        self.efferents = []

    @property
    def _object(self):
        return self._mechanism

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

    def _update_llvm_param_gradients(self, ctx, builder, state, params, node_delta_w, z_value, backpropagated_error, *, tags=frozenset()):
        # psyneulink functions expect a 2d input, where index 0 is the vector
        fun = ctx.import_llvm_function(self._mechanism.function, tags=frozenset({"derivative"}))
        fun_input_ty = fun.args[2].type.pointee

        mech_input = builder.alloca(fun_input_ty)
        mech_input_ptr = builder.gep(mech_input, [ctx.int32_ty(0),
                                                  ctx.int32_ty(0)])
        builder.store(builder.load(z_value), mech_input_ptr)

        mech_params = builder.gep(params, [ctx.int32_ty(0),
                                           ctx.int32_ty(0),
                                           ctx.int32_ty(self._idx)])

        mech_state = builder.gep(state, [ctx.int32_ty(0),
                                         ctx.int32_ty(0),
                                         ctx.int32_ty(self._idx)])

        f_params_ptr = pnlvm.helpers.get_param_ptr(builder, self._mechanism, mech_params, "function")
        f_params, builder = self._mechanism._gen_llvm_param_ports_for_obj(
                self._mechanism.function, f_params_ptr, ctx, builder, mech_params, mech_state, mech_input)
        f_state = pnlvm.helpers.get_state_ptr(builder, self._mechanism, mech_state, "function")

        output, _ = self._mechanism._gen_llvm_invoke_function(ctx, builder, self._mechanism.function, f_params, f_state, mech_input, tags=frozenset({"derivative"}))
        activation_func_derivative = builder.gep(output, [ctx.int32_ty(0),
                                                          ctx.int32_ty(0)])

        error_val = gen_inject_vec_hadamard(ctx, builder, activation_func_derivative, backpropagated_error)
        return error_val

    def _extract_llvm_param_ptr(self, ctx, builder, params, param_id):
        mechanism_params = builder.gep(params, [ctx.int32_ty(0),
                                           ctx.int32_ty(0),
                                           ctx.int32_ty(self._idx)])

        param_ptr = pnlvm.helpers.get_param_ptr(builder, self._mechanism, mechanism_params, param_id)

        return param_ptr

    def __repr__(self):
        return "PytorchWrapper for: " +self._mechanism.__repr__()

class PytorchLSTMMechanismWrapper(PytorchMechanismWrapper): #lgtm [py/missing-call-to-init]
    """
    An interpretation of a LSTM mechanism as an equivalent pytorch object
    """
    def __init__(self, mechanism, component_idx, device, context=None):
        self._mechanism = mechanism
        self._idx = component_idx
        self._context = context

        self.value = torch.tensor(mechanism.defaults.value.copy(), device=device, requires_grad=True)
        self.afferents = []
        self.efferents = []
        self._lstm = torch.nn.LSTMCell(mechanism.input_size, mechanism.hidden_size, bias=False)
        self._copy_params_from_psyneulink(device)

    def _get_learnable_param_ids(self):
        return ["i_input_matrix",
                "i_hidden_matrix",
                "f_input_matrix",
                "f_hidden_matrix",
                "g_input_matrix",
                "g_hidden_matrix",
                "o_input_matrix",
                "o_hidden_matrix"]

    def _copy_params_from_psyneulink(self, device):
        # copy weights
        i_input_matrix = self._mechanism.parameters.i_input_matrix._get(self._context)
        i_hidden_matrix = self._mechanism.parameters.i_hidden_matrix._get(self._context)
        f_input_matrix = self._mechanism.parameters.f_input_matrix._get(self._context)
        f_hidden_matrix = self._mechanism.parameters.f_hidden_matrix._get(self._context)
        g_input_matrix = self._mechanism.parameters.g_input_matrix._get(self._context)
        g_hidden_matrix = self._mechanism.parameters.g_hidden_matrix._get(self._context)
        o_input_matrix = self._mechanism.parameters.o_input_matrix._get(self._context)
        o_hidden_matrix = self._mechanism.parameters.o_hidden_matrix._get(self._context)

        w_ih = np.concatenate([i_input_matrix, f_input_matrix, g_input_matrix, o_input_matrix])
        w_hh = np.concatenate([i_hidden_matrix, f_hidden_matrix, g_hidden_matrix, o_hidden_matrix])
        w_ih_tensor = torch.nn.Parameter(torch.tensor(w_ih, dtype=torch.double, device=device))
        w_hh_tensor = torch.nn.Parameter(torch.tensor(w_hh, dtype=torch.double, device=device))
        self._lstm.weight_ih = w_ih_tensor
        self._lstm.weight_hh = w_hh_tensor

        # copy initial recurrent state
        self.hidden_state = torch.tensor(self._mechanism.initial_hidden_state, dtype=torch.double, device=device)
        self.cell_state = torch.tensor(self._mechanism.initial_cell_state, dtype=torch.double, device=device)

    def _copy_params_to_psyneulink(self):
        pass

    def _update_llvm_param_gradients(self, ctx, builder, state, params, node_delta_w, z_value, backpropagated_error, *, tags=frozenset()):
        # Setup function values
        tags = tags.union({"derivative"})
        func = ctx.import_llvm_function(self._mechanism.function, tags=tags)
        fun_input_ty = func.args[2].type.pointee

        mech_input = builder.alloca(fun_input_ty)
        mech_input_ptr = builder.gep(mech_input, [ctx.int32_ty(0),
                                                  ctx.int32_ty(0)])
        builder.store(builder.load(z_value), mech_input_ptr)

        params = builder.gep(params, [ctx.int32_ty(0),
                                      ctx.int32_ty(0),
                                      ctx.int32_ty(self._idx)])

        state = builder.gep(state, [ctx.int32_ty(0),
                                    ctx.int32_ty(0),
                                    ctx.int32_ty(self._idx)])

        f_params_ptr = pnlvm.helpers.get_param_ptr(builder, self._mechanism, params, "function")
        f_params, builder = self._mechanism._gen_llvm_param_ports_for_obj(
                self._mechanism.function, f_params_ptr, ctx, builder, params, state, mech_input)
        f_state = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "function")

        logistic_params_ptr = pnlvm.helpers.get_param_ptr(builder, self._mechanism, params, "logistic")
        logistic_params, builder = self._mechanism._gen_llvm_param_ports_for_obj(
                self._mechanism.logistic, logistic_params_ptr, ctx, builder, params, state, mech_input)
        logistic_state = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "logistic")
        logistic_derivative_func = ctx.import_llvm_function(self._mechanism.logistic, tags=tags)

        tanh_params_ptr = pnlvm.helpers.get_param_ptr(builder, self._mechanism, params, "tanh")
        tanh_params, builder = self._mechanism._gen_llvm_param_ports_for_obj(
                self._mechanism.tanh, tanh_params_ptr, ctx, builder, params, state, mech_input)
        tanh_state = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "tanh")
        tanh_func = ctx.import_llvm_function(self._mechanism.tanh)
        tanh_derivative_func = ctx.import_llvm_function(self._mechanism.tanh, tags=tags)

        # extract stored intermediates
        i_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "i")
        i_bar_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "i_bar")
        f_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "f")
        f_bar_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "f_bar")
        g_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "g")
        g_bar_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "g_bar")
        o_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "o")
        o_bar_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "o_bar")
        c_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "c")
        c_prev_ptr = pnlvm.helpers.get_state_ptr(builder, self._mechanism, state, "c_prev")

        #1) Calculate delta_o_bar
        c_tanh = builder.alloca(c_ptr.type.pointee)
        builder.call(tanh_func, [tanh_params, tanh_state, c_ptr, c_tanh])
        o_bar_logistic_derivative = builder.alloca(o_bar_ptr.type.pointee)
        builder.call(logistic_derivative_func, [logistic_params, logistic_state, o_bar_ptr, o_bar_logistic_derivative])

        delta_o_bar = gen_inject_vec_hadamard(ctx, builder, backpropagated_error, c_tanh)
        delta_o_bar = gen_inject_vec_hadamard(ctx, builder, delta_o_bar, o_bar_logistic_derivative)

        #2) Calculate delta_c
        c_tanh_derivative = builder.alloca(c_ptr.type.pointee)
        builder.call(tanh_derivative_func, [tanh_params, tanh_state, c_ptr, c_tanh_derivative])
        delta_c = gen_inject_vec_hadamard(ctx, builder, backpropagated_error, o_ptr)
        delta_c = gen_inject_vec_hadamard(ctx, builder, delta_c, c_tanh_derivative)

        #3) Calculate delta_f_bar
        f_bar_logistic_deriviative = builder.alloca(f_bar_ptr.type.pointee)
        builder.call(logistic_derivative_func, [logistic_params, logistic_state, f_bar_ptr, f_bar_logistic_deriviative])
        delta_f_bar = gen_inject_vec_hadamard(ctx, builder, delta_c, c_prev_ptr)
        delta_f_bar = gen_inject_vec_hadamard(ctx, builder, delta_f_bar, f_bar_logistic_deriviative)

        #3) Calculate delta_i_bar
        i_bar_logistic_deriviative = builder.alloca(i_bar_ptr.type.pointee)
        builder.call(logistic_derivative_func, [logistic_params, logistic_state, i_bar_ptr, i_bar_logistic_deriviative])
        delta_i_bar = gen_inject_vec_hadamard(ctx, builder, delta_c, g_ptr)
        delta_i_bar = gen_inject_vec_hadamard(ctx, builder, delta_i_bar, i_bar_logistic_deriviative)
        pnlvm.helpers.printf_float_array(builder, delta_i_bar, prefix=f"{self} delta_i_bar\t", override_debug=True)

        #4) Calculate delta_g_bar
        g_bar_tanh_deriviative = builder.alloca(g_bar_ptr.type.pointee)
        builder.call(tanh_derivative_func, [tanh_params, tanh_state, g_bar_ptr, g_bar_tanh_deriviative])
        delta_g_bar = gen_inject_vec_hadamard(ctx, builder, delta_c, i_ptr)
        delta_g_bar = gen_inject_vec_hadamard(ctx, builder, delta_g_bar, g_bar_tanh_deriviative)


        #5) Calculate weight gradients
        delta_w_i_input_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("i_input_matrix"))])
        i_input_matrix_gradient = gen_inject_vec_outer_product(ctx, builder, delta_i_bar, z_value)
        pnlvm.helpers.printf_float_array(builder, z_value, prefix=f"{self} z_value\t", override_debug=True)
        gen_inject_mat_add(ctx, builder, delta_w_i_input_matrix, i_input_matrix_gradient, delta_w_i_input_matrix)
        pnlvm.helpers.printf_float_matrix(builder, i_input_matrix_gradient, prefix=f"{self} i_input_matrix\t", override_debug=True)

        # i_hidden_matrix = pnlvm.helpers.get_param_ptr(builder, self._mechanism, "i_hidden_matrix")
        # delta_w_i_hidden_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("i_hidden_matrix"))])
        # i_hidden_matrix_gradient = gen_inject_vec_outer_product(ctx, builder, delta_i_bar, z_value)

        delta_w_f_input_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("f_input_matrix"))])
        f_input_matrix_gradient = gen_inject_vec_outer_product(ctx, builder, delta_f_bar, z_value)
        gen_inject_mat_add(ctx, builder, delta_w_f_input_matrix, f_input_matrix_gradient, delta_w_f_input_matrix)

        # f_hidden_matrix = pnlvm.helpers.get_param_ptr(builder, self._mechanism, "f_hidden_matrix")
        # delta_w_f_hidden_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("f_hidden_matrix"))])
        # f_hidden_matrix_gradient = gen_inject_vec_outer_product()

        delta_w_g_input_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("g_input_matrix"))])
        g_input_matrix_gradient = gen_inject_vec_outer_product(ctx, builder, delta_g_bar, z_value)
        gen_inject_mat_add(ctx, builder, delta_w_g_input_matrix, g_input_matrix_gradient, delta_w_g_input_matrix)

        # g_hidden_matrix = pnlvm.helpers.get_param_ptr(builder, self._mechanism, "g_hidden_matrix")
        # delta_w_g_hidden_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("g_hidden_matrix"))])
        # g_hidden_matrix_gradient = gen_inject_vec_outer_product()

        delta_w_o_input_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("o_input_matrix"))])
        o_input_matrix_gradient = gen_inject_vec_outer_product(ctx, builder, delta_o_bar, z_value)
        gen_inject_mat_add(ctx, builder, delta_w_o_input_matrix, o_input_matrix_gradient, delta_w_o_input_matrix)

        # o_hidden_matrix = pnlvm.helpers.get_param_ptr(builder, self._mechanism, "o_hidden_matrix")
        # delta_w_o_hidden_matrix = builder.gep(node_delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index("o_hidden_matrix"))])
        # o_hidden_matrix_gradient = gen_inject_vec_outer_product()

        output, _ = self._mechanism._gen_llvm_invoke_function(ctx, builder, self._mechanism.function, f_params, f_state, mech_input, tags=tags)
        activation_func_derivative = builder.gep(output, [ctx.int32_ty(0),
                                                          ctx.int32_ty(0)])

        error_val = gen_inject_vec_hadamard(ctx, builder, activation_func_derivative, backpropagated_error)
        return error_val

    def execute(self, variable):
        variable = torch.reshape(variable, (1, variable.size(0)))
        hidden = torch.reshape(self.hidden_state, (1, self.hidden_state.size(0)))
        cell_state = torch.reshape(self.cell_state, (1, self.cell_state.size(0)))

        hid, cell = self._lstm(variable, (hidden, cell_state))
        self.hidden_state = hid[0]
        self.cell_state = cell[0]
        self.value = cell[0]

        return self.value

    def _get_pytorch_params(self):
        return [self._lstm.weight_ih, self._lstm.weight_hh]

class PytorchProjectionWrapper(PytorchWrapper):
    """
    An interpretation of a projection as an equivalent pytorch object
    """
    def __init__(self, projection, component_idx, device, sender=None, receiver=None, context=None):
        self._projection = projection
        self._idx = component_idx
        self._context = context

        self.sender = sender
        self.receiver = receiver

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
    @property
    def _sender_port_idx(self):
        port = self._projection.sender
        return port.owner.output_ports.index(port)

    @property
    def _receiver_port_idx(self):
        port = self._projection.receiver
        return port.owner.input_ports.index(port)

    @property
    def _object(self):
        return self._projection

    def _get_learnable_param_ids(self):
        return ["matrix"]

    def _get_pytorch_params(self):
        return [self.matrix]

    def execute(self, variable):
        return torch.matmul(variable, self.matrix)

    def log_matrix(self):
        if self._projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self._projection.parameters.matrix._set(detached_matrix, context=self._context)
            self._projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)

    def _extract_llvm_param_ptr(self, ctx, builder, params, param_id):
        proj_params = builder.gep(params, [ctx.int32_ty(0),
                                           ctx.int32_ty(1),
                                           ctx.int32_ty(self._idx)])

        param_ptr = pnlvm.helpers.get_param_ptr(builder, self._projection, proj_params, param_id)

        if param_id == "matrix":
            dim_x, dim_y = self.matrix.detach().numpy().shape
            param_ptr = builder.bitcast(param_ptr, pnlvm.ir.types.ArrayType(
                        pnlvm.ir.types.ArrayType(ctx.float_ty, dim_y), dim_x).as_pointer())

        return param_ptr

    def _gen_llvm_execute(self, ctx, builder, state, params, data):
        proj_matrix = self._extract_llvm_param_ptr(ctx, builder, params, "matrix")

        input_vec = builder.gep(data, [ctx.int32_ty(0),
                                       ctx.int32_ty(0),
                                       ctx.int32_ty(self.sender._idx),
                                       ctx.int32_ty(self._sender_port_idx)])

        output_vec = gen_inject_vxm(ctx, builder, input_vec, proj_matrix)

        pnlvm.helpers.printf_float_array(builder, input_vec, prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} input:\n", override_debug=False)
        pnlvm.helpers.printf_float_matrix(builder, proj_matrix, prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} mat:\n", override_debug=False)
        pnlvm.helpers.printf_float_array(builder, output_vec, prefix=f"{self.sender._mechanism} -> {self.receiver._mechanism} output:\n", override_debug=False)

        return output_vec

    def _update_llvm_param_gradients(self, ctx, builder, state, params, delta_w, node_input, error_val):
        if self._projection.learnable:
            node_delta_w = builder.gep(delta_w, [ctx.int32_ty(0), ctx.int32_ty(self._get_learnable_param_ids().index('matrix'))])

            outer_product = gen_inject_vec_outer_product(ctx, builder, node_input, error_val)
            gen_inject_mat_add(ctx, builder, outer_product, node_delta_w, node_delta_w)

    def __repr__(self):
        return "PytorchWrapper for: " +self._projection.__repr__()
