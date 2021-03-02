from psyneulink.core import llvm as pnlvm
from psyneulink.library.compositions.pytorchllvmhelper import *

__all__ = ['AdamOptimizer', 'SGDOptimizer']

class Optimizer():

    def __init__(self, pytorch_model):
        self._pytorch_model = pytorch_model
        self._composition = pytorch_model._composition

        self._DELTA_W_NUM = 0

    # gets the type of the delta_w struct
    def _get_param_gradients_struct_type(self, ctx):
        gradients_struct = pnlvm.ir.types.LiteralStructType((component._get_learnable_param_struct_type(ctx) for component in self._pytorch_model.components))
        return gradients_struct

    def _get_optimizer_struct_type(self, ctx, extra_types=[]):
        structs = (self._get_param_gradients_struct_type(ctx), *extra_types)
        return pnlvm.ir.types.LiteralStructType(structs)

    # inserts logic that zeroes out a gradient struct
    def _gen_zero_gradient_struct(self, ctx, builder, grad_struct):
        builder.store(grad_struct.type.pointee(None), grad_struct)

    def zero_grad(self, ctx):
        name = self._composition.name + "_ZERO_GRAD"

        args = [self._get_optimizer_struct_type(ctx).as_pointer()]
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        optim_struct = llvm_func.args[0]

        delta_w_struct = builder.gep(
            optim_struct, [ctx.int32_ty(0), ctx.int32_ty(self._DELTA_W_NUM)])
        self._gen_zero_gradient_struct(ctx, builder, delta_w_struct)
        builder.ret_void()

        return llvm_func

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        return self.step(ctx)

    # to be implemented by child classes - steps the optimizer
    def step(self, ctx):
        raise Exception("Unimplemented method!")

# Class that is used to represent a compiled optimizer - aims to reimplement the logic of torch.optimizer in the form of llvmlite compileable code


class AdamOptimizer(Optimizer):
    """Implements compiled ADAM Optimizer ( from paper https://arxiv.org/pdf/1412.6980.pdf  )"""
    # sets up parameters of model & the information required for forward computation
    def __init__(self, pytorch_model, lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0,):
        super().__init__(pytorch_model)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self._M_T_NUM = 1
        self._V_T_NUM = 2
        self._T_NUM = 3

    def _get_optimizer_struct_type(self, ctx):
        m_t = self._get_param_gradients_struct_type(ctx)  # current timestep first moment
        v_t = m_t  # current timestep second moment
        time_counter = ctx.float_ty  # keeps track of timestep

        extra_types = [m_t, v_t, time_counter]
        return super()._get_optimizer_struct_type(ctx, extra_types=extra_types)

    # steps the adam optimizer (methodology: https://arxiv.org/pdf/1412.6980.pdf )
    def step(self, ctx):
        name = self._composition.name + "_ADAM_STEP"

        args = [self._get_optimizer_struct_type(ctx).as_pointer(),
                ctx.get_param_struct_type(self._composition).as_pointer()]
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        optim_struct, params = llvm_func.args

        # setup values
        zero = ctx.int32_ty(0)
        one_float = ctx.float_ty(1)

        delta_w = builder.gep(optim_struct, [zero, ctx.int32_ty(self._DELTA_W_NUM)])
        m_t = builder.gep(optim_struct, [zero, ctx.int32_ty(self._M_T_NUM)])
        v_t = builder.gep(optim_struct, [zero, ctx.int32_ty(self._V_T_NUM)])
        t = builder.gep(optim_struct, [zero, ctx.int32_ty(self._T_NUM)])

        # get methods needed
        pow = ctx.import_llvm_function("__pnl_builtin_pow")
        sqrt = ctx.get_builtin("sqrt", [ctx.float_ty])

        lr = ctx.float_ty(self.lr)
        eps = ctx.float_ty(self.eps)
        b1 = ctx.float_ty(self.betas[0])
        b2 = ctx.float_ty(self.betas[1])
        one_minus_b1 = builder.fsub(one_float, b1)
        one_minus_b2 = builder.fsub(one_float, b2)

        # 1) increment t
        builder.store(builder.fadd(builder.load(t), one_float), t)
        t_val = builder.load(t)
        # 1.5) calculate values to be used later (based on incremented t)
        b1_pow = builder.call(pow, [b1, t_val])
        b2_pow = builder.call(pow, [b2, t_val])
        one_minus_b1_pow = builder.fsub(one_float, b1_pow)
        one_minus_b2_pow = builder.fsub(one_float, b2_pow)

        pnlvm.helpers.printf(
                builder, f"%f b1_pow_sub %f\nb2 pow sub %f\n",t_val, one_minus_b1_pow, one_minus_b2_pow)

        # 2) update first moments
        for idx, component in enumerate(self._pytorch_model.components):
            for param_idx, param_id in enumerate(component._get_learnable_param_ids()):
                gradient_ptr = builder.gep(delta_w, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])
                m_t_ptr = builder.gep(m_t, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])

                # m_t = m_t * b1
                gen_inject_mat_scalar_mult(ctx, builder, m_t_ptr, b1, m_t_ptr)

                # (1 - b1)*g_t
                tmp_val = gen_inject_mat_scalar_mult(ctx, builder, gradient_ptr, one_minus_b1)

                # m_t = m_t + (1-b1)*g_t
                gen_inject_mat_add(ctx, builder, m_t_ptr, tmp_val, m_t_ptr)

                pnlvm.helpers.printf_float_matrix(builder, m_t_ptr, prefix=f"mt val: {component}\n", override_debug=False)

        # 3) update second moments
        for idx, component in enumerate(self._pytorch_model.components):
            for param_idx, param_id in enumerate(component._get_learnable_param_ids()):
                gradient_ptr = builder.gep(delta_w, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])
                v_t_ptr = builder.gep(v_t, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])

                # v_t = v_t * b2
                gen_inject_mat_scalar_mult(ctx, builder, v_t_ptr, b2, v_t_ptr)

                # g_t * g_t
                delta_w_sqrd = gen_inject_mat_hadamard(ctx, builder, gradient_ptr, gradient_ptr)

                # (1-b2)*(g_t)^2
                gen_inject_mat_scalar_mult(ctx, builder, delta_w_sqrd, one_minus_b2, delta_w_sqrd)

                # v_t = v_t + (1-b2)*(g_t)^2
                gen_inject_mat_add(ctx, builder, v_t_ptr, delta_w_sqrd, v_t_ptr)

        # 4) update weights
        # this is the new learning rate to use
        # NOTE: This differs from the version specified in the paper to numerically match with pytorch's implementation
        step_size = builder.fdiv(lr, one_minus_b1_pow)
        step_size = pnlvm.helpers.fneg(builder, step_size)

        for idx, component in enumerate(self._pytorch_model.components):
            for param_idx, param_id in enumerate(component._get_learnable_param_ids()):
                gradient_ptr = builder.gep(delta_w, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])
                m_t_ptr = builder.gep(m_t, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])
                v_t_ptr = builder.gep(v_t, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])

                pnlvm.helpers.printf_float_matrix(builder, gradient_ptr, prefix=f"grad val: {component}\n", override_debug=False)

                # this is messy - #TODO - cleanup this
                param_ptr = component._extract_llvm_param_ptr(ctx, builder, params, param_id)

                pnlvm.helpers.printf(builder, "biascorr2 %.20f\n", one_minus_b2_pow, override_debug=False)
                with pnlvm.helpers.array_ptr_loop(builder, param_ptr, "optimizer_w_upd_outer") as (b1, row_idx):
                    param_row = builder.gep(param_ptr, [ctx.int32_ty(0), row_idx])
                    with pnlvm.helpers.array_ptr_loop(b1, param_row, "optimizer_w_upd_inner") as (b2, column_idx):
                        # sqrt(v_t) + eps
                        v_t_value = b2.load(b2.gep(v_t_ptr, [zero, row_idx, column_idx]))
                        value = b2.call(sqrt, [v_t_value])
                        denom = b2.call(sqrt, [one_minus_b2_pow])
                        value = b2.fdiv(value, denom)
                        value = b2.fadd(value, eps)
                        pnlvm.helpers.printf(builder, "val %.20f\n", value, override_debug=False)
                        # alpha_t * m_t
                        m_t_value = b2.load(b2.gep(
                            m_t_ptr, [zero, row_idx, column_idx]))

                        # value = alpha_t * m_t / (sqrt(v_t) + eps)
                        value = b2.fdiv(m_t_value, value)
                        value = b2.fmul(step_size, value)

                        old_weight_ptr = b2.gep(
                            param_ptr, [zero, row_idx, column_idx])

                        # new_weight = old_weight - value
                        value = b2.fadd(b2.load(old_weight_ptr), value)
                        b2.store(value, old_weight_ptr)

                    pnlvm.helpers.printf(b1, "\n", override_debug=False)

        pnlvm.helpers.printf(builder, f"\t\t\tOPTIM DONE UPDATE\n",override_debug=False)

        builder.ret_void()

        return llvm_func


class SGDOptimizer(Optimizer):
    """Implements compiled Stocastic Gradient Descent optimizer (without momentum)"""
    # sets up parameters of model & the information required for forward computation
    def __init__(self, pytorch_model, lr=1e-3):
        super().__init__(pytorch_model)
        self.lr = lr

    # steps the sgd optimizer (methodology: https://arxiv.org/pdf/1412.6980.pdf )
    def step(self, ctx):
        name = self._composition.name + "_SGD_STEP"

        args = [self._get_optimizer_struct_type(ctx).as_pointer(),
                ctx.get_param_struct_type(self._composition).as_pointer()]
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        optim_struct, params = llvm_func.args

        zero = ctx.int32_ty(0)
        component_gradients = builder.gep(optim_struct, [zero, ctx.int32_ty(self._DELTA_W_NUM)])

        lr = ctx.float_ty(self.lr)

        # update weights
        for idx, component in enumerate(self._pytorch_model.components):
            for param_idx, param_id in enumerate(component._get_learnable_param_ids()):
                gradient_ptr = builder.gep(component_gradients, [zero, ctx.int32_ty(idx), ctx.int32_ty(param_idx)])
                param_ptr = component._extract_llvm_param_ptr(ctx, builder, params, param_id)

                multiplied_delta_w = gen_inject_mat_scalar_mult(ctx, builder, gradient_ptr, lr)
                gen_inject_mat_sub(ctx, builder, param_ptr, multiplied_delta_w, param_ptr)

        builder.ret_void()

        return llvm_func
