from psyneulink.core import llvm as pnlvm

__all__ = ['AdamOptimizer','SGDOptimizer']


class Optimizer():

    def __init__(self, pytorch_model):
        self._pytorch_model = pytorch_model
        self._composition = pytorch_model._composition
        self._structs = []

        self._DELTA_W_NUM = 0

    # gets the type of the delta_w struct
    def _get_delta_w_struct_type(self, ctx):
        delta_w = [None]*len(self._composition.nodes)
        for node in self._composition.nodes:
            node_idx = self._composition._get_node_index(node)
            afferent_nodes = self._pytorch_model._get_afferent_nodes(node)
            delta_w[node_idx] = [None] * len(afferent_nodes)
            for (afferent_node, matrix) in afferent_nodes:
                afferent_node_index = self._pytorch_model._get_afferent_node_index(
                    node, afferent_node)
                weights_dim_x, weights_dim_y = matrix.shape
                delta_w_array = pnlvm.ir.ArrayType(
                    pnlvm.ir.ArrayType(
                        ctx.float_ty,
                        weights_dim_y
                    ),
                    weights_dim_x
                )
                delta_w[node_idx][afferent_node_index] = delta_w_array
            delta_w[node_idx] = pnlvm.ir.types.LiteralStructType(delta_w[node_idx])
        delta_w = pnlvm.ir.types.LiteralStructType(delta_w)
        return delta_w

    def _get_optimizer_struct_type(self, ctx, extra_types=[]):
        structs = []
        structs.append(self._get_delta_w_struct_type(ctx))
        structs += extra_types
        return pnlvm.ir.types.LiteralStructType(structs)

    def _get_listof_gradient_struct_values(self):
        values = []
        for node in self._composition.nodes:
            node_idx = self._composition._get_node_index(node)
            afferent_nodes = self._pytorch_model._get_afferent_nodes(node)
            for (afferent_node, matrix) in afferent_nodes:
                afferent_node_index = self._pytorch_model._get_afferent_node_index(
                    node, afferent_node)
                weights_dim_x, weights_dim_y = matrix.shape
                values.append((node, node_idx, afferent_node,
                               afferent_node_index, matrix, weights_dim_x, weights_dim_y))
        return values
    # inserts logic that zeroes out a gradient struct

    def _gen_zero_gradient_struct(self, ctx, builder, grad_struct):
        builder.store(grad_struct.type.pointee(None),grad_struct)

    def zero_grad(self, ctx, builder, optim_struct):
        delta_w_struct = builder.gep(
            optim_struct, [ctx.int32_ty(0), ctx.int32_ty(self._DELTA_W_NUM)])
        self._gen_zero_gradient_struct(ctx, builder, delta_w_struct)

    # to be implemented by child classes - sets the initial values for the optim struct
    def initialize_optimizer_struct(self, ctx, builder, optim_struct):
        raise Exception("Unimplemented method!")

    # to be implemented by child classes - steps the optimizer
    def step(self, ctx, builder, optim_struct, model_params):
        raise Exception("Unimplemented method!")

# Class that is used to represent a compiled optimizer - aims to reimplement the logic of torch.optimizer in the form of llvmlite compileable code


class AdamOptimizer(Optimizer):
    """Implements compiled ADAM Optimizer ( from paper https://arxiv.org/pdf/1412.6980.pdf  )"""
    # sets up parameters of model & the information required for forward computation
    def __init__(self, pytorch_model, lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0,):
        super().__init__(pytorch_model)
        self.lr = lr
        self.betas = betas
        self.eps = 1e-8
        self.weight_decay = weight_decay

        self._M_T_NUM = 1
        self._V_T_NUM = 2
        self._T_NUM = 3

    def _get_optimizer_struct_type(self, ctx):
        extra_types = []
        m_t = self._get_delta_w_struct_type(
            ctx)  # current timestep first moment
        v_t = self._get_delta_w_struct_type(
            ctx)  # current timestep second moment
        time_counter = ctx.float_ty  # keeps track of timestep

        extra_types += [m_t, v_t, time_counter]
        return super()._get_optimizer_struct_type(ctx, extra_types=extra_types)

    def initialize_optimizer_struct(self, ctx, builder, optim_struct):
        # set initial moments to 0
        delta_w = builder.gep(
            optim_struct, [ctx.int32_ty(0), ctx.int32_ty(self._DELTA_W_NUM)])
        m_t = builder.gep(
            optim_struct, [ctx.int32_ty(0), ctx.int32_ty(self._M_T_NUM)])
        v_t = builder.gep(
            optim_struct, [ctx.int32_ty(0), ctx.int32_ty(self._V_T_NUM)])
        t = builder.gep(
            optim_struct, [ctx.int32_ty(0), ctx.int32_ty(self._T_NUM)])
        self._gen_zero_gradient_struct(ctx, builder, delta_w)
        self._gen_zero_gradient_struct(ctx, builder, m_t)
        self._gen_zero_gradient_struct(ctx, builder, v_t)
        builder.store(ctx.float_ty(0), t)


    # steps the adam optimizer (methodology: https://arxiv.org/pdf/1412.6980.pdf )
    def step(self, ctx):
        name = self._composition.name+"_ADAM_STEP"
        # try:
        #     llvm_func = ctx.get_llvm_function(name)
        #     return llvm_func
        # except Exception as e:
        #     pass

        args = [self._get_optimizer_struct_type(ctx).as_pointer(),
                self._pytorch_model._get_param_struct_type(ctx).as_pointer()]
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        llvm_func.attributes.add('alwaysinline')
        optim_struct, model_params = llvm_func.args

        # setup values
        zero = ctx.int32_ty(0)
        one_float = ctx.float_ty(1)

        delta_w = builder.gep(optim_struct, [zero, ctx.int32_ty(self._DELTA_W_NUM)])
        m_t = builder.gep(optim_struct, [zero, ctx.int32_ty(self._M_T_NUM)])
        v_t = builder.gep(optim_struct, [zero, ctx.int32_ty(self._V_T_NUM)])
        t = builder.gep(optim_struct, [zero, ctx.int32_ty(self._T_NUM)])

        # get methods needed
        pow = ctx.get_llvm_function("__pnl_builtin_pow")
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
        
        ctx.inject_printf(
                builder, f"%f b1_pow_sub %f\nb2 pow sub %f\n",t_val, one_minus_b1_pow,one_minus_b2_pow)
        alpha_mult = builder.call(sqrt, [one_minus_b2_pow])
        ctx.inject_printf(
                builder, f"%f\n",alpha_mult)
        alpha_mult = builder.fdiv(alpha_mult, one_minus_b1_pow)

        # this is the new learning rate to use
        alpha_t = builder.fmul(alpha_mult, lr)

        gradient_struct_values = self._get_listof_gradient_struct_values()

        # 2) update first moments
        for (node, node_idx, afferent_node, afferent_node_index, matrix, weights_dim_x, weights_dim_y) in gradient_struct_values:
            ctx.inject_printf(
                builder, f"\t\t\t\tOPTIM UPDATE FIRST MOMENT {afferent_node.name} {node.name}\n")

            node_idx_ir = ctx.int32_ty(node_idx)
            afferent_node_index_ir = ctx.int32_ty(afferent_node_index)

            m_t_ptr = builder.gep(
                m_t, [zero, node_idx_ir, afferent_node_index_ir])
            delta_w_ptr = builder.gep(
                delta_w, [zero, node_idx_ir, afferent_node_index_ir])

            # m_t = m_t * b1
            self._pytorch_model._gen_inject_mat_scalar_mult(ctx, builder, m_t_ptr, b1, m_t_ptr)

            # (1 - b1)*g_t
            tmp_val = self._pytorch_model._gen_inject_mat_scalar_mult(ctx, builder, delta_w_ptr, one_minus_b1)

            # m_t = m_t + (1-b1)*g_t
            self._pytorch_model._gen_inject_mat_add(ctx, builder, m_t_ptr, tmp_val, m_t_ptr)

        # 3) update second moments
        for (node, node_idx, afferent_node, afferent_node_index, matrix, weights_dim_x, weights_dim_y) in gradient_struct_values:
            ctx.inject_printf(
                builder, f"\t\t\t\tOPTIM UPDATE SECOND MOMENT {afferent_node.name} {node.name}\n")

            node_idx_ir = ctx.int32_ty(node_idx)
            afferent_node_index_ir = ctx.int32_ty(afferent_node_index)

            v_t_ptr = builder.gep(
                v_t, [zero, node_idx_ir, afferent_node_index_ir])
            delta_w_ptr = builder.gep(
                delta_w, [zero, node_idx_ir, afferent_node_index_ir])

            # v_t = v_t * b2
            self._pytorch_model._gen_inject_mat_scalar_mult(ctx, builder, v_t_ptr, b2, v_t_ptr)

            # g_t * g_t
            delta_w_sqrd = self._pytorch_model._gen_inject_mat_hadamard(ctx, builder, delta_w_ptr, delta_w_ptr)

            # (1-b2)*(g_t)^2
            self._pytorch_model._gen_inject_mat_scalar_mult(ctx, builder, delta_w_sqrd, one_minus_b2, delta_w_sqrd)

            # v_t = v_t + (1-b2)*(g_t)^2
            self._pytorch_model._gen_inject_mat_add(ctx, builder, v_t_ptr, delta_w_sqrd, v_t_ptr)

        # 4) update weights

        for (node, node_idx, afferent_node, afferent_node_index, matrix, weights_dim_x, weights_dim_y) in gradient_struct_values:
            node_idx_ir = ctx.int32_ty(node_idx)
            afferent_node_index_ir = ctx.int32_ty(afferent_node_index)
            
            m_t_ptr = builder.gep(
                m_t, [zero, node_idx_ir, afferent_node_index_ir])
            v_t_ptr = builder.gep(
                v_t, [zero, node_idx_ir, afferent_node_index_ir])
            delta_w_ptr = builder.gep(
                delta_w, [zero, node_idx_ir, afferent_node_index_ir])
            # this is messy - #TODO - cleanup this
            weights_llvmlite, weights_dim_x, weights_dim_y = self._pytorch_model._gen_get_node_weight_ptr(
                ctx, builder, model_params, node, afferent_node)
            ctx.inject_printf(
                builder, f"OPTIM UPDATE WEIGHTS {afferent_node.name} {node.name}\n",override_debug=False)
            weight_row = None
            with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(weights_dim_x), "optimizer_w_upd_outer") as (b1, weight_row):
                weight_column = None
                with pnlvm.helpers.for_loop_zero_inc(b1, ctx.int32_ty(weights_dim_y), "optimizer_w_upd_inner") as (b2, weight_column):
                    # sqrt(v_t) + eps
                    v_t_value = b2.load(b2.gep(
                        v_t_ptr, [zero, weight_row, weight_column]))
                    value = b2.call(sqrt, [v_t_value])
                    value = b2.fadd(value, eps)

                    # alpha_t * m_t
                    m_t_value = b2.load(b2.gep(
                        m_t_ptr, [zero, weight_row, weight_column]))
                    m_t_value = b2.fmul(alpha_t, m_t_value)

                    # value = alpha_t * m_t / (sqrt(v_t) + eps)
                    value = b2.fdiv(m_t_value, value)

                    old_weight_ptr = b2.gep(
                        weights_llvmlite, [zero, weight_row, weight_column])
                    
                    # new_weight = old_weight - value
                    value = b2.fsub(b2.load(old_weight_ptr), value)
                    b2.store(value, old_weight_ptr)

                    delta_w_val = b2.load(b2.gep(delta_w_ptr,[zero, weight_row, weight_column]))
                    ctx.inject_printf(b2,"%f ",delta_w_val,override_debug=False)
                ctx.inject_printf(b1,"\n",override_debug=False)
                
        ctx.inject_printf(builder, f"\t\t\tOPTIM DONE UPDATE\n",override_debug=False)

        builder.ret_void()

        return llvm_func


class SGDOptimizer(Optimizer):
    """Implements compiled Stocastic Gradient Descent optimizer (without momentum)"""
    # sets up parameters of model & the information required for forward computation
    def __init__(self, pytorch_model, lr=1e-3):
        super().__init__(pytorch_model)
        self.lr = lr

    def _get_optimizer_struct_type(self, ctx):
        return super()._get_optimizer_struct_type(ctx)

    def initialize_optimizer_struct(self, ctx, builder, optim_struct):
        pass

    # steps the adam optimizer (methodology: https://arxiv.org/pdf/1412.6980.pdf )
    def step(self, ctx):
        name = self._composition.name+"_SGD_STEP"
        # try:
        #     llvm_func = ctx.get_llvm_function(name)
        #     return llvm_func
        # except Exception as e:
        #     pass

        args = [self._get_optimizer_struct_type(ctx).as_pointer(),
                self._pytorch_model._get_param_struct_type(ctx).as_pointer()]
        builder = ctx.create_llvm_function(args, self, name)
        llvm_func = builder.function
        llvm_func.attributes.add('alwaysinline')
        optim_struct, model_params = llvm_func.args

        zero = ctx.int32_ty(0)
        delta_w = builder.gep(optim_struct, [zero, ctx.int32_ty(self._DELTA_W_NUM)])

        lr = ctx.float_ty(self.lr)
       
        gradient_struct_values = self._get_listof_gradient_struct_values()
        
        # update weights
        for (node, node_idx, afferent_node, afferent_node_index, matrix, _, _) in gradient_struct_values:
            node_idx_ir = ctx.int32_ty(node_idx)
            afferent_node_index_ir = ctx.int32_ty(afferent_node_index)

            delta_w_ptr = builder.gep(delta_w,[zero,node_idx_ir,afferent_node_index_ir])            
            weights_llvmlite, _, _ = self._pytorch_model._gen_get_node_weight_ptr(ctx, builder, model_params, node, afferent_node)
            
            multiplied_delta_w = self._pytorch_model._gen_inject_mat_scalar_mult(ctx, builder, delta_w_ptr, lr)
            self._pytorch_model._gen_inject_mat_sub(ctx, builder, weights_llvmlite, multiplied_delta_w, weights_llvmlite)

        builder.ret_void()

        return llvm_func
