from psyneulink.core import llvm as pnlvm


__all__ = ['MSELoss']


class Loss():

    def __init__(self, pytorch_model):
        self._pytorch_model = pytorch_model
        self._composition = pytorch_model._composition
        self._structs = []

        self._DELTA_W_NUM = 0

    def _gen_inject_lossfunc_call(self, ctx, builder, bin_func, value, target):
        return builder.call(bin_func, [builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(0)]),
                                       ctx.int32_ty(len(value.type.pointee)),
                                       builder.gep(target, [ctx.int32_ty(0), ctx.int32_ty(0)])])
# Class that is used to represent a compiled optimizer - aims to reimplement the logic of torch.optimizer in the form of llvmlite compileable code


class MSELoss(Loss):
    """Implements compiled MSE Loss"""
    def __init__(self,pytorch_model, reduction='sum'):
        if reduction not in ['sum']:
            raise Exception("Unsupported compiled reduction type "+reduction)
        
        super().__init__(pytorch_model)
        self.reduction = reduction

    # creates a bin func that returns the mse loss 
    def _gen_call_function(self,ctx):
        name = self._composition.name+"_MSE_CALL"

        # args:
        # 1) pointer to network output
        # 2) pointer to target
        # 3) dimensionality
        args = [ctx.float_ty.as_pointer(), ctx.int32_ty, ctx.float_ty.as_pointer()]
        builder = ctx.create_llvm_function(args, self, name,return_type=ctx.float_ty)
        value, dim, target = builder.function.args

        sum = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(-0.0), sum)

        with pnlvm.helpers.for_loop_zero_inc(builder, dim, "mse_sum_loop") as (b1, index):
            value_ptr = b1.gep(value,[index])
            target_ptr = b1.gep(target,[index])
            diff = b1.fsub(b1.load(value_ptr),b1.load(target_ptr))
            diff = b1.fmul(diff,diff)
            b1.store(b1.fadd(b1.load(sum),diff),sum)

        builder.ret(builder.load(sum))

        return builder.function

    # inserts the computation for dC/da
    def _gen_inject_loss_differential(self, ctx, builder, value, target, output=None, sum_loss=False):
        dim = len(value.type.pointee)
        assert len(target.type.pointee) == dim
        if output is None:
            output = builder.alloca(pnlvm.ir.types.ArrayType(ctx.float_ty, dim))
            # zero output vector
            builder.store(output.type.pointee(None), output)
        assert len(output.type.pointee) == dim

        if sum_loss is False:
            self._pytorch_model._gen_inject_vec_sub(ctx, builder, value, target, output)
        else:
            # in this case, we add the loss
            tmp = self._pytorch_model._gen_inject_vec_sub(ctx, builder, value, target)
            self._pytorch_model._gen_inject_vec_add(ctx, builder, output, tmp, output)
        return output
