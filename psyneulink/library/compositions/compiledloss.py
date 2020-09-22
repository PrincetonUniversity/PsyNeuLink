from psyneulink.core import llvm as pnlvm
from psyneulink.library.compositions.pytorchllvmhelper import *

__all__ = ['MSELoss']


class Loss():

    def __init__(self):
        self._DELTA_W_NUM = 0


    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        return self._gen_loss_function(ctx)


    def gen_inject_lossfunc_call(self, ctx, builder, bin_func, value, target):
        return builder.call(bin_func, [builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(0)]),
                                       ctx.int32_ty(len(value.type.pointee)),
                                       builder.gep(target, [ctx.int32_ty(0), ctx.int32_ty(0)])])

class MSELoss(Loss):
    """Implements compiled MSE Loss"""
    def __init__(self, reduction='sum'):
        if reduction not in ['sum']:
            raise Exception("Unsupported compiled reduction type " + reduction)

        super().__init__()
        self.reduction = reduction

    def _gen_loss_function(self, ctx):
        name = "LEARNING_MSE_CALL"

        # args:
        # 1) pointer to network output
        # 2) pointer to target
        # 3) dimensionality
        args = [ctx.float_ty.as_pointer(), ctx.int32_ty, ctx.float_ty.as_pointer()]
        builder = ctx.create_llvm_function(args, self, name, return_type=ctx.float_ty)
        value, dim, target = builder.function.args

        sum = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(-0.0), sum)

        with pnlvm.helpers.for_loop_zero_inc(builder, dim, "mse_sum_loop") as (b1, index):
            value_ptr = b1.gep(value,[index])
            target_ptr = b1.gep(target,[index])
            diff = b1.fsub(b1.load(value_ptr),b1.load(target_ptr))
            diff = b1.fmul(diff,diff)
            b1.store(b1.fadd(b1.load(sum),diff),sum)

        # Average the values in sum by dimensionality
        builder.store(builder.fdiv(builder.load(sum),builder.uitofp(dim, ctx.float_ty)), sum)

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
            # we take mean
            gen_inject_vec_sub(ctx, builder, value, target, output)
            # multiply each element i by 2/n to get dC/da_i
            scalar_mult = builder.fdiv(ctx.float_ty(2), ctx.float_ty(dim))
            with pnlvm.helpers.for_loop_zero_inc(builder, ctx.int32_ty(dim), "mse_mean_mult_loop") as (b1, index):
                element_ptr = b1.gep(output, [ctx.int32_ty(0), index])
                b1.store(b1.fmul(b1.load(element_ptr),scalar_mult),element_ptr)
        else:
            # in this case, we add the loss
            tmp = gen_inject_vec_sub(ctx, builder, value, target)
            gen_inject_vec_add(ctx, builder, output, tmp, output)
        return output
