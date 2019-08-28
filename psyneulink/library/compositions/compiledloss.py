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


__all__ = ['MSELoss']


class Loss():

    def __init__(self, pytorch_model):
        self._pytorch_model = pytorch_model
        self._composition = pytorch_model._composition
        self._structs = []

        self._DELTA_W_NUM = 0

    def _gen_inject_lossfunc_call(self,ctx,builder,bin_func,value,target,dim):
        return builder.call(bin_func, [builder.bitcast(value, ctx.float_ty.as_pointer()), builder.bitcast(target, ctx.float_ty.as_pointer()), ctx.int32_ty(dim)])
# Class that is used to represent a compiled optimizer - aims to reimplement the logic of torch.optimizer in the form of llvmlite compileable code


class MSELoss(Loss):
    '''
    Implements compiled MSE Loss
    '''
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
        args = [ctx.float_ty.as_pointer(),
                ctx.float_ty.as_pointer(),
                ctx.int32_ty
                ]
        builder = ctx.create_llvm_function(args, self, name,return_type=ctx.float_ty)
        llvm_func = builder.function
        llvm_func.attributes.add('alwaysinline')
        value, target, dim = llvm_func.args

        sum = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(0),sum)

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, dim, "mse_sum_loop") as (builder, index):
            value_ptr = builder.gep(value,[index])
            target_ptr = builder.gep(target,[index])
            diff = builder.fsub(builder.load(value_ptr),builder.load(target_ptr))
            diff = builder.fmul(diff,diff)
            builder.store(builder.fadd(builder.load(sum),diff),sum)

        builder.ret(builder.load(sum))

        return llvm_func

    # inserts the computation for dC/da
    def _gen_inject_loss_differential(self,ctx,builder,value,target,dim,output=None,sum_loss=False):
        
        if output is None:
            output = builder.alloca(
                ir.types.ArrayType(
                    ctx.float_ty,
                    dim
                )
            )
            # zero output vector
            self._pytorch_model._gen_inject_vec_scalar_mult(ctx,builder,output,ctx.float_ty(0),dim,output)

        if sum_loss is False:
            self._pytorch_model._gen_inject_vec_sub(ctx,builder,value,target,dim,output)
            self._pytorch_model._gen_inject_vec_scalar_mult(ctx,builder,output,ctx.float_ty(2),dim,output)
        else:
            # in this case, we add the loss
            tmp = self._pytorch_model._gen_inject_vec_sub(ctx,builder,value,target,dim)
            self._pytorch_model._gen_inject_vec_scalar_mult(ctx,builder,tmp,ctx.float_ty(2),dim,tmp)
            self._pytorch_model._gen_inject_vec_add(ctx,builder,output,tmp,dim,output)
        return output
