#!/usr/bin/python3

import copy
import ctypes
import psyneulink.llvm as pnlvm
import numpy as np
from llvmlite import ir
import pytest
import functools


DIM_X=1000
TST_MIN=1.0
TST_MAX=3.0

vector = np.random.rand(DIM_X)

@pytest.mark.llvm
def test_helper_fclamp():

    with pnlvm.LLVMBuilderContext() as ctx:
        local_vec = copy.deepcopy(vector)
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, ctx.int32_ty))

        # Create clamp function
        custom_name = ctx.module.get_unique_name("clamp")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        vec, count = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        def _clamp_wrap(builder, index, minv, maxv, vec):
            val_ptr = builder.gep(vec, [index])
            val = builder.load(val_ptr)
            val = pnlvm.helpers.fclamp(builder, val, minv, maxv)
            builder.store(val, val_ptr)
        kwargs = {"minv": ctx.float_ty(TST_MIN), "maxv":ctx.float_ty(TST_MAX), "vec":vec}
        inner = functools.partial(_clamp_wrap, **kwargs)

        builder = pnlvm.helpers.for_loop_zero_inc(builder, count, inner, "linear")


        builder.ret_void()
    ref = np.clip(vector, TST_MIN, TST_MAX)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    ct_ty = pnlvm._convert_llvm_ir_to_ctype(double_ptr_ty)
    ct_vec = local_vec.ctypes.data_as(ct_ty)

    bin_f(ct_vec, DIM_X)

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
def test_helper_fclamp_const():

    with pnlvm.LLVMBuilderContext() as ctx:
        local_vec = copy.deepcopy(vector)
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, ctx.int32_ty))

        # Create clamp function
        custom_name = ctx.module.get_unique_name("clamp")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        vec, count = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        def _clamp_wrap(builder, index, vec):
            val_ptr = builder.gep(vec, [index])
            val = builder.load(val_ptr)
            val = pnlvm.helpers.fclamp_const(builder, val, TST_MIN, TST_MAX)
            builder.store(val, val_ptr)
        kwargs = {"vec":vec}
        inner = functools.partial(_clamp_wrap, **kwargs)

        builder = pnlvm.helpers.for_loop_zero_inc(builder, count, inner, "linear")


        builder.ret_void()
    ref = np.clip(vector, TST_MIN, TST_MAX)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    ct_ty = pnlvm._convert_llvm_ir_to_ctype(double_ptr_ty)
    ct_vec = local_vec.ctypes.data_as(ct_ty)

    bin_f(ct_vec, DIM_X)

    assert np.array_equal(local_vec, ref)
