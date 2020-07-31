import ctypes
import ctypes.util
import copy
import numpy as np
import pytest
import sys

from psyneulink.core import llvm as pnlvm
from llvmlite import ir


DIM_X = 1000
TST_MIN = 1.0
TST_MAX = 3.0

VECTOR = np.random.rand(DIM_X)

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_fclamp(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        local_vec = copy.deepcopy(VECTOR)
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, ctx.int32_ty,
                                                  double_ptr_ty))

        # Create clamp function
        custom_name = ctx.get_unique_name("clamp")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        vec, count, bounds = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        tst_min = builder.load(builder.gep(bounds, [ctx.int32_ty(0)]))
        tst_max = builder.load(builder.gep(bounds, [ctx.int32_ty(1)]))

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "linear") as (b1, index):
            val_ptr = b1.gep(vec, [index])
            val = b1.load(val_ptr)
            val = pnlvm.helpers.fclamp(b1, val, tst_min, tst_max)
            b1.store(val, val_ptr)

        builder.ret_void()

    ref = np.clip(VECTOR, TST_MIN, TST_MAX)
    bounds = np.asfarray([TST_MIN, TST_MAX])
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = pnlvm._convert_llvm_ir_to_ctype(double_ptr_ty)
        ct_vec = local_vec.ctypes.data_as(ct_ty)
        ct_bounds = bounds.ctypes.data_as(ct_ty)

        bin_f(ct_vec, DIM_X, ct_bounds)
    else:
        bin_f.cuda_wrap_call(local_vec, np.int32(DIM_X), bounds)

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_fclamp_const(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        local_vec = copy.deepcopy(VECTOR)
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, ctx.int32_ty))

        # Create clamp function
        custom_name = ctx.get_unique_name("clamp")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        vec, count = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "linear") as (b1, index):
            val_ptr = b1.gep(vec, [index])
            val = b1.load(val_ptr)
            val = pnlvm.helpers.fclamp(b1, val, TST_MIN, TST_MAX)
            b1.store(val, val_ptr)

        builder.ret_void()

    ref = np.clip(VECTOR, TST_MIN, TST_MAX)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = pnlvm._convert_llvm_ir_to_ctype(double_ptr_ty)
        ct_vec = local_vec.ctypes.data_as(ct_ty)

        bin_f(ct_vec, DIM_X)
    else:
        bin_f.cuda_wrap_call(local_vec, np.int32(DIM_X))

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_is_close(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), [double_ptr_ty, double_ptr_ty,
                                                  double_ptr_ty, ctx.int32_ty])

        # Create clamp function
        custom_name = ctx.get_unique_name("all_close")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        in1, in2, out, count = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "compare") as (b1, index):
            val1_ptr = b1.gep(in1, [index])
            val2_ptr = b1.gep(in2, [index])
            val1 = b1.load(val1_ptr)
            val2 = b1.load(val2_ptr)
            close = pnlvm.helpers.is_close(b1, val1, val2)
            out_ptr = b1.gep(out, [index])
            out_val = b1.select(close, ctx.float_ty(1), ctx.float_ty(0))
            b1.store(out_val, out_ptr)

        builder.ret_void()

    vec1 = copy.deepcopy(VECTOR)
    tmp = np.random.rand(DIM_X)
    tmp[0::2] = vec1[0::2]
    vec2 = np.asfarray(tmp)
    assert len(vec1) == len(vec2)
    res = np.empty_like(vec2)

    ref = np.isclose(vec1, vec2)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = pnlvm._convert_llvm_ir_to_ctype(double_ptr_ty)
        ct_vec1 = vec1.ctypes.data_as(ct_ty)
        ct_vec2 = vec2.ctypes.data_as(ct_ty)
        ct_res = res.ctypes.data_as(ct_ty)

        bin_f(ct_vec1, ct_vec2, ct_res, DIM_X)
    else:
        bin_f.cuda_wrap_call(vec1, vec2, res, np.int32(DIM_X))

    assert np.array_equal(res, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_all_close(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        arr_ptr_ty = ir.ArrayType(ctx.float_ty, DIM_X).as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), [arr_ptr_ty, arr_ptr_ty,
                                                  ir.IntType(32).as_pointer()])

        custom_name = ctx.get_unique_name("all_close")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        in1, in2, out = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        all_close = pnlvm.helpers.all_close(builder, in1, in2)
        res = builder.select(all_close, out.type.pointee(1), out.type.pointee(0))
        builder.store(res, out)
        builder.ret_void()

    vec1 = copy.deepcopy(VECTOR)
    vec2 = copy.deepcopy(VECTOR)

    ref = np.allclose(vec1, vec2)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = pnlvm._convert_llvm_ir_to_ctype(arr_ptr_ty)
        ct_vec1 = vec1.ctypes.data_as(ct_ty)
        ct_vec2 = vec2.ctypes.data_as(ct_ty)
        res = ctypes.c_int32()

        bin_f(ct_vec1, ct_vec2, ctypes.byref(res))
    else:
        res = np.array([5], dtype=np.int32)
        bin_f.cuda_wrap_call(vec1, vec2, res)
        res = res[0]

    assert np.array_equal(res, ref)

@pytest.mark.llvm
@pytest.mark.parametrize("ir_argtype,format_spec,values_to_check", [
    (pnlvm.ir.IntType(32), "%u", range(0, 100)),
    (pnlvm.ir.IntType(64), "%ld", [int(-4E10), int(-3E10), int(-2E10)]),
    (pnlvm.ir.DoubleType(), "%lf", [x *.5 for x in range(0, 10)]),
    ], ids=["i32", "i64", "double"])
@pytest.mark.skipif(sys.platform == 'win32', reason="Loading C library is complicated on windows")
def test_helper_printf(capfd, ir_argtype, format_spec, values_to_check):
    format_str = f"Hello {(format_spec+' ')*len(values_to_check)} \n"
    with pnlvm.LLVMBuilderContext() as ctx:
        func_ty = ir.FunctionType(ir.VoidType(), [])
        ir_values_to_check = [ir_argtype(i) for i in values_to_check]
        custom_name = ctx.get_unique_name("test_printf")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        pnlvm.helpers.printf(builder, format_str, *ir_values_to_check, override_debug=True)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)


    # Printf is buffered in libc.
    bin_f()
    libc = ctypes.util.find_library("c")
    libc = ctypes.CDLL(libc)
    libc.fflush(0)
    assert capfd.readouterr().out == format_str % tuple(values_to_check)
