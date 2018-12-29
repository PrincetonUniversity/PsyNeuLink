#!/usr/bin/python3

import ctypes
import numpy as np
import pytest
import timeit

from psyneulink.core import llvm as pnlvm

from llvmlite import ir

DIM_X=1000
DIM_Y=2000

matrix = np.random.rand(DIM_X, DIM_Y)
vector = np.random.rand(DIM_X)
llvm_res = np.random.rand(DIM_Y)
result = np.dot(vector, matrix)

@pytest.mark.llvm
@pytest.mark.benchmark
def test_matmul_numpy(benchmark):
    numpy_res = benchmark(np.dot, vector, matrix)
    assert np.allclose(numpy_res, result)

#start = timeit.default_timer()
ct_vec = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_mat = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x, y = matrix.shape
#stop = timeit.default_timer()
#print("Convert time elapsed {:f}".format(stop-start))

@pytest.mark.llvm
@pytest.mark.benchmark
def test_matmul_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    benchmark(llvm_fun, ct_vec, ct_mat, x, y, ct_res)
    assert np.allclose(llvm_res, result)

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.benchmark
def test_matmul_cuda(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    cuda_vec = pnlvm.jit_engine.pycuda.driver.In(vector)
    cuda_mat = pnlvm.jit_engine.pycuda.driver.In(matrix)
    cuda_res = pnlvm.jit_engine.pycuda.driver.Out(llvm_res)
    benchmark(llvm_fun.cuda_call, cuda_vec, cuda_mat, np.int32(x), np.int32(y), cuda_res)
    assert np.allclose(llvm_res, result)

@pytest.mark.llvm
@pytest.mark.benchmark
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_matmul_llvm_constant_dim(benchmark, mode):
    custom_name = None

    with pnlvm.LLVMBuilderContext() as ctx:
        custom_name = ctx.get_unique_name("vxsqm")
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, double_ptr_ty))

        # get builtin IR
        builtin = ctx.get_llvm_function('__pnl_builtin_vxm')

        # Create square vector matrix multiply
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        _x = ctx.int32_ty(x)
        _y = ctx.int32_ty(y)
        _v, _m, _o = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.call(builtin, [_v, _m, _x, _y, _o])
        builder.ret_void()

    binf2 = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        benchmark(binf2, ct_vec, ct_mat, ct_res)
    else:
        import pycuda
        cuda_vec = pycuda.driver.In(vector)
        cuda_mat = pycuda.driver.In(matrix)
        cuda_res = pycuda.driver.Out(llvm_res)
        benchmark(binf2.cuda_call, cuda_vec, cuda_mat, cuda_res)
    assert np.allclose(llvm_res, result)
