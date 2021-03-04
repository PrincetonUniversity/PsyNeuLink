import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm


DIM_X=1000


u = np.random.rand(DIM_X)
v = np.random.rand(DIM_X)
scalar = np.random.rand()


llvm_res = np.random.rand(DIM_X)
add_res = np.add(u, v)
sub_res = np.subtract(u, v)
mul_res = np.multiply(u, v)
smul_res = np.multiply(u, scalar)


ct_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.mark.benchmark(group="Hadamard")
@pytest.mark.parametrize("op, y, llvm_y, builtin, result", [
                         (np.add, v, ct_v, "__pnl_builtin_vec_add", add_res),
                         (np.subtract, v, ct_v, "__pnl_builtin_vec_sub", sub_res),
                         (np.multiply, v, ct_v, "__pnl_builtin_vec_hadamard", mul_res),
                         (np.multiply, scalar, scalar, "__pnl_builtin_vec_scalar_mult", smul_res),
                         ], ids=["ADD", "SUB", "MUL", "SMUL"])
def test_vector_op(benchmark, op, y, llvm_y, builtin, result, func_mode):
    if func_mode == 'Python':
        def ex():
            return op(u, y)
    elif func_mode == 'LLVM':
        bin_f = pnlvm.LLVMBinaryFunction.get(builtin)
        def ex():
            bin_f(ct_u, llvm_y, DIM_X, ct_res)
            return llvm_res
    elif func_mode == 'PTX':
        bin_f = pnlvm.LLVMBinaryFunction.get(builtin)
        cuda_u = pnlvm.jit_engine.pycuda.driver.In(u)
        cuda_y = np.float64(y) if np.isscalar(y) else pnlvm.jit_engine.pycuda.driver.In(y)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(llvm_res)
        def ex():
            bin_f.cuda_call(cuda_u, cuda_y, np.int32(DIM_X), cuda_res)
            return llvm_res

    res = benchmark(ex)
    assert np.allclose(res, result)


@pytest.mark.benchmark(group="Sum")
def test_vector_sum(benchmark, func_mode):
    if func_mode == 'Python':
        def ex():
            return np.sum(u)
    elif func_mode == 'LLVM':
        bin_f = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum")
        def ex():
            bin_f(ct_u, DIM_X, ct_res)
            return llvm_res[0]
    elif func_mode == 'PTX':
        bin_f = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum")
        cuda_u = pnlvm.jit_engine.pycuda.driver.In(u)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(llvm_res)
        def ex():
            bin_f.cuda_call(cuda_u, np.int32(DIM_X), cuda_res)
            return llvm_res[0]

    res = benchmark(ex)
    assert np.allclose(res, sum(u))
