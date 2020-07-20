import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm


DIM_X=1000
DIM_Y=10

u = np.random.rand(DIM_X)
v = np.random.rand(DIM_X)
w = np.random.rand(DIM_Y)
scalar = np.random.rand()


llvm_res = np.random.rand(DIM_X)
add_res = np.add(u, v)
sub_res = np.subtract(u, v)
mul_res = np.multiply(u, v)
smul_res = np.multiply(u, scalar)
outer_res = np.outer(u, w)
llvm_outer_res = np.random.rand(DIM_X, DIM_Y)

ct_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_w = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_outer_res = llvm_outer_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

@pytest.mark.benchmark(group="Hadamard")
@pytest.mark.parametrize("op, y, llvm_y, builtin, result", [
                         (np.add, v, ct_v, "__pnl_builtin_vec_add", add_res),
                         (np.subtract, v, ct_v, "__pnl_builtin_vec_sub", sub_res),
                         (np.multiply, v, ct_v, "__pnl_builtin_vec_hadamard", mul_res),
                         (np.multiply, scalar, scalar, "__pnl_builtin_vec_scalar_mult", smul_res),
                         ], ids=["ADD", "SUB", "MUL", "SMUL"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm)])
def test_vector_op(benchmark, op, y, llvm_y, builtin, result, mode):
    if mode == 'Python':
        res = benchmark(op, u, y)
    elif mode == 'LLVM':
        llvm_fun = pnlvm.LLVMBinaryFunction.get(builtin)
        benchmark(llvm_fun, ct_u, llvm_y, DIM_X, ct_res)
        res = llvm_res
    assert np.allclose(res, result)

@pytest.mark.benchmark(group="Sum")
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm)])
def test_vector_sum(benchmark, mode):
    if mode == 'Python':
        res = benchmark(np.sum, u)
    elif mode == 'LLVM':
        llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum")
        benchmark(llvm_fun, ct_u, DIM_X, ct_res)
        res = llvm_res[0]
    assert np.allclose(res, sum(u))

@pytest.mark.benchmark(group="VecOuter")
def test_vec_outer_numpy(benchmark):
    numpy_res = benchmark(np.outer, u, w)
    assert np.allclose(numpy_res, outer_res)

@pytest.mark.llvm
@pytest.mark.benchmark(group="VecOuter")
def test_vec_outer_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_outer_product")
    benchmark(llvm_fun, ct_u, ct_w, DIM_X, DIM_Y, ct_outer_res)
    assert np.allclose(llvm_outer_res, outer_res)

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.benchmark(group="VecOuter")
def test_vec_outer_cuda(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_outer_product")
    cuda_m1 = pnlvm.jit_engine.pycuda.driver.In(u)
    cuda_m2 = pnlvm.jit_engine.pycuda.driver.In(w)
    cuda_res = pnlvm.jit_engine.pycuda.driver.Out(llvm_outer_res)
    benchmark(llvm_fun.cuda_call, cuda_m1, cuda_m2, np.int32(DIM_X), np.int32(DIM_Y), cuda_res)
    assert np.allclose(llvm_outer_res, outer_res)
