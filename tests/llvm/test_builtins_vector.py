import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm


DIM_X=1500
# These are just basic tests to check that vector indexing and operations
# work correctly when compiled. The values don't matter much.
# Might as well make them representable in fp32 for single precision testing.
u = np.random.rand(DIM_X).astype(np.float32).astype(np.float64)
v = np.random.rand(DIM_X).astype(np.float32).astype(np.float64)
scalar = np.random.rand()


add_res = np.add(u, v)
sub_res = np.subtract(u, v)
mul_res = np.multiply(u, v)
smul_res = np.multiply(u, scalar)


@pytest.mark.benchmark(group="Hadamard")
@pytest.mark.parametrize("op, v, builtin, result", [
                         (np.add, v, "__pnl_builtin_vec_add", add_res),
                         (np.subtract, v, "__pnl_builtin_vec_sub", sub_res),
                         (np.multiply, v, "__pnl_builtin_vec_hadamard", mul_res),
                         (np.multiply, scalar, "__pnl_builtin_vec_scalar_mult", smul_res),
                         ], ids=["ADD", "SUB", "MUL", "SMUL"])
def test_vector_op(benchmark, op, v, builtin, result, func_mode):
    if func_mode == 'Python':
        def ex():
            return op(u, v)
    elif func_mode == 'LLVM':
        bin_f = pnlvm.LLVMBinaryFunction.get(builtin)
        dty = np.dtype(bin_f.byref_arg_types[0])
        assert dty == np.dtype(bin_f.byref_arg_types[1])
        assert dty == np.dtype(bin_f.byref_arg_types[3])

        lu = u.astype(dty)
        lv = dty.type(v) if np.isscalar(v) else v.astype(dty)
        lres = np.empty_like(lu)

        ct_u = lu.ctypes.data_as(bin_f.c_func.argtypes[0])
        ct_v = lv if np.isscalar(lv) else lv.ctypes.data_as(bin_f.c_func.argtypes[1])
        ct_res = lres.ctypes.data_as(bin_f.c_func.argtypes[3])

        def ex():
            bin_f(ct_u, ct_v, DIM_X, ct_res)
            return lres

    elif func_mode == 'PTX':
        bin_f = pnlvm.LLVMBinaryFunction.get(builtin)
        dty = np.dtype(bin_f.byref_arg_types[0])
        assert dty == np.dtype(bin_f.byref_arg_types[1])
        assert dty == np.dtype(bin_f.byref_arg_types[3])

        lu = u.astype(dty)
        lv = dty.type(v) if np.isscalar(v) else v.astype(dty)
        lres = np.empty_like(lu)

        cuda_u = pnlvm.jit_engine.pycuda.driver.In(lu)
        cuda_v = lv if np.isscalar(lv) else pnlvm.jit_engine.pycuda.driver.In(lv)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(lres)
        def ex():
            bin_f.cuda_call(cuda_u, cuda_v, np.int32(DIM_X), cuda_res)
            return lres

    res = benchmark(ex)
    assert np.allclose(res, result)


@pytest.mark.benchmark(group="Sum")
def test_vector_sum(benchmark, func_mode):
    if func_mode == 'Python':
        def ex():
            return np.sum(u)
    elif func_mode == 'LLVM':
        bin_f = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum")

        lu = u.astype(np.dtype(bin_f.byref_arg_types[0]))
        llvm_res = np.empty(1, dtype=lu.dtype)

        ct_u = lu.ctypes.data_as(bin_f.c_func.argtypes[0])
        ct_res = llvm_res.ctypes.data_as(bin_f.c_func.argtypes[2])

        def ex():
            bin_f(ct_u, DIM_X, ct_res)
            return llvm_res[0]
    elif func_mode == 'PTX':
        bin_f = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum")
        lu = u.astype(np.dtype(bin_f.byref_arg_types[0]))
        cuda_u = pnlvm.jit_engine.pycuda.driver.In(lu)
        res = np.empty(1, dtype=lu.dtype)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(res)
        def ex():
            bin_f.cuda_call(cuda_u, np.int32(DIM_X), cuda_res)
            return res[0]

    res = benchmark(ex)
    assert np.allclose(res, sum(u))
