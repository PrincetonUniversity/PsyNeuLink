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

    def _numpy_args(bin_f):
        np_u = u.astype(bin_f.np_arg_dtypes[0])
        np_v = bin_f.np_arg_dtypes[1].type(v) if np.isscalar(v) else v.astype(bin_f.np_arg_dtypes[1])
        np_res = np.empty_like(np_u)

        return np_u, np_v, np_res

    if func_mode == 'Python':
        def ex():
            return op(u, v)

    elif func_mode == 'LLVM':
        bin_f = pnlvm.LLVMBinaryFunction.get(builtin, ctype_ptr_args=(0, 1, 3))
        lu, lv, lres = _numpy_args(bin_f)

        ct_u = lu.ctypes.data_as(bin_f.c_func.argtypes[0])
        ct_v = lv if np.isscalar(lv) else lv.ctypes.data_as(bin_f.c_func.argtypes[1])
        ct_res = lres.ctypes.data_as(bin_f.c_func.argtypes[3])

        def ex():
            bin_f(ct_u, ct_v, DIM_X, ct_res)
            return lres

    elif func_mode == 'PTX':
        bin_f = pnlvm.LLVMBinaryFunction.get(builtin)
        lu, lv, lres = _numpy_args(bin_f)

        cuda_u = pnlvm.jit_engine.pycuda.driver.In(lu)
        cuda_v = lv if np.isscalar(lv) else pnlvm.jit_engine.pycuda.driver.In(lv)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(lres)

        def ex():
            bin_f.cuda_call(cuda_u, cuda_v, np.int32(DIM_X), cuda_res)
            return lres

    res = benchmark(ex)
    np.testing.assert_allclose(res, result)


@pytest.mark.benchmark(group="Sum")
def test_vector_sum(benchmark, func_mode):

    if func_mode == 'Python':
        def ex():
            return np.sum(u)

    elif func_mode == 'LLVM':
        bin_f = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum", ctype_ptr_args=(0,))

        np_u = u.astype(bin_f.np_arg_dtypes[0])
        np_res = bin_f.np_buffer_for_arg(2)

        ct_u = np_u.ctypes.data_as(bin_f.c_func.argtypes[0])

        def ex():
            bin_f(ct_u, DIM_X, np_res)
            return np_res

    elif func_mode == 'PTX':
        bin_f = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_sum")

        np_u = u.astype(bin_f.np_arg_dtypes[0])
        np_res = bin_f.np_buffer_for_arg(2)

        cuda_u = pnlvm.jit_engine.pycuda.driver.In(np_u)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(np_res)

        def ex():
            bin_f.cuda_call(cuda_u, np.int32(DIM_X), cuda_res)
            return np_res

    res = benchmark(ex)
    np.testing.assert_allclose(res, np.sum(u))
