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
