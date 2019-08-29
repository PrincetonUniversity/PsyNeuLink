import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

ITERATIONS=100
DIM_X=1000
DIM_Y=2000

matrix = np.random.rand(DIM_X, DIM_Y)
vector = np.random.rand(DIM_X)
llvm_res = np.random.rand(DIM_Y)

ct_vec = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_mat = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x, y = matrix.shape

@pytest.mark.llvm
def test_recompile():
    # The original builtin mxv function
    binf = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    orig_res = np.empty_like(llvm_res)
    ct_res = orig_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    binf.c_func(ct_vec, ct_mat, x, y, ct_res)

    # Rebuild and try again
    # This is not a public API
    pnlvm._llvm_build()

    rebuild_res = np.empty_like(llvm_res)
    ct_res = rebuild_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    binf.c_func(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(orig_res, rebuild_res)

    # Get a new pointer
    binf2 = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    new_res = np.empty_like(llvm_res)
    ct_res = new_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    binf.c_func(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(rebuild_res, new_res)

    callable_res = np.empty_like(llvm_res)
    ct_res = callable_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    binf(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(new_res, callable_res)
