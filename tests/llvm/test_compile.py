import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

DIM_X=1000
DIM_Y=2000

@pytest.mark.llvm
def test_recompile():
    # The original builtin mxv function
    bin_f = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')

    vector = np.random.rand(DIM_X).astype(bin_f.np_params[0].base)
    matrix = np.random.rand(DIM_X, DIM_Y).astype(bin_f.np_params[1].base)
    llvm_res = np.empty(DIM_Y, dtype=bin_f.np_params[4].base)

    x, y = matrix.shape

    ct_vec = vector.ctypes.data_as(bin_f.c_func.argtypes[0])
    ct_mat = matrix.ctypes.data_as(bin_f.c_func.argtypes[1])

    orig_res = np.empty_like(llvm_res)
    ct_res = orig_res.ctypes.data_as(bin_f.c_func.argtypes[4])

    bin_f.c_func(ct_vec, ct_mat, x, y, ct_res)

    # Rebuild and try again
    # This is not a public API
    pnlvm._llvm_build()

    rebuild_res = np.empty_like(llvm_res)
    ct_res = rebuild_res.ctypes.data_as(bin_f.c_func.argtypes[4])

    bin_f.c_func(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(orig_res, rebuild_res)

    # Get a new pointer
    bin_f2 = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    new_res = np.empty_like(llvm_res)
    ct_res = new_res.ctypes.data_as(bin_f2.c_func.argtypes[4])

    bin_f2.c_func(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(rebuild_res, new_res)

    callable_res = np.empty_like(llvm_res)
    ct_res = callable_res.ctypes.data_as(bin_f.c_func.argtypes[4])

    bin_f2(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(new_res, callable_res)
