import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

ITERATIONS=100
DIM_X=1000
DIM_Y=2000

@pytest.mark.llvm
def test_recompile():
    # The original builtin mxv function
    binf = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    dty = np.dtype(binf.byref_arg_types[0])
    assert dty == np.dtype(binf.byref_arg_types[1])
    assert dty == np.dtype(binf.byref_arg_types[4])

    matrix = np.random.rand(DIM_X, DIM_Y).astype(dty)
    vector = np.random.rand(DIM_X).astype(dty)
    llvm_res = np.empty(DIM_Y, dtype=dty)

    x, y = matrix.shape

    ct_vec = vector.ctypes.data_as(binf.c_func.argtypes[0])
    ct_mat = matrix.ctypes.data_as(binf.c_func.argtypes[1])

    orig_res = np.empty_like(llvm_res)
    ct_res = orig_res.ctypes.data_as(binf.c_func.argtypes[4])

    binf.c_func(ct_vec, ct_mat, x, y, ct_res)

    # Rebuild and try again
    # This is not a public API
    pnlvm._llvm_build()

    rebuild_res = np.empty_like(llvm_res)
    ct_res = rebuild_res.ctypes.data_as(binf.c_func.argtypes[4])

    binf.c_func(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(orig_res, rebuild_res)

    # Get a new pointer
    binf2 = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')
    new_res = np.empty_like(llvm_res)
    ct_res = new_res.ctypes.data_as(binf2.c_func.argtypes[4])

    binf2.c_func(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(rebuild_res, new_res)

    callable_res = np.empty_like(llvm_res)
    ct_res = callable_res.ctypes.data_as(binf.c_func.argtypes[4])

    binf2(ct_vec, ct_mat, x, y, ct_res)
    assert np.array_equal(new_res, callable_res)
