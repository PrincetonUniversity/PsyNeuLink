#!/usr/bin/python3

import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

from llvmlite import ir

DIM_X = 1000
DIM_Y = 2000
u = np.random.rand(DIM_X,DIM_Y)
scalar = np.random.rand()

llvm_res = np.random.rand(DIM_X,DIM_Y)
result = np.multiply(u,scalar)

@pytest.mark.llvm
@pytest.mark.benchmark
def test_matadd_numpy(benchmark):
    numpy_res = benchmark(np.multiply, u, scalar)
    assert np.allclose(numpy_res, result)

ct_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x = DIM_X
y = DIM_Y
@pytest.mark.llvm
@pytest.mark.benchmark
def test_matadd_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_mat_scalar_mult")
    benchmark(llvm_fun, ct_u, scalar, x, y, ct_res)
    assert np.allclose(llvm_res, result)
