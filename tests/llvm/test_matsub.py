#!/usr/bin/python3

import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

from llvmlite import ir

DIM_X = 1000
DIM_Y = 2000
u = np.random.rand(DIM_X,DIM_Y)
v = np.random.rand(DIM_X,DIM_Y)

llvm_res = np.random.rand(DIM_X,DIM_Y)
result = np.subtract(u,v)

@pytest.mark.llvm
@pytest.mark.benchmark
def test_matsub_numpy(benchmark):
    numpy_res = benchmark(np.subtract, u, v)
    assert np.allclose(numpy_res, result)

ct_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x = DIM_X
y = DIM_Y
@pytest.mark.llvm
@pytest.mark.benchmark
def test_matsub_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_mat_sub")
    benchmark(llvm_fun, ct_u, ct_v, x, y, ct_res)
    assert np.allclose(llvm_res, result)
