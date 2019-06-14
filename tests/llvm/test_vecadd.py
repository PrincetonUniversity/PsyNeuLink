#!/usr/bin/python3

import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

from llvmlite import ir

DIM_X=1000

u = np.random.rand(DIM_X)
v = np.random.rand(DIM_X)

llvm_res = np.random.rand(DIM_X)
result = np.add(u,v)

@pytest.mark.llvm
@pytest.mark.benchmark
def test_vecadd_numpy(benchmark):
    numpy_res = benchmark(np.add, u, v)
    assert np.allclose(numpy_res, result)

ct_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x = DIM_X

@pytest.mark.llvm
@pytest.mark.benchmark
def test_vecadd_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vec_add")
    benchmark(llvm_fun, ct_u, ct_v, x, ct_res)
    assert np.allclose(llvm_res, result)
