import numpy as np
import pytest

import psyneulink.core.components.functions.function as Function
import psyneulink.core.globals.keywords as kw
import psyneulink.core.llvm as pnlvm

@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.benchmark
def test_basic(size, benchmark):
    variable = np.random.rand(size)
    f = Function.Identity(default_variable=variable)
    res = benchmark(f.function, variable)
    assert np.allclose(res, variable)

@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.benchmark
def test_llvm(size, benchmark):
    variable = np.random.rand(size)
    f = Function.Identity(default_variable=variable)
    m = pnlvm.execution.FuncExecution(f, None)
    res = benchmark(m.execute, variable)
    assert np.allclose(res, variable)

@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.benchmark
@pytest.mark.skipif(not pnlvm.ptx_enabled, reason="PTX engine not enabled/available")
def test_ptx_cuda(size, benchmark):
    variable = np.random.rand(size)
    f = Function.Identity(default_variable=variable)
    m = pnlvm.execution.FuncExecution(f, None)
    res = benchmark(m.cuda_execute, variable)
    assert np.allclose(res, variable)
