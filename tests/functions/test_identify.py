import numpy as np
import psyneulink.core.components.functions.function as Function
import psyneulink.core.globals.keywords as kw
import pytest

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
    res = benchmark(f.bin_function, variable)
    assert np.allclose(res, variable)
