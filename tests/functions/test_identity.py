import numpy as np
import pytest

import psyneulink.core.components.functions.nonstateful.transferfunctions as Functions

@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.benchmark(group="IdentityFunction")
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_basic(size, benchmark, func_mode):
    variable = np.random.rand(size)
    f = Functions.Identity(default_variable=variable)
    EX = pytest.helpers.get_func_execution(f, func_mode)

    res = benchmark(EX, variable)
    np.testing.assert_allclose(res, variable)
