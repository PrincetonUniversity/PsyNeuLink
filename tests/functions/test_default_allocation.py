import numpy as np
import pytest

import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import DefaultAllocationFunction

@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.benchmark(group="IdentityFunction")
def test_basic(benchmark, func_mode):
    variable = np.random.rand(1)
    f = DefaultAllocationFunction()
    EX = pytest.helpers.get_func_execution(f, func_mode)

    res = benchmark(EX, variable)
    assert np.allclose(res, variable)
