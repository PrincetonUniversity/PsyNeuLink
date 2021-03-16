import numpy as np
import pytest

import psyneulink.core.components.functions.transferfunctions as Functions
import psyneulink.core.llvm as pnlvm

@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.benchmark(group="IdentityFunction")
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_basic(size, benchmark, func_mode):
    variable = np.random.rand(size)
    f = Functions.Identity(default_variable=variable)
    if func_mode == 'Python':
        EX = f.function
    elif func_mode == 'LLVM':
        EX = pnlvm.execution.FuncExecution(f).execute
    elif func_mode == 'PTX':
        EX = pnlvm.execution.FuncExecution(f).cuda_execute

    res = benchmark(EX, variable)
    assert np.allclose(res, variable)
