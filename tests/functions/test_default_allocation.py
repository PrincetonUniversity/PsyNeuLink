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
    if func_mode == 'Python':
        EX = f.function
    elif func_mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif func_mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    res = benchmark(EX, variable)
    assert np.allclose(res, variable)
