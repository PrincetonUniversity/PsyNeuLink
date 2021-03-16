import numpy as np
import pytest

import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import DefaultAllocationFunction

@pytest.mark.function
@pytest.mark.identity_function
@pytest.mark.benchmark(group="IdentityFunction")
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])
                                  ])
def test_basic(benchmark, mode):
    variable = np.random.rand(1)
    f = DefaultAllocationFunction()
    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    res = benchmark(EX, variable)
    assert np.allclose(res, variable)
