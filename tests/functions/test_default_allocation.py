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
        res = benchmark(f.function, variable)
    elif mode == 'LLVM':
        m = pnlvm.execution.FuncExecution(f)
        res = benchmark(m.execute, variable)
    elif mode == 'PTX':
        m = pnlvm.execution.FuncExecution(f)
        res = benchmark(m.cuda_execute, variable)
    assert np.allclose(res, variable)
