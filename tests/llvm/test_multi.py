import pytest
import psyneulink.core.llvm as pnlvm

import numpy as np
import psyneulink.core.components.functions.function as Function
import psyneulink.core.components.functions.objectivefunctions as Functions
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
import psyneulink.core.globals.keywords as kw

SIZE=10
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = [np.random.rand(SIZE) + Function.EPSILON, np.random.rand(SIZE) + Function.EPSILON]
v1 = test_var[0]
v2 = test_var[1]
expected = np.linalg.norm(v1 - v2)

@pytest.mark.cuda
@pytest.mark.llvm
@pytest.mark.multi
@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.benchmark(group="DistanceFunction multi")
@pytest.mark.parametrize("executions", [1,5,100])
@pytest.mark.skipif(not pnlvm.ptx_enabled, reason="PTX engine not enabled/available")
def test_ptx_cuda_multi(benchmark, executions):
    f = Functions.Distance(default_variable=test_var, metric=kw.EUCLIDEAN)
    e = pnlvm.execution.FuncExecution(f, [None for _ in range(executions)])
    res = benchmark(e.cuda_execute, [test_var for _ in range(executions)])
    assert np.allclose(res, [expected for _ in range(executions)])
    assert executions == 1 or len(res) == executions

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.mechanism
@pytest.mark.multi
@pytest.mark.transfer_mechanism
@pytest.mark.benchmark(group="TransferMechanism multi")
@pytest.mark.parametrize("executions", [1,5,100])
@pytest.mark.skipif(not pnlvm.ptx_enabled, reason="PTX engine not enabled/available")
def test_transfer_mech_multi(benchmark, executions):
    variable = [0 for _ in range(SIZE)]
    T = TransferMechanism(
        name='T',
        default_variable=variable,
        integration_rate=1.0,
        noise=-2.0,
        integrator_mode=True
    )
    var = [[10.0 for _ in range(SIZE)] for _ in range(executions)]
    expected = [[8.0 for i in range(SIZE)]]
    e = pnlvm.execution.MechExecution(T, [None for _ in range(executions)])
    res = benchmark(e.cuda_execute, var)
    if executions > 1:
        expected = [expected for _ in range(executions)]

    assert np.allclose(res, expected)
    assert len(res) == executions
