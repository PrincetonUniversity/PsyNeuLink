import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.distributionfunctions as Functions
import psyneulink.core.globals.keywords as kw
import pytest

from math import e, pi, sqrt

np.random.seed(0)
test_var = np.random.rand()

RAND1 = np.random.rand()
RAND2 = np.random.rand()
RAND3 = np.random.rand()
RAND4 = np.random.rand()
RAND5 = np.random.rand()

test_data = [
    (Functions.DriftDiffusionAnalytical, test_var, {}, "Not Implemented",
     (1.9774974807292212, 0.012242689689501842, 1.9774974807292207, 1.3147677945132479, 1.7929299891370192, 1.9774974807292207, 1.3147677945132479, 1.7929299891370192)),
    (Functions.DriftDiffusionAnalytical, test_var, {"drift_rate": RAND1, "threshold": RAND2, "starting_point": RAND3, "t0":RAND4, "noise": RAND5}, "Not Implemented",
     (0.4236547993389047, -2.7755575615628914e-17, 0.5173675420165031, 0.06942854144616283, 6.302631815990666, 1.4934079600147951, 0.4288991185241868, 1.7740760781361433)),
]

# use list, naming function produces ugly names
names = [
    "DriftDiffusionAnalytical-DefaultParameters",
    "DriftDiffusionAnalytical-RandomParameters",
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, llvm_fail, expected", test_data, ids=names)
@pytest.mark.parametrize("mode", [
    "Python",
    pytest.param("LLVM", marks=pytest.mark.llvm),
    pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_execute(func, variable, params, llvm_fail, expected, benchmark, mode):
    if mode == "LLVM" and llvm_fail:
        pytest.xfail(llvm_fail)
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName
    if mode == "Python":
        ex = f
    elif mode == "LLVM":
        ex = pnlvm.execution.FuncExecution(f).execute
    elif mode == "PTX":
        ex = pnlvm.execution.FuncExecution(f).cude_execute
    res = ex(variable)
    assert np.allclose(res, expected)
    benchmark(f.function, variable)
