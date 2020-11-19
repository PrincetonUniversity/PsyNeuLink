
import numpy as np

import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.function as Function
import psyneulink.core.components.functions.objectivefunctions as Functions
import psyneulink.core.components.functions.transferfunctions
import psyneulink.core.globals.keywords as kw
import pytest

SIZE=10
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = np.random.rand(SIZE) + Function.EPSILON
hollow_matrix = Function.get_matrix(kw.HOLLOW_MATRIX, SIZE, SIZE)
v1 = test_var
v2 = np.dot(hollow_matrix * hollow_matrix, v1)
norm = len(v1)

test_data = [
    (kw.ENTROPY, False, -np.sum(v1 * np.log(v2))),
    (kw.ENTROPY, True, -np.sum(v1 * np.log(v2)) / norm),
    (kw.ENERGY, False, -np.sum(v1 * v2) / 2),
    (kw.ENERGY, True, (-np.sum(v1 * v2) / 2) / norm**2),
]

# use list, naming function produces ugly names
names = [
    "ENTROPY",
    "ENTROPY NORMALIZED",
    "ENERGY",
    "ENERGY NORMALIZED",
]

@pytest.mark.function
@pytest.mark.stability_function
@pytest.mark.benchmark
@pytest.mark.parametrize("metric, normalize, expected", test_data, ids=names)
@pytest.mark.parametrize("variable", [test_var, test_var.astype(np.float32)], ids=["float", "float32"] )
@pytest.mark.parametrize('mode', ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_basic(variable, metric, normalize, expected, benchmark, mode):
    f = Functions.Stability(default_variable=variable, metric=metric, normalize=normalize)
    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    benchmark.group = "DistanceFunction " + metric + ("-normalized" if normalize else "")
    res = benchmark(EX, variable)
    assert np.allclose(res, expected)
    assert np.isscalar(res) or len(res) == 1
