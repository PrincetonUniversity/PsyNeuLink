import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.function as Function
import psyneulink.core.components.functions.objectivefunctions as Functions
import psyneulink.core.globals.keywords as kw
import pytest

SIZE=1000
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = [np.random.rand(SIZE) + Function.EPSILON, np.random.rand(SIZE) + Function.EPSILON]
v1 = test_var[0]
v2 = test_var[1]
norm = len(test_var[0])

def correlation(v1,v2):
    v1_norm = v1 - np.mean(v1)
    v2_norm = v2 - np.mean(v2)
    return np.sum(v1_norm * v2_norm) / np.sqrt(np.sum(v1_norm**2) * np.sum(v2_norm**2))

test_data = [
    (test_var, kw.MAX_ABS_DIFF, False, None, np.max(abs(v1 - v2))),
    (test_var, kw.MAX_ABS_DIFF, True,  None, np.max(abs(v1 - v2))),
    (test_var, kw.DIFFERENCE, False, None, np.sum(np.abs(v1 - v2))),
    (test_var, kw.DIFFERENCE, True,  None, np.sum(np.abs(v1 - v2)) / norm),
    (test_var, kw.COSINE, False, None, 1 - np.abs(np.sum(v1 * v2) / (
                                             np.sqrt(np.sum(v1**2)) *
                                             np.sqrt(np.sum(v2**2))) )),
    (test_var, kw.NORMED_L0_SIMILARITY, False, None, 1 - np.sum(np.abs(v1 - v2) / 4)),
    (test_var, kw.NORMED_L0_SIMILARITY, True, None, (1 - np.sum(np.abs(v1 - v2) / 4)) / norm),
    (test_var, kw.EUCLIDEAN, False, None, np.linalg.norm(v1 - v2)),
    (test_var, kw.EUCLIDEAN, True,  None, np.linalg.norm(v1 - v2) / norm),
    (test_var, kw.ANGLE, False, "Needs sci-py", 0),
    (test_var, kw.ANGLE, True,  "Needs sci-py", 0 / norm),
    (test_var, kw.CORRELATION, False, None, 1 - np.abs(correlation(v1,v2))),
    (test_var, kw.CORRELATION, True,  None, 1 - np.abs(correlation(v1,v2))),
    (test_var, kw.CROSS_ENTROPY, False, None, -np.sum(v1 * np.log(v2))),
    (test_var, kw.CROSS_ENTROPY, True,  None, -np.sum(v1 * np.log(v2)) / norm),
    (test_var, kw.ENERGY, False, None, -np.sum(v1 * v2) / 2),
    (test_var, kw.ENERGY, True, None, (-np.sum(v1 * v2) / 2) / norm**2),
]

# use list, naming function produces ugly names
names = [
    "MAX_ABS_DIFF",
    "MAX_ABS_DIFF NORMALIZED",
    "DIFFERENCE",
    "DIFFERENCE NORMALIZED",
    "COSINE",
    "NORMED_L0_SIMILARITY",
    "NORMED_L0_SIMILARITY NORMALIZED",
    "EUCLIDEAN",
    "EUCLIDEAN NORMALIZED",
    "ANGLE",
    "ANGLE NORMALIZED",
    "CORRELATION",
    "CORRELATION NORMALIZED",
    # "PEARSON",
    # "PEARSON NORMALIZED",
    "CROSS_ENTROPY",
    "CROSS_ENTROPY NORMALIZED",
    "ENERGY",
    "ENERGY NORMALIZED",
]

@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.parametrize("variable, metric, normalize, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(variable, metric, normalize, fail, expected, benchmark):
    if fail is not None:
        pytest.xfail(fail)

    benchmark.group = "DistanceFunction " + metric + ("-normalized" if normalize else "")
    f = Functions.Distance(default_variable=variable, metric=metric, normalize=normalize)
    res = benchmark(f.function, variable)
    assert np.allclose(res, expected)
    assert np.isscalar(res) or len(res) == 1 or (metric == kw.PEARSON and res.size == 4)


@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.parametrize("variable, metric, normalize, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(variable, metric, normalize, fail, expected, benchmark):
    if fail is not None:
        pytest.xfail(fail)

    benchmark.group = "DistanceFunction " + metric + ("-normalized" if normalize else "")
    f = Functions.Distance(default_variable=variable, metric=metric, normalize=normalize)
    e = pnlvm.execution.FuncExecution(f)
    res = benchmark(e.execute, variable)
    assert np.allclose(res, expected)
    assert np.isscalar(res) or len(res) == 1 or (metric == kw.PEARSON and res.size == 4)

@pytest.mark.cuda
@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.parametrize("variable, metric, normalize, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_ptx_cuda(variable, metric, normalize, fail, expected, benchmark):
    if fail is not None:
        pytest.xfail(fail)

    benchmark.group = "DistanceFunction " + metric + ("-normalized" if normalize else "")
    f = Functions.Distance(default_variable=variable, metric=metric, normalize=normalize)
    e = pnlvm.execution.FuncExecution(f)
    res = benchmark(e.cuda_execute, variable)
    assert np.allclose(res, expected)
    assert np.isscalar(res) or len(res) == 1 or (metric == kw.PEARSON and res.size == 4)
