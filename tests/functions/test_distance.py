import numpy as np
import psyneulink.core.components.functions as Functions
import psyneulink.core.globals.keywords as kw
import pytest

SIZE=1000
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = np.random.rand(2, SIZE) + Functions.EPSILON
v1 = test_var[0]
v2 = test_var[1]
norm = len(test_var[0])

def correlation(v1, v2):
    v1_norm = v1 - np.mean(v1)
    v2_norm = v2 - np.mean(v2)
    return np.sum(v1_norm * v2_norm) / np.sqrt(np.sum(v1_norm**2) * np.sum(v2_norm**2))


test_data = [
    pytest.param(kw.MAX_ABS_DIFF, False, np.max(abs(v1 - v2)), id="MAX_ABS_DIFF"),
    pytest.param(kw.MAX_ABS_DIFF, True,  np.max(abs(v1 - v2)), id="MAX_ABS_DIFF NORMALIZED"),
    pytest.param(kw.DIFFERENCE, False, np.sum(np.abs(v1 - v2)), id="DIFFERENCE"),
    pytest.param(kw.DIFFERENCE, True,  np.sum(np.abs(v1 - v2)) / norm, id="DIFFERENCE NORMALIZED"),
    pytest.param(kw.COSINE, False, 1 - np.abs(np.sum(v1 * v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))), id="COSINE"),
    pytest.param(kw.COSINE, True, 1 - np.abs(np.sum(v1 * v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))), id="COSINE NORMALIZED"),
    pytest.param(kw.NORMED_L0_SIMILARITY, False, 1 - np.sum(np.abs(v1 - v2) / 4), id="NORMED_L0_SIMILARITY"),
    pytest.param(kw.NORMED_L0_SIMILARITY, True, (1 - np.sum(np.abs(v1 - v2) / 4)) / norm, id="NORMED_L0_SIMILARITY NORMALIZED"),
    pytest.param(kw.EUCLIDEAN, False, np.linalg.norm(v1 - v2), id="EUCLIDEAN"),
    pytest.param(kw.EUCLIDEAN, True,  np.linalg.norm(v1 - v2) / norm, id="EUCLIDEAN NORMALIZED"),
    pytest.param(kw.CORRELATION, False, 1 - np.abs(correlation(v1, v2)), id="CORRELATION"),
    pytest.param(kw.CORRELATION, True,  1 - np.abs(correlation(v1, v2)), id="CORRELATION NORMALIZED"),
    pytest.param(kw.CROSS_ENTROPY, False, -np.sum(v1 * np.log(v2)), id="CROSS_ENTROPY"),
    pytest.param(kw.CROSS_ENTROPY, True,  -np.sum(v1 * np.log(v2)) / norm, id="CROSS_ENTROPY NORMALIZED"),
    pytest.param(kw.ENERGY, False, -np.sum(v1 * v2) / 2, id="ENERGY"),
    pytest.param(kw.ENERGY, True, (-np.sum(v1 * v2) / 2) / norm ** 2, id="ENERGY NORMALIZED"),
    pytest.param(kw.DOT_PRODUCT, False, np.dot(v1, v2), id="DOT_PRODUCT"),
    pytest.param(kw.DOT_PRODUCT, True, np.dot(v1, v2) / norm, id="DOT_PRODUCT NORMALIZED"),
]

@pytest.mark.function
@pytest.mark.distance_function
@pytest.mark.benchmark
@pytest.mark.parametrize("metric, normalize, expected", test_data)
@pytest.mark.parametrize("variable", [test_var, test_var.astype(np.float32), test_var.tolist()], ids=["np.default", "np.float32", "list"])
def test_basic(variable, metric, normalize, expected, benchmark, func_mode):

    benchmark.group = "DistanceFunction " + metric + ("-normalized" if normalize else "")
    f = Functions.Distance(default_variable=variable, metric=metric, normalize=normalize)
    EX = pytest.helpers.get_func_execution(f, func_mode)

    res = benchmark(EX, variable)

    # FIXME: Python calculation of COSINE using fp32 inputs are not accurate.
    #        LLVM calculations of most metrics using fp32 are not accurate.
    tol = {'rtol':1e-5, 'atol':1e-8} if metric == kw.COSINE or pytest.helpers.llvm_current_fp_precision() == 'fp32' else {}
    np.testing.assert_allclose(res, expected, **tol)
    assert np.isscalar(res) or res.ndim == 0 or len(res) == 1
