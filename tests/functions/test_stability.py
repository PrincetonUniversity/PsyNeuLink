
import psyneulink.components.functions.function as Function
import psyneulink.globals.keywords as kw
import numpy as np
import pytest

SIZE=10
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = np.random.rand(SIZE) + Function.EPSILON
hollow_matrix=Function.get_matrix(Function.HOLLOW_MATRIX, SIZE, SIZE)
v1 = test_var
v2 = np.dot(hollow_matrix * hollow_matrix, v1)
norm = len(v1)

test_data = [
    (test_var, kw.ENTROPY, False, None, -np.sum(v1*np.log(v2))),
    (test_var, kw.ENTROPY, True,  None, -np.sum(v1*np.log(v2))/norm),
    (test_var, kw.ENERGY, False, None, -np.sum(v1*v2)/2),
    (test_var, kw.ENERGY, True, None, (-np.sum(v1*v2)/2)/norm**2),
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
@pytest.mark.parametrize("variable, metric, normalize, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(variable, metric, normalize, fail, expected, benchmark):
    f = Function.Stability(default_variable=variable, metric=metric, normalize=normalize)
    benchmark.group = "DistanceFunction " + metric + ("-normalized" if normalize else "")
    res = benchmark(f.function, variable)
    assert np.allclose(res, expected)
    assert np.isscalar(res)
