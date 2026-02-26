import numpy as np
import pytest

import psyneulink as pnl

np.random.seed(0)
SIZE=10
test_var = np.random.rand(2, SIZE)

test_data = [
    (pnl.Loss.L0, test_var, 0.6842291758389014),
    (pnl.Loss.L1, test_var, 2.9046204135831926),
    (pnl.Loss.SSE, test_var, 1.1252771382029314),
    (pnl.Loss.MSE, test_var, 0.11252771382029314),
    (pnl.Loss.POISSON_NLL, test_var, 0.8733498011268741),
    pytest.param(pnl.Loss.CROSS_ENTROPY, test_var, -2.3336335487949333, marks=pytest.mark.llvm_not_implemented),
]

GROUP_PREFIX="SelectionFunction "

@pytest.mark.function
@pytest.mark.benchmark
@pytest.mark.parametrize("loss, variable, expected", test_data, ids=[getattr(x, 'values', x)[0] for x in test_data])
@pytest.mark.parametrize("normalize", ["normalize", "no-normalize"])
def test_basic(loss, variable, normalize, expected, benchmark, func_mode):
    do_normalize = normalize == "normalize"
    f = pnl.LossFunction(default_variable=variable, loss=loss, normalize=do_normalize)

    EX = pytest.helpers.get_func_execution(f, func_mode)
    res = benchmark(EX, variable)

    expected = expected if not do_normalize else expected / SIZE
    np.testing.assert_allclose(res, expected)

@pytest.mark.mechanism
@pytest.mark.benchmark
@pytest.mark.parametrize("loss, variable, expected", test_data, ids=[getattr(x, 'values', x)[0] for x in test_data])
@pytest.mark.parametrize("normalize", ["normalize", "no-normalize"])
def test_in_mechanism(loss, variable, normalize, expected, benchmark, mech_mode):
    do_normalize = normalize == "normalize"
    f = pnl.LossFunction(default_variable=variable, loss=loss, normalize=do_normalize)
    m = pnl.ProcessingMechanism(function=f, default_variable=variable)

    EX = pytest.helpers.get_mech_execution(m, mech_mode)
    res = benchmark(EX, variable)

    expected = expected if not do_normalize else expected / SIZE
    np.testing.assert_allclose(res, expected)
