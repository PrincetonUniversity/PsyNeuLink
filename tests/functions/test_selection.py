import numpy as np
import pytest

import psyneulink.core.components.functions.nonstateful.selectionfunctions as Functions
import psyneulink.core.globals.keywords as kw
import psyneulink.core.llvm as pnlvm
from psyneulink.core.globals.utilities import _SeededPhilox

np.random.seed(0)
SIZE=10
test_var = np.random.rand(SIZE) * 2.0 - 1.0

# the sum of probs should be 1.0
test_prob = np.random.rand(SIZE)
test_prob /= sum(test_prob)
test_philox = np.random.rand(SIZE)
test_philox /= sum(test_philox)

test_data = [
    (Functions.OneHot, test_var, {'mode':kw.MAX_VAL}, [0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MAX_ABS_VAL}, [0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MAX_INDICATOR}, [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MAX_ABS_INDICATOR}, [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_VAL}, [0., 0., 0., 0., 0., 0., 0., 0., 0., -0.23311696]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_ABS_VAL}, [0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_INDICATOR}, [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_ABS_INDICATOR}, [0., 0., 0., 1.,0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, [test_var, test_prob], {'mode':kw.PROB}, [0., 0., 0., 0.08976636599379373, 0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, [test_var, test_prob], {'mode':kw.PROB_INDICATOR}, [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, [test_var, test_philox], {'mode':kw.PROB}, [0., 0.43037873274483895, 0., 0., 0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, [test_var, test_philox], {'mode':kw.PROB_INDICATOR}, [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
]

# use list, naming function produces ugly names
names = [
    "OneHot MAX_VAL",
    "OneHot MAX_ABS_VAL",
    "OneHot MAX_INDICATOR",
    "OneHot MAX_ABS_INDICATOR",
    "OneHot MIN_VAL",
    "OneHot MIN_ABS_VAL",
    "OneHot MIN_INDICATOR",
    "OneHot MIN_ABS_INDICATOR",
    "OneHot PROB",
    "OneHot PROB_INDICATOR",
    "OneHot PROB PHILOX",
    "OneHot PROB_INDICATOR PHILOX",
]

GROUP_PREFIX="SelectionFunction "

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
def test_basic(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = GROUP_PREFIX + func.componentName + params['mode']

    f = func(default_variable=variable, **params)
    if len(variable) == 2 and variable[1] is test_philox:
        f.parameters.random_state.set(_SeededPhilox([0]))

    EX = pytest.helpers.get_func_execution(f, func_mode)

    EX(variable)
    res = EX(variable)
    assert np.allclose(res, expected)
    if benchmark.enabled:
        benchmark(EX, variable)
