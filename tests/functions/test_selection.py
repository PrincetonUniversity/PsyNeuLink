from unittest import mock

import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.globals.keywords as kw
from psyneulink.core.globals.utilities import _SeededPhilox

np.random.seed(0)
SIZE=10
test_var = np.random.rand(SIZE) * 2.0 - 1.0

# the sum of probs should be 1.0
test_prob = np.random.rand(SIZE)
test_prob /= sum(test_prob)
test_philox = np.random.rand(SIZE)
test_philox /= sum(test_philox)

expected_philox_prob = (0., 0.43037873274483895, 0., 0., 0., 0., 0., 0., 0., 0.)
expected_philox_ind = (0., 1., 0., 0., 0., 0., 0., 0., 0., 0.)

llvm_res = {'fp32': {}, 'fp64': {}}
llvm_res['fp32'][expected_philox_prob] = (0.09762700647115707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
llvm_res['fp32'][expected_philox_ind] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

test_data = [
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MAX}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), id="OneHot ARG_MAX"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MAX_ABS}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), id="OneHot ARG MAX_ABS"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.ARG_MAX_ABS}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), id="OneHot ARG MAX_ABS Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MAX_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), id="OneHot ARG_MAX_INDICATOR"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), id="OneHot ARG_MAX_ABS_INDICATOR"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.ARG_MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), id="OneHot ARG_MAX_ABS_INDICATOR Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MIN}, (0., 0., 0., 0., 0., 0., 0., 0., 0, -0.23311696), id="OneHot ARG_MIN"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MIN_ABS}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.ARG_MIN_ABS}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MIN_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 0., 1.), id="OneHot ARG_MIN_INDICATOR"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.ARG_MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS_INDICATOR"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.ARG_MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS_INDICATOR Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MAX_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.),  id="OneHot MAX_VAL"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MAX_ABS_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.),  id="OneHot MAX_ABS_VAL"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.MAX_ABS_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.),  id="OneHot MAX_ABS_VAL Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MAX_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.),  id="OneHot MAX_INDICATOR"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.),  id="OneHot MAX_ABS_INDICATOR"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.),  id="OneHot MAX_ABS_INDICATOR Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MIN_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0., -0.23311696),  id="OneHot MIN_VAL"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MIN_ABS_VAL}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.),  id="OneHot MIN_ABS_VAL"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.MIN_ABS_VAL}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.),  id="OneHot MIN_ABS_VAL Neg"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MIN_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 0., 1.),  id="OneHot MIN_INDICATOR"),
    pytest.param(pnl.OneHot, test_var, {'mode':kw.MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.),  id="OneHot MIN_ABS_INDICATOR"),
    pytest.param(pnl.OneHot, -test_var, {'mode':kw.MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.),  id="OneHot MIN_ABS_INDICATOR Neg"),
    pytest.param(pnl.OneHot, [test_var, test_prob], {'mode':kw.PROB}, (0., 0., 0., 0.08976636599379373, 0., 0., 0., 0., 0., 0.), id="OneHot PROB"),
    pytest.param(pnl.OneHot, [test_var, test_prob], {'mode':kw.PROB_INDICATOR}, (0., 0., 0., 1., 0., 0., 0., 0., 0., 0.), id="OneHot PROB_INDICATOR"),
    pytest.param(pnl.OneHot, [test_var, test_philox], {'mode':kw.PROB}, expected_philox_prob, id="OneHot PROB Philox"),
    pytest.param(pnl.OneHot, [test_var, test_philox], {'mode':kw.PROB_INDICATOR}, expected_philox_ind, id="OneHot PROB_INDICATOR Philox"),

]

GROUP_PREFIX="SelectionFunction "

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data)
def test_basic(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = GROUP_PREFIX + func.componentName + params['mode']

    f = func(default_variable=variable, **params)
    if len(variable) == 2 and variable[1] is test_philox:
        f.parameters.random_state.set(_SeededPhilox([0]))

    if func_mode != 'Python':
        precision = pytest.helpers.llvm_current_fp_precision()
        expected = llvm_res[precision].get(expected, expected)

    EX = pytest.helpers.get_func_execution(f, func_mode)

    EX(variable)
    res = benchmark(EX, variable)

    np.testing.assert_allclose(res, expected)


test_var3 = np.append(np.append(test_var, test_var), test_var)
test_var_2d = np.atleast_2d(test_var)
test_var3_2d = np.append(np.append(test_var_2d, test_var_2d, axis=0), test_var_2d, axis=0)


@pytest.mark.benchmark
@pytest.mark.parametrize("variable, direction, abs_val, tie, expected",
[
    # simple
    *[(test_var, kw.MAX, "absolute", tie, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(test_var, kw.MAX, "original", tie, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(test_var, kw.MIN, "absolute", tie, [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(test_var, kw.MIN, "original", tie, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],

    # negated
    *[(-test_var, kw.MAX, "absolute", tie, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(-test_var, kw.MAX, "original", tie, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(-test_var, kw.MIN, "absolute", tie, [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(-test_var, kw.MIN, "original", tie, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],

    # 2d
    *[(test_var_2d, kw.MAX, "absolute", tie, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(test_var_2d, kw.MAX, "original", tie, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(test_var_2d, kw.MIN, "absolute", tie, [[0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(test_var_2d, kw.MIN, "original", tie, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],

    # 2d negated
    *[(-test_var_2d, kw.MAX, "absolute", tie, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(-test_var_2d, kw.MAX, "original", tie, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(-test_var_2d, kw.MIN, "absolute", tie, [[0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],
    *[(-test_var_2d, kw.MIN, "original", tie, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0]]) for tie in [kw.FIRST,kw.LAST,kw.RANDOM,kw.ALL]],

    # multiple extreme values
    *[(test_var3, kw.MAX, abs_val, kw.FIRST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      for abs_val in ("absolute", "original")],
    *[(test_var3, kw.MAX, abs_val, kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0])
      for abs_val in ("absolute", "original")],
    *[(test_var3, kw.MAX, abs_val, kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      for abs_val in ("absolute", "original")],
    *[(test_var3, kw.MAX, abs_val, kw.ALL, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0])
      for abs_val in ("absolute", "original")],

    (test_var3, kw.MIN, "absolute", kw.FIRST, [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (test_var3, kw.MIN, "absolute", kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (test_var3, kw.MIN, "absolute", kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (test_var3, kw.MIN, "absolute", kw.ALL, [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    (test_var3, kw.MIN, "original", kw.FIRST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (test_var3, kw.MIN, "original", kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446]),
    (test_var3, kw.MIN, "original", kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (test_var3, kw.MIN, "original", kw.ALL, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446]),

    # multiple extreme values negated
    (-test_var3, kw.MAX, "absolute", kw.FIRST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MAX, "absolute", kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]),
    (-test_var3, kw.MAX, "absolute", kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MAX, "absolute", kw.ALL, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]),

    (-test_var3, kw.MAX, "original", kw.FIRST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MAX, "original", kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446]),
    (-test_var3, kw.MAX, "original", kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MAX, "original", kw.ALL, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2331169623484446]),

    (-test_var3, kw.MIN, "absolute", kw.FIRST, [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MIN, "absolute", kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MIN, "absolute", kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MIN, "absolute", kw.ALL, [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    (-test_var3, kw.MIN, "original", kw.FIRST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MIN, "original", kw.LAST, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0]),
    (-test_var3, kw.MIN, "original", kw.RANDOM, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (-test_var3, kw.MIN, "original", kw.ALL, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9273255210020586, 0.0]),

    # multiple extreme values 2d
    *[(test_var3_2d, kw.MAX, abs_val, kw.FIRST, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      for abs_val in ("absolute", "original")],
    *[(test_var3_2d, kw.MAX, abs_val, kw.LAST, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]])
      for abs_val in ("absolute", "original")],
    *[(test_var3_2d, kw.MAX, abs_val, kw.RANDOM, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      for abs_val in ("absolute", "original")],
    *[(test_var3_2d, kw.MAX, abs_val, kw.ALL, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9273255210020586, 0.0]])
      for abs_val in ("absolute", "original")],

    (test_var3_2d, kw.MIN, "absolute", kw.FIRST, [[0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    (test_var3_2d, kw.MIN, "absolute", kw.LAST, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    (test_var3_2d, kw.MIN, "absolute", kw.RANDOM, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    (test_var3_2d, kw.MIN, "absolute", kw.ALL, [[0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.08976636599379373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),

    (test_var3_2d, kw.MIN, "original", kw.FIRST, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    (test_var3_2d, kw.MIN, "original", kw.LAST, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446]]),
    (test_var3_2d, kw.MIN, "original", kw.RANDOM, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    (test_var3_2d, kw.MIN, "original", kw.ALL, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2331169623484446]]),
], ids=lambda x: x if isinstance(x, str) else str(getattr(x, 'shape', '')) )
@pytest.mark.parametrize("indicator", ["indicator", "value"])
def test_one_hot_mode_deterministic(benchmark, variable, tie, indicator, direction, abs_val, expected, func_mode):

    f = pnl.OneHot(default_variable=np.zeros_like(variable),
                   mode=kw.DETERMINISTIC,
                   tie=tie,
                   indicator=indicator=="indicator",
                   abs_val=abs_val=="absolute",
                   direction=direction,
                   seed=5) # seed to select middle of the 3 ties

    EX = pytest.helpers.get_func_execution(f, func_mode)

    EX(variable)
    res = benchmark(EX, variable)

    if indicator == "indicator":
        expected = np.where(np.asarray(expected) != 0, np.ones_like(expected), expected)

    np.testing.assert_allclose(res, expected)


# Regression test for the float-rounding edge case in OneHot's PROB sampling.
# np.cumsum of a probability distribution can land just below 1.0 due to
# float-rounding (e.g. cumsum([0.7, 0.2, 0.1])[-1] == 0.9999999999999999).
# random_state.uniform() draws from [0, 1), so it can return a value >=
# cum_sum[-1]. The pre-fix code did
#     next(element for element in cum_sum if element > random_value)
# which raised StopIteration when no element of cum_sum was strictly greater
# than the draw. The fix uses np.searchsorted with a clamp so the last index
# is selected in that case.
@pytest.mark.function
@pytest.mark.parametrize("mode, expected_factory", [
    (kw.PROB, lambda data: data * np.array([0., 0., 1.])),
    (kw.PROB_INDICATOR, lambda data: np.array([0., 0., 1.])),
])
def test_one_hot_prob_cumsum_below_one_due_to_float_rounding(mode, expected_factory):
    # cumsum drifts below 1.0 in float64 -- np.allclose(sum, 1) is still True
    # so OneHot validation accepts it.
    prob_dist = np.array([0.7, 0.2, 0.1])
    cum_sum_last = float(np.cumsum(prob_dist)[-1])
    assert cum_sum_last < 1.0, "test premise: cumsum should drift under 1.0"

    data = np.array([10.0, 20.0, 30.0])
    f = pnl.OneHot(default_variable=[data, prob_dist], mode=mode)

    # Force uniform() to return cum_sum[-1] exactly (a value uniform() can
    # legally produce). Pre-fix this triggered StopIteration; post-fix the
    # last index wins.
    class _StubRandomState:
        used_seed = [0]
        def uniform(self, *args, **kwargs):
            return cum_sum_last

    real = pnl.OneHot._get_current_parameter_value
    def patched(self, name, context=None):
        if name == "random_state":
            return _StubRandomState()
        return real(self, name, context)

    with mock.patch.object(pnl.OneHot, "_get_current_parameter_value", patched):
        res = f([data, prob_dist])

    np.testing.assert_allclose(res, expected_factory(data))


# Same boundary case, exercised on both Python and LLVM paths. Validation runs
# at construction against `default_variable`, so we construct OneHot with a
# well-formed prob_dist and then call it with a runtime prob_dist whose
# cumulative sum falls strictly under 1.0. About half of all uniform() draws
# then land in the (cum_sum[-1], 1.0) gap. Pre-fix the Python path raised
# StopIteration on those draws and the LLVM path silently produced an
# all-zero output; post-fix both paths fall back to the last index and
# return a valid one-hot.
@pytest.mark.function
def test_one_hot_prob_runtime_cumsum_under_one_never_all_zero(func_mode):
    default_prob = np.array([0.7, 0.2, 0.1])
    default_data = np.array([10.0, 20.0, 30.0])
    f = pnl.OneHot(default_variable=[default_data, default_prob],
                   mode=kw.PROB_INDICATOR)

    # cum_sum = [0.5, 0.5, 0.5]; any uniform draw >= 0.5 misses every bucket.
    runtime_prob = np.array([0.5, 0.0, 0.0])
    runtime_data = np.array([10.0, 20.0, 30.0])

    EX = pytest.helpers.get_func_execution(f, func_mode)

    # 32 draws is enough that with overwhelming probability some draws fall
    # in the gap (the seed is fixed by the conftest, so this is deterministic
    # per CI run).
    for _ in range(32):
        res = np.asarray(EX([runtime_data, runtime_prob]))
        assert res.sum() == 1.0, f"unexpected output (all-zero or invalid): {res}"
