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
