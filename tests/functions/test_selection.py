import numpy as np
import pytest

import psyneulink.core.components.functions.nonstateful.selectionfunctions as Functions
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
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MAX}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), id="OneHot ARG_MAX"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MAX_ABS}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), id="OneHot ARG MAX_ABS"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.ARG_MAX_ABS}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), id="OneHot ARG MAX_ABS Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MAX_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), id="OneHot ARG_MAX_INDICATOR"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), id="OneHot ARG_MAX_ABS_INDICATOR"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.ARG_MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), id="OneHot ARG_MAX_ABS_INDICATOR Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MIN}, (0., 0., 0., 0., 0., 0., 0., 0., 0, -0.23311696), id="OneHot ARG_MIN"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MIN_ABS}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.ARG_MIN_ABS}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MIN_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 0., 1.), id="OneHot ARG_MIN_INDICATOR"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.ARG_MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS_INDICATOR"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.ARG_MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.), id="OneHot ARG_MIN_ABS_INDICATOR Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MAX_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MAX_VAL"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MAX_ABS_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MAX_ABS_VAL"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.MAX_ABS_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MAX_ABS_VAL Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MAX_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MAX_INDICATOR"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MAX_ABS_INDICATOR"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.MAX_ABS_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 1., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MAX_ABS_INDICATOR Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MIN_VAL}, (0., 0., 0., 0., 0., 0., 0., 0., 0., -0.23311696), marks=pytest.mark.llvm_not_implemented, id="OneHot MIN_VAL"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MIN_ABS_VAL}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MIN_ABS_VAL"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.MIN_ABS_VAL}, (0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MIN_ABS_VAL Neg"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MIN_INDICATOR}, (0., 0., 0., 0., 0., 0., 0., 0., 0., 1.), marks=pytest.mark.llvm_not_implemented, id="OneHot MIN_INDICATOR"),
    pytest.param(Functions.OneHot, test_var, {'mode':kw.MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MIN_ABS_INDICATOR"),
    pytest.param(Functions.OneHot, -test_var, {'mode':kw.MIN_ABS_INDICATOR}, (0., 0., 0., 1.,0., 0., 0., 0., 0., 0.), marks=pytest.mark.llvm_not_implemented, id="OneHot MIN_ABS_INDICATOR Neg"),
    pytest.param(Functions.OneHot, [test_var, test_prob], {'mode':kw.PROB}, (0., 0., 0., 0.08976636599379373, 0., 0., 0., 0., 0., 0.), id="OneHot PROB"),
    pytest.param(Functions.OneHot, [test_var, test_prob], {'mode':kw.PROB_INDICATOR}, (0., 0., 0., 1., 0., 0., 0., 0., 0., 0.), id="OneHot PROB_INDICATOR"),
    pytest.param(Functions.OneHot, [test_var, test_philox], {'mode':kw.PROB}, expected_philox_prob, id="OneHot PROB Philox"),
    pytest.param(Functions.OneHot, [test_var, test_philox], {'mode':kw.PROB_INDICATOR}, expected_philox_ind, id="OneHot PROB_INDICATOR Philox"),

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
