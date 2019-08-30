import numpy as np
import pytest

import psyneulink.core.components.functions.selectionfunctions as Functions
import psyneulink.core.globals.keywords as kw
import psyneulink.core.llvm as pnlvm

np.random.seed(0)
SIZE=10
test_var = np.random.rand(SIZE) * 2.0 - 1.0
test_prob = np.random.rand(SIZE)

test_data = [
    (Functions.OneHot, test_var, {'mode':kw.MAX_VAL}, [0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MAX_ABS_VAL}, [0., 0., 0., 0., 0., 0., 0., 0., 0.92732552, 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MAX_INDICATOR}, [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MAX_ABS_INDICATOR}, [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_VAL}, [0., 0., 0., 0., 0., 0., 0., 0., 0., -0.23311696]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_ABS_VAL}, [0., 0., 0., 0.08976637, 0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_INDICATOR}, [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
    (Functions.OneHot, test_var, {'mode':kw.MIN_ABS_INDICATOR}, [0., 0., 0., 1.,0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, [test_var, test_prob], {'mode':kw.PROB}, [0.09762701, 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    (Functions.OneHot, [test_var, test_prob], {'mode':kw.PROB_INDICATOR}, [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
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
]

GROUP_PREFIX="SelectionFunction "

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(func, variable, params, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = GROUP_PREFIX + func.componentName + params['mode']
    f(variable)
    res = f(variable)
    assert np.allclose(res, expected)
    benchmark(f, variable)


@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(func, variable, params, expected, benchmark):
    benchmark.group = GROUP_PREFIX + func.componentName + params['mode']
    f = func(default_variable=variable, **params)
    m = pnlvm.execution.FuncExecution(f)
    m.execute(variable)
    res = m.execute(variable)
    assert np.allclose(res, expected)
    benchmark(m.execute, variable)


@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_ptx_cuda(func, variable, params, expected, benchmark):
    benchmark.group = GROUP_PREFIX + func.componentName + params['mode']
    f = func(default_variable=variable, **params)
    m = pnlvm.execution.FuncExecution(f)
    m.cuda_execute(variable)
    res = m.cuda_execute(variable)
    assert np.allclose(res, expected)
    benchmark(m.cuda_execute, variable)
