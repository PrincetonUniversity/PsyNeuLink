import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.transferfunctions as Functions
import psyneulink.core.globals.keywords as kw
import pytest

from math import e, pi, sqrt

SIZE=5
test_var = np.random.rand(SIZE)
test_matrix = np.random.rand(SIZE, SIZE)
test_matrix_s = np.random.rand(SIZE, SIZE // 4)
test_matrix_l = np.random.rand(SIZE, 3 * SIZE)

RAND1 = np.random.rand()
RAND2 = np.random.rand()
RAND3 = np.random.rand()
RAND4 = np.random.rand()

softmax_helper = RAND1 * test_var
softmax_helper = softmax_helper - np.max(softmax_helper)
softmax_helper = np.exp(softmax_helper) / np.sum(np.exp(softmax_helper))

tanh_helper = -2*(RAND1 * (test_var + RAND2 - RAND3) + RAND4)
tanh_helper = (1 - e**tanh_helper)/ (1 + e**tanh_helper)

gaussian_helper = e**(-(test_var-RAND2)**2/(2*RAND1**2)) / sqrt(2*pi*RAND1)
gaussian_helper = RAND3 * gaussian_helper + RAND4

def gaussian_distort_helper(seed):
    state = np.random.RandomState(np.asarray([seed]))
    # compensate for construction
    state.normal(test_var + RAND1, RAND2)
    return RAND4 * state.normal(test_var + RAND1, RAND2) + RAND3


test_data = [
    (Functions.Linear, test_var, {'slope':RAND1, 'intercept':RAND2}, None, test_var * RAND1 + RAND2),
    (Functions.Exponential, test_var, {'scale':RAND1, 'rate':RAND2}, None, RAND1 * np.exp(RAND2 * test_var)),
    (Functions.Logistic, test_var, {'gain':RAND1, 'x_0':RAND2, 'offset':RAND3, 'scale':RAND4}, None, RAND4 / (1 + np.exp(-(RAND1 * (test_var - RAND2)) + RAND3))),
    (Functions.Tanh, test_var, {'gain':RAND1, 'bias':RAND2, 'x_0':RAND3, 'offset':RAND4}, None, tanh_helper),
    (Functions.ReLU, test_var, {'gain':RAND1, 'bias':RAND2, 'leak':RAND3}, None, np.maximum(RAND1 * (test_var - RAND2), RAND3 * RAND1 *(test_var - RAND2))),
    (Functions.Gaussian, test_var, {'standard_deviation':RAND1, 'bias':RAND2, 'scale':RAND3, 'offset':RAND4}, None, gaussian_helper),
    (Functions.GaussianDistort, test_var.tolist(), {'bias': RAND1, 'variance':RAND2, 'offset':RAND3, 'scale':RAND4 }, None, gaussian_distort_helper(0)),
    (Functions.GaussianDistort, test_var.tolist(), {'bias': RAND1, 'variance':RAND2, 'offset':RAND3, 'scale':RAND4, 'seed':0 }, None, gaussian_distort_helper(0)),
    (Functions.SoftMax, test_var, {'gain':RAND1, 'per_item': False}, None, softmax_helper),
    (Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.MAX_VAL}, 'per_item': False}, None, np.where(softmax_helper == np.max(softmax_helper), np.max(softmax_helper), 0)),
    (Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.MAX_INDICATOR}, 'per_item': False}, None, np.where(softmax_helper == np.max(softmax_helper), 1, 0)),
    (Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix.tolist()}, None, np.dot(test_var, test_matrix)),
    (Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix_l.tolist()}, None, np.dot(test_var, test_matrix_l)),
    (Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix_s.tolist()}, None, np.dot(test_var, test_matrix_s)),
]

# use list, naming function produces ugly names
names = [
    "LINEAR",
    "EXPONENTIAL",
    "LOGISTIC",
    "TANH",
    "RELU",
    "GAUSIAN",
    "GAUSSIAN DISTORT GLOBAL SEED",
    "GAUSSIAN DISTORT",
    "SOFT_MAX ALL",
    "SOFT_MAX MAX_VAL",
    "SOFT_MAX MAX_INDICATOR",
    "LINEAR_MATRIX SQUARE",
    "LINEAR_MATRIX WIDE",
    "LINEAR_MATRIX TALL",
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.parametrize("func, variable, params, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(func, variable, params, fail, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName
    res = f.function(variable)
    benchmark(f.function, variable)
    assert np.allclose(res, expected)


@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.parametrize("func, variable, params, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(func, variable, params, fail, expected, benchmark):
    if fail is not None:
        pytest.xfail(fail)

    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName
    m = pnlvm.execution.FuncExecution(f)
    res = m.execute(variable)
    benchmark(m.execute, variable)
    assert np.allclose(res, expected)

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.parametrize("func, variable, params, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_ptx_cuda(func, variable, params, fail, expected, benchmark):
    if fail is not None:
        pytest.xfail(fail)

    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName
    m = pnlvm.execution.FuncExecution(f)
    res = m.cuda_execute(variable)
    benchmark(m.cuda_execute, variable)
    assert np.allclose(res, expected)

def test_transfer_with_costs_function():
    from psyneulink.core.components.functions.transferfunctions import TransferWithCosts, CostFunctions
    f = TransferWithCosts()
    result = f(1)
    assert np.allclose(result, 1)
    f.toggle_cost_function(CostFunctions.INTENSITY)
    f = TransferWithCosts(enabled_cost_functions=CostFunctions.INTENSITY)
    result = f(2)
    assert np.allclose(result, 2)
    assert np.allclose(f.intensity_cost, 7.38905609893065)
    assert f.adjustment_cost == None
    assert f.duration_cost == None
    assert np.allclose(np.float(f.combined_costs), 7.38905609893065)
    f.toggle_cost_function(CostFunctions.ADJUSTMENT)
    result = f(3)
    assert np.allclose(result, 3)
    assert np.allclose(f.intensity_cost, 20.085536923187668)
    assert np.allclose(f.adjustment_cost, 1)
    assert f.duration_cost == None
    assert np.allclose(np.float(f.combined_costs), 21.085536923187668)
    f.toggle_cost_function(CostFunctions.DURATION)
    result = f(5)
    assert np.allclose(result, 5)
    assert np.allclose(f.intensity_cost, 148.413159102576603)
    assert np.allclose(f.adjustment_cost, 2)
    assert np.allclose(f.duration_cost, 5)
    assert np.allclose(np.float(f.combined_costs), 155.413159102576603)
    result = f(1)
    assert np.allclose(result, 1)
    assert np.allclose(f.intensity_cost, 2.718281828459045)
    assert np.allclose(f.adjustment_cost, 4)
    assert np.allclose(f.duration_cost, 6)
    assert np.allclose(np.float(f.combined_costs), 12.718281828459045)
