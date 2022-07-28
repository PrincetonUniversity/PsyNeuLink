import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.nonstateful.transferfunctions as Functions
import psyneulink.core.globals.keywords as kw
import pytest

from math import e, pi, sqrt

SIZE=10
np.random.seed(0)
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

tanh_helper = (RAND1 * (test_var + RAND2 - RAND3) + RAND4)
tanh_helper = np.tanh(tanh_helper)

gaussian_helper = e**(-(test_var - RAND2)**2 / (2 * RAND1**2)) / sqrt(2 * pi * RAND1)
gaussian_helper = RAND3 * gaussian_helper + RAND4

def gaussian_distort_helper(seed):
    state = np.random.RandomState([seed])
    # compensate for construction
    state.normal(test_var + RAND1, RAND2)
    return RAND4 * state.normal(test_var + RAND1, RAND2) + RAND3


test_data = [
    pytest.param(Functions.Linear, test_var, {'slope':RAND1, 'intercept':RAND2}, test_var * RAND1 + RAND2, id="LINEAR"),
    pytest.param(Functions.Exponential, test_var, {'scale':RAND1, 'rate':RAND2}, RAND1 * np.exp(RAND2 * test_var), id="EXPONENTIAL"),
    pytest.param(Functions.Logistic, test_var, {'gain':RAND1, 'x_0':RAND2, 'offset':RAND3, 'scale':RAND4}, RAND4 / (1 + np.exp(-(RAND1 * (test_var - RAND2)) + RAND3)), id="LOGISTIC"),
    pytest.param(Functions.Tanh, test_var, {'gain':RAND1, 'bias':RAND2, 'x_0':RAND3, 'offset':RAND4}, tanh_helper, id="TANH"),
    pytest.param(Functions.ReLU, test_var, {'gain':RAND1, 'bias':RAND2, 'leak':RAND3}, np.maximum(RAND1 * (test_var - RAND2), RAND3 * RAND1 *(test_var - RAND2)), id="RELU"),
    pytest.param(Functions.Angle, [0.5488135,  0.71518937, 0.60276338, 0.54488318, 0.4236548,
                                   0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152], {},
                 [0.85314409, 0.00556188, 0.01070476, 0.0214405,  0.05559454,
                  0.08091079, 0.21657281, 0.19296643, 0.21343805, 0.92738261, 0.00483101],
                 id="ANGLE"),
    pytest.param(Functions.Gaussian, test_var, {'standard_deviation':RAND1, 'bias':RAND2, 'scale':RAND3, 'offset':RAND4}, gaussian_helper, id="GAUSSIAN"),
    pytest.param(Functions.GaussianDistort, test_var.tolist(), {'bias': RAND1, 'variance':RAND2, 'offset':RAND3, 'scale':RAND4 }, gaussian_distort_helper(0), id="GAUSSIAN DISTORT GLOBAL SEED"),
    pytest.param(Functions.GaussianDistort, test_var.tolist(), {'bias': RAND1, 'variance':RAND2, 'offset':RAND3, 'scale':RAND4, 'seed':0 }, gaussian_distort_helper(0), id="GAUSSIAN DISTORT"),
    pytest.param(Functions.SoftMax, test_var, {'gain':RAND1, 'per_item': False}, softmax_helper, id="SOFT_MAX ALL"),
    pytest.param(Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.MAX_VAL}, 'per_item': False}, np.where(softmax_helper == np.max(softmax_helper), np.max(softmax_helper), 0), id="SOFT_MAX MAX_VAL"),
    pytest.param(Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.MAX_INDICATOR}, 'per_item': False}, np.where(softmax_helper == np.max(softmax_helper), 1, 0), id="SOFT_MAX MAX_INDICATOR"),
    pytest.param(Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.PROB}, 'per_item': False},
                 [0.0, 0.0, 0.0, 0.0, test_var[4], 0.0, 0.0, 0.0, 0.0, 0.0], id="SOFT_MAX PROB"),
    pytest.param(Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix.tolist()}, np.dot(test_var, test_matrix), id="LINEAR_MATRIX SQUARE"),
    pytest.param(Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix_l.tolist()}, np.dot(test_var, test_matrix_l), id="LINEAR_MATRIX WIDE"),
    pytest.param(Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix_s.tolist()}, np.dot(test_var, test_matrix_s), id="LINEAR_MATRIX TALL"),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data)
def test_execute(func, variable, params, expected, benchmark, func_mode):
    if 'Angle' in func.componentName and func_mode != 'Python':
        pytest.skip('Angle not yet supported by LLVM or PTX')
    benchmark.group = "TransferFunction " + func.componentName
    f = func(default_variable=variable, **params)
    ex = pytest.helpers.get_func_execution(f, func_mode)

    res = ex(variable)
    assert np.allclose(res, expected)
    if benchmark.enabled:
        benchmark(ex, variable)


relu_derivative_helper = lambda x : RAND1 if x > 0 else RAND1 * RAND3
logistic_helper = RAND4 / (1 + np.exp(-(RAND1 * (test_var - RAND2)) + RAND3))
tanh_derivative_helper = (RAND1 * (test_var + RAND2) + RAND3)
tanh_derivative_helper = (1 - np.tanh(tanh_derivative_helper)**2) * RAND4 * RAND1
derivative_test_data = [
    (Functions.Linear, test_var, {'slope':RAND1, 'intercept':RAND2}, RAND1),
    (Functions.Exponential, test_var, {'scale':RAND1, 'rate':RAND2}, RAND1 * RAND2 * np.exp(RAND2 * test_var)),
    (Functions.Logistic, test_var, {'gain':RAND1, 'x_0':RAND2, 'offset':RAND3, 'scale':RAND4}, RAND1 * RAND4 * logistic_helper * (1 - logistic_helper)),
    (Functions.ReLU, test_var, {'gain':RAND1, 'bias':RAND2, 'leak':RAND3}, list(map(relu_derivative_helper, test_var))),
    (Functions.Tanh, test_var, {'gain':RAND1, 'bias':RAND2, 'offset':RAND3, 'scale':RAND4}, tanh_derivative_helper),
]

derivative_names = [
    "LINEAR_DERIVATIVE",
    "EXPONENTIAL_DERIVATIVE",
    "LOGISTIC_DERIVATIVE",
    "RELU_DERIVATIVE",
    "TANH_DERIVATIVE",
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", derivative_test_data, ids=derivative_names)
def test_execute_derivative(func, variable, params, expected, benchmark, func_mode):
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName + " Derivative"
    if func_mode == 'Python':
        ex = f.derivative
    elif func_mode == 'LLVM':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative"})).execute
    elif func_mode == 'PTX':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative"})).cuda_execute

    res = benchmark(ex, variable)
    assert np.allclose(res, expected)


def test_transfer_with_costs_function():
    f = Functions.TransferWithCosts()
    result = f(1)
    assert np.allclose(result, 1)
    f.toggle_cost(Functions.CostFunctions.INTENSITY)
    f = Functions.TransferWithCosts(enabled_cost_functions=Functions.CostFunctions.INTENSITY)
    result = f(2)
    assert np.allclose(result, 2)
    assert np.allclose(f.intensity_cost, 7.38905609893065)
    assert f.adjustment_cost is None
    assert f.duration_cost is None
    assert np.allclose(f.combined_costs, 7.38905609893065)
    f.toggle_cost(Functions.CostFunctions.ADJUSTMENT)
    result = f(3)
    assert np.allclose(result, 3)
    assert np.allclose(f.intensity_cost, 20.085536923187668)
    assert np.allclose(f.adjustment_cost, 1)
    assert f.duration_cost is None
    assert np.allclose(f.combined_costs, 21.085536923187668)
    f.toggle_cost(Functions.CostFunctions.DURATION)
    result = f(5)
    assert np.allclose(result, 5)
    assert np.allclose(f.intensity_cost, 148.413159102576603)
    assert np.allclose(f.adjustment_cost, 2)
    assert np.allclose(f.duration_cost, 5)
    assert np.allclose(f.combined_costs, 155.413159102576603)
    result = f(1)
    assert np.allclose(result, 1)
    assert np.allclose(f.intensity_cost, 2.718281828459045)
    assert np.allclose(f.adjustment_cost, 4)
    assert np.allclose(f.duration_cost, 6)
    assert np.allclose(f.combined_costs, 12.718281828459045)


@pytest.mark.parametrize(
    'default_variable, func_name, expected_func_variable, expected_func_value',
    [
        ([1, 2, 3], 'transfer_fct', [1, 2, 3], [1, 2, 3])
    ]
)
def test_transfer_with_costs_shapes(
    default_variable,
    func_name,
    expected_func_variable,
    expected_func_value
):
    twc = Functions.TransferWithCosts(default_variable=default_variable)

    np.testing.assert_array_equal(
        getattr(twc.parameters, func_name).get().defaults.variable,
        expected_func_variable
    )
    np.testing.assert_array_equal(
        getattr(twc.parameters, func_name).get().defaults.value,
        expected_func_value
    )
