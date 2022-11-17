import numpy as np
import pytest

import psyneulink.core.components.functions.nonstateful.transferfunctions as Functions
import psyneulink.core.globals.keywords as kw
import psyneulink.core.llvm as pnlvm

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

softmax_helper  = RAND1 * test_var
softmax_helper  = softmax_helper - np.max(softmax_helper)
softmax_helper  = np.exp(softmax_helper) / np.sum(np.exp(softmax_helper))
softmax_helper2 = np.array((softmax_helper, softmax_helper)).reshape(2, -1)

tanh_helper = (RAND1 * (test_var + RAND2 - RAND3) + RAND4)
tanh_helper = np.tanh(tanh_helper)

gaussian_helper = np.e**(-(test_var - RAND2)**2 / (2 * RAND1**2)) / np.sqrt(2 * np.pi * RAND1)
gaussian_helper = RAND3 * gaussian_helper + RAND4

relu_helper = np.maximum(RAND1 * (test_var - RAND2), RAND3 * RAND1 *(test_var - RAND2))
logistic_helper = RAND4 / (1 + np.exp(-(RAND1 * (test_var - RAND2)) + RAND3))

def gaussian_distort_helper(seed):
    state = np.random.RandomState([seed])
    # compensate for construction
    state.normal(test_var + RAND1, RAND2)
    return RAND4 * state.normal(test_var + RAND1, RAND2) + RAND3


test_data = [
    pytest.param(Functions.Linear, test_var, {kw.SLOPE:RAND1, kw.INTERCEPT:RAND2}, test_var * RAND1 + RAND2, id="LINEAR"),
    pytest.param(Functions.Exponential, test_var, {kw.SCALE:RAND1, kw.RATE:RAND2}, RAND1 * np.exp(RAND2 * test_var), id="EXPONENTIAL"),
    pytest.param(Functions.Logistic, test_var, {kw.GAIN:RAND1, kw.X_0:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, logistic_helper, id="LOGISTIC"),
    pytest.param(Functions.Tanh, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.X_0:RAND3, kw.OFFSET:RAND4}, tanh_helper, id="TANH"),
    pytest.param(Functions.ReLU, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.LEAK:RAND3}, relu_helper, id="RELU"),
    # Angle doesn't have a helper using 'test_var', hardcode the input as well
    pytest.param(Functions.Angle, [0.5488135,  0.71518937, 0.60276338, 0.54488318, 0.4236548,
                                   0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152], {},
                 [0.85314409, 0.00556188, 0.01070476, 0.0214405,  0.05559454,
                  0.08091079, 0.21657281, 0.19296643, 0.21343805, 0.92738261, 0.00483101],
                 id="ANGLE"),

    pytest.param(Functions.Gaussian, test_var, {kw.STANDARD_DEVIATION:RAND1, kw.BIAS:RAND2, kw.SCALE:RAND3, kw.OFFSET:RAND4}, gaussian_helper, id="GAUSSIAN"),
    pytest.param(Functions.GaussianDistort, test_var, {kw.BIAS: RAND1, kw.VARIANCE:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4 }, gaussian_distort_helper(0), id="GAUSSIAN DISTORT GLOBAL SEED"),
    pytest.param(Functions.GaussianDistort, test_var, {kw.BIAS: RAND1, kw.VARIANCE:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4, 'seed':0 }, gaussian_distort_helper(0), id="GAUSSIAN DISTORT"),

    # SoftMax 1D input
    pytest.param(Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.PER_ITEM:False}, softmax_helper, id="SOFT_MAX ALL"),
    pytest.param(Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:False}, np.where(softmax_helper == np.max(softmax_helper), softmax_helper, 0), id="SOFT_MAX MAX_VAL"),
    pytest.param(Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:False}, np.where(softmax_helper == np.max(softmax_helper), 1, 0), id="SOFT_MAX MAX_INDICATOR"),
    pytest.param(Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.PROB, kw.PER_ITEM:False},
                 [0.0, 0.0, 0.0, 0.0, test_var[4], 0.0, 0.0, 0.0, 0.0, 0.0], id="SOFT_MAX PROB"),

    # SoftMax 2D testing per-item
    pytest.param(Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.PER_ITEM:True}, [softmax_helper], id="SOFT_MAX ALL 2D"),
    pytest.param(Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:True},
                 [np.where(softmax_helper == np.max(softmax_helper), softmax_helper, 0)], id="SOFT_MAX MAX_VAL 2D"),
    pytest.param(Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:True},
                 [np.where(softmax_helper == np.max(softmax_helper), 1, 0)], id="SOFT_MAX MAX_INDICATOR 2D"),
    pytest.param(Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.PROB, kw.PER_ITEM:True},
                 [[0.0, 0.0, 0.0, 0.0, test_var[4], 0.0, 0.0, 0.0, 0.0, 0.0]], id="SOFT_MAX PROB 2D"),

    # SoftMax per-item with 2 elements in input
    pytest.param(Functions.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.PER_ITEM: True}, softmax_helper2, id="SOFT_MAX ALL PER_ITEM"),
    pytest.param(Functions.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM: True},
                 np.where(softmax_helper2 == np.max(softmax_helper2), softmax_helper2, 0), id="SOFT_MAX MAX_VAL PER_ITEM"),
    pytest.param(Functions.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM: True},
                 np.where(softmax_helper2 == np.max(softmax_helper2), 1, 0), id="SOFT_MAX MAX_INDICATOR PER_ITEM"),

    pytest.param(Functions.LinearMatrix, test_var, {kw.MATRIX:test_matrix}, np.dot(test_var, test_matrix), id="LINEAR_MATRIX SQUARE"),
    pytest.param(Functions.LinearMatrix, test_var, {kw.MATRIX:test_matrix_l}, np.dot(test_var, test_matrix_l), id="LINEAR_MATRIX WIDE"),
    pytest.param(Functions.LinearMatrix, test_var, {kw.MATRIX:test_matrix_s}, np.dot(test_var, test_matrix_s), id="LINEAR_MATRIX TALL"),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data)
def test_execute(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = "TransferFunction " + func.componentName
    f = func(default_variable=variable, **params)
    ex = pytest.helpers.get_func_execution(f, func_mode)

    res = benchmark(ex, variable)
    assert np.allclose(res, expected)


tanh_derivative_helper = (RAND1 * (test_var + RAND2) + RAND3)
tanh_derivative_helper = (1 - np.tanh(tanh_derivative_helper)**2) * RAND4 * RAND1


derivative_test_data = [
    (Functions.Linear, test_var, {kw.SLOPE:RAND1, kw.INTERCEPT:RAND2}, RAND1),
    (Functions.Exponential, test_var, {kw.SCALE:RAND1, kw.RATE:RAND2}, RAND1 * RAND2 * np.exp(RAND2 * test_var)),
    (Functions.Logistic, test_var, {kw.GAIN:RAND1, kw.X_0:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, RAND1 * RAND4 * logistic_helper * (1 - logistic_helper)),
    (Functions.ReLU, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.LEAK:RAND3}, np.where((test_var - RAND2) > 0, RAND1, RAND1 * RAND3)),
    (Functions.Tanh, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, tanh_derivative_helper),

    # SoftMax per-item=False
    (Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:False},
     [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
      -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]),
    (Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:False},
     [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
      -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]),
    (Functions.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.ALL, kw.PER_ITEM:False},
     [[ 0.08863569, -0.01005855, -0.00978921, -0.00965338, -0.00937495, -0.00989168, -0.00940653, -0.01049662, -0.01068039, -0.00928437],
      [-0.01005855,  0.09185608, -0.01019041, -0.01004901, -0.00975917, -0.01029708, -0.00979205, -0.01092681, -0.01111811, -0.00966488],
      [-0.00978921, -0.01019041,  0.08966934, -0.00977993, -0.00949785, -0.01002135, -0.00952985, -0.01063423, -0.0108204,  -0.00940609],
      [-0.00965338, -0.01004901, -0.00977993,  0.08856078, -0.00936606, -0.0098823,  -0.00939761, -0.01048667, -0.01067026, -0.00927557],
      [-0.00937495, -0.00975917, -0.00949785, -0.00936606,  0.08627659, -0.00959726, -0.00912656, -0.0101842,  -0.0103625,  -0.00900804],
      [-0.00989168, -0.01029708, -0.01002135, -0.0098823,  -0.00959726,  0.09050301, -0.0096296,  -0.01074554, -0.01093366, -0.00950454],
      [-0.00940653, -0.00979205, -0.00952985, -0.00939761, -0.00912656, -0.0096296,   0.08653653, -0.01021852, -0.01039741, -0.00903839],
      [-0.01049662, -0.01092681, -0.01063423, -0.01048667, -0.0101842,  -0.01074554, -0.01021852,  0.09538073, -0.01160233, -0.01008581],
      [-0.01068039, -0.01111811, -0.0108204,  -0.01067026, -0.0103625,  -0.01093366, -0.01039741, -0.01160233,  0.09684744, -0.01026238],
      [-0.00928437, -0.00966488, -0.00940609, -0.00927557, -0.00900804, -0.00950454, -0.00903839, -0.01008581, -0.01026238,  0.08553008]]),

      # SoftMax per-tem=True 2D single element
    (Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:True},
     [[-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]]),
    (Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:True},
     [[-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]]),
    (Functions.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.ALL, kw.PER_ITEM:True},
     [[ 0.08863569, -0.01005855, -0.00978921, -0.00965338, -0.00937495, -0.00989168, -0.00940653, -0.01049662, -0.01068039, -0.00928437],
      [-0.01005855,  0.09185608, -0.01019041, -0.01004901, -0.00975917, -0.01029708, -0.00979205, -0.01092681, -0.01111811, -0.00966488],
      [-0.00978921, -0.01019041,  0.08966934, -0.00977993, -0.00949785, -0.01002135, -0.00952985, -0.01063423, -0.0108204,  -0.00940609],
      [-0.00965338, -0.01004901, -0.00977993,  0.08856078, -0.00936606, -0.0098823,  -0.00939761, -0.01048667, -0.01067026, -0.00927557],
      [-0.00937495, -0.00975917, -0.00949785, -0.00936606,  0.08627659, -0.00959726, -0.00912656, -0.0101842,  -0.0103625,  -0.00900804],
      [-0.00989168, -0.01029708, -0.01002135, -0.0098823,  -0.00959726,  0.09050301, -0.0096296,  -0.01074554, -0.01093366, -0.00950454],
      [-0.00940653, -0.00979205, -0.00952985, -0.00939761, -0.00912656, -0.0096296,   0.08653653, -0.01021852, -0.01039741, -0.00903839],
      [-0.01049662, -0.01092681, -0.01063423, -0.01048667, -0.0101842,  -0.01074554, -0.01021852,  0.09538073, -0.01160233, -0.01008581],
      [-0.01068039, -0.01111811, -0.0108204,  -0.01067026, -0.0103625,  -0.01093366, -0.01039741, -0.01160233,  0.09684744, -0.01026238],
      [-0.00928437, -0.00966488, -0.00940609, -0.00927557, -0.00900804, -0.00950454, -0.00903839, -0.01008581, -0.01026238,  0.08553008]]),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", derivative_test_data, ids=lambda x: getattr(x, 'name', None) or getattr(x, 'get', lambda p, q: None)(kw.OUTPUT_TYPE, None))
def test_transfer_derivative(func, variable, params, expected, benchmark, func_mode):
    if func == Functions.SoftMax and params[kw.OUTPUT_TYPE] == kw.ALL and func_mode != "Python":
        pytest.skip("Compiled derivative using 'ALL' is not implemented")

    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName + " Derivative"
    if func_mode == 'Python':
        ex = f.derivative
    elif func_mode == 'LLVM':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative"})).execute
    elif func_mode == 'PTX':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative"})).cuda_execute
    else:
        assert False, "unknown function mode: {}".format(func_mode)

    res = benchmark(ex, variable)
    assert np.allclose(res, expected)


derivative_out_test_data = [
    (Functions.Logistic, logistic_helper, {kw.GAIN:RAND1, kw.X_0:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, RAND1 * RAND4 * logistic_helper * (1 - logistic_helper)),
    (Functions.ReLU, relu_helper, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.LEAK:RAND3}, np.where((test_var - RAND2) > 0, RAND1, RAND1 * RAND3)),
    (Functions.SoftMax, softmax_helper, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:False},
     [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
      -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]),
    (Functions.SoftMax, [softmax_helper], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:True},
     [[-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]]),
]
@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", derivative_out_test_data, ids=lambda x: getattr(x, 'name', None) or getattr(x, 'get', lambda p, q: None)(kw.OUTPUT_TYPE, None))
def test_transfer_derivative_out(func, variable, params, expected, benchmark, func_mode):
    if func == Functions.SoftMax and params[kw.OUTPUT_TYPE] == kw.ALL and func_mode != "Python":
        pytest.skip("Compiled derivative using 'ALL' is not implemented")

    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName + " Derivative"
    if func_mode == 'Python':
        def ex(x):
            return f.derivative(input=None, output=x)
    elif func_mode == 'LLVM':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative_out"})).execute
    elif func_mode == 'PTX':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative_out"})).cuda_execute
    else:
        assert False, "unknown function mode: {}".format(func_mode)

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
