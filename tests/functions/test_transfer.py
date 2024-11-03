import functools
import numpy as np
import pytest

import psyneulink as pnl
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


def binomial_distort_helper(seed):
    state = np.random.RandomState([seed])
    # compensate for construction
    state.binomial(1, p=RAND1, size=len(test_var))
    return state.binomial(1, p=(1 - RAND1), size=len(test_var)) * test_var


test_data = [
    pytest.param(pnl.Linear, test_var, {kw.SLOPE:RAND1, kw.INTERCEPT:RAND2}, test_var * RAND1 + RAND2, id="LINEAR"),
    pytest.param(pnl.Exponential, test_var, {kw.SCALE:RAND1, kw.RATE:RAND2}, RAND1 * np.exp(RAND2 * test_var), id="EXPONENTIAL"),
    pytest.param(pnl.Logistic, test_var, {kw.GAIN:RAND1, kw.X_0:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, logistic_helper, id="LOGISTIC"),
    pytest.param(pnl.Tanh, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.X_0:RAND3, kw.OFFSET:RAND4}, tanh_helper, id="TANH"),
    pytest.param(pnl.ReLU, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.LEAK:RAND3}, relu_helper, id="RELU"),

    # Angle doesn't have a helper using 'test_var', hardcode bopth the input and output
    pytest.param(pnl.Angle,
                 [0.5488135,  0.71518937, 0.60276338, 0.54488318, 0.4236548,
                  0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152],
                  {},
                 [0.85314409, 0.00556188, 0.01070476, 0.0214405,  0.05559454,
                  0.08091079, 0.21657281, 0.19296643, 0.21343805, 0.92738261, 0.00483101],
                 id="ANGLE"),

    # Distort
    pytest.param(pnl.Gaussian, test_var, {kw.STANDARD_DEVIATION:RAND1, kw.BIAS:RAND2, kw.SCALE:RAND3, kw.OFFSET:RAND4}, gaussian_helper, id="GAUSSIAN"),
    pytest.param(pnl.GaussianDistort, test_var, {kw.BIAS: RAND1, kw.VARIANCE:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4 }, gaussian_distort_helper(0), id="GAUSSIAN DISTORT GLOBAL SEED"),
    pytest.param(pnl.GaussianDistort, test_var, {kw.BIAS: RAND1, kw.VARIANCE:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4, 'seed':0 }, gaussian_distort_helper(0), id="GAUSSIAN DISTORT"),
    pytest.param(pnl.BinomialDistort, test_var, {'seed':0, 'p':RAND1 }, binomial_distort_helper(0), id="BINOMIAL DISTORT"),

    # SoftMax 1D input
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.PER_ITEM:False}, softmax_helper, id="SOFT_MAX ALL"),
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:pnl.ARG_MAX, kw.PER_ITEM:False},
                 np.where(softmax_helper == np.max(softmax_helper), softmax_helper, 0), id="SOFT_MAX ARG_MAX"),
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:pnl.ARG_MAX_INDICATOR, kw.PER_ITEM:False},
                 np.where(softmax_helper == np.max(softmax_helper), 1, 0), id="SOFT_MAX ARG_MAX_INDICATOR"),
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:False},
                 np.where(softmax_helper == np.max(softmax_helper), softmax_helper, 0), id="SOFT_MAX MAX_VAL"),
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:False},
                 np.where(softmax_helper == np.max(softmax_helper), 1, 0), id="SOFT_MAX MAX_INDICATOR"),
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.PROB, kw.PER_ITEM:False},
                 [0.0, 0.0, 0.0, 0.0, test_var[4], 0.0, 0.0, 0.0, 0.0, 0.0], id="SOFT_MAX PROB"),

    # SoftMax 2D testing per-item
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.PER_ITEM:True}, [softmax_helper], id="SOFT_MAX ALL 2D"),
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:pnl.ARG_MAX, kw.PER_ITEM:True},
                 [np.where(softmax_helper == np.max(softmax_helper), softmax_helper, 0)], id="SOFT_MAX ARG_MAX 2D"),
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:pnl.ARG_MAX_INDICATOR, kw.PER_ITEM:True},
                 [np.where(softmax_helper == np.max(softmax_helper), 1, 0)], id="SOFT_MAX ARG_MAX_INDICATOR 2D"),
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:True},
                 [np.where(softmax_helper == np.max(softmax_helper), softmax_helper, 0)], id="SOFT_MAX MAX_VAL 2D"),
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:True},
                 [np.where(softmax_helper == np.max(softmax_helper), 1, 0)], id="SOFT_MAX MAX_INDICATOR 2D"),
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.PROB, kw.PER_ITEM:True},
                 [[0.0, 0.0, 0.0, 0.0, test_var[4], 0.0, 0.0, 0.0, 0.0, 0.0]], id="SOFT_MAX PROB 2D"),

    # SoftMax per-item with 2 elements in input
    pytest.param(pnl.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.PER_ITEM: True}, softmax_helper2, id="SOFT_MAX ALL PER_ITEM"),
    pytest.param(pnl.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:pnl.ARG_MAX, kw.PER_ITEM: True},
                 np.where(softmax_helper2 == np.max(softmax_helper2), softmax_helper2, 0), id="SOFT_MAX ARG_MAX PER_ITEM"),
    pytest.param(pnl.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:pnl.ARG_MAX_INDICATOR, kw.PER_ITEM: True},
                 np.where(softmax_helper2 == np.max(softmax_helper2), 1, 0), id="SOFT_MAX ARG_MAX_INDICATOR PER_ITEM"),
    pytest.param(pnl.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM: True},
                 np.where(softmax_helper2 == np.max(softmax_helper2), softmax_helper2, 0), id="SOFT_MAX MAX_VAL PER_ITEM"),
    pytest.param(pnl.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM: True},
                 np.where(softmax_helper2 == np.max(softmax_helper2), 1, 0), id="SOFT_MAX MAX_INDICATOR PER_ITEM"),

    # Linear Matrix
    pytest.param(pnl.MatrixTransform, test_var, {kw.MATRIX:test_matrix}, np.dot(test_var, test_matrix), id="LINEAR_MATRIX SQUARE"),
    pytest.param(pnl.MatrixTransform, test_var, {kw.MATRIX:test_matrix_l}, np.dot(test_var, test_matrix_l), id="LINEAR_MATRIX WIDE"),
    pytest.param(pnl.MatrixTransform, test_var, {kw.MATRIX:test_matrix_s}, np.dot(test_var, test_matrix_s), id="LINEAR_MATRIX TALL"),

    # Dropout is just identity in non-learning mode
    pytest.param(pnl.Dropout, test_var, {}, test_var, id="DROPOUT"),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data)
def test_execute(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = "TransferFunction " + func.componentName

    if func_mode != 'Python':
        if ('output' in params
                and params['output'] in {kw.MAX_VAL, kw.MAX_ABS_VAL, kw.MAX_INDICATOR, kw.MAX_ABS_INDICATOR,
                                         kw.MIN_VAL, kw.MIN_ABS_VAL, kw.MIN_INDICATOR, kw.MIN_ABS_INDICATOR}):
            pytest.skip("{params['mode']} is not supported in {func_mode}")

    f = func(default_variable=variable, **params)
    ex = pytest.helpers.get_func_execution(f, func_mode)

    res = benchmark(ex, variable)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-8)


tanh_derivative_helper = (RAND1 * (test_var + RAND2) + RAND3)
tanh_derivative_helper = (1 - np.tanh(tanh_derivative_helper)**2) * RAND4 * RAND1


derivative_test_data = [
    (pnl.Linear, test_var, {kw.SLOPE:RAND1, kw.INTERCEPT:RAND2}, RAND1),
    (pnl.Exponential, test_var, {kw.SCALE:RAND1, kw.RATE:RAND2}, RAND1 * RAND2 * np.exp(RAND2 * test_var)),
    (pnl.Logistic, test_var, {kw.GAIN:RAND1, kw.X_0:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, RAND1 * RAND4 * logistic_helper * (1 - logistic_helper)),
    (pnl.ReLU, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.LEAK:RAND3}, np.where((test_var - RAND2) > 0, RAND1, RAND1 * RAND3)),
    (pnl.Tanh, test_var, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, tanh_derivative_helper),

    # SoftMax per-item=False
    (pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:False},
     [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
      -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]),
    (pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:False},
     [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
      -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]),
    pytest.param(pnl.SoftMax, test_var, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.ALL, kw.PER_ITEM:False},
                 [[ 0.088635686173821480, -0.010058549286956951, -0.009789214523259433, -0.009653377599514660, -0.009374948470179183,
                   -0.009891677863509920, -0.009406534609578588, -0.010496622361458180, -0.010680386821751540, -0.009284374637613039],
                  [-0.010058549286956951,  0.091856076128865180, -0.010190413769852785, -0.010049009732287338, -0.009759169518165271,
                   -0.010297076447528582, -0.009792050177702091, -0.010926813872042194, -0.011118109698906910, -0.009664883625423075],
                  [-0.009789214523259433, -0.010190413769852785,  0.089669339130699100, -0.009779930406389987, -0.009497851156931268,
                   -0.010021354713444461, -0.009529851380888969, -0.010634229847424508, -0.010820403403188785, -0.009406089929318929],
                  [-0.009653377599514660, -0.010049009732287338, -0.009779930406389987,  0.088560779144081720, -0.009366057244326959,
                   -0.009882296570138368, -0.009397613427348460, -0.010486667337129447, -0.010670257514724050, -0.009275569312222474],
                  [-0.009374948470179183, -0.009759169518165271, -0.009497851156931268, -0.009366057244326959,  0.08627659236704915,
                   -0.009597264807784339, -0.009126561218167337, -0.010184203911638403, -0.010362498859374313, -0.009008037180482098],
                  [-0.009891677863509920, -0.010297076447528582, -0.010021354713444461, -0.009882296570138368, -0.009597264807784339,
                    0.090503011588098000, -0.009629599976882700, -0.010745537931292683, -0.010933660158663310, -0.009504543118853646],
                  [-0.009406534609578588, -0.009792050177702091, -0.009529851380888969, -0.009397613427348460, -0.009126561218167337,
                   -0.009629599976882700,  0.086536526770559590, -0.010218516599910580, -0.010397412260182810, -0.009038387119898062],
                  [-0.010496622361458180, -0.010926813872042194, -0.010634229847424508, -0.010486667337129447, -0.010184203911638403,
                   -0.010745537931292683, -0.010218516599910580,  0.095380732590004670, -0.011602329078808723, -0.01008581165029997],
                  [-0.010680386821751540, -0.011118109698906910, -0.010820403403188785, -0.010670257514724050, -0.010362498859374313,
                   -0.010933660158663310, -0.010397412260182810, -0.011602329078808723,  0.096847441839448930, -0.010262384043848514],
                  [-0.009284374637613039, -0.009664883625423075, -0.009406089929318929, -0.009275569312222474, -0.009008037180482098,
                   -0.009504543118853646, -0.009038387119898062, -0.010085811650299970, -0.010262384043848514,  0.08553008061795979]],
                 marks=pytest.mark.llvm_not_implemented),
    # SoftMax per-tem=True
    (pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:True},
     [[-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]]),
    (pnl.SoftMax, [test_var, test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_INDICATOR, kw.PER_ITEM:True},
     [[-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513],
      [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]]),
    pytest.param(pnl.SoftMax, [test_var], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.ALL, kw.PER_ITEM:True},
                 [[ 0.088635686173821480, -0.010058549286956951, -0.009789214523259433, -0.009653377599514660, -0.009374948470179183,
                    -0.009891677863509920, -0.009406534609578588, -0.010496622361458180, -0.010680386821751540, -0.009284374637613039],
                   [-0.010058549286956951,  0.091856076128865180, -0.010190413769852785, -0.010049009732287338, -0.009759169518165271,
                    -0.010297076447528582, -0.009792050177702091, -0.010926813872042194, -0.011118109698906910, -0.009664883625423075],
                   [-0.009789214523259433, -0.010190413769852785,  0.089669339130699100, -0.009779930406389987, -0.009497851156931268,
                    -0.010021354713444461, -0.009529851380888969, -0.010634229847424508, -0.010820403403188785, -0.009406089929318929],
                   [-0.009653377599514660, -0.010049009732287338, -0.009779930406389987,  0.088560779144081720, -0.009366057244326959,
                    -0.009882296570138368, -0.009397613427348460, -0.010486667337129447, -0.010670257514724050, -0.009275569312222474],
                   [-0.009374948470179183, -0.009759169518165271, -0.009497851156931268, -0.009366057244326959,  0.08627659236704915,
                    -0.009597264807784339, -0.009126561218167337, -0.010184203911638403, -0.010362498859374313, -0.009008037180482098],
                   [-0.009891677863509920, -0.010297076447528582, -0.010021354713444461, -0.009882296570138368, -0.009597264807784339,
                     0.090503011588098000, -0.009629599976882700, -0.010745537931292683, -0.010933660158663310, -0.009504543118853646],
                   [-0.009406534609578588, -0.009792050177702091, -0.009529851380888969, -0.009397613427348460, -0.009126561218167337,
                    -0.009629599976882700,  0.086536526770559590, -0.010218516599910580, -0.010397412260182810, -0.009038387119898062],
                   [-0.010496622361458180, -0.010926813872042194, -0.010634229847424508, -0.010486667337129447, -0.010184203911638403,
                    -0.010745537931292683, -0.010218516599910580,  0.095380732590004670, -0.011602329078808723, -0.01008581165029997],
                   [-0.010680386821751540, -0.011118109698906910, -0.010820403403188785, -0.010670257514724050, -0.010362498859374313,
                    -0.010933660158663310, -0.010397412260182810, -0.011602329078808723,  0.096847441839448930, -0.010262384043848514],
                   [-0.009284374637613039, -0.009664883625423075, -0.009406089929318929, -0.009275569312222474, -0.009008037180482098,
                    -0.009504543118853646, -0.009038387119898062, -0.010085811650299970, -0.010262384043848514,  0.08553008061795979]],
                 marks=pytest.mark.llvm_not_implemented),
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected",
                         derivative_test_data,
                         ids=lambda x: getattr(x, 'name', None) or getattr(x, 'get', lambda p, q: None)(kw.OUTPUT_TYPE, None))
def test_transfer_derivative(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = "TransferFunction " + func.componentName + " Derivative"

    f = func(default_variable=variable, **params)

    if func_mode == 'Python':
        ex = f.derivative

    elif func_mode == 'LLVM':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative"})).execute

    elif func_mode == 'PTX':
        ex = pnlvm.execution.FuncExecution(f, tags=frozenset({"derivative"})).cuda_execute

    else:
        assert False, "unknown function mode: {}".format(func_mode)

    res = benchmark(ex, variable)

    # Tanh and Logistic need reduced accuracy in single precision mode
    if func_mode != 'Python' and pytest.helpers.llvm_current_fp_precision() == 'fp32' and func in {pnl.Tanh, pnl.Logistic}:
        tolerance = {'rtol': 5e-7, 'atol': 1e-8}
    else:
        tolerance = {}

    np.testing.assert_allclose(res, expected, **tolerance)


derivative_out_test_data = [
    (pnl.Logistic, logistic_helper, {kw.GAIN:RAND1, kw.X_0:RAND2, kw.OFFSET:RAND3, kw.SCALE:RAND4}, RAND1 * RAND4 * logistic_helper * (1 - logistic_helper)),
    (pnl.ReLU, relu_helper, {kw.GAIN:RAND1, kw.BIAS:RAND2, kw.LEAK:RAND3}, np.where((test_var - RAND2) > 0, RAND1, RAND1 * RAND3)),
    (pnl.SoftMax, softmax_helper, {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:False},
     [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
      -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]),
    (pnl.SoftMax, [softmax_helper, softmax_helper], {kw.GAIN:RAND1, kw.OUTPUT_TYPE:kw.MAX_VAL, kw.PER_ITEM:True},
     [[-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513],
      [-0.010680386821751537, -0.011118109698906909, -0.01082040340318878, -0.010670257514724047, -0.010362498859374309,
       -0.010933660158663306, -0.010397412260182806, -0.011602329078808718, 0.09684744183944892, -0.010262384043848513]]),
]
@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", derivative_out_test_data, ids=lambda x: getattr(x, 'name', None) or getattr(x, 'get', lambda p, q: None)(kw.OUTPUT_TYPE, None))
def test_transfer_derivative_out(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = "TransferFunction " + func.componentName + " Derivative"

    f = func(default_variable=variable, **params)

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

    # Logistic needs reduced accuracy in single precision mode because it uses exp()
    if func_mode != 'Python' and func is pnl.Logistic and pytest.helpers.llvm_current_fp_precision() == 'fp32' and func is pnl.Logistic:
        tolerance = {'rtol': 1e-7, 'atol': 1e-8}
    else:
        tolerance = {}

    np.testing.assert_allclose(res, expected, **tolerance)


def combine_costs(costs):
    return functools.reduce(lambda x, y: x | y, costs, pnl.CostFunctions.NONE)

@pytest.mark.parametrize("cost_functions",
                         map(combine_costs, pytest.helpers.power_set(cf for cf in pnl.CostFunctions if cf != pnl.CostFunctions.NONE and cf != pnl.CostFunctions.ALL)))
@pytest.mark.benchmark
@pytest.mark.function
def test_transfer_with_costs(cost_functions, func_mode, benchmark):

    f = pnl.TransferWithCosts(enabled_cost_functions=cost_functions)

    def check(cost_function, if_enabled, if_disabled, observed):
        if cost_function in cost_functions:
            np.testing.assert_allclose(observed, if_enabled)
        else:
            assert np.array_equal(observed, if_disabled)

            # HACK: workaround intensity cost returning [1] when disabled
            nonlocal total_cost
            total_cost -= observed or 0

    ex = pytest.helpers.get_func_execution(f, func_mode)

    res = ex(10)
    total_cost = (f.intensity_cost or 0) + (f.adjustment_cost or 0) + (f.duration_cost or 0)

    assert res == [10]

    # TODO : Intensity cost is [1] even when disabled
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2711
    check(pnl.CostFunctions.INTENSITY,
          [22026.465794806703],
          None if cost_functions == pnl.CostFunctions.NONE else [1],
          f.intensity_cost)
    check(pnl.CostFunctions.ADJUSTMENT, [10], None, f.adjustment_cost)
    check(pnl.CostFunctions.DURATION, [10], None, f.duration_cost)

    # TODO: Combined costs are not supported in compiled mode
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2712
    if func_mode == "Python":
        assert np.array_equal(total_cost, f.combined_costs or 0)


    # Second run with positive adjustment
    res = ex(15)
    total_cost = (f.intensity_cost or 0) + (f.adjustment_cost or 0) + (f.duration_cost or 0)

    assert res == [15]

    # TODO : Intensity cost is [1] even when disabled
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2711
    check(pnl.CostFunctions.INTENSITY,
          [3269017.372472108],
          None if cost_functions == pnl.CostFunctions.NONE else [1],
          f.intensity_cost)
    check(pnl.CostFunctions.ADJUSTMENT, [5], None, f.adjustment_cost)
    check(pnl.CostFunctions.DURATION, [25], None, f.duration_cost)

    # TODO: Combined costs are not supported in compiled mode
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2712
    if func_mode == "Python":
        assert np.array_equal(total_cost, f.combined_costs or 0)


    # Third run with negative adjustment
    res = ex(7)
    total_cost = (f.intensity_cost or 0) + (f.adjustment_cost or 0) + (f.duration_cost or 0)

    assert res == [7]

    # TODO : Intensity cost is [1] even when disabled
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2711
    check(pnl.CostFunctions.INTENSITY,
          [1096.6331584284583],
          None if cost_functions == pnl.CostFunctions.NONE else [1],
          f.intensity_cost)
    check(pnl.CostFunctions.ADJUSTMENT, [8], None, f.adjustment_cost)
    check(pnl.CostFunctions.DURATION, [32], None, f.duration_cost)

    # TODO: Combined costs are not supported in compiled mode
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2712
    if func_mode == "Python":
        assert np.array_equal(total_cost, f.combined_costs or 0)

    benchmark(ex, 10)


def test_transfer_with_costs_toggle():
    f = pnl.TransferWithCosts()
    result = f(1)
    np.testing.assert_allclose(result, 1)
    f.toggle_cost(pnl.CostFunctions.INTENSITY)

    f = pnl.TransferWithCosts(enabled_cost_functions=pnl.CostFunctions.INTENSITY)
    result = f(2)
    np.testing.assert_allclose(result, 2)
    np.testing.assert_allclose(f.intensity_cost, 7.38905609893065)
    assert f.adjustment_cost is None
    assert f.duration_cost is None
    np.testing.assert_allclose(f.combined_costs, 7.38905609893065)

    f.toggle_cost(pnl.CostFunctions.ADJUSTMENT)
    result = f(3)
    np.testing.assert_allclose(result, 3)
    np.testing.assert_allclose(f.intensity_cost, 20.085536923187668)
    np.testing.assert_allclose(f.adjustment_cost, 1)
    assert f.duration_cost is None
    np.testing.assert_allclose(f.combined_costs, 21.085536923187668)

    f.toggle_cost(pnl.CostFunctions.DURATION)
    result = f(5)
    np.testing.assert_allclose(result, 5)
    np.testing.assert_allclose(f.intensity_cost, 148.413159102576603)
    np.testing.assert_allclose(f.adjustment_cost, 2)
    np.testing.assert_allclose(f.duration_cost, 5)
    np.testing.assert_allclose(f.combined_costs, 155.413159102576603)
    result = f(1)
    np.testing.assert_allclose(result, 1)
    np.testing.assert_allclose(f.intensity_cost, 2.718281828459045)
    np.testing.assert_allclose(f.adjustment_cost, 4)
    np.testing.assert_allclose(f.duration_cost, 6)
    np.testing.assert_allclose(f.combined_costs, 12.718281828459045)


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
    twc = pnl.TransferWithCosts(default_variable=default_variable)

    np.testing.assert_array_equal(
        getattr(twc.parameters, func_name).get().defaults.variable,
        expected_func_variable
    )
    np.testing.assert_array_equal(
        getattr(twc.parameters, func_name).get().defaults.value,
        expected_func_value
    )
