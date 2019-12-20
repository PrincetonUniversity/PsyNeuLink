
import numpy as np
import pytest

import psyneulink.core.components.functions.statefulfunctions.integratorfunctions as Functions
import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.functions.function import FunctionError

np.random.seed(0)
SIZE=10
test_var = np.random.rand(SIZE)
test_initializer = np.random.rand(SIZE)
test_noise_arr = np.random.rand(SIZE)

RAND0_1 = np.random.random()
RAND2 = np.random.rand()
RAND3 = np.random.rand()

def SimpleIntFun(init, value, iterations, rate, noise, offset, **kwargs):
    val = np.full_like(value, init)
    for i in range(iterations):
        val = val + (rate * value) + noise + offset
    return val


def AdaptiveIntFun(init, value, iterations, rate, noise, offset, **kwargs):
    val = np.full_like(value, init)
    for i in range(iterations):
        val = (1 - rate) * val + rate * value + noise + offset
    return val


def DriftIntFun(init, value, iterations, **kwargs):
    assert iterations == 3
    if np.isscalar(kwargs["noise"]):
        if "initializer" not in kwargs:
            return ([0.35782281, 4.03326927, 4.90427264, 0.90944534, 1.45943493,
                     2.31791882, 3.05580281, 1.20089146, 2.8408554 , 1.93964773],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
        else:
            return ([1.14954785, 4.56216419, 5.4723172 , 1.83504198, 1.53047099,
                     2.40504812, 3.07602121, 2.0335113 , 3.61901215, 2.80965988],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
    else:
        if "initializer" not in kwargs:
            return ([0.17810305, 4.06675934, 4.20730295, 0.90582833, 1.60883329,
                     2.27822395, 2.2923697 , 1.10933472, 2.71418965, 1.86808107],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
        else:
            return ([0.96982809, 4.59565426, 4.77534751, 1.83142497, 1.67986935,
                     2.36535325, 2.3125881 , 1.94195457, 3.4923464 , 2.73809322],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])


GROUP_PREFIX="IntegratorFunction "


@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("variable, params", [
    (test_var, {'rate':RAND0_1, 'noise':RAND2, 'offset':RAND3}),
    (test_var, {'rate':RAND0_1, 'noise':test_noise_arr, 'offset':RAND3}),
    (test_var, {'initializer':test_initializer, 'rate':RAND0_1, 'noise':RAND2, 'offset':RAND3}),
    (test_var, {'initializer':test_initializer, 'rate':RAND0_1, 'noise':test_noise_arr, 'offset':RAND3}),
    ], ids=["SNOISE", "VNOISE", "Initializer-SNOISE", "Initializer-VNOISE"])
@pytest.mark.parametrize("func", [
    (Functions.AdaptiveIntegrator, AdaptiveIntFun),
    (Functions.SimpleIntegrator, SimpleIntFun),
    (Functions.DriftDiffusionIntegrator, DriftIntFun),
    ], ids=lambda x: x[0])
@pytest.mark.parametrize("mode", [
    "Python",
    pytest.param("LLVM", marks=pytest.mark.llvm),
    pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda])])
@pytest.mark.benchmark
def test_execute(func, mode, variable, params, benchmark):
    benchmark.group = GROUP_PREFIX + func[0].componentName
    f = func[0](default_variable=variable, **params)
    if mode == "Python":
        ex = f
    elif mode == "LLVM":
        ex = pnlvm.execution.FuncExecution(f).execute
    elif mode == "PTX":
        ex = pnlvm.execution.FuncExecution(f).cuda_execute
    ex(variable)
    ex(variable)
    res = ex(variable)
    expected = func[1](f.initializer, variable, 3, **params)
    for r, e in zip(res, expected):
        assert np.allclose(r, e)
    benchmark(ex, variable)


def test_integrator_function_no_default_variable_and_params_len_more_than_1():
    I = Functions.AdaptiveIntegrator(rate=[.1, .2, .3])
    I.defaults.variable = np.array([0,0,0])

def test_integrator_function_default_variable_len_1_but_user_specified_and_params_len_more_than_1():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[1], rate=[.1, .2, .3])
    error_msg_a = 'The length (3) of the array specified for the rate parameter'
    error_msg_b = 'must match the length (1) of the default input ([1])'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_default_variable_and_params_len_more_than_1_error():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[0,0], rate=[.1, .2, .3])
    error_msg_a = 'The length (3) of the array specified for the rate parameter'
    error_msg_b = 'must match the length (2) of the default input ([0 0])'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_with_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(rate=[.1, .2, .3], offset=[.4,.5])
    error_msg_a = "The parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "(['rate', 'offset']) don't all have the same length"
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_with_default_variable_and_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        Functions.AdaptiveIntegrator(default_variable=[0,0,0], rate=[.1, .2, .3], offset=[.4,.5])
    error_msg_a = "The following parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "don't have the same length as its 'default_variable' (3): ['offset']."
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)
