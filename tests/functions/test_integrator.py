
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
    if "initializer" not in kwargs:
        return ([0.52012043, 0.65216743, 0.56293865, 0.51700106, 0.42078612,
                 0.59717009, 0.43184381, 0.79231601, 0.84937253, 0.38887017],
                [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
    else:
        return ([1.31184547, 1.18106235, 1.13098321, 1.4425977 , 0.49182217,
                 0.68429939, 0.45206221, 1.62493585, 1.62752928, 1.25888232],
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
    # Filter out illegal combinations
    if func[0] is Functions.DriftDiffusionIntegrator and not np.isscalar(params["noise"]):
        pytest.skip("DDI needs scalar noise")
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
