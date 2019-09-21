
import numpy as np
import pytest

import psyneulink.core.components.functions.statefulfunctions.integratorfunctions as Functions
import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.function import FunctionError

SIZE=1000
test_var = np.random.rand(SIZE)
test_initializer = np.random.rand(SIZE)
test_noise_arr = np.random.rand(SIZE)

RAND0_1 = np.random.random()
RAND2 = np.random.rand()
RAND3 = np.random.rand()

def AdaptiveIntFun(init, value, iterations, rate, noise, offset, **kwargs):
    val = np.full_like(value, init)
    for i in range(iterations):
        val = (1 - rate) * val + rate * value + noise + offset
    return val

test_data = [
    (Functions.AdaptiveIntegrator, test_var, {'rate':RAND0_1, 'noise':RAND2, 'offset':RAND3}, AdaptiveIntFun),
    (Functions.AdaptiveIntegrator, test_var, {'rate':RAND0_1, 'noise':test_noise_arr, 'offset':RAND3}, AdaptiveIntFun),
    (Functions.AdaptiveIntegrator, test_var, {'initializer':test_initializer, 'rate':RAND0_1, 'noise':RAND2, 'offset':RAND3}, AdaptiveIntFun),
    (Functions.AdaptiveIntegrator, test_var, {'initializer':test_initializer, 'rate':RAND0_1, 'noise':test_noise_arr, 'offset':RAND3}, AdaptiveIntFun),
]

# use list, naming function produces ugly names
names = [
    "AdaptiveIntegrator",
    "AdaptiveIntegrator Noise Array",
    "AdaptiveIntegrator Initializer",
    "AdaptiveIntegrator Initializer Noise Array",
]

GROUP_PREFIX="IntegratorFunction "

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(func, variable, params, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = GROUP_PREFIX + func.componentName
    f(variable)
    f(variable)
    res = benchmark(f, variable)
    # This is rather hacky. it might break with pytest benchmark update
    iterations = 3 if benchmark.disabled else benchmark.stats.stats.rounds + 2
    assert np.allclose(res, expected(f.initializer, variable, iterations, **params))


@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(func, variable, params, expected, benchmark):
    benchmark.group = GROUP_PREFIX + func.componentName
    f = func(default_variable=variable, **params)
    m = pnlvm.execution.FuncExecution(f)
    m.execute(variable)
    m.execute(variable)
    res = benchmark(m.execute, variable)
    # This is rather hacky. it might break with pytest benchmark update
    iterations = 3 if benchmark.disabled else benchmark.stats.stats.rounds + 2
    assert np.allclose(res, expected(f.initializer, variable, iterations, **params))

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_ptx_cuda(func, variable, params, expected, benchmark):
    benchmark.group = GROUP_PREFIX + func.componentName
    f = func(default_variable=variable, **params)
    m = pnlvm.execution.FuncExecution(f)
    m.cuda_execute(variable)
    m.cuda_execute(variable)
    res = benchmark(m.cuda_execute, variable)
    # This is rather hacky. it might break with pytest benchmark update
    iterations = 3 if benchmark.disabled else benchmark.stats.stats.rounds + 2
    assert np.allclose(res, expected(f.initializer, variable, iterations, **params))

def test_integrator_function_no_default_variable_and_params_len_more_than_1():
    I = AdaptiveIntegrator(rate=[.1, .2, .3])
    I.defaults.variable = np.array([0,0,0])

def test_integrator_function_default_variable_len_1_but_user_specified_and_params_len_more_than_1():
    with pytest.raises(FunctionError) as error_text:
        AdaptiveIntegrator(default_variable=[1], rate=[.1, .2, .3])
    error_msg_a = 'The length (3) of the array specified for the rate parameter'
    error_msg_b = 'must match the length (1) of the default input ([1])'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_default_variable_and_params_len_more_than_1_error():
    with pytest.raises(FunctionError) as error_text:
        AdaptiveIntegrator(default_variable=[0,0], rate=[.1, .2, .3])
    error_msg_a = 'The length (3) of the array specified for the rate parameter'
    error_msg_b = 'must match the length (2) of the default input ([0 0])'
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_with_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        AdaptiveIntegrator(rate=[.1, .2, .3], offset=[.4,.5])
    error_msg_a = "The parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "(['rate', 'offset']) don't all have the same length"
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)

def test_integrator_function_with_default_variable_and_params_of_different_lengths():
    with pytest.raises(FunctionError) as error_text:
        AdaptiveIntegrator(default_variable=[0,0,0], rate=[.1, .2, .3], offset=[.4,.5])
    error_msg_a = "The following parameters with len>1 specified for AdaptiveIntegrator Function"
    error_msg_b = "don't have the same length as its 'default_variable' (3): ['offset']."
    assert error_msg_a in str(error_text.value)
    assert error_msg_b in str(error_text.value)
