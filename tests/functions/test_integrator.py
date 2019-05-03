
import numpy as np
import pytest

import psyneulink.core.components.functions.statefulfunctions.integratorfunctions as Functions
import psyneulink.core.llvm as pnlvm

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
    benchmark.group = GROUP_PREFIX + func.componentName;
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
    benchmark.group = GROUP_PREFIX + func.componentName;
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
    benchmark.group = GROUP_PREFIX + func.componentName;
    f = func(default_variable=variable, **params)
    m = pnlvm.execution.FuncExecution(f)
    m.cuda_execute(variable)
    m.cuda_execute(variable)
    res = benchmark(m.cuda_execute, variable)
    # This is rather hacky. it might break with pytest benchmark update
    iterations = 3 if benchmark.disabled else benchmark.stats.stats.rounds + 2
    assert np.allclose(res, expected(f.initializer, variable, iterations, **params))
