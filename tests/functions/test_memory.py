import numpy as np
import pytest

import psyneulink.core.components.functions.statefulfunctions.memoryfunctions as Functions
import psyneulink.core.llvm as pnlvm

np.random.seed(0)
SIZE=10
test_var = np.random.rand(2, SIZE)
test_initializer = {tuple(test_var[0]): test_var[1]}
test_noise_arr = np.random.rand(SIZE)

RAND1 = np.random.random(1)
RAND2 = np.random.rand()
RAND3 = np.random.rand()

test_data = [
    (Functions.DND, test_var, {'rate':RAND1}, [[
       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215 ],[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777]]),
    (Functions.DND, test_var, {'initializer':test_initializer, 'rate':RAND1}, [[
       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215 ],[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777
]]),
# Disable noise tests for now as they trigger failure in DND lookup
#    (Functions.DND, test_var, {'rate':RAND1, 'noise':RAND2}, [[
#       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215 ],[
#       1.3230471933615413, 1.4894230558066361, 1.3769970655058605, 1.3191168724311135, 1.1978884887731214, 1.4201278025008728, 1.2118209006969092, 1.6660066902162964, 1.737896449935246, 1.1576752082599944
#]]),
#    (Functions.DND, test_var, {'initializer':test_initializer, 'rate':RAND1, 'noise':RAND2}, [[
#       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215 ],[
#       1.3230471933615413, 1.4894230558066361, 1.3769970655058605, 1.3191168724311135, 1.1978884887731214, 1.4201278025008728, 1.2118209006969092, 1.6660066902162964, 1.737896449935246, 1.1576752082599944
#]]),
]

# use list, naming function produces ugly names
names = [
    "DND",
    "DND Initializer",
#    "DND Noise",
#    "DND Initializer Noise",
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
    res = f(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    benchmark(f, variable)


@pytest.mark.skip
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
    res = m.execute(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    benchmark(m.execute, variable)


@pytest.mark.skip
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
    res = m.cuda_execute(variable)
    assert np.allclose(res, expected)
    benchmark(m.cuda_execute, variable)
