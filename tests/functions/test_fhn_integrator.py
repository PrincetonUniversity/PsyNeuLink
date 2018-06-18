import psyneulink.components.functions.function as Function
import psyneulink.globals.keywords as kw
import numpy as np
import pytest

np.random.seed(0)
params = {'a_v' : np.random.rand(),
          'b_v' : np.random.rand(),
          'c_v' : np.random.rand(),
          'd_v' : np.random.rand(),
          'e_v' : np.random.rand(),
          'f_v' : np.random.rand(),
          'a_w' : np.random.rand(),
          'b_w' : np.random.rand(),
          'c_w' : np.random.rand(),
          'time_constant_v' : np.random.rand(),
          'time_constant_w' : np.random.rand(),
          'threshold' : np.random.rand(),
          'uncorrelated_activity' : np.random.rand(),
          'mode' : np.random.rand(),
          }

SIZE=8
test_var = np.random.rand(SIZE)

test_data = [
    (Function.FHNIntegrator, test_var, "RK4", params, ([0.23621415, 0.24033985, 0.22322577, 0.43877457, 0.42374559,
       0.44914662, 0.47952703, 0.42953011], [0.21400386, 0.21415157, 0.2135381 , 0.22111574, 0.22059828,
       0.22147186, 0.22251029, 0.22079764], 0.15000000000000002)),
    (Function.FHNIntegrator, test_var, "EULER", params, ([0.23686576, 0.24093183, 0.22404678, 0.43291206, 0.41863405,
       0.44273909, 0.47139546, 0.42413492], [0.20757016, 0.2076755 , 0.20723764, 0.21257185, 0.21221299,
       0.21281834, 0.21353476, 0.21235135], 0.15000000000000002)),
]

# use list, naming function produces ugly names
names = [
    "FHNIntegrator RK4",
    "FHNIntegrator EULER",
]

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, integration_method, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(func, variable, integration_method, params, expected, benchmark):
    f = func(default_variable=variable, integration_method=integration_method, params=params)
    res = f.function(variable)
    res = f.function(variable)
    res = f.function(variable)

    benchmark(f.function, variable)

    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    assert np.allclose(res[2], expected[2])


@pytest.mark.function
@pytest.mark.skip
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, integration_method, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(func, variable, integration_method, params, expected, benchmark):
    f = func(default_variable=variable, integration_method=integration_method, params=params)
    res = f.bin_function(variable)
    res = f.bin_function(variable)
    res = f.bin_function(variable)

    benchmark(f.bin_function, variable)

    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    assert np.allclose(res[2], expected[2])
