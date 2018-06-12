
import psyneulink.components.functions.function as Function
import psyneulink.globals.keywords as kw
import numpy as np
import pytest

SIZE=8
np.random.seed(0)
test_var = np.random.rand(SIZE)
test_initializer = np.random.rand(SIZE)

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

test_data = [
    (Function.FHNIntegrator, test_var, "RK4", params, ([0.33511552, 0.36644674, 0.34524828, 0.33437834, 0.31170683,
       0.35336752, 0.31430587, 0.39997071], [0.69061155, 0.69494508, 0.69201482, 0.69050939, 0.68736324,
       0.69313801, 0.68772434, 0.69956405], 0.15000000000000002)),
    (Function.FHNIntegrator, test_var, "EULER", params, ([0.33322092, 0.36410292, 0.34322017, 0.33249301, 0.31007781,
       0.35122431, 0.31265031, 0.39702677], [0.5978193 , 0.60006729, 0.59854804, 0.59776621, 0.59612942,
       0.59913078, 0.59631748, 0.6024552 ], 0.15000000000000002)),
]

# use list, naming function produces ugly names
names = [
    "FHNIntegrator RK4",
    "FHNIntegrator EULER",
]

GROUP_PREFIX="FHNIntegrator "

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.parametrize("func, variable, integration_method, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(func, variable, integration_method, params, expected, benchmark):
    f = func(default_variable=variable, integration_method=integration_method, params=params)
    benchmark.group = GROUP_PREFIX + func.componentName;
    f.function(variable)
    f.function(variable)
    res = benchmark(f.function, variable)
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
    f.bin_function(variable)
    f.bin_function(variable)
    res = benchmark(f.bin_function, variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    assert np.allclose(res[2], expected[2])
