import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.statefulfunctions.integratorfunctions
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
test_scalar = np.random.rand()

test_data = [
    (psyneulink.core.components.functions.statefulfunctions.integratorfunctions.FitzHughNagumoIntegrator, test_var, "RK4", params, ([0.23619944, 0.24032298, 0.22321782, 0.43865125, 0.42363054,
                                                                                                                          0.44901757, 0.47938108, 0.42941189], [0.21378097, 0.21388886, 0.21344061, 0.21894107, 0.21856817,
       0.21919746, 0.21994384, 0.2187119 ], 0.15000000000000002)),
    (psyneulink.core.components.functions.statefulfunctions.integratorfunctions.FitzHughNagumoIntegrator, test_scalar, "RK4", params,
     ([0.33803257], [0.21641212], 0.15000000000000002)),
    (psyneulink.core.components.functions.statefulfunctions.integratorfunctions.FitzHughNagumoIntegrator, test_var, "EULER", params, ([0.23686576, 0.24093183, 0.22404678, 0.43291206, 0.41863405,
                                                                                                                            0.44273909,
                                                                                                                            0.47139546, 0.42413492], [0.20757016, 0.2076755 , 0.20723764, 0.21257185, 0.21221299,
       0.21281834, 0.21353476, 0.21235135], 0.15000000000000002)),
    (psyneulink.core.components.functions.statefulfunctions.integratorfunctions.FitzHughNagumoIntegrator, test_scalar, "EULER", params, ([0.33642314], [0.21013003], 0.15000000000000002)),
]

# use list, naming function produces ugly names
names = [
    "FitzHughNagumoIntegrator RK4 VECTOR",
    "FitzHughNagumoIntegrator RK4 SCALAR",
    "FitzHughNagumoIntegrator EULER VECTOR",
    "FitzHughNagumoIntegrator EULER SCALAR",
]

@pytest.mark.function
@pytest.mark.integrator_function
@pytest.mark.fitzHughNagumo_integrator_function
@pytest.mark.benchmark(group="FitzHughNagumoIntegrator")
@pytest.mark.parametrize("func, variable, integration_method, params, expected", test_data, ids=names)
@pytest.mark.parametrize('mode', ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_basic(func, variable, integration_method, params, expected, benchmark, mode):
    f = func(default_variable=variable, integration_method=integration_method, params=params)
    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    res = EX(variable)
    res = EX(variable)
    res = EX(variable)

    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    assert np.allclose(res[2], expected[2])

    if benchmark.enabled:
        benchmark(f, variable)
