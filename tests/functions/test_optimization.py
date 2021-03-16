import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.function as Function
import psyneulink.core.components.functions.objectivefunctions as Functions
import psyneulink.core.components.functions.optimizationfunctions as OPTFunctions
import psyneulink.core.globals.keywords as kw
from psyneulink.core.globals.sampleiterator import SampleIterator, SampleSpec
import pytest

SIZE=5
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = np.random.rand(SIZE) + Function.EPSILON
EPS = float(Function.EPSILON)
search_space = [SampleIterator([EPS, 1.0] if i % 2 == 0 else SampleSpec(start=EPS, stop=1.0, num=2)) for i in range(SIZE)]
results = {
    Functions.Stability: {
        kw.ENERGY: {
            True: {
                OPTFunctions.MINIMIZE: {
                    'FIRST': ((1.0, 1.0, 1.0, 1.0, 1.0), -0.4, [], []),
                    'RANDOM': ((1.0, 1.0, 1.0, 1.0, 1.0), -0.4, [], []),
                },
                OPTFunctions.MAXIMIZE: {
                    'FIRST': ((EPS, EPS, EPS, EPS, EPS), -1.9721522630525296e-32, [], []),
                    'RANDOM': ((1.0, EPS, EPS, EPS, EPS), -1.9721522630525296e-32, [], []),
                },
            },
            False: {
                OPTFunctions.MINIMIZE: {
                    'FIRST': ((1.0, 1.0, 1.0, 1.0, 1.0), -10.0, [], []),
                    'RANDOM': ((1.0, 1.0, 1.0, 1.0, 1.0), -10.0, [], []),
                },
                OPTFunctions.MAXIMIZE: {
                    'FIRST': ((EPS, EPS, EPS, EPS, EPS), -4.930380657631324e-31, [], []),
                    'RANDOM': ((1.0, EPS, EPS, EPS, EPS), -4.930380657631324e-31, [], []),
                },
            },
        },
        kw.ENTROPY: {
            True: {
                OPTFunctions.MINIMIZE: {
                    'FIRST': ((1.0, 1.0, 1.0, 1.0, 1.0), -1.3862943611198906, [], []),
                    'RANDOM': ((1.0, 1.0, 1.0, 1.0, 1.0), -1.3862943611198906, [], []),
                },
                OPTFunctions.MAXIMIZE: {
                    'FIRST': ((EPS, EPS, EPS, EPS, 1.0), 6.931471805599453, [], []),
                    'RANDOM': ((EPS, EPS, 1.0, EPS, EPS), 6.931471805599453, [], []),
                },
            },
            False: {
                OPTFunctions.MINIMIZE: {
                    'FIRST': ((1.0, 1.0, 1.0, 1.0, 1.0), -6.931471805599453, [], []),
                    'RANDOM': ((1.0, 1.0, 1.0, 1.0, 1.0), -6.931471805599453, [], []),
                },
                OPTFunctions.MAXIMIZE: {
                    'FIRST': ((EPS, EPS, EPS, EPS, 1.0), 34.657359027997266, [], []),
                    'RANDOM': ((EPS, EPS, 1.0, EPS, EPS), 34.657359027997266, [], []),
                },
            },
        },
    },
}


@pytest.mark.function
@pytest.mark.benchmark
@pytest.mark.optimization_function
@pytest.mark.parametrize("selection", ['FIRST', 'RANDOM'])
@pytest.mark.parametrize("direction", [OPTFunctions.MINIMIZE, OPTFunctions.MAXIMIZE])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("metric", [kw.ENERGY, kw.ENTROPY])
@pytest.mark.parametrize("obj_func", [Functions.Stability])
@pytest.mark.parametrize('mode', ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_grid_search(obj_func, metric, normalize, direction, selection, benchmark, mode):
    variable = test_var
    result = results[obj_func][metric][normalize][direction][selection]
    benchmark.group = "OptimizationFunction " + str(obj_func) + " " + metric

    of = obj_func(default_variable=variable, metric=metric, normalize=normalize)
    f = OPTFunctions.GridSearch(objective_function=of, default_variable=variable,
                                search_space=search_space, direction=direction,
                                select_randomly_from_optimal_values=(selection=='RANDOM'),
                                seed=0)
    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    res = EX(variable)

    assert np.allclose(res[0], result[0])
    assert np.allclose(res[1], result[1])
    if mode == 'Python':
        assert np.allclose(res[2], result[2])
        assert np.allclose(res[3], result[3])

    if benchmark.enabled:
        benchmark(f.function, variable)
