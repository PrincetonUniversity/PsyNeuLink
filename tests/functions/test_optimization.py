import numpy as np
import pytest

import psyneulink as pnl

SIZE = 5
# Some metrics (CROSS_ENTROPY) don't like 0s
test_var = np.random.rand(SIZE) + pnl.EPSILON
EPS = float(pnl.EPSILON)
search_space = [pnl.SampleIterator([EPS, 1.0] if i % 2 == 0 else pnl.SampleSpec(start=EPS, stop=1.0, num=2)) for i in range(SIZE)]
results = {
    pnl.Stability: {
        pnl.ENERGY: {
            True: {
                pnl.MINIMIZE: {
                    pnl.FIRST: ((1.0, 1.0, 1.0, 1.0, 1.0), -0.4, [], []),
                    pnl.RANDOM: ((1.0, 1.0, 1.0, 1.0, 1.0), -0.4, [], []),
                },
                pnl.MAXIMIZE: {
                    pnl.FIRST: ((EPS, EPS, EPS, EPS, EPS), -1.9721522630525296e-32, [], []),
                    pnl.RANDOM: ((1.0, EPS, EPS, EPS, EPS), -1.9721522630525296e-32, [], []),
                },
            },
            False: {
                pnl.MINIMIZE: {
                    pnl.FIRST: ((1.0, 1.0, 1.0, 1.0, 1.0), -10.0, [], []),
                    pnl.RANDOM: ((1.0, 1.0, 1.0, 1.0, 1.0), -10.0, [], []),
                },
                pnl.MAXIMIZE: {
                    pnl.FIRST: ((EPS, EPS, EPS, EPS, EPS), -4.930380657631324e-31, [], []),
                    pnl.RANDOM: ((1.0, EPS, EPS, EPS, EPS), -4.930380657631324e-31, [], []),
                },
            },
        },
        # ENTROPY now = Shannon entropy in bits of p=v/sum(v): min (0) at one-hot points,
        # max (log2(5)=2.3219) at the uniform point; normalize=True divides by length (5).
        pnl.ENTROPY: {
            True: {
                pnl.MINIMIZE: {
                    pnl.FIRST: ((EPS, EPS, EPS, EPS, 1.0), 0.0, [], []),
                    pnl.RANDOM: ((EPS, EPS, 1.0, EPS, EPS), 0.0, [], []),
                },
                pnl.MAXIMIZE: {
                    pnl.FIRST: ((EPS, EPS, EPS, EPS, EPS), 0.4643856189774724, [], []),
                    pnl.RANDOM: ((EPS, EPS, EPS, EPS, EPS), 0.4643856189774724, [], []),
                },
            },
            False: {
                pnl.MINIMIZE: {
                    pnl.FIRST: ((EPS, EPS, EPS, EPS, 1.0), 0.0, [], []),
                    pnl.RANDOM: ((EPS, EPS, 1.0, EPS, EPS), 0.0, [], []),
                },
                pnl.MAXIMIZE: {
                    pnl.FIRST: ((EPS, EPS, EPS, EPS, EPS), 2.321928094887362, [], []),
                    pnl.RANDOM: ((EPS, EPS, EPS, EPS, EPS), 2.321928094887362, [], []),
                },
            },
        },
    },
}


@pytest.mark.function
@pytest.mark.benchmark
@pytest.mark.optimization_function
@pytest.mark.parametrize("selection", [pnl.FIRST, pnl.RANDOM])
@pytest.mark.parametrize("direction", [pnl.MINIMIZE, pnl.MAXIMIZE])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("metric", [pnl.ENERGY, pnl.ENTROPY])
@pytest.mark.parametrize("obj_func", [pnl.Stability])
def test_grid_search(obj_func, metric, normalize, direction, selection, benchmark, func_mode):
    variable = test_var
    result = results[obj_func][metric][normalize][direction][selection]
    benchmark.group = "OptimizationFunction " + str(obj_func) + " " + metric

    of = obj_func(default_variable=variable, metric=metric, normalize=normalize)
    f = pnl.GridSearch(objective_function=of,
                       default_variable=variable,
                       search_space=search_space,
                       direction=direction,
                       select_randomly_from_optimal_values=(selection==pnl.RANDOM),
                       seed=0,
                       save_values=False)

    EX = pytest.helpers.get_func_execution(f, func_mode)

    res = benchmark(EX, variable)

    np.testing.assert_allclose(res[0], result[0], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(res[1], result[1], rtol=1e-5, atol=1e-8)

    if func_mode == 'Python':
        np.testing.assert_allclose(res[2], result[2])
        np.testing.assert_allclose(res[3], result[3])

    else:
        assert len(res) == 2
