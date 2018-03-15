import psyneulink as pnl
import psyneulink.components.functions.function as Function
import psyneulink.globals.keywords as kw
import numpy as np
import pytest

class TestReduce:

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_single_array(self):
        R_function = pnl.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
                                              default_variable=[[1, 2, 3, 4, 5]],
                                              name="R_mechanism")

        assert np.allclose(R_function.execute([1, 2, 3, 4, 5]), [15.0])
        assert np.allclose(R_function.execute([[1, 2, 3, 4, 5]]), [15.0])
        assert np.allclose(R_function.execute([[[1, 2, 3, 4, 5]]]), [1, 2, 3, 4, 5])
        # assert np.allclose(R_function.execute([[[1, 2, 3, 4, 5]]]), [15.0])

        assert np.allclose(R_mechanism.execute([1, 2, 3, 4, 5]), [[15.0]])
        assert np.allclose(R_mechanism.execute([[1, 2, 3, 4, 5]]), [[15.0]])
        assert np.allclose(R_mechanism.execute([1, 2, 3, 4, 5]), [15.0])
        # assert np.allclose(R_mechanism.execute([[1, 2, 3, 4, 5]]), [15.0])

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_column_vector(self):
        R_function = pnl.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
                                              default_variable=[[1], [2], [3], [4], [5]],
                                              name="R_mechanism")

        assert np.allclose(R_function.execute([[1], [2], [3], [4], [5]]), [1, 2, 3, 4, 5])
        # assert np.allclose(R_function.execute([[1], [2], [3], [4], [5]]), [15.0])
        assert np.allclose(R_function.execute([[[1], [2], [3], [4], [5]]]), [15.0])

        assert np.allclose(R_mechanism.execute([[1], [2], [3], [4], [5]]), [1, 2, 3, 4, 5])
        # assert np.allclose(R_mechanism.execute([[1], [2], [3], [4], [5]]), [15.0])

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_matrix(self):
        R_function = pnl.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
                                              default_variable=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                              name="R_mechanism")

        assert np.allclose(R_function.execute([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [6, 15, 24])
        assert np.allclose(R_function.execute([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]), [12, 15, 18])

        assert np.allclose(R_mechanism.execute([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [6, 15, 24])

    # def test_heterogeneous_arrays(self):
    #     R_function = pnl.Reduce(operation=pnl.SUM)
    #     # R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
    #     #                                       default_variable=[[1, 2], [3, 4, 5], [6, 7, 8, 9]],
    #     #                                       name="R_mechanism")
    #     print(R_function.execute([[1, 2], [3, 4, 5], [6, 7, 8, 9]]))
    #     print(R_function.execute([[[1, 2], [3, 4, 5], [6, 7, 8, 9]]]))
    #
    #     # print("mech = ", R_mechanism.execute([[1, 2], [3, 4, 5], [6, 7, 8, 9]]))
    #     # print("mech = ", R_mechanism.execute([[[1, 2], [3, 4, 5], [6, 7, 8, 9]]]))
    #     # print("mech = ", R_mechanism.execute([[[1, 2], [3, 4, 5], [6, 7, 8, 9]]]))
    #


SIZE=5
#This gives us the correct 2d array
test_var = np.random.rand(1, SIZE)

RAND1_V = np.random.rand(1, SIZE)
RAND2_V = np.random.rand(1, SIZE)
RAND3_V = np.random.rand(1, SIZE)

RAND1_S = np.random.rand()
RAND2_S = np.random.rand()
RAND3_S = np.random.rand()

test_linear_combination_data = [
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.SUM}, test_var * RAND1_S + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.SUM}, test_var * RAND1_S + RAND2_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.SUM}, test_var * RAND1_V + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.SUM}, test_var * RAND1_V + RAND2_V),

    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, test_var * RAND1_S + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, test_var * RAND1_S + RAND2_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, test_var * RAND1_V + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, test_var * RAND1_V + RAND2_V),
]

# use list, naming function produces ugly names
linear_combination_names = [
    "COMBINE-1 SUM",
    "COMBINE-1 SUM VECTOR OFFSET",
    "COMBINE-1 SUM VECTOR SCALE",
    "COMBINE-1 SUM VECTOR OFFSET SCALE",

    "COMBINE-1 PRODUCT",
    "COMBINE-1 PRODUCT VECTOR OFFSET",
    "COMBINE-1 PRODUCT VECTOR SCALE",
    "COMBINE-1 PRODUCT VECTOR OFFSET SCALE",
]

@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("func, variable, params, expected", test_linear_combination_data, ids=linear_combination_names)
@pytest.mark.benchmark
def test_linear_combination_function(func, variable, params, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName;
    res = benchmark(f.function, variable)
    assert np.allclose(res, expected)
