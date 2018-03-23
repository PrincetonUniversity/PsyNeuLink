import psyneulink as pnl
import psyneulink.components.functions.function as Function
import psyneulink.globals.keywords as kw
import numpy as np
import pytest

from itertools import product

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
test_var2 = np.random.rand(2, SIZE)

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

    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_S + RAND2_S),
# TODO: enable vector scale/offset when the validation is fixed
#    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_S + RAND2_V),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_V + RAND2_S),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_V + RAND2_V),

    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_S + RAND2_S),
# TODO: enable vector scale/offset when the validation is fixed
#    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_S + RAND2_V),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_V + RAND2_S),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_V + RAND2_V),
]

# pytest naming function produces ugly names
def _naming_function(config):
    _, var, params, _, form = config
    inputs = var.shape[0]
    op = params['operation']
    vector_string = ""
    if not np.isscalar(params['scale']):
        vector_string += " SCALE"
    if not np.isscalar(params['offset']):
        vector_string += " OFFSET"
    if vector_string != "":
        vector_string = " VECTOR" + vector_string
    return "COMBINE-{} {}{} {}".format(inputs, op, vector_string, form)

_data =[a + (b,) for a, b in  product(test_linear_combination_data, ['Python', 'LLVM'])]

@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("func, variable, params, expected, bin_execute", _data, ids=list(map(_naming_function, _data)))
@pytest.mark.benchmark
def test_linear_combination_function(func, variable, params, expected, bin_execute, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "LinearCombinationFunction " + func.componentName;
    if (bin_execute == 'LLVM'):
        res = benchmark(f.bin_function, variable)
    else:
        res = benchmark(f.function, variable)
    assert np.allclose(res, expected)
