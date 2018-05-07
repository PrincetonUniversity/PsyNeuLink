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
test_var2 = np.random.rand(2, SIZE)

RAND1_V = np.random.rand(1, SIZE)
RAND2_V = np.random.rand(1, SIZE)
RAND3_V = np.random.rand(1, SIZE)

RAND1_S = np.random.rand()
RAND2_S = np.random.rand()
RAND3_S = np.random.rand()

test_linear_combination_data = [
    (Function.LinearCombination, test_var, {'scale':None, 'offset':None, 'operation':pnl.SUM}, test_var),
    (Function.LinearCombination, test_var, {'scale':None, 'offset':RAND2_S, 'operation':pnl.SUM}, test_var + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':None, 'offset':RAND2_V, 'operation':pnl.SUM}, test_var + RAND2_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':None, 'operation':pnl.SUM}, test_var * RAND1_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.SUM}, test_var * RAND1_S + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.SUM}, test_var * RAND1_S + RAND2_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':None, 'operation':pnl.SUM}, test_var * RAND1_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.SUM}, test_var * RAND1_V + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.SUM}, test_var * RAND1_V + RAND2_V),

    (Function.LinearCombination, test_var, {'scale':None, 'offset':None, 'operation':pnl.PRODUCT}, test_var),
    (Function.LinearCombination, test_var, {'scale':None, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, test_var + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':None, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, test_var + RAND2_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':None, 'operation':pnl.PRODUCT}, test_var * RAND1_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, test_var * RAND1_S + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, test_var * RAND1_S + RAND2_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':None, 'operation':pnl.PRODUCT}, test_var * RAND1_V),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, test_var * RAND1_V + RAND2_S),
    (Function.LinearCombination, test_var, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, test_var * RAND1_V + RAND2_V),

    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_S + RAND2_S),
# TODO: enable vector scale/offset when the validation is fixed
#    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_S + RAND2_V),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_V + RAND2_S),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.SUM}, np.sum(test_var2, axis=0) * RAND1_V + RAND2_V),

    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_S + RAND2_S),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_S, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_S + RAND2_V),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_S, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_V + RAND2_S),
#    (Function.LinearCombination, test_var2, {'scale':RAND1_V, 'offset':RAND2_V, 'operation':pnl.PRODUCT}, np.product(test_var2, axis=0) * RAND1_V + RAND2_V),
    (Function.LinearCombination, test_var2, {'exponents': -1., 'operation': pnl.SUM}, 1 / test_var2[0] + 1 / test_var2[1]),
]

# pytest naming function produces ugly names
def _naming_function(config):
    _, var, params, _ = config
    inputs = var.shape[0]
    op = params['operation']
    param_string = ""
    for p in 'scale','offset':
        if p not in params or params[p] is None:
            param_string += " NO "
        elif np.isscalar(params[p]):
            param_string += " SCALAR "
        else:
            param_string += " VECTOR "
        param_string += p.upper()

    return "COMBINE-{} {}{}".format(inputs, op, param_string)


@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("func, variable, params, expected", test_linear_combination_data, ids=list(map(_naming_function, test_linear_combination_data)))
@pytest.mark.benchmark
def test_linear_combination_function(func, variable, params, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "CombinationFunction " + func.componentName
    res = benchmark(f.function, variable)
    assert np.allclose(res, expected)

# ------------------------------------

# testing within a mechanism using various input states
input_1 = np.array([[1, 2, 3, 4]])

test_linear_comb_data_2 = [
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], None, None, [[1, 2, 3, 4]]),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], 2, None, [[2, 4, 6, 8]]),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], [1, 2, -1, 0], None, [1, 4, -3, 0]),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], None, 2, [3, 4, 5, 6]),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], -2, 3, None),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], [1, 2.5, 0, 0], 1.5, [2.5, 6.5, 1.5, 1.5]),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], None, [1, 0, -1, 0], [2, 2, 2, 4]),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], -2, [1, 0, -1, 0], None),
    (pnl.SUM, [[1, 2, 3, 4]], 4, ['hi'], [1, 2.5, 0, 0], [1, 0, -1, 0], None),

    (pnl.PRODUCT, [[1, 2, 3, 4]], 4, ['hi'], [1, 2.5, 0, 0], [1, 0, -1, 0], None),

    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], None, None, [[15, 18, 21, 24]]),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], 2, None, [[30, 36, 42, 48]]),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], [1, 2, -1, 0], None, [[15, 36, -21, 0]]),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], None, 2, [[17, 20, 23, 26]]),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], -2, 3, None),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], [1, 2.5, 0, 0], 1.5, None),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], None, [1, 0, -1, 0], [[16, 18, 20, 24]]),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], -2, [1, 0, -1, 0], None),
    (pnl.SUM, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 4, ['1', '2', '3'], [1, 2.5, 0, 0], [1, 0, -1, 0], None),

    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], None, None, [[0, 0, 21, 64]]),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], 2, None, [[0, 0, 42, 128]]),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], [1, 2, -1, 0], None, [[0, 0, -21, 0]]),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], None, 2, [[2, 2, 23, 66]]),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], -2, 3, None),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], [1, 2.5, 0, 0], 1.5, None),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], None, [1, 0, -1, 0], [[1, 0, 20, 64]]),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], -2, [1, 0, -1, 0], None),
    (pnl.PRODUCT, [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], 4, ['1', '2', '3'], [1, 2.5, 0, 0], [1, 0, -1, 0], None),

]

linear_comb_names_2 = [
    'sum_one_input_no_scale_no_offset',
    'sum_one_input_scalar_scale_no_offset',
    'sum_one_input_hadamard_scale_no_offset',
    'sum_one_input_no_scale_scalar_offset',
    'sum_one_input_scalar_scale_scalar_offset',
    'sum_one_input_hadamard_scale_scalar_offset',
    'sum_one_input_no_scale_hadamard_offset',
    'sum_one_input_scalar_scale_hadamard_offset',
    'sum_one_input_hadamard_scale_hadamard_offset',

    'product_one_input_hadamard_scale_hadamard_offset',

    'sum_3_input_no_scale_no_offset',
    'sum_3_input_scalar_scale_no_offset',
    'sum_3_input_hadamard_scale_no_offset',
    'sum_3_input_no_scale_scalar_offset',
    'sum_3_input_scalar_scale_scalar_offset',
    'sum_3_input_hadamard_scale_scalar_offset',
    'sum_3_input_no_scale_hadamard_offset',
    'sum_3_input_scalar_scale_hadamard_offset',
    'sum_3_input_hadamard_scale_hadamard_offset',

    'product_3_input_no_scale_no_offset',
    'product_3_input_scalar_scale_no_offset',
    'product_3_input_hadamard_scale_no_offset',
    'product_3_input_no_scale_scalar_offset',
    'product_3_input_scalar_scale_scalar_offset',
    'product_3_input_hadamard_scale_scalar_offset',
    'product_3_input_no_scale_hadamard_offset',
    'product_3_input_scalar_scale_hadamard_offset',
    'product_3_input_hadamard_scale_hadamard_offset',
]

@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("operation, input, size, input_states, scale, offset, expected", test_linear_comb_data_2, ids=linear_comb_names_2)
@pytest.mark.benchmark
def test_linear_combination_function_in_mechanism(operation, input, size, input_states, scale, offset, expected, benchmark):
    f = pnl.LinearCombination(default_variable=input, operation=operation, scale=scale, offset=offset)
    p = pnl.ProcessingMechanism(size=[size] * len(input_states), function=f, input_states=input_states)
    benchmark.group = "CombinationFunction " + pnl.LinearCombination.componentName + "in Mechanism"
    res = benchmark(f.execute, input)
    if expected is None:
        if operation == pnl.SUM:
            expected = np.sum(input, axis=0) * scale + offset
        if operation == pnl.PRODUCT:
            expected = np.product(input, axis=0) * scale + offset

    assert np.allclose(res, expected)
