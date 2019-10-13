import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.llvm as pnlvm

class TestRearrange:

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_no_default_variable(self):
        R_function = pnl.Rearrange(arrangement=[(1,2),0])
        result = R_function.execute([[0,0],[1,1],[2,2]])
        for exp,act in zip(result, [[ 1.,  1.,  2.,  2.],[ 0.,  0.]]):
            assert np.allclose(exp,act)

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_with_default_variable(self):
        R_function = pnl.Rearrange(default_variable=[[0],[0],[0]], arrangement=[(1,2),0])
        result = R_function.execute([[0,0],[1,1],[2,2]])
        for exp,act in zip(result, [[ 1.,  1.,  2.,  2.],[ 0.,  0.]]):
            assert np.allclose(exp,act)

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_arrangement_has_out_of_bounds_index(self):
        with pytest.raises(pnl.FunctionError) as error_text:
            pnl.Rearrange(default_variable=[0,0], arrangement=[(1,2),0])
        error_msg = "'default_variable' for Rearrange must be at least 2d."
        assert error_msg in str(error_text.value)

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_default_variable_mismatches_arrangement(self):
        with pytest.raises(pnl.FunctionError) as error_text:
            pnl.Rearrange(default_variable=[[0],[0]], arrangement=[(1,2),0])
        error_msg_a = "'arrangement' arg for Rearrange"
        error_msg_b = "is out of bounds for its 'default_variable' arg (max index = 1)."
        assert error_msg_a in str(error_text.value)
        assert error_msg_b in str(error_text.value)

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_default_variable_has_non_numeric_index(self):
        # with pytest.raises(pnl.FunctionError) as error_text:
        with pytest.raises(pnl.UtilitiesError) as error_text:
            pnl.Rearrange(default_variable=[[0],['a']], arrangement=[(1,2),0])
        # error_msg = "All elements of 'default_variable' for Rearrange must be scalar values."
        error_msg = "[['0']\\n ['a']] has non-numeric entries"
        assert error_msg in str(error_text.value)

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_arrangement_has_non_numeric_index(self):
        with pytest.raises(pnl.FunctionError) as error_text:
            pnl.Rearrange(default_variable=[[0],[0],[0]], arrangement=[(1,2),'a'])
        error_msg_a = "Index specified in 'arrangement' arg"
        error_msg_b = "('a') is not an int."
        assert error_msg_a in str(error_text.value)
        assert error_msg_b in str(error_text.value)

    # @pytest.mark.function
    # @pytest.mark.combination_function
    # def test_column_vector(self):
    #     R_function = pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM)
    #     R_mechanism = pnl.ProcessingMechanism(function=pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM),
    #                                           default_variable=[[1], [2], [3], [4], [5]],
    #                                           name="R_mechanism")
    #
    #     assert np.allclose(R_function.execute([[1], [2], [3], [4], [5]]), [1, 2, 3, 4, 5])
    #     # assert np.allclose(R_function.execute([[1], [2], [3], [4], [5]]), [15.0])
    #     assert np.allclose(R_function.execute([[[1], [2], [3], [4], [5]]]), [15.0])
    #
    #     assert np.allclose(R_mechanism.execute([[1], [2], [3], [4], [5]]), [1, 2, 3, 4, 5])
    #     # assert np.allclose(R_mechanism.execute([[1], [2], [3], [4], [5]]), [15.0])
    #
    # @pytest.mark.function
    # @pytest.mark.combination_function
    # def test_matrix(self):
    #     R_function = pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM)
    #     R_mechanism = pnl.ProcessingMechanism(function=pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM),
    #                                           default_variable=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #                                           name="R_mechanism")
    #
    #     assert np.allclose(R_function.execute([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [6, 15, 24])
    #     assert np.allclose(R_function.execute([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]), [12, 15, 18])
    #
    #     assert np.allclose(R_mechanism.execute([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [6, 15, 24])


class TestReduce:

    @pytest.mark.function
    @pytest.mark.combination_function
    def test_single_array(self):
        R_function = pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM),
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
        R_function = pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM),
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
        R_function = pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.core.components.functions.combinationfunctions.Reduce(operation=pnl.SUM),
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
np.random.seed(0)
#This gives us the correct 2d array
test_var = np.random.rand(1, SIZE)
test_var2 = np.random.rand(2, SIZE)

RAND1_V = np.random.rand(SIZE)
RAND2_V = np.random.rand(SIZE)
RAND3_V = np.random.rand(SIZE)

RAND1_S = np.random.rand()
RAND2_S = np.random.rand()
RAND3_S = np.random.rand()

@pytest.mark.benchmark
@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("variable", [test_var, test_var2], ids=["VAR1D", "VAR2D"])
@pytest.mark.parametrize("operation", [pnl.SUM, pnl.PRODUCT])
@pytest.mark.parametrize("exponents", [None, 2.0], ids=["E_NONE", "E_SCALAR"])
@pytest.mark.parametrize("weights", [None, 0.5, [[-1],[1]]], ids=["W_NONE", "W_SCALAR", "W_VECTOR"])
@pytest.mark.parametrize("scale", [None, RAND1_S, RAND1_V], ids=["S_NONE", "S_SCALAR", "S_VECTOR"])
@pytest.mark.parametrize("offset", [None, RAND2_S, RAND2_V], ids=["O_NONE", "O_SCALAR", "O_VECTOR"])
@pytest.mark.parametrize("bin_execute", ["Python",
                                        pytest.param("LLVM", marks=pytest.mark.llvm),
                                        pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_linear_combination_function(variable, operation, exponents, weights, scale, offset, bin_execute, benchmark):
    if weights is not None and not np.isscalar(weights) and len(variable) != len(weights):
        pytest.skip("variable/weights mismatch")

    f = pnl.core.components.functions.combinationfunctions.LinearCombination(default_variable=variable,
                                                                             operation=operation,
                                                                             exponents=exponents,
                                                                             weights=weights,
                                                                             scale=scale,
                                                                             offset=offset)
    benchmark.group = "LinearCombinationFunction"
    if (bin_execute == 'LLVM'):
        e = pnlvm.execution.FuncExecution(f)
        res = benchmark(e.execute, variable)
    elif (bin_execute == 'PTX'):
        e = pnlvm.execution.FuncExecution(f)
        res = benchmark(e.cuda_execute, variable)
    else:
        res = benchmark(f.function, variable)

    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    exponent = 1.0 if exponents is None else exponents
    weights = 1.0 if weights is None else weights

    tmp = (variable ** exponent) * weights
    if operation == pnl.SUM:
        expected = np.sum(tmp, axis=0) * scale + offset
    if operation == pnl.PRODUCT:
        expected = np.product(tmp, axis=0) * scale + offset

    assert np.allclose(res, expected)

# ------------------------------------

@pytest.mark.benchmark
@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("operation", [pnl.SUM, pnl.PRODUCT])
@pytest.mark.parametrize("input, input_ports", [ ([[1,2,3,4]], ["hi"]), ([[1,2,3,4], [5,6,7,8], [9,10,11,12]], ['1','2','3']), ([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], ['1','2','3']) ], ids=["1S", "2S", "3S"])
@pytest.mark.parametrize("scale", [None, 2.5, [1,2.5,0,0]], ids=["S_NONE", "S_SCALAR", "S_VECTOR"])
@pytest.mark.parametrize("offset", [None, 1.5, [1,2.5,0,0]], ids=["O_NONE", "O_SCALAR", "O_VECTOR"])
@pytest.mark.parametrize("mode", ["Python",
                                  pytest.param("LLVM", marks=pytest.mark.llvm),
                                  pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_linear_combination_function_in_mechanism(operation, input, input_ports, scale, offset, benchmark, mode):
    f = pnl.core.components.functions.combinationfunctions.LinearCombination(default_variable=input, operation=operation, scale=scale, offset=offset)
    p = pnl.ProcessingMechanism(size=[len(input[0])] * len(input), function=f, input_ports=input_ports)
    benchmark.group = "CombinationFunction " + pnl.core.components.functions.combinationfunctions.LinearCombination.componentName + "in Mechanism"

    if mode == "Python":
        res = benchmark(f.execute, input)
    elif mode == "LLVM":
        e = pnlvm.execution.FuncExecution(f)
        res = benchmark(e.execute, input)
    elif mode == "PTX":
        e = pnlvm.execution.FuncExecution(f)
        res = benchmark(e.cuda_execute, input)


    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    if operation == pnl.SUM:
        expected = np.sum(input, axis=0) * scale + offset
    if operation == pnl.PRODUCT:
        expected = np.product(input, axis=0) * scale + offset

    assert np.allclose(res, expected)
