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
        with pytest.raises(pnl.FunctionError) as error_text:
            pnl.Rearrange(default_variable=[[0],['a']], arrangement=[(1,2),0])
        # error_msg = "All elements of 'default_variable' for Rearrange must be scalar values."
        error_msg = "must be scalar values"
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
np.random.seed(0)
#This gives us the correct 2d array
test_varr1 = np.random.rand(1, SIZE)
test_varr2 = np.random.rand(2, SIZE)
test_varr3 = np.random.rand(3, SIZE)

#This gives us the correct 2d column array
test_varc1 = np.random.rand(SIZE, 1)
test_varc2 = np.random.rand(SIZE, 1)
test_varc3 = np.random.rand(SIZE, 1)

#This gives us the correct 2d matrix array
test_varm1 = np.random.rand(SIZE, 3)
test_varm2 = np.random.rand(SIZE, 3)
test_varm3 = np.random.rand(SIZE, 3)

RAND1_V = np.random.rand(SIZE)
RAND2_V = np.random.rand(SIZE)
RAND3_V = np.random.rand(SIZE)

RAND1_S = np.random.rand()
RAND2_S = np.random.rand()
RAND3_S = np.random.rand()


@pytest.mark.benchmark(group="ReduceFunction")
@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("variable", [test_varr1, test_varr2, test_varr3,
                                      test_varc1, test_varc2, test_varc3,
                                      test_varm1, test_varm2, test_varm3,
                                     ], ids=["VAR1", "VAR2", "VAR3",
                                             "VAR1c", "VAR2c", "VAR3c",
                                             "VAR1m", "VAR2m", "VAR3m",
                                            ])
@pytest.mark.parametrize("operation", [pnl.SUM, pnl.PRODUCT])
@pytest.mark.parametrize("exponents", [None, 2.0, [3.0], 'V'], ids=["E_NONE", "E_SCALAR", "E_VECTOR1", "E_VECTORN"])
@pytest.mark.parametrize("weights", [None, 0.5, 'VC', 'VR'], ids=["W_NONE", "W_SCALAR", "W_VECTORN", "W_VECTORM"])
@pytest.mark.parametrize("scale", [RAND1_S, RAND1_V], ids=["S_SCALAR", "S_VECTOR"])
@pytest.mark.parametrize("offset", [RAND2_S, RAND2_V], ids=["O_SCALAR", "O_VECTOR"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_reduce_function(variable, operation, exponents, weights, scale, offset, mode, benchmark):
    if weights == 'VC':
        weights = [[(-1) ** i] for i, v in enumerate(variable)]
    if weights == 'VR':
        weights = [(-1) ** i for i, v in enumerate(variable[0])]
    if exponents == 'V':
        exponents = [[v[0]] for v in variable]

    try:
        f = pnl.Reduce(default_variable=variable,
                       operation=operation,
                       exponents=exponents,
                       weights=weights,
                       scale=scale,
                       offset=offset)
    except ValueError as e:
        if not np.isscalar(scale) and "The truth value of an array" in str(e):
            pytest.xfail("vector scale is not supported")
        if not np.isscalar(offset) and "The truth value of an array" in str(e):
            pytest.xfail("vector offset is not supported")
        raise e from None

    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    res = benchmark(EX, variable)

    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    exponent = 1.0 if exponents is None else exponents
    weights = 1.0 if weights is None else weights

    tmp = (variable ** exponent) * weights
    if operation == pnl.SUM:
        expected = np.sum(tmp, axis=1) * scale + offset
    if operation == pnl.PRODUCT:
        expected = np.product(tmp, axis=1) * scale + offset

    assert np.allclose(res, expected)


@pytest.mark.benchmark(group="LinearCombinationFunction")
@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("variable", [test_varr1, test_varr2], ids=["VAR1", "VAR2"])
@pytest.mark.parametrize("operation", [pnl.SUM, pnl.PRODUCT])
@pytest.mark.parametrize("exponents", [None, 2.0, [3.0], 'V'], ids=["E_NONE", "E_SCALAR", "E_VECTOR1", "E_VECTORN"])
@pytest.mark.parametrize("weights", [None, 0.5, 'V'], ids=["W_NONE", "W_SCALAR", "W_VECTORN"])
@pytest.mark.parametrize("scale", [None, RAND1_S, RAND1_V], ids=["S_NONE", "S_SCALAR", "S_VECTOR"])
@pytest.mark.parametrize("offset", [None, RAND2_S, RAND2_V], ids=["O_NONE", "O_SCALAR", "O_VECTOR"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_linear_combination_function(variable, operation, exponents, weights, scale, offset, mode, benchmark):
    if weights == 'V':
        weights = [[-1 ** i] for i, v in enumerate(variable)]
    if exponents == 'V':
        exponents = [[v[0]] for v in variable]

    f = pnl.LinearCombination(default_variable=variable,
                              operation=operation,
                              exponents=exponents,
                              weights=weights,
                              scale=scale,
                              offset=offset)
    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute

    res = benchmark(EX, variable)

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


@pytest.mark.benchmark(group="LinearCombinationFunction in Mechanism")
@pytest.mark.function
@pytest.mark.combination_function
@pytest.mark.parametrize("operation", [pnl.SUM, pnl.PRODUCT])
@pytest.mark.parametrize("input, input_ports", [ ([[1,2,3,4]], ["hi"]), ([[1,2,3,4], [5,6,7,8], [9,10,11,12]], ['1','2','3']), ([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2]], ['1','2','3']) ], ids=["1S", "2S", "3S"])
@pytest.mark.parametrize("scale", [None, 2.5, [1,2.5,0,0]], ids=["S_NONE", "S_SCALAR", "S_VECTOR"])
@pytest.mark.parametrize("offset", [None, 1.5, [1,2.5,0,0]], ids=["O_NONE", "O_SCALAR", "O_VECTOR"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_linear_combination_function_in_mechanism(operation, input, input_ports, scale, offset, benchmark, mode):
    f = pnl.LinearCombination(default_variable=input, operation=operation, scale=scale, offset=offset)
    p = pnl.ProcessingMechanism(size=[len(input[0])] * len(input), function=f, input_ports=input_ports)

    if mode == 'Python':
        EX = p.execute
    elif mode == 'LLVM':
        e = pnlvm.execution.MechExecution(p)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.MechExecution(p)
        EX = e.cuda_execute

    res = benchmark(EX, input)

    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    if operation == pnl.SUM:
        expected = np.sum(input, axis=0) * scale + offset
    if operation == pnl.PRODUCT:
        expected = np.product(input, axis=0) * scale + offset

    assert np.allclose(res, expected)
