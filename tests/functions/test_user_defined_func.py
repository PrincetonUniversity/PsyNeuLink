import math

import numpy as np
import pytest

from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear, Logistic
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.processing import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing import TransferMechanism
from psyneulink.core.compositions.composition import Composition

import psyneulink.core.llvm as pnlvm


# default val is same shape as expected output
# we only use param1 and param2 to avoid automatic shape changes of the variable
def binAdd(_, param1, param2):
    return param1 + param2


def binSub(_, param1, param2):
    return param1 - param2


def binMul(_, param1, param2):
    return param1 * param2


def binDiv(_, param1, param2):
    return param1 / param2


def binPow(_, param1, param2):
    return param1 ** param2


@pytest.mark.parametrize("param1", [1, np.ones(2), np.ones((2, 2))], ids=['scalar', 'vector', 'matrix'])
@pytest.mark.parametrize("param2", [2, np.ones(2) * 2, np.ones((2, 2)) * 2], ids=['scalar', 'vector', 'matrix'])
@pytest.mark.parametrize("func", [binAdd, binSub, binMul, binDiv, binPow])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_bin_arith(param1, param2, func, func_mode, benchmark):

    U = UserDefinedFunction(custom_function=func, param1=param1, param2=param2)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, 0)
    assert np.allclose(val, func(0, param1=param1, param2=param2))


def unNot(variable):
    var1 = False
    # compiled UDFs don't support python bool type outputs
    if not var1:
        return 0.0
    else:
        return 1.0


def binAnd(variable):
    var1 = True
    var2 = False
    # compiled UDFs don't support python bool type outputs
    if var1 and var2:
        return 0.0
    else:
        return 1.0

def triAnd(variable):
    var1 = True
    var2 = False
    # compiled UDFs don't support python bool type outputs
    if var1 and var2 and False:
        return 0.0
    else:
        return 1.0


def binOr(variable):
    var1 = True
    var2 = False
    # compiled UDFs don't support python bool type outputs
    if var1 or var2:
        return 1.0
    else:
        return 0.0


def triOr(variable):
    var1 = True
    var2 = False
    # compiled UDFs don't support python bool type outputs
    if var1 or var2 or True:
        return 1.0
    else:
        return 0.0


def multiAndOr(variable):
    var1 = True
    var2 = False
    # compiled UDFs don't support python bool type outputs
    if var1 or var2 and True or True and var1:
        return 1.0
    else:
        return 0.0


def varAnd(var):
    return var[0] and var[1]


def varOr(var):
    return var[0] or var[1]


def varNot(var):
    # compiled UDFs don't support python bool type outputs
    if not var[0]:
        return 0.0
    else:
        return 1.0


@pytest.mark.parametrize("op,var, expected", [
    (unNot, 0, 0.0),
    (binAnd, 0, 1.0),
    (binOr, 0, 1.0),
    (triAnd, 0, 1.0),
    (triOr, 0, 1.0),
    (multiAndOr, 0, 1.0),
    (varAnd, [0.0, 0.0], 0.0),
    (varAnd, [5.0, -0.0], -0.0),
    (varAnd, [0.0, -2.0], 0.0),
    (varAnd, [5.0, -2.0], -2.0),
    (varOr, [-0.0, 0.0], 0.0),
    (varOr, [3.0, 0.0], 3.0),
    (varOr, [0.0, -3.0], -3.0),
    (varOr, [1.5, -1.0], 1.5),
    (varNot, [1.5], 1.0),
    (varNot, [0.0], 0.0),
    ])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_boolop(op, var, expected, func_mode, benchmark):
    U = UserDefinedFunction(custom_function=op, default_variable=var)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, var)
    assert val == expected


def binEQ(variable, var1, var2):
    if var1 == var2:
        return 1.0
    else:
        return 0.0


def binNE(variable, var1, var2):
    if var1 != var2:
        return 1.0
    else:
        return 0.0


def binLT(variable, var1, var2):
    if var1 < var2:
        return 1.0
    else:
        return 0.0


def binLE(variable, var1, var2):
    if var1 <= var2:
        return 1.0
    else:
        return 0.0


def binGT(variable, var1, var2):
    if var1 > var2:
        return 1.0
    else:
        return 0.0


def binGE(variable, var1, var2):
    if var1 >= var2:
        return 1.0
    else:
        return 0.0

@pytest.mark.parametrize("func,var1,var2,expected", [
                    (binEQ, 1.0, 2.0, 0.0),
                    (binNE, 1.0, 2.0, 1.0),
                    (binLT, 1.0, 2.0, 1.0),
                    (binLE, 1.0, 2.0, 1.0),
                    (binGT, 1.0, 2.0, 0.0),
                    (binGE, 1.0, 2.0, 0.0),
                    ])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_cmpop(func, var1, var2, expected, func_mode, benchmark):
    U = UserDefinedFunction(custom_function=func, default_variable=[0], var1=var1, var2=var2)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, [0])
    assert np.allclose(expected, val)

@pytest.mark.parametrize("op,var1,var2,expected", [ # parameter is string since compiled udf doesn't support closures as of present
                    ("Eq", 1.0, 2.0, 0.0),
                    ("Eq", [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]),
                    ("Eq", 1.0, [1.0, 2.0], [1.0, 0.0]),
                    ("Eq", [2.0, 1.0], 1.0, [0.0, 1.0]),
                    ("Eq", [[1.0, 2.0], [3.0, 4.0]], 1.0, [[1.0, 0.0], [0.0, 0.0]]),
                    ("Eq", 1.0, [[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 0.0]]),
                    ("Eq", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]),
                    ("NotEq", 1.0, 2.0, 1.0),
                    ("NotEq", [1.0, 2.0], [1.0, 2.0], [0.0, 0.0]),
                    ("NotEq", 1.0, [1.0, 2.0], [0.0, 1.0]),
                    ("NotEq", [2.0, 1.0], 1.0, [1.0, 0.0]),
                    ("NotEq", [[1.0, 2.0], [3.0, 4.0]], 1.0, [[0.0, 1.0], [1.0, 1.0]]),
                    ("NotEq", 1.0, [[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 1.0]]),
                    ("NotEq", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]),
                    ("Lt", 1.0, 2.0, 1.0),
                    ("Lt", [1.0, 2.0], [1.0, 2.0], [0.0, 0.0]),
                    ("Lt", 1.0, [1.0, 2.0], [0.0, 1.0]),
                    ("Lt", [2.0, 1.0], 1.0, [0.0, 0.0]),
                    ("Lt", [[1.0, 2.0], [3.0, 4.0]], 1.0, [[0.0, 0.0], [0.0, 0.0]]),
                    ("Lt", 1.0, [[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 1.0]]),
                    ("Lt", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]),
                    ("LtE", 1.0, 2.0, 1.0),
                    ("LtE", [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]),
                    ("LtE", 1.0, [1.0, 2.0], [1.0, 1.0]),
                    ("LtE", [2.0, 1.0], 1.0, [0.0, 1.0]),
                    ("LtE", [[1.0, 2.0], [3.0, 4.0]], 1.0, [[1.0, 0.0], [0.0, 0.0]]),
                    ("LtE", 1.0, [[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]),
                    ("LtE", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]),
                    ("Gt", 1.0, 2.0, 0.0),
                    ("Gt", [1.0, 2.0], [1.0, 2.0], [0.0, 0.0]),
                    ("Gt", 1.0, [1.0, 2.0], [0.0, 0.0]),
                    ("Gt", [2.0, 1.0], 1.0, [1.0, 0.0]),
                    ("Gt", [[1.0, 2.0], [3.0, 4.0]], 1.0, [[0.0, 1.0], [1.0, 1.0]]),
                    ("Gt", 1.0, [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]),
                    ("Gt", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]),
                    ("GtE", 1.0, 2.0, 0.0),
                    ("GtE", [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]),
                    ("GtE", 1.0, [1.0, 2.0], [1.0, 0.0]),
                    ("GtE", [2.0, 1.0], 1.0, [1.0, 1.0]),
                    ("GtE", [[1.0, 2.0], [3.0, 4.0]], 1.0, [[1.0, 1.0], [1.0, 1.0]]),
                    ("GtE", 1.0, [[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 0.0]]),
                    ("GtE", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]),
                    ])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_cmpop_numpy(op, var1, var2, expected, func_mode, benchmark):
    # we explicitly use np here to ensure that the result is castable to float in the scalar-scalar case
    if op == "Eq":
        def myFunction(variable, var1, var2):
            return np.equal(var1, var2).astype(float)
    elif op == "NotEq":
        def myFunction(variable, var1, var2):
            return np.not_equal(var1, var2).astype(float)
    elif op == "Lt":
        def myFunction(variable, var1, var2):
            return np.less(var1, var2).astype(float)
    elif op == "LtE":
        def myFunction(variable, var1, var2):
            return np.less_equal(var1, var2).astype(float)
    elif op == "Gt":
        def myFunction(variable, var1, var2):
            return np.greater(var1, var2).astype(float)
    elif op == "GtE":
        def myFunction(variable, var1, var2):
            return np.greater_equal(var1, var2).astype(float)

    U = UserDefinedFunction(custom_function=myFunction, default_variable=[0], var1=var1, var2=var2)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, [0])
    assert np.allclose(expected, val)


def simpleFun(variable, param1, param2):
    return variable * 2 + param2


def condReturn(variable, param1, param2):
    if variable[0]:
        return param1 + 0.5
    return param2 + 0.3


def condValReturn(variable, param1, param2):
    if variable[0]:
        val = param1 + 0.5
    else:
        val = param2 + 0.3
    return val

def lambdaGen():
    return lambda var, param1, param2: var + param1 * param2


@pytest.mark.parametrize("func,var,params,expected", [
    (simpleFun, [1, 3], {"param1":None, "param2":3}, [5, 9]),
    (condReturn, [0], {"param1":1, "param2":2}, [2.3]),
    (condReturn, [1], {"param1":1, "param2":2}, [1.5]),
    (condValReturn, [0], {"param1":1, "param2":2}, [2.3]),
    (condValReturn, [1], {"param1":1, "param2":2}, [1.5]),
    (lambdaGen(), [3], {"param1":3, "param2":-0.5}, [1.5]),
])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func(func, var, params, expected, func_mode, benchmark):

    U = UserDefinedFunction(custom_function=func, default_variable=var, **params)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, var)
    assert np.allclose(val, expected)


def branchOnVarCmp(variable, param1, param2):
    if variable[0][0] > 0 and variable[0][1] > 0:
        return variable * 2 + param2
    else:
        return variable * -2 + param2


def branchOnVarFloat(variable, param1, param2):
    if variable[0]:
        return 1.0
    else:
        return 0.0


def swap(variable, param1, param2):
    val = variable[0]
    variable[0] = variable[1]
    variable[1] = val
    return variable


def indexLit(variable, param1, param2):
    return [1,2,3,4][variable[0] + 2]


@pytest.mark.parametrize("func,var,expected", [
    (branchOnVarCmp, [[1, 3]], [[5, 9]]),
    (branchOnVarCmp, [[-1, 3]], [[5, -3]]),
    (branchOnVarFloat, [0.0], [0.0]),
    (branchOnVarFloat, [-0.0], [0.0]),
    (branchOnVarFloat, [-1.5], [1.0]),
    (branchOnVarFloat, [1.5], [1.0]),
    (branchOnVarFloat, [float("Inf")], [1.0]),
    (branchOnVarFloat, [float("-Inf")], [1.0]),
    (branchOnVarFloat, [float("NaN")], [1.0]),
    (branchOnVarFloat, [float("-NaN")], [1.0]),
    (swap, [-1, 3], [3, -1]),
    (indexLit, [0], [3]),
])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_branching(func, var, expected, func_mode, benchmark):

    U = UserDefinedFunction(custom_function=func, default_variable=var, param2=3)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, var)
    assert np.allclose(val, expected)


@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_variable_index(func_mode, benchmark):
    def myFunction(variable):
        variable[0][0] = variable[0][0] + 5
        variable[0][1] = variable[0][1] + 7
        return variable

    U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]])
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, [[1, 3]])
    assert np.allclose(val, [[6, 10]])


def unarySubVar(variable, param):
    return -variable


def unarySubParam(variable, param):
    return -param


def unaryAddVar(variable, param):
    return +variable


def unaryAddParam(variable, param):
    return +param


@pytest.mark.parametrize("variable", [
                    1,
                    np.ones(2),
                    np.ones((2)),
                    np.ones((2, 2))
                    ], ids=["scalar", "vec", "vec-2d", "mat"])
@pytest.mark.parametrize("func", [unarySubVar, unarySubParam, unaryAddVar, unaryAddParam])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_unary(func, variable, func_mode, benchmark):
    U = UserDefinedFunction(custom_function=func, default_variable=variable, param=variable)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, variable)
    assert np.allclose(val, func(variable, param=variable))


@pytest.mark.benchmark(group="Function UDF")
def test_user_def_reward_func(func_mode, benchmark):
    variable = [[1,2,3,4]]
    def myFunction(x,t0=0.48):
        return (x[0][0]>0).astype(float) * (x[0][2]>0).astype(float) / (np.max([x[0][1],x[0][3]]) + t0)
    U = UserDefinedFunction(custom_function=myFunction, default_variable=variable, param=variable)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, variable)
    assert np.allclose(val, 0.2232142857142857)


@pytest.mark.parametrize("dtype, expected", [ # parameter is string since compiled udf doesn't support closures as of present
                    ("SCALAR_VAR", 1.0),
                    ("VECTOR_VAR", [1,2]),
                    ("MATRIX_VAR", [[1,2],[3,4]]),
                    ("BOOL_VAR", 1.0),
                    ("TUPLE_VAR", (1, 2, 3, 4)),
                    ("SCALAR_LIT", 1.0),
                    ("VECTOR_LIT", [1,2]),
                    ("MATRIX_LIT", [[1,2],[3,4]]),
                    ("TUPLE_LIT", (1, 2, 3, 4)),
                    ])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_return(dtype, expected, func_mode, benchmark):
    if dtype == "SCALAR_VAR":
        def myFunction(variable):
            var = 1.0
            return var
    elif dtype == "VECTOR_VAR":
        def myFunction(variable):
            var = [1,2]
            return var
    elif dtype == "MATRIX_VAR":
        def myFunction(variable):
            var = [[1,2],[3,4]]
            return var
    elif dtype == "BOOL_VAR":
        def myFunction(variable):
            var = True
            return 1.0
    elif dtype == "TUPLE_VAR":
        def myFunction(variable):
            var = (1, 2, 3, 4)
            return var
    elif dtype == "SCALAR_LIT":
        def myFunction(variable):
            return 1.0
    elif dtype == "VECTOR_LIT":
        def myFunction(variable):
            return [1,2]
    elif dtype == "MATRIX_LIT":
        def myFunction(variable):
            return [[1,2],[3,4]]
    elif dtype == "TUPLE_LIT":
        def myFunction(variable):
            return (1, 2, 3, 4)

    U = UserDefinedFunction(custom_function=myFunction, default_variable=0)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, 0)
    assert np.allclose(val, expected)


@pytest.mark.parametrize("op,variable,expected", [ # parameter is string since compiled udf doesn't support closures as of present
                ("TANH", [[1, 3]], [0.76159416, 0.99505475]),
                ("EXP", [[1, 3]], [2.71828183, 20.08553692]),
                ("SQRT", [[1, 3]], [1.0, 1.7320508075688772]),
                ("SHAPE", [1, 2], [2]),
                ("SHAPE", [[1, 3]], [1, 2]),
                ("ASTYPE_FLOAT", [1], [1.0]),
                ("ASTYPE_INT", [-1.5], [-1.0]),
                ("NP_MAX", 5.0, 5.0),
                ("NP_MAX", [0.0, 0.0], 0),
                ("NP_MAX", [1.0, 2.0], 2),
                ("NP_MAX", [1.0, 2.0, float("-Inf"), float("Inf"), float("NaN")], float("NaN")),
                ("NP_MAX", [[2.0, 1.0], [6.0, 2.0]], 6),
                ("NP_MAX", [[[-2.0, -1.0], [-6.0, -2.0]],[[2.0, 1.0], [6.0, 2.0]]], 6),
                ("NP_MAX", [[float('-Inf'), 1.0], [6.0, 2.0]], 6),
                ("NP_MAX", [[float('Inf'), 1.0], [6.0, 2.0]], float('Inf')),
                ("NP_MAX", [[float('NaN'), 1.0], [6.0, 2.0]], float('NaN')),
                ("NP_MAX", [[float('-NaN'), 1.0], [6.0, 2.0]], float('-NaN')),
                ("NP_MAX", [[5.0, float('-Inf'), 1.0], [3.0, 6.0, 2.0]], 6),
                ("NP_MAX", [[5.0, float('Inf'), 1.0], [3.0, 6.0, 2.0]], float('Inf')),
                ("NP_MAX", [[5.0, float('NaN'), 1.0], [3.0, 6.0, 2.0]], float('NaN')),
                ("NP_MAX", [[5.0, float('NaN'), 1.0], [3.0, 6.0, 2.0]], float('-NaN')),
                ("FLATTEN", [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0, 4.0])
                ])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_numpy(op, variable, expected, func_mode, benchmark):
    if op == "TANH":
        def myFunction(variable):
            return np.tanh(variable)
    elif op == "EXP":
        def myFunction(variable):
            return np.exp(variable)
    elif op == "SQRT":
        def myFunction(variable):
            return np.sqrt(variable)
    elif op == "SHAPE":
        def myFunction(variable):
            return variable.shape
    elif op == "ASTYPE_FLOAT":
        def myFunction(variable):
            return variable.astype(float)
    elif op == "ASTYPE_INT":
        # return types cannot be integers, so we cast back to float and check for truncation
        def myFunction(variable):
            return variable.astype(int).astype(float)
    elif op == "NP_MAX":
        def myFunction(variable):
            return np.max(variable)
    elif op == "FLATTEN":
        def myFunction(variable):
            return variable.flatten()

    U = UserDefinedFunction(custom_function=myFunction, default_variable=variable)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, variable)
    assert np.allclose(val, expected, equal_nan=True)


@pytest.mark.benchmark(group="UDF in Mechanism")
def test_udf_in_mechanism(mech_mode, benchmark):
    def myFunction(variable, param1, param2):
        return sum(variable[0]) + 2

    myMech = ProcessingMechanism(function=myFunction, size=4, name='myMech')
    # assert 'param1' in myMech.parameter_ports.names # <- FIX reinstate when problem with function params is fixed
    # assert 'param2' in myMech.parameter_ports.names # <- FIX reinstate when problem with function params is fixed
    e = pytest.helpers.get_mech_execution(myMech, mech_mode)

    val = benchmark(e, [-1, 2, 3, 4])
    assert np.allclose(val, [[10]])


@pytest.mark.parametrize("op,variable,expected", [ # parameter is string since compiled udf doesn't support closures as of present
                ("SUM", [1.0, 3.0], 12),
                ("SUM", [[1.0], [3.0]], [12.0]),
                ("SUM", [[1.0, 2.0], [3.0, 4.0]], [12.0, 18.0]),
                ("LEN", [1.0, 3.0], 8),
                ("LEN", [[1.0], [3.0]], 8),
                ("MAX_MULTI", [1.0, 3.0, 2.0], 6),
                ("MAX_TUPLE", [1.0, 3.0, 2.0], 6),
                ("MAX", [1.0, 3.0, 2.0], 3.0),
                ("MAX", [1.0, float("Inf"), 2.0], float("Inf")),
                ("MAX", [1.0, float("NaN"), 2.0], 2.0),
                ("MAX", [1.0, float("-Inf"), 2.0], 2.0),
                ])
@pytest.mark.benchmark(group="Function UDF")
def test_user_def_func_builtin(op, variable, expected, func_mode, benchmark):
    if op == "SUM":
        def myFunction(var):
            return sum(var) + sum((var[0], var[1])) + sum([var[0], var[1]])
    elif op == "LEN":
        def myFunction(var):
            return len(var) + len((var[0], var[1])) + len([var[0], var[1]]) + len((1.0, (1,2)))
    elif op == "MAX":
        def myFunction(variable):
            return max(variable)
    elif op == "MAX_MULTI":
        # special cased, since passing in multiple variables without a closure is hard
        def myFunction(variable):
            return max(variable[0], variable[1], variable[2], -5, 6)
    elif op == "MAX_TUPLE":
        def myFunction(variable):
            return max((variable[0], variable[1], variable[2], -5, 6))

    U = UserDefinedFunction(custom_function=myFunction, default_variable=variable)
    e = pytest.helpers.get_func_execution(U, func_mode)

    val = benchmark(e, variable)
    assert np.allclose(val, expected)


@pytest.mark.parametrize(
    "func, args, expected",
    [
        (math.fsum, ([1, 2, 3], ), 6),
        (math.sin, (math.pi / 2, ), 1),
    ]
)
@pytest.mark.benchmark(group="Function UDF")
# incompatible with compilation currently
def test_user_def_func_builtin_direct(func, args, expected, benchmark):
    func = UserDefinedFunction(func)

    val = benchmark(func, *args)
    assert np.allclose(val, expected)

@pytest.mark.benchmark(group="UDF as Composition Origin")
def test_udf_composition_origin(comp_mode, benchmark):
    def myFunction(variable, context):
        return [variable[0][1], variable[0][0]]

    myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
    T = TransferMechanism(size=2, function=Linear)
    c = Composition(pathways=[myMech, T])
    benchmark(c.run, inputs={myMech: [[1, 3, 5]]}, execution_mode=comp_mode)
    assert np.allclose(c.results[0][0], [3, 1])


@pytest.mark.benchmark(group="UDF as Composition Terminal")
def test_udf_composition_terminal(comp_mode, benchmark):
    def myFunction(variable, context):
        return [variable[0][2], variable[0][0]]

    myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
    T2 = TransferMechanism(size=3, function=Linear)
    c2 = Composition(pathways=[[T2, myMech]])
    benchmark(c2.run, inputs={T2: [[1, 2, 3]]}, execution_mode=comp_mode)
    assert np.allclose(c2.results[0][0], [3, 1])


def test_udf_with_pnl_func():
    L = Logistic(gain=2)

    def myFunction(variable, context):
        return L(variable) + 2

    U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0, 0]])
    myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
    val1 = myMech.execute(input=[1, 2, 3])
    val2 = U.execute(variable=[[1, 2, 3]])
    assert np.allclose(val1, val2)
    assert np.allclose(val1, L([1, 2, 3]) + 2)


def test_udf_runtime_params_reset():
    def myFunction(variable, x):
        return variable + x

    U = UserDefinedFunction(custom_function=myFunction, x=0)
    assert U.function(0) == 0
    assert U.function(0, params={'x': 1}) == 1
    assert U.function(0) == 0


@pytest.mark.parametrize(
    'expression, parameters, result',
    [
        ('x + y', {'x': 2, 'y': 4}, 6),
        ('(x + y) * z', {'x': 2, 'y': 4, 'z': 2}, 12),
        ('x + f(3)', {'x': 1, 'f': lambda x: x}, 4),
        ('x + f (3)', {'x': 1, 'f': lambda x: x}, 4),
        ('np.sum([int(x), 2])', {'x': 1, 'np': np}, 3),
        (
            '(x * y) / 3 + f(z_0, z) + z0 - (x**y) * VAR',
            {'x': 2, 'y': 3, 'f': lambda a, b: a + b, 'z_0': 1, 'z0': 1, 'z': 1, 'VAR': 1},
            -3
        )
    ]
)
@pytest.mark.parametrize('explicit_udf', [True, False])
def test_expression_execution(expression, parameters, result, explicit_udf):
    if explicit_udf:
        u = UserDefinedFunction(custom_function=expression, **parameters)
    else:
        m = ProcessingMechanism(function=expression, **parameters)
        u = m.function

    for p in parameters:
        assert p in u.cust_fct_params

    assert u.execute() == result


def _function_test_integration(variable, x, y, z):
    return x * y + z


@pytest.mark.parametrize(
    'function',
    [
        (lambda variable, x, y, z: x * y + z),
        'x * y + z',
        _function_test_integration
    ]
)
def test_integration(function):
    u = UserDefinedFunction(
        custom_function=function,
        x=2,
        y=3,
        z=5,
        stateful_parameter='x'
    )

    assert u.execute() == 11
    assert u.execute() == 38


class TestUserDefFunc:
    def test_udf_creates_parameter_ports(self):
        def func(input=[[0], [0]], p=0, q=1):
            return (p + q) * input

        m = ProcessingMechanism(
            default_variable=[[0], [0]],
            function=UserDefinedFunction(func)
        )

        assert len(m.parameter_ports) == 2
        assert 'p' in m.parameter_ports.names
        assert 'q' in m.parameter_ports.names

    @pytest.fixture
    def mech_with_autogenerated_udf(self):
        def func(input=[[0], [0]], p=0, q=1):
            return (p + q) * input

        m = ProcessingMechanism(
            default_variable=[[0], [0]],
            function=func
        )

        return m

    def test_mech_autogenerated_udf_execute(self, mech_with_autogenerated_udf):
        # Test if execute is working with auto-defined udf's
        val1 = mech_with_autogenerated_udf.execute(input=[[1], [1]])
        assert np.allclose(val1, np.array([[1], [1]]))

    def test_autogenerated_udf(self, mech_with_autogenerated_udf):
        assert isinstance(mech_with_autogenerated_udf.function, UserDefinedFunction)

    def test_autogenerated_udf_creates_parameter_ports(self, mech_with_autogenerated_udf):
        assert len(mech_with_autogenerated_udf.parameter_ports) == 2
        assert 'p' in mech_with_autogenerated_udf.parameter_ports.names
        assert 'q' in mech_with_autogenerated_udf.parameter_ports.names

    def test_autogenerated_udf_creates_parameters(self, mech_with_autogenerated_udf):
        assert hasattr(mech_with_autogenerated_udf.function.parameters, 'p')
        assert hasattr(mech_with_autogenerated_udf.function.parameters, 'q')

        assert mech_with_autogenerated_udf.function.parameters.p.default_value == 0
        assert mech_with_autogenerated_udf.function.parameters.q.default_value == 1

        assert not hasattr(mech_with_autogenerated_udf.function.class_parameters, 'p')
        assert not hasattr(mech_with_autogenerated_udf.function.class_parameters, 'q')

    def test_autogenerated_udf_parameters_states_have_source(self, mech_with_autogenerated_udf):
        for p in mech_with_autogenerated_udf.parameter_ports:
            assert p.source is getattr(mech_with_autogenerated_udf.function.parameters, p.name)
