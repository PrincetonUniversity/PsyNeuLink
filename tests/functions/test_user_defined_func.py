import numpy as np
import pytest

from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.processing import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing import TransferMechanism
from psyneulink.core.compositions.composition import Composition

import psyneulink.core.llvm as pnlvm

class TestBinaryOperations:
    @pytest.mark.parametrize("param1, param2", [
                        (1, 2), # scalar - scalar
                        (np.ones(2), 2), # vec - scalar
                        (2, np.ones(2)), # scalar - vec
                        (np.ones((2, 2)), 2), # mat - scalar
                        (2, np.ones((2, 2))), # scalar - mat
                        (np.ones(2), np.array([1, 2])), # vec - vec
                        (np.ones((2, 2)), np.array([[1, 2], [3, 4]])), # mat - mat
                        ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_add(self, param1, param2, bin_execute, benchmark):
        # default val is same shape as expected output
        def myFunction(_, param1, param2):
            # we only use param1 and param2 to avoid automatic shape changes of the variable
            return param1 + param2

        U = UserDefinedFunction(custom_function=myFunction, param1=param1, param2=param2)
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, 0)
        assert np.allclose(val, param1 + param2)

    @pytest.mark.parametrize("param1, param2", [
                        (1, 2), # scalar - scalar
                        (np.ones(2), 2), # vec - scalar
                        (2, np.ones(2)), # scalar - vec
                        (np.ones((2, 2)), 2), # mat - scalar
                        (2, np.ones((2, 2))), # scalar - mat
                        (np.ones(2), np.array([1, 2])), # vec - vec
                        (np.ones((2, 2)), np.array([[1, 2], [3, 4]])), # mat - mat
                        ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_mul(self, param1, param2, bin_execute, benchmark):
        # default val is same shape as expected output
        def myFunction(_, param1, param2):
            # we only use param1 and param2 to avoid automatic shape changes of the variable
            return param1 * param2

        U = UserDefinedFunction(custom_function=myFunction, param1=param1, param2=param2)
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, 0)
        assert np.allclose(val, param1 * param2)

    @pytest.mark.parametrize("op", [ # parameter is string since compiled udf doesn't support closures as of present
                        "AND",
                        "OR",
                        ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_boolop(self, op, bin_execute, benchmark):
        if op == "AND":
            def myFunction(variable):
                var1 = True
                var2 = False
                # compiled UDFs don't support python bool type outputs
                if var1 and var2:
                    return 0.0
                else:
                    return 1.0
        elif op == "OR":
            def myFunction(variable):
                var1 = True
                var2 = False
                # compiled UDFs don't support python bool type outputs
                if var1 or var2:
                    return 1.0
                else:
                    return 0.0

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[0])
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, [0])
        assert val == 1.0

    @pytest.mark.parametrize("op", [ # parameter is string since compiled udf doesn't support closures as of present
                        "Eq",
                        "NotEq",
                        "Lt",
                        "LtE",
                        "Gt",
                        "GtE",
                        ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_cmpop(self, op, bin_execute, benchmark):
        if op == "Eq":
            def myFunction(variable):
                var1 = 1.0
                var2 = 1.0
                if var1 == var2:
                    return 1.0
                else:
                    return 0.0
        elif op == "NotEq":
            def myFunction(variable):
                var1 = 1.0
                var2 = 2.0
                if var1 != var2:
                    return 1.0
                else:
                    return 0.0
        elif op == "Lt":
            def myFunction(variable):
                var1 = 1.0
                var2 = 2.0
                if var1 < var2:
                    return 1.0
                else:
                    return 0.0
        elif op == "LtE":
            def myFunction(variable):
                var1 = 1.0
                var2 = 2.0
                var3 = 1.0
                if var1 <= var2 and var1 <= var3:
                    return 1.0
                else:
                    return 0.0
        elif op == "Gt":
            def myFunction(variable):
                var1 = 2.0
                var2 = 1.0
                if var1 > var2:
                    return 1.0
                else:
                    return 0.0
        elif op == "GtE":
            def myFunction(variable):
                var1 = 3.0
                var2 = 2.0
                var3 = 3.0
                if var1 >= var2 and var1 >= var3:
                    return 1.0
                else:
                    return 0.0

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[0])
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, [0])
        assert val == 1.0

class TestUserDefFunc:

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func(self, bin_execute, benchmark):
        def myFunction(variable, param1, param2):
            return variable * 2 + param2

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]], param2=3)
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, [1, 3])
        assert np.allclose(val, [[5, 9]])

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_branching(self, bin_execute, benchmark):
        def myFunction(variable, param1, param2):
            if variable[0][0] > 0 and variable[0][1] > 0:
                return variable * 2 + param2
            else:
                return variable * -2 + param2

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]], param2=3)
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, [[1, 3]])
        assert np.allclose(val, [[5, 9]])
        val2 = e([[-1, 3]])
        assert np.allclose(val2, [[5, -3]])

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_variable_index(self, bin_execute, benchmark):
        def myFunction(variable):
            variable[0][0] = variable[0][0] + 5
            variable[0][1] = variable[0][1] + 7
            return variable

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]])
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, [[1, 3]])
        assert np.allclose(val, [[6, 10]])

    @pytest.mark.parametrize("variable", [
                        (1), # scalar
                        (np.ones((2))), # vec-2d
                        (np.ones((2, 2))) # mat
                        ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_usub(self, variable, bin_execute, benchmark):
        def myFunction(variable, param):
            return -param

        U = UserDefinedFunction(custom_function=myFunction, default_variable=variable, param=variable)
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, variable)
        assert np.allclose(val, -variable)

    @pytest.mark.parametrize("dtype, expected", [ # parameter is string since compiled udf doesn't support closures as of present
                        ("SCALAR", 1.0),
                        ("VECTOR", [1,2]),
                        ("MATRIX", [[1,2],[3,4]]),
                        ("BOOL", 1.0),
                        ("TUPLE", (1, 2, 3, 4))
                        ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_assign(self, dtype, expected, bin_execute, benchmark):
        if dtype == "SCALAR":
            def myFunction(variable):
                var = 1.0
                return var
        elif dtype == "VECTOR":
            def myFunction(variable):
                var = [1,2]
                return var
        elif dtype == "MATRIX":
            def myFunction(variable):
                var = [[1,2],[3,4]]
                return var
        elif dtype == "BOOL":
            def myFunction(variable):
                var = True
                return 1.0
        elif dtype == "TUPLE":
            def myFunction(variable):
                var = (1, 2, 3, 4)
                return var

        U = UserDefinedFunction(custom_function=myFunction, default_variable=0)
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, 0)
        assert np.allclose(val, expected)

    @pytest.mark.parametrize("op,expected", [ # parameter is string since compiled udf doesn't support closures as of present
                    ("TANH", [0.76159416, 0.99505475]),
                    ("EXP", [2.71828183, 20.08553692]),
                    ])
    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func_numpy(self, op, expected, bin_execute, benchmark):
        variable = [[1, 3]]
        if op == "TANH":
            def myFunction(variable):
                return np.tanh(variable)
        elif op == "EXP":
            def myFunction(variable):
                return np.exp(variable)

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]])
        if bin_execute == 'LLVM':
            e = pnlvm.execution.FuncExecution(U).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.FuncExecution(U).cuda_execute
        else:
            e = U
        val = benchmark(e, variable)
        assert np.allclose(val, expected)

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="UDF in Mechanism")
    def test_udf_in_mechanism(self, bin_execute, benchmark):
        def myFunction(variable, param1, param2):
            return sum(variable[0]) + 2

        myMech = ProcessingMechanism(function=myFunction, size=4, name='myMech')
        # assert 'param1' in myMech.parameter_ports.names # <- FIX reinstate when problem with function params is fixed
        # assert 'param2' in myMech.parameter_ports.names # <- FIX reinstate when problem with function params is fixed
        if bin_execute == 'LLVM':
            e = pnlvm.execution.MechExecution(myMech).execute
        elif bin_execute == 'PTX':
            e = pnlvm.execution.MechExecution(myMech).cuda_execute
        else:
            e = myMech.execute
        val = benchmark(e, [-1, 2, 3, 4])
        assert np.allclose(val, [[10]])


    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="UDF as Composition Origin")
    def test_udf_composition_origin(self, bin_execute, benchmark):
        def myFunction(variable, context):
            return [variable[0][1], variable[0][0]]

        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        T = TransferMechanism(size=2, function=Linear)
        c = Composition(pathways=[myMech, T])
        benchmark(c.run, inputs={myMech: [[1, 3, 5]]}, bin_execute=bin_execute)
        assert np.allclose(c.results[0][0], [3, 1])

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="UDF as Composition Terminal")
    def test_udf_composition_terminal(self, bin_execute, benchmark):
        def myFunction(variable, context):
            return [variable[0][2], variable[0][0]]

        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        T2 = TransferMechanism(size=3, function=Linear)
        c2 = Composition(pathways=[[T2, myMech]])
        benchmark(c2.run, inputs={T2: [[1, 2, 3]]}, bin_execute=bin_execute)
        assert(np.allclose(c2.results[0][0], [3, 1]))

    def test_udf_with_pnl_func(self):
        L = Logistic(gain=2)

        def myFunction(variable, context):
            return L(variable) + 2

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0, 0]])
        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        val1 = myMech.execute(input=[1, 2, 3])
        val2 = U.execute(variable=[[1, 2, 3]])
        assert np.allclose(val1, val2)
        assert np.allclose(val1, L([1, 2, 3]) + 2)

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
