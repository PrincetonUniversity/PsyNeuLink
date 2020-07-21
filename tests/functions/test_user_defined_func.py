import numpy as np
import pytest

from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.processing import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing import TransferMechanism
from psyneulink.core.compositions.composition import Composition

import psyneulink.core.llvm as pnlvm

class TestUserDefFunc:

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             #pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="Function UDF")
    def test_user_def_func(self, bin_execute, benchmark):
        def myFunction(variable, param1, param2):
            return variable * 2 + param2

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]], param2=3)
        if (bin_execute == 'LLVM'):
            e = pnlvm.execution.FuncExecution(U).execute
        else:
            e = U
        val = benchmark(e, [1, 3])
        assert np.allclose(val, [[5, 9]])

    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             #pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="UDF in Mechanism")
    def test_udf_in_mechanism(self, bin_execute, benchmark):
        def myFunction(variable, param1, param2):
            return sum(variable[0]) + 2

        myMech = ProcessingMechanism(function=myFunction, size=4, name='myMech')
        # assert 'param1' in myMech.parameter_ports.names # <- FIX reinstate when problem with function params is fixed
        # assert 'param2' in myMech.parameter_ports.names # <- FIX reinstate when problem with function params is fixed
        if (bin_execute == 'LLVM'):
            e = pnlvm.execution.MechExecution(myMech).execute
        else:
            e = myMech.execute
        val = benchmark(e, [-1, 2, 3, 4])
        assert np.allclose(val, [[10]])


    @pytest.mark.parametrize("bin_execute", ['Python',
                                             pytest.param('LLVM', marks=pytest.mark.llvm),
                                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                             #pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="UDF as Composition Origin")
    def test_udf_composition_origin(self, bin_execute, benchmark):
        def myFunction(variable, params, context):
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
                                             #pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    @pytest.mark.benchmark(group="UDF as Composition Terminal")
    def test_udf_composition_terminal(self, bin_execute, benchmark):
        def myFunction(variable, params, context):
            return [variable[0][2], variable[0][0]]

        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        T2 = TransferMechanism(size=3, function=Linear)
        c2 = Composition(pathways=[[T2, myMech]])
        benchmark(c2.run, inputs={T2: [[1, 2, 3]]}, bin_execute=bin_execute)
        assert(np.allclose(c2.results[0][0], [3, 1]))

    def test_udf_with_pnl_func(self):
        L = Logistic(gain=2)

        def myFunction(variable, params, context):
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
            assert p.source is mech_with_autogenerated_udf.function
