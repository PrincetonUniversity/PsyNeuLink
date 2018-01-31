import pytest
import numpy as np

from psyneulink.library.mechanisms.processing.transfer.lca import LCA
from psyneulink.components.mechanisms.processing import ProcessingMechanism
from psyneulink.components.mechanisms.processing import TransferMechanism
from psyneulink.components.functions.function import Linear, Logistic, UserDefinedFunction
from psyneulink.components.process import Process
from psyneulink.components.system import System

#

class TestUserDefFunc:

    def test_python_func(self):
        def myFunction(variable, params, context):
            return sum(variable[0]) + 2
        myMech = ProcessingMechanism(function=myFunction, size=4, name='myMech')
        val = myMech.execute(input=[-1, 2, 3, 4])
        assert np.allclose(val, [[10]])

    def test_user_def_func(self):
        def myFunction(variable, params, context):
            return variable * 2 + 3
        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0]])
        myMech = ProcessingMechanism(function=myFunction, size=2, name='myMech')
        val = myMech.execute([1, 3])

    def test_udf_system_origin(self):
        def myFunction(variable, params, context):
            return [variable[0][1], variable[0][0]]
        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        T = TransferMechanism(size=2, function=Linear)
        p = Process(pathway=[myMech, T])
        s = System(processes=[p])
        s.run(inputs = {myMech: [[1, 3, 5]]})
        assert np.allclose(s.results[0][0], [3, 1])

    def test_udf_system_terminal(self):
        def myFunction(variable, params, context):
            return [variable[0][2], variable[0][0]]
        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        T2 = TransferMechanism(size=3, function=Linear)
        p2 = Process(pathway=[T2, myMech])
        s2 = System(processes=[p2])
        s2.run(inputs = {T2: [[1, 2, 3]]})
        assert(np.allclose(s2.results[0][0], [3, 1]))

    def test_udf_with_pnl_func(self):
        L = Logistic(gain=2)

        def myFunction(variable, params, context):
            return L.function(variable) + 2

        U = UserDefinedFunction(custom_function=myFunction, default_variable=[[0, 0, 0]])
        myMech = ProcessingMechanism(function=myFunction, size=3, name='myMech')
        val1 = myMech.execute(input=[1, 2, 3])
        val2 = U.execute(variable=[[1, 2, 3]])
        assert np.allclose(val1, val2)
        assert np.allclose(val1, L.function([1, 2, 3]) + 2)