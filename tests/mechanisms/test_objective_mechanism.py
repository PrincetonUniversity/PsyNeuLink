import numpy as np
import pytest

#from psyneulink.components.component import ComponentError
#from psyneulink.components.functions.function import FunctionError
#from psyneulink.components.functions.function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, SoftMax, UserDefinedFunction
#from psyneulink.components.functions.function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist, UniformToNormalDist
#from psyneulink.components.mechanisms.mechanism import MechanismError
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
#from psyneulink.globals.utilities import UtilitiesError
#from psyneulink.components.process import Process
#from psyneulink.components.system import System

VECTOR_SIZE=4

class TestObjectiveMechanism:
    # VALID INPUTS

    @pytest.mark.mechanism
    @pytest.mark.objective_mechanism
    @pytest.mark.benchmark(group="ObjectiveMechanism")
    @pytest.mark.parametrize("mode", ['Python', 'LLVM'])
    def test_objective_mech_inputs_list_of_ints(self, benchmark, mode):

        O = ObjectiveMechanism(
            name='O',
            default_variable=[0 for i in range(VECTOR_SIZE)],
        )
        val = benchmark(O.execute, [10.0 for i in range(VECTOR_SIZE)], bin_execute=(mode == 'LLVM'))
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
