import numpy as np
import pytest

import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism

VECTOR_SIZE=4

class TestObjectiveMechanism:
    # VALID INPUTS

    @pytest.mark.mechanism
    @pytest.mark.objective_mechanism
    @pytest.mark.benchmark(group="ObjectiveMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=[pytest.mark.llvm]),
                                      pytest.param('PTX', marks=[pytest.mark.cuda, pytest.mark.skipif(not pnlvm.ptx_enabled, reason="PTX engine not enabled/available")])])
    def test_objective_mech_inputs_list_of_ints(self, benchmark, mode):

        O = ObjectiveMechanism(
            name='O',
            default_variable=[0 for i in range(VECTOR_SIZE)],
        )
        if mode == 'Python':
            val = benchmark(O.execute, [10.0 for i in range(VECTOR_SIZE)])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(O, None)
            val =  benchmark(e.execute, [10.0 for i in range(VECTOR_SIZE)])
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(O, None)
            val =  benchmark(e.cuda_execute, [10.0 for i in range(VECTOR_SIZE)])

        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
