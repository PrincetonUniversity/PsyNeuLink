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
    def test_objective_mech_inputs_list_of_ints(self, benchmark, mech_mode):

        O = ObjectiveMechanism(
            name='O',
            default_variable=[0 for i in range(VECTOR_SIZE)],
        )
        if mech_mode == 'Python':
            EX = O.execute
        elif mech_mode == 'LLVM':
            e = pnlvm.execution.MechExecution(O)
            EX = e.execute
        elif mech_mode == 'PTX':
            e = pnlvm.execution.MechExecution(O)
            EX = e.cuda_execute

        val = benchmark(EX, [10.0 for i in range(VECTOR_SIZE)])

        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
