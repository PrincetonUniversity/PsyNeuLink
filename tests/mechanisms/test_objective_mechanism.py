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
        EX = pytest.helpers.get_mech_execution(O, mech_mode)

        val = benchmark(EX, [10.0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
