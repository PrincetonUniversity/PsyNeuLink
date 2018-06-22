import numpy as np
import pytest

from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism

VECTOR_SIZE=4

class TestObjectiveMechanism:
    # VALID INPUTS

    @pytest.mark.mechanism
    @pytest.mark.objective_mechanism
    @pytest.mark.benchmark(group="ObjectiveMechanism")
    @pytest.mark.parametrize("mode", ['Python'])
    def test_objective_mech_inputs_list_of_ints(self, benchmark, mode):

        O = ObjectiveMechanism(
            name='O',
            default_variable=[0 for i in range(VECTOR_SIZE)],
        )
        val = benchmark(O.execute, [10.0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
