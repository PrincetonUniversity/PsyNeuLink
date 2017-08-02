import numpy as np
import pytest

from PsyNeuLink.Components.Functions.Function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, FunctionError, ExponentialDist, NormalDist
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.KWTA import KWTA, KWTAError
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferError, TransferMechanism
from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Globals.Keywords import MATRIX_KEYWORD_VALUES, RANDOM_CONNECTIVITY_MATRIX
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import REPORT_OUTPUT_PREF, VERBOSE_PREF
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Scheduling.TimeScale import TimeScale
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

class TestKWTAInputs:

    def test_kwta_empty_spec(self):
        K = KWTA()
        assert(K.value is None)
        assert(K.variable.tolist() == [[0], [0]])
        assert(K.size.tolist() == [1, 1])
        assert(K.matrix.tolist() == [[0]])

    def test_kwta_check_attrs(self):
        K = KWTA(
            name='K',
            size=3
        )
        assert(K.value is None)
        assert(K.variable.tolist() == [[0., 0., 0.], [0., 0., 0.]])
        assert(K.size.tolist() == [3, 3])
        assert(K.matrix.tolist() == [[0., -1., -1.], [-1., 0., -1.], [-1., -1., 0.]])
        assert(K.recurrent_projection.sender is K.output_state)
        assert(K.recurrent_projection.receiver is K.input_states[1])

    def test_kwta_inputs_list_of_ints(self):
        K = KWTA(
            name='K',
            default_variable=[0, 0, 0, 0]
        )
        val = K.execute([10, 12, 0, -1]).tolist()
        assert(val == [[0.9933071490757153, 0.9990889488055994, 0.0066928509242848554, 0.0024726231566347743]])
        val = K.execute([1, 2, 3, 0]).tolist()
        assert(val == [[0.3775406687981454, 0.6224593312018546, 0.8175744761936437, 0.18242552380635635]])

    def test_kwta_no_inputs(self):
        K = KWTA(
            name='K'
        )
        assert(K.variable.tolist() == [[0], [0]])
        val = K.execute([10]).tolist()
        assert(val == [[0.5]])

    def test_kwta_inputs_list_of_strings(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size = 4,
            )
            K.execute(["one", "two", "three", "four"]).tolist()
        assert("which is not supported for KWTA" in str(error_text.value))

    def test_kwta_var_list_of_strings(self):
        with pytest.raises(UtilitiesError) as error_text:
            K = KWTA(
                name='K',
                default_variable=['a', 'b', 'c', 'd'],
                time_scale=TimeScale.TIME_STEP
            )
        assert("has non-numeric entries" in str(error_text.value))

    def test_recurrent_mech_inputs_mismatched_with_default_longer(self):
        with pytest.raises(MechanismError) as error_text:
            K = KWTA(
                name='K',
                size=4
            )
            K.execute([1, 2, 3, 4, 5]).tolist()
        assert("does not match required length" in str(error_text.value))

    def test_recurrent_mech_inputs_mismatched_with_default_shorter(self):
        with pytest.raises(MechanismError) as error_text:
            K = KWTA(
                name='K',
                size=6
            )
            K.execute([1, 2, 3, 4, 5]).tolist()
        assert("does not match required length" in str(error_text.value))

class TestKWTAMatrix:

    def test_kwta_matrix_keyword_spec(self):

        for m in MATRIX_KEYWORD_VALUES:
            if m != RANDOM_CONNECTIVITY_MATRIX:
                K = KWTA(
                    name='K',
                    size=4,
                    matrix=m
                )
                val = K.execute([10, 10, 10, 10]).tolist()
                assert(val == [[.5, .5, .5, .5]])

    def test_kwta_matrix_auto_hetero_spec(self):
        K = KWTA(
            name='K',
            size=4,
            auto=3,
            hetero=2
        )
        assert(K.recurrent_projection.matrix.tolist() == [[3, 2, 2, 2], [2, 3, 2, 2], [2, 2, 3, 2], [2, 2, 2, 3]])

    def test_kwta_matrix_hetero_spec(self):
        K = KWTA(
            name='K',
            size=3,
            hetero=-.5,
        )
        assert(K.recurrent_projection.matrix.tolist() == [[0, -.5, -.5], [-.5, 0, -.5], [-.5, -.5, 0]])

    def test_kwta_matrix_auto_spec(self):
        K = KWTA(
            name='K',
            size=3,
            auto=-.5,
        )
        assert(K.recurrent_projection.matrix.tolist() == [[-.5, -1, -1], [-1, -.5, -1], [-1, -1, -.5]])

    def test_kwta_matrix_mixed_sign(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                matrix=[[-1, 2, -2, 4]] * 4
            )
        assert "Mixing positive and negative values can create non-supported inhibition vector" in str(error_text.value)

    def test_kwta_auto_hetero_mixed_sign(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                matrix=[[-1, 2, -2, 4]] * 4
            )
        assert "Mixing positive and negative values can create non-supported inhibition vector" in str(error_text.value)
