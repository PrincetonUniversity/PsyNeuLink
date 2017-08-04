import numpy as np
import pytest
import typecheck

from PsyNeuLink.Components.Component import ComponentError
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.KWTA import KWTA, KWTAError
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferError, TransferMechanism
from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Globals.Keywords import MATRIX_KEYWORD_VALUES, RANDOM_CONNECTIVITY_MATRIX
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import REPORT_OUTPUT_PREF, VERBOSE_PREF
from PsyNeuLink.Globals.Run import RunError
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Scheduling.TimeScale import TimeScale
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection


class TestKWTAInputs:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

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

    def test_kwta_naive_input(self):
        with pytest.raises(RunError) as error_text:
            K = KWTA(
                name='K',
                size=4
            )
            p = process(pathway=[K], prefs=TestKWTAInputs.simple_prefs)
            s = system(processes=[p], prefs=TestKWTAInputs.simple_prefs)
            s.run(inputs={K: [[np.array([2, 4, 1, 6])]]})
            # (correct input should be [[np.array([2, 4, 1, 6]), [0, 0, 0, 0]]]
        assert "For KWTA mechanisms, remember to append an array of zeros" in str(error_text.value)


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
                auto=3,
                hetero=-2.2
            )
        assert "Mixing positive and negative values can create non-supported inhibition vector" in str(error_text.value)

    def test_kwta_matrix_hetero_mixed_sign(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=2,
                matrix = [[1, 3], [1, 3]],
                hetero=-2.2
            )
        assert "Mixing positive and negative values can create non-supported inhibition vector" in str(error_text.value)


class TestKWTAFunction:

    def test_kwta_gain(self):
        K = KWTA(
            name='K',
            size=3,
            gain=2
        )
        val = K.execute(input = [1, 2, 3]).tolist()
        assert val == [[0.2689414213699951, 0.7310585786300049, 0.9525741268224334]]

    def test_kwta_bias(self):
        K = KWTA(
            name='K',
            size=3,
            bias = -.2
        )
        val = K.execute(input=[1, 2, 3]).tolist()
        assert val == [[0.425557483188341, 0.6681877721681662, 0.84553473491646523]]

    # the inhibition would have to be positive in order to get the desired activity level: thus, inhibition is set to 0
    def test_kwta_gain_bias(self):
        K = KWTA(
            name='K',
            size=2,
            gain=-.2,
            bias=4
        )
        val = K.execute(input = [.1, -4]).tolist()
        assert val == [[0.017636340339722684, 0.039165722796764356]]

    def test_kwta_gain_string(self):
        with pytest.raises(typecheck.framework.InputParameterError) as error_text:
            K = KWTA(
                name='K',
                size=3,
                gain='some string'
            )
        assert "has got an incompatible value for gain: some string" in str(error_text.value)


class TestKWTARatio:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_ratio_empty(self):
        K = KWTA(
            name='K',
            size=4
        )
        p = process(pathway = [K], prefs = TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs = TestKWTARatio.simple_prefs)
        s.run(inputs = {K: [[[2, 4, 1, 6], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.2689414213699951, 0.7310585786300049, 0.11920292202211755, 0.9525741268224334]]
        s.run(inputs = {K: [[[1, 2, 3, 4], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[.1506327210799811, .49250998958935144, .5109072669826713, .9093258490277805]]

    def test_kwta_ratio_1(self):
        K = KWTA(
            name='K',
            size=4,
            ratio=1
        )
        p = process(pathway = [K], prefs = TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs = TestKWTARatio.simple_prefs)
        s.run(inputs = {K: [[[2, 4, 1, 6], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.5, 0.8807970779778823, 0.2689414213699951, 0.9820137900379085]]
        s.run(inputs = {K: [[[1, 2, 3, 4], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[.19232715234771303, .5, .57471576588477, .8924114570116725]]

    def test_kwta_ratio_0(self):
        K = KWTA(
            name='K',
            size=4,
            ratio=0
        )
        p = process(pathway = [K], prefs = TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs = TestKWTARatio.simple_prefs)
        s.run(inputs = {K: [[[2, 4, 1, 6], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.11920292202211755, 0.5, 0.04742587317756678, 0.8807970779778823]]
        s.run(inputs = {K: [[[1, 2, 3, 4], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[.1351178490620377, .4763048259160004, .5, .935035431964823]]

    # answers for this tests should be exactly 60% of the way between the answers for ratio=0 and ratio=1
    def test_kwta_ratio_0_3(self):
        K = KWTA(
            name='K',
            size=4,
            ratio=0.3
        )
        p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
        s.run(inputs={K: [[[2, 4, 1, 6], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.19781611144141825, 0.6456563062257954, 0.08317269649392238, 0.9308615796566533]]
        s.run(inputs={K: [[[1, 2, 3, 4], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[.14355829498333966, .49143167747317057, .5053768901517178, .9201571487736694]]

    def test_kwta_ratio_2(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                ratio=2
            )
        assert "must be between 0 and 1" in str(error_text.value)

    def test_kwta_ratio_neg_1(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                ratio=-1
            )
        assert "must be between 0 and 1" in str(error_text.value)


class TestKWTAKValue:
    def test_kwta_k_value_empty_size_4(self):
        K = KWTA(
            name='K',
            size=4
        )
        assert K.k_value == 0.5
        assert K.int_k == 2
        p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
        s.run(inputs={K: [[[1, 2, 3, 4], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.18242552380635635, 0.3775406687981454, 0.6224593312018546, 0.8175744761936437]]

    def test_kwta_k_value_empty_size_6(self):
        K = KWTA(
            name='K',
            size=6
        )
        assert K.k_value == 0.5
        assert K.int_k == 3
        p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
        s.run(inputs={K: [[[1, 2, 2, 3, 3, 4], [0, 0, 0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.18242552380635635, 0.3775406687981454, 0.3775406687981454,
                                     0.6224593312018546, 0.6224593312018546, 0.8175744761936437]]

    def test_kwta_k_value_int_size_5(self):
        K = KWTA(
            name='K',
            size=5,
            k_value=3
        )
        assert K.k_value == 3
        assert K.int_k == 3

    def test_kwta_k_value_float_0_4(self):
        size_and_int_k_pairs = [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 2), (7, 3)]
        for size_val, expected_int_k in size_and_int_k_pairs:
            K = KWTA(
                name='K',
                size=size_val,
                k_value=0.4
            )
            assert K.k_value == 0.4
            assert K.int_k == expected_int_k
            p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
            s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
            input1 = range(size_val)
            input2 = [0] * size_val
            s.run(inputs={K: [[input1, input2]]})

    def test_kwta_k_value_bad_float(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                k_value=2.5
            )
        assert "must be an integer, or between 0 and 1." in str(error_text.value)

    def test_kwta_k_value_list(self):
        with pytest.raises(ComponentError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                k_value=[1, 2]
            )
        assert "must be compatible with float" in str(error_text.value)

    def test_kwta_k_value_too_large(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                k_value=5
            )
        assert "was larger than the total number of elements" in str(error_text.value)

    def test_kwta_k_value_too_low(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTA(
                name='K',
                size=4,
                k_value=-5
            )
        assert "was larger than the total number of elements" in str(error_text.value)


class TestKWTAThreshold:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_threshold_empty(self):
        K = KWTA(
            name='K',
            size=4
        )
        assert K.threshold == 0

    def test_kwta_threshold_int(self):
        K = KWTA(
            name='K',
            size=4,
            threshold=-1
        )
        p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
        s.run(inputs={K: [[[1, 2, 3, 4], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.07585818002124359, 0.1824255238063564, 0.3775406687981456, 0.6224593312018547]]

    def test_kwta_threshold_float(self):
        K = KWTA(
            name='K',
            size=4,
            threshold=0.5
        )
        p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
        s.run(inputs={K: [[[1, 2, 3, 3], [0, 0, 0, 0]]]})
        assert K.value.tolist() == [[0.2689414213699951, 0.5, 0.7310585786300049, 0.7310585786300049]]


class TestKWTAControl:
    pass


class TestKWTALongTerm:

    def test_kwta_size_10_k_3_threshold_1(self):
        K = KWTA(
            name='K',
            size=10,
            k_value=3,
            threshold=1,
            decay=0.3,
            time_scale=TimeScale.TIME_STEP
        )
        p = process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
        s = system(processes=[p], prefs=TestKWTARatio.simple_prefs)
        kwta_input = {K: [[[-1, -.5, 0, 0, 0, 1, 1, 2, 3, 3], ([0] * 10)]]}
        print("")
        for i in range(20):
            s.run(inputs=kwta_input)
            print('\ntrial number', i)
            print('K.value: ', K.value)
        kwta_input2 = {K: [[([0] * 10), ([0] * 10)]]}
        print('turning to zero-inputs now:')
        for i in range(20):
            s.run(inputs=kwta_input2)
            print('\ntrial number', i)
            print('K.value: ', K.value)
