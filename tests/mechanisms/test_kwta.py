import pytest

import numpy as np

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.globals.keywords import MATRIX_KEYWORD_VALUES, RANDOM_CONNECTIVITY_MATRIX
from psyneulink.core.globals.preferences.basepreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.core.globals.parameters import ParameterError
from psyneulink.library.components.mechanisms.processing.transfer.kwtamechanism import KWTAError, KWTAMechanism

class TestKWTAInputs:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_empty_spec(self):
        K = KWTAMechanism()
        np.testing.assert_allclose(K.value, K.defaults.value)
        assert(K.defaults.variable == [[0]])
        assert(K.size == [1])
        assert(K.matrix.base == [[5]])

    def test_kwta_check_attrs(self):
        K = KWTAMechanism(
            name='K',
            size=3
        )
        np.testing.assert_allclose(K.value, K.defaults.value)
        assert(np.allclose(K.defaults.variable, [[0., 0., 0.]]))
        assert(K.size == [3])
        assert(np.allclose(K.matrix.base, [[5, 0, 0], [0, 5, 0], [0, 0, 5]]))
        assert(K.recurrent_projection.sender is K.output_port)
        assert(K.recurrent_projection.receiver is K.input_port)

    def test_kwta_inputs_list_of_ints(self):
        K = KWTAMechanism(
            name='K',
            default_variable=[0, 0, 0, 0]
        )
        val = K.execute([10, 12, 0, -1])
        assert(np.allclose(val, [[0.9933071490757153, 0.9990889488055994, 0.0066928509242848554, 0.0024726231566347743]]))
        val = K.execute([1, 2, 3, 0])
        assert(np.allclose(val, [[0.3775406687981454, 0.6224593312018546, 0.8175744761936437, 0.18242552380635635]]))

    def test_kwta_no_inputs(self):
        K = KWTAMechanism(
            name='K'
        )
        assert(K.defaults.variable == [[0]])
        val = K.execute([10])
        assert(np.allclose(val, [[0.5]]))

    def test_kwta_inputs_list_of_strings(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size = 4,
            )
            K.execute(["one", "two", "three", "four"])
        assert("which is not supported for KWTA" in str(error_text.value))

    def test_kwta_var_list_of_strings(self):
        with pytest.raises(ParameterError) as error_text:
            K = KWTAMechanism(
                name='K',
                default_variable=['a', 'b', 'c', 'd'],
                integrator_mode=True
            )
        assert("non-numeric entries" in str(error_text.value))

    def test_recurrent_mech_inputs_mismatched_with_default_longer(self):
        with pytest.raises(MechanismError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4
            )
            K.execute([1, 2, 3, 4, 5])
        assert("does not match required length" in str(error_text.value))

    def test_recurrent_mech_inputs_mismatched_with_default_shorter(self):
        with pytest.raises(MechanismError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=6
            )
            K.execute([1, 2, 3, 4, 5])
        assert("does not match required length" in str(error_text.value))


class TestKWTAFunction:

    def test_kwta_function_various_spec(self):
        specs = [Logistic, Linear, Linear(slope=3), Logistic(gain=2, offset=-4.2)]
        for s in specs:
            K = KWTAMechanism(
                name='K',
                size=5,
                function=s,
                k_value=4
            )
            K.execute([1, 2, 5, -2, .3])

    def test_kwta_log_gain(self):
        K = KWTAMechanism(
            name='K',
            size=3,
            function=Logistic(gain=2),
            k_value=2
        )
        val = K.execute(input = [1, 2, 3])
        assert np.allclose(val, [[0.2689414213699951, 0.7310585786300049, 0.9525741268224334]])

    def test_kwta_log_offset(self):
        K = KWTAMechanism(
            name='K',
            size=3,
            function=Logistic(offset=-.2),
            k_value=2
        )
        val = K.execute(input=[1, 2, 3])
        assert np.allclose(val, [[0.425557483188341, 0.6681877721681662, 0.84553473491646523]])

    # the inhibition would have to be positive in order to get the desired activity level: thus, inhibition is set to 0
    def test_kwta_log_gain_offset(self):
        K = KWTAMechanism(
            name='K',
            size=2,
            function=Logistic(gain=-.2, offset=4),
            k_value=1
        )
        val = K.execute(input = [.1, -4])
        assert np.allclose(val, [[0.017636340339722684, 0.039165722796764356]])

    def test_kwta_linear(self): # inhibition would be positive: so instead it is set to zero
        K = KWTAMechanism(
            name='K',
            threshold=3,
            size=3,
            k_value=2,
            function=Linear
        )
        val = K.execute(input=[1, 3, 4])
        assert np.allclose(val, [[1, 3, 4]])

    def test_kwta_linear_slope(self):
        K = KWTAMechanism(
            name='K',
            threshold=.5,
            size=5,
            k_value=2,
            function=Linear(slope=2)
        )
        val = K.execute(input=[1, 3, 4, 2, 1])
        assert np.allclose(val, [[-2, 2, 4, 0, -2]])

    def test_kwta_linear_system(self):
        K=KWTAMechanism(
            name='K',
            size=4,
            k_value=3,
            function=Linear
        )

class TestKWTAMatrix:

    def test_kwta_matrix_keyword_spec(self):

        for m in MATRIX_KEYWORD_VALUES:
            if m != RANDOM_CONNECTIVITY_MATRIX:
                K = KWTAMechanism(
                    name='K',
                    size=4,
                    matrix=m
                )
                val = K.execute([10, 10, 10, 10])
                assert(np.allclose(val, [[.5, .5, .5, .5]]))

    def test_kwta_matrix_auto_hetero_spec(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            auto=3,
            hetero=2
        )
        assert(np.allclose(K.recurrent_projection.matrix.base, [[3, 2, 2, 2], [2, 3, 2, 2], [2, 2, 3, 2], [2, 2, 2, 3]]))

    def test_kwta_matrix_hetero_spec(self):
        K = KWTAMechanism(
            name='K',
            size=3,
            hetero=-.5,
        )
        assert(np.allclose(K.recurrent_projection.matrix.base, [[5, -.5, -.5], [-.5, 5, -.5], [-.5, -.5, 5]]))

    def test_kwta_matrix_auto_spec(self):
        K = KWTAMechanism(
            name='K',
            size=3,
            auto=-.5,
        )
        assert(np.allclose(K.recurrent_projection.matrix.base, [[-.5, 0, 0], [0, -.5, 0], [0, 0, -.5]]))


class TestKWTARatio:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_ratio_empty(self):
        K = KWTAMechanism(
            name='K',
            size=4
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs = {K: [2, 4, 1, 6]})
        assert np.allclose(K.parameters.value.get(c), [[0.2689414213699951, 0.7310585786300049,
                                                        0.11920292202211755, 0.9525741268224334]])
        c.run(inputs = {K: [1, 2, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.09271329298112314, 0.7368459299092773,
                                                        0.2631540700907225, 0.9842837170829899]])


    def test_kwta_ratio_1(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            ratio=1
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs = {K: [2, 4, 1, 6]})
        assert np.allclose(K.parameters.value.get(c), [[0.5, 0.8807970779778823,
                                                        0.2689414213699951, 0.9820137900379085]])
        c.run(inputs = {K: [1, 2, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.30054433998850033, 0.8868817857039745,
                                                        0.5, 0.9897010588046231]])


    def test_kwta_ratio_0(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            ratio=0
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs = {K: [2, 4, 1, 6]})
        assert np.allclose(K.parameters.value.get(c), [[0.11920292202211755, 0.5,
                                                        0.04742587317756678, 0.8807970779778823]])
        c.run(inputs = {K: [1, 2, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.051956902301427035, 0.5,
                                                        0.22048012438199008, 0.9802370486903237]])

    # answers for this tests should be exactly 70% of the way between the answers for ratio=0 and ratio=1
    # (after taking the inverse of the Logistic function on the output)
    def test_kwta_ratio_0_3(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            ratio=0.3
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs={K: [2, 4, 1, 6]})
        assert np.allclose(K.parameters.value.get(c), [[0.19781611144141834, 0.6456563062257956,
                                                        0.08317269649392241, 0.9308615796566533]])
        c.run(inputs={K: [1, 2, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.06324086143390241, 0.6326786177649943,
                                                        0.21948113371757957, 0.9814716617176014]])


    def test_kwta_ratio_2(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4,
                ratio=2
            )
        assert "must be between 0 and 1" in str(error_text.value)

    def test_kwta_ratio_neg_1(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4,
                ratio=-1
            )
        assert "must be between 0 and 1" in str(error_text.value)


class TestKWTAKValue:
    def test_kwta_k_value_empty_size_4(self):
        K = KWTAMechanism(
            name='K',
            size=4
        )
        assert K.k_value.base == 0.5
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs={K: [1, 2, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.18242552380635635, 0.3775406687981454,
                                                        0.6224593312018546, 0.8175744761936437]])


    def test_kwta_k_value_empty_size_6(self):
        K = KWTAMechanism(
            name='K',
            size=6
        )
        assert K.k_value.base == 0.5
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs={K: [1, 2, 2, 3, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.18242552380635635, 0.3775406687981454,
                                                        0.3775406687981454, 0.6224593312018546,
                                                        0.6224593312018546, 0.8175744761936437]])


    def test_kwta_k_value_int_size_5(self):
        K = KWTAMechanism(
            name='K',
            size=5,
            k_value=3
        )
        assert K.k_value.base == 3

    # This is a deprecated test used when the int_k optimization was being used. It's no longer useful since int_k is
    # dynamically calculated as of 8/9/17 -CW
    # def test_kwta_k_value_float_0_4(self):
    #     size_and_int_k_pairs = [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 2), (7, 3)]
    #     for size_val, expected_int_k in size_and_int_k_pairs:
    #         K = KWTA(
    #             name='K',
    #             size=size_val,
    #             k_value=0.4
    #         )
    #         assert K.k_value.base == 0.4
    #         assert K.int_k == expected_int_k
    #         p = Process(pathway=[K], prefs=TestKWTARatio.simple_prefs)
    #         s = System(processes=[p], prefs=TestKWTARatio.simple_prefs)
    #         input1 = list(range(size_val))
    #         s.run(inputs={K: input1})

    def test_kwta_k_value_bad_float(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4,
                k_value=2.5
            )
        assert "must be an integer, or between 0 and 1." in str(error_text.value)

    def test_kwta_k_value_list(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4,
                k_value=[1, 2]
            )
        assert "must be a single number" in str(error_text.value)

    def test_kwta_k_value_too_large(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4,
                k_value=5
            )
        assert "was larger than the total number of elements" in str(error_text.value)

    def test_kwta_k_value_too_low(self):
        with pytest.raises(KWTAError) as error_text:
            K = KWTAMechanism(
                name='K',
                size=4,
                k_value=-5
            )
        assert "was larger than the total number of elements" in str(error_text.value)


class TestKWTAThreshold:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_threshold_empty(self):
        K = KWTAMechanism(
            name='K',
            size=4
        )
        assert K.threshold.base == 0

    def test_kwta_threshold_int(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            threshold=-1
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs={K: [1, 2, 3, 4]})
        assert np.allclose(K.parameters.value.get(c), [[0.07585818002124355, 0.18242552380635635,
                                                        0.3775406687981454, 0.6224593312018546]])

    def test_kwta_threshold_float(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            threshold=0.5
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)

        c.run(inputs={K: [1, 2, 3, 3]})
        assert np.allclose(K.parameters.value.get(c), [[0.2689414213699951, 0.5,
                                                        0.7310585786300049, 0.7310585786300049]])


class TestKWTAControl:
    pass


class TestKWTALongTerm:

    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_size_10_k_3_threshold_1(self):
        K = KWTAMechanism(
            name='K',
            size=10,
            k_value=3,
            threshold=1,
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)
        kwta_input = {K: [[-1, -.5, 0, 0, 0, 1, 1, 2, 3, 3]]}
        print("")
        for i in range(20):
            c.run(inputs=kwta_input)
            print('\ntrial number', i)
            print('K.parameters.value.get(c): ', K.parameters.value.get(c))
        assert np.allclose(K.parameters.value.get(c), [[0.012938850123312412, 0.022127587008877226,
                                                        0.039010157367582114, 0.039010157367582114,
                                                        0.039010157367582114, 0.19055156271846602,
                                                        0.19055156271846602, 0.969124504436019,
                                                        0.9895271824560731, 0.9895271824560731]])
        kwta_input2 = {K: [0] * 10}

        print('\n\nturning to zero-inputs now:')
        for i in range(20):
            c.run(inputs=kwta_input2)
            print('\ntrial number', i)
            print('K.parameters.value.get(c): ', K.parameters.value.get(c))
        assert np.allclose(K.parameters.value.get(c), [[0.13127237999481228, 0.13130057846907178,
                                                        0.1313653354768465, 0.1313653354768465,
                                                        0.1313653354768465, 0.5863768938723602,
                                                        0.5863768938723602, 0.8390251365605804,
                                                        0.8390251603214743, 0.8390251603214743]])

class TestKWTAAverageBased:

    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_kwta_average_k_2(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            k_value=2,
            threshold=0,
            function=Linear,
            average_based=True
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)
        kwta_input = {K: [[1, 2, 3, 4]]}
        c.run(inputs=kwta_input)
        assert np.allclose(K.parameters.value.get(c), [[-1.5, -0.5, 0.5, 1.5]])

    def test_kwta_average_k_1(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            k_value=1,
            threshold=0,
            function=Linear,
            average_based=True
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)
        kwta_input = {K: [[1, 2, 3, 4]]}
        c.run(inputs=kwta_input)
        assert np.allclose(K.parameters.value.get(c), [[-2, -1, 0, 1]])

    def test_kwta_average_k_1_ratio_0_2(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            k_value=1,
            threshold=0,
            ratio=0.2,
            function=Linear,
            average_based=True
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)
        kwta_input = {K: [[1, 2, 3, 4]]}
        c.run(inputs=kwta_input)
        assert np.allclose(K.parameters.value.get(c), [[-2.6, -1.6, -0.6000000000000001, 0.3999999999999999]])

    def test_kwta_average_k_1_ratio_0_8(self):
        K = KWTAMechanism(
            name='K',
            size=4,
            k_value=1,
            threshold=0,
            ratio=0.8,
            function=Linear,
            average_based=True
        )
        c = Composition(pathways=[K],
                        prefs=TestKWTARatio.simple_prefs)
        kwta_input = {K: [[1, 2, 3, 4]]}
        c.run(inputs=kwta_input)
        assert np.allclose(K.parameters.value.get(c), [[-1.4, -0.3999999999999999, 0.6000000000000001, 1.6]])

# class TestClip:
#     def test_clip_float(self):
#         K = KWTA(clip=[-2.0, 2.0],
#                  integrator_mode=False)
#         assert np.allclose(K.execute(3.0), 2.0)
#         assert np.allclose(K.execute(-3.0), -2.0)
#
#     def test_clip_array(self):
#         K = KWTA(default_variable=[[0.0, 0.0, 0.0]],
#                  clip=[-2.0, 2.0],
#                  integrator_mode=False)
#         assert np.allclose(K.execute([3.0, 0.0, -3.0]), [2.0, 0.0, -2.0])
#
#     def test_clip_2d_array(self):
#         K = KWTA(default_variable=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#                  clip=[-2.0, 2.0],
#                  integrator_mode=False)
#         assert np.allclose(K.execute([[-5.0, -1.0, 5.0], [5.0, -5.0, 1.0], [1.0, 5.0, 5.0]]),
#                            [[-2.0, -1.0, 2.0], [2.0, -2.0, 1.0], [1.0, 2.0, 2.0]])
