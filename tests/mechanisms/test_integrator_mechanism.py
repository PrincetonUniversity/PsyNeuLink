import numpy as np
import pytest

from PsyNeuLink.Components.Functions.Function import AccumulatorIntegrator, ConstantIntegrator, NormalDist, \
    SimpleIntegrator, FHNIntegrator
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, DriftDiffusionIntegrator, \
    OrnsteinUhlenbeckIntegrator
from PsyNeuLink.Components.Functions.Function import FunctionError
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism \
    import IntegratorMechanism
from PsyNeuLink.Scheduling.TimeScale import TimeScale


# ======================================= FUNCTION TESTS ============================================

class TestIntegratorFunctions:

    def test_simple_integrator(self):
        I = IntegratorMechanism(
            function=SimpleIntegrator(
                initializer=10.0,
                rate=5.0,
                offset=10,
            )
        )
        # P = process(pathway=[I])
        val = I.execute(1)
        assert val == 25

    def test_constant_integrator(self):
        I = IntegratorMechanism(
            function=ConstantIntegrator(
                initializer=10.0,
                rate=5.0,
                offset=10
            )
        )
        # P = process(pathway=[I])
        # constant integrator does not use input value (variable)

        # step 1:
        val = I.execute(20000)
        # value = 10 + 5
        # adjusted_value = 15 + 10
        # previous_value = 25
        # RETURN 25

        # step 2:
        val2 = I.execute(70000)
        # value = 25 + 5
        # adjusted_value = 30 + 10
        # previous_value = 30
        # RETURN 40
        assert (val, val2) == (25, 40)

    def test_adaptive_integrator(self):
        I = IntegratorMechanism(
            function=AdaptiveIntegrator(
                initializer=10.0,
                rate=0.5,
                offset=10,
            )
        )
        # P = process(pathway=[I])
        # 10*0.5 + 1*0.5 + 10
        val = I.execute(1)
        assert val == 15.5

    def test_drift_diffusion_integrator(self):
        I = IntegratorMechanism(
            function=DriftDiffusionIntegrator(
                initializer=10.0,
                rate=10,
                time_step_size=0.5,
                offset=10,
            )
        )
        # P = process(pathway=[I])
        # 10 + 10*0.5 + 0 + 10 = 25
        val = I.execute(1)
        assert val == 25

    def test_ornstein_uhlenbeck_integrator(self):
        I = IntegratorMechanism(
            function=OrnsteinUhlenbeckIntegrator(
                initializer=10.0,
                rate=10,
                time_step_size=0.5,
                decay=0.1,
                offset=10,
            )
        )
        # P = process(pathway=[I])
        # value = previous_value + decay * rate * new_value * time_step_size + np.sqrt(
        # time_step_size * noise) * np.random.normal()
        # step 1:
        val = I.execute(1)
        # value = 10 + 0.1*10*1*0.5 + 0
        # adjusted_value = 10.5 + 10
        # previous_value = 20.5
        # RETURN 20.5

        # step 2:
        val2 = I.execute(1)
        # value = 20.5 + 0.1*10*1*0.5 + 0
        # adjusted_value = 21 + 10
        # previous_value = 31
        # RETURN 31

        # step 3:
        val3 = I.execute(1)
        # value = 31 + 0.1*10*1*0.5 + 0
        # adjusted_value = 31.5 + 10
        # previous_value = 41.5
        # RETURN 41.5

        assert (val, val2, val3) == (20.5, 31, 41.5)

    def test_integrator_no_function(self):
        I = IntegratorMechanism(time_scale=TimeScale.TIME_STEP)
        # P = process(pathway=[I])
        val = float(I.execute(10))
        assert val == 5


class TestResetInitializer:

    def test_integrator_simple_with_reset_intializer(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
            ),
            time_scale=TimeScale.TIME_STEP
        )
    #     # P = process(pathway=[I])

        #  returns previous_value + rate*variable + noise
        # so in this case, returns 10.0
        val = float(I.execute(10))

        # testing initializer
        I.function_object.reset_initializer = 5.0

        val2 = float(I.execute(0))

        assert [val, val2] == [10.0, 5.0]

    def test_integrator_adaptive_with_reset_intializer(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=0.5
            ),
            time_scale=TimeScale.TIME_STEP
        )
        # val = float(I.execute(10)[0])
        # P = process(pathway=[I])
        val = float(I.execute(10))
        # returns (rate)*variable + (1-rate*previous_value) + noise
        # rate = 1, noise = 0, so in this case, returns 10.0

        # testing initializer
        I.function_object.reset_initializer = 1.0
        val2 = float(I.execute(1))

        assert [val, val2] == [5.0, 1.0]

    def test_integrator_constant_with_reset_intializer(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=ConstantIntegrator(
                rate=1.0
            ),
            time_scale=TimeScale.TIME_STEP
        )
        # val = float(I.execute(10)[0])
        # P = process(pathway=[I])
        val = float(I.execute())
        # returns previous_value + rate + noise
        # rate = 1.0, noise = 0, so in this case returns 1.0

        # testing initializer
        I.function_object.reset_initializer = 10.0
        val2 = float(I.execute())

        assert [val, val2] == [1.0, 11.0]

    def test_integrator_diffusion_with_reset_intializer(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=DriftDiffusionIntegrator(
            ),
            time_scale=TimeScale.TIME_STEP
        )
        # val = float(I.execute(10)[0])
        # P = process(pathway=[I])
        val = float(I.execute(10))

        # testing initializer
        I.function_object.reset_initializer = 1.0
        val2 = float(I.execute(0))

        assert [val, val2] == [10.0, 1.0]

# ======================================= INPUT TESTS ============================================


class TestIntegratorInputs:
    # Part 1: VALID INPUT:

    # input = float

    def test_integrator_input_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
            )
        )
        # P = process(pathway=[I])
        val = float(I.execute(10.0))
        assert val == 10.0

    # input = list of length 1

    def test_integrator_input_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
            )
        )
        # P = process(pathway=[I])
        val = float(I.execute([10.0]))
        assert val == 10.0

    # input = list of length 5

    def test_integrator_input_list_len_5(self):

        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0, 0],
            function=SimpleIntegrator(
            )
        )
        # P = process(pathway=[I])
        val = I.execute([10.0, 5.0, 2.0, 1.0, 0.0])[0]
        expected_output = [10.0, 5.0, 2.0, 1.0, 0.0]

        for i in range(len(expected_output)):
            v = val[i]
            e = expected_output[i]
            np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    # input = numpy array of length 5

    def test_integrator_input_array_len_5(self):

        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0, 0],
            function=SimpleIntegrator(
            )
        )
        # P = process(pathway=[I])
        input_array = np.array([10.0, 5.0, 2.0, 1.0, 0.0])
        val = I.execute(input_array)[0]
        expected_output = [10.0, 5.0, 2.0, 1.0, 0.0]

        for i in range(len(expected_output)):
            v = val[i]
            e = expected_output[i]
            np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    # Part 2: INVALID INPUT

    # input = list of length > default length

    def test_integrator_input_array_greater_than_default(self):

        with pytest.raises(MechanismError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                default_variable=[0, 0, 0]
            )
            # P = process(pathway=[I])
            I.execute([10.0, 5.0, 2.0, 1.0, 0.0])
        assert "does not match required length" in str(error_text)

    # input = list of length < default length

    def test_integrator_input_array_less_than_default(self):

        with pytest.raises(MechanismError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                default_variable=[0, 0, 0, 0, 0]
            )
            # P = process(pathway=[I])
            I.execute([10.0, 5.0, 2.0])
        assert "does not match required length" in str(error_text)


# ======================================= RATE TESTS ============================================
class TestIntegratorRate:
    # VALID RATE:

    # rate = float, integration_type = simple

    def test_integrator_type_simple_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                rate=5.0
            )
        )
        # P = process(pathway=[I])
        val = float(I.execute(10.0))
        assert val == 50.0

    # rate = float, integration_type = constant

    def test_integrator_type_constant_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=ConstantIntegrator(
                rate=5.0
            )
        )
        # P = process(pathway=[I])
        val = float(I.execute(10.0))
        assert val == 5.0

    # rate = float, integration_type = diffusion

    def test_integrator_type_diffusion_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=DriftDiffusionIntegrator(
                rate=5.0
            )
        )
        # P = process(pathway=[I])
        val = float(I.execute(10.0))
        assert val == 50.0

    # rate = list, integration_type = simple

    def test_integrator_type_simple_rate_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0],
            function=SimpleIntegrator(
                rate=[5.0, 5.0, 5.0]
            )
        )
        # P = process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [50.0, 50.0, 50.0]

    # rate = list, integration_type = constant

    def test_integrator_type_constant_rate_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=ConstantIntegrator(
                rate=[5.0, 5.0, 5.0]
            )
        )
        # P = process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [5.0, 5.0, 5.0]

    # rate = list, integration_type = diffusion

    def test_integrator_type_diffusion_rate_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=DriftDiffusionIntegrator(
                rate=[5.0, 5.0, 5.0]
            )
        )
        # P = process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [50.0, 50.0, 50.0]

    # rate = list, integration_type = diffusion

    def test_integrator_type_adaptive_rate_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=[0.5, 0.5, 0.5]
            )
        )
        # P = process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [5.0, 5.0, 5.0]

    # rate = float, integration_type = adaptive

    def test_integrator_type_adaptive_rate_float_input_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=0.5
            )
        )
        # P = process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [5.0, 5.0, 5.0]

    # rate = float, integration_type = adaptive

    def test_integrator_type_adaptive_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=0.5
            )
        )
        # P = process(pathway=[I])
        val = list(I.execute(10.0))
        assert val == [5.0]

    # INVALID RATE:

    def test_integrator_type_simple_rate_list_input_float(self):
        with pytest.raises(FunctionError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                function=SimpleIntegrator(

                    rate=[5.0, 5.0, 5.0]
                )
            )
            # P = process(pathway=[I])
            float(I.execute(10.0))
        assert (
            "array specified for the rate parameter" in str(error_text)
            and "must match the length" in str(error_text)
            and "of the default input" in str(error_text)
        )

    def test_integrator_type_constant_rate_list_input_float(self):
        with pytest.raises(FunctionError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                function=ConstantIntegrator(
                    rate=[5.0, 5.0, 5.0]
                )
            )
            # P = process(pathway=[I])
            float(I.execute(10.0))
        assert (
            "array specified for the rate parameter" in str(error_text)
            and "must match the length" in str(error_text)
            and "of the default input" in str(error_text)
        )

    def test_integrator_type_diffusion_rate_list_input_float(self):
        with pytest.raises(FunctionError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                function=DriftDiffusionIntegrator(
                    rate=[5.0, 5.0, 5.0]
                )
            )
            # P = process(pathway=[I])
            float(I.execute(10.0))
        assert (
            "array specified for the rate parameter" in str(error_text)
            and "must match the length" in str(error_text)
            and "of the default input" in str(error_text)
        )

    # def test_accumulator_integrator(self):
    #     I = IntegratorMechanism(
    #             function = AccumulatorIntegrator(
    #                 initializer = 10.0,
    #                 rate = 5.0,
    #                 increment= 1.0
    #             )
    #         )
    # #     P = process(pathway=[I])

    #     # value = previous_value * rate + noise + increment
    #     # step 1:
    #     val = I.execute()
    #     # value = 10.0 * 5.0 + 0 + 1.0
    #     # RETURN 51

    #     # step 2:
    #     val2 = I.execute(2000)
    #     # value = 51*5 + 0 + 1.0
    #     # RETURN 256
    #     assert (val, val2) == (51, 256)


class TestIntegratorNoise:

    def test_integrator_simple_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                noise=NormalDist().function
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = float(I.execute(10))

        I.function_object.reset_initializer = 5.0

        val2 = float(I.execute(0))

        np.testing.assert_allclose(val, 11.86755799)
        np.testing.assert_allclose(val2, 4.022722120123589)

    def test_integrator_simple_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=SimpleIntegrator(
                noise=NormalDist().function,
                default_variable=[0, 0, 0, 0]
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = I.execute([10, 10, 10, 10])[0]

        np.testing.assert_allclose(val, [10.12167502,  10.44386323,  10.33367433,  11.49407907])

    def test_integrator_accumulator_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(
                noise=NormalDist().function
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 1.8675579901499675)

    def test_integrator_accumulator_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=AccumulatorIntegrator(
                noise=NormalDist().function,
                default_variable=[0, 0, 0, 0]
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = I.execute([10, 10, 10, 10])[0]
        print(val)
        np.testing.assert_allclose(val, [-0.15135721, -0.10321885,  0.4105985,   0.14404357])

    def test_integrator_constant_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=ConstantIntegrator(
                noise=NormalDist().function
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 1.8675579901499675)

    def test_integrator_constant_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=ConstantIntegrator(
                noise=NormalDist().function,
                default_variable=[0, 0, 0, 0]
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = I.execute([10, 10, 10, 10])[0]

        np.testing.assert_allclose(val, [0.12167502,  0.44386323,  0.33367433,  1.49407907])

    def test_integrator_adaptive_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                noise=NormalDist().function
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 11.86755799)

    def test_integrator_adaptive_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=AdaptiveIntegrator(
                noise=NormalDist().function,
                default_variable=[0, 0, 0, 0]
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = I.execute([10, 10, 10, 10])[0]

        np.testing.assert_allclose(val, [10.12167502,  10.44386323,  10.33367433,  11.49407907])

    def test_integrator_drift_diffusion_noise_val(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=DriftDiffusionIntegrator(
                noise=5.0,
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 15.010789523731438)

    def test_integrator_ornstein_uhlenbeck_noise_val(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=OrnsteinUhlenbeckIntegrator(
                noise=5.0,
            ),
            time_scale=TimeScale.TIME_STEP
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 15.010789523731438)


class TestFHN:


    def test_FHN_defaults(self):

        F = IntegratorMechanism(
            name='IntegratorMech-FHNFunction',
            function=FHNIntegrator(

            )
        )
        plot_v_list = []
        plot_w_list = []

        expected_v_list = []
        expected_w_list = []
        stimulus = 1.0
        for i in range(10):
            for j in range(10):
                new_v = F.execute(stimulus)[0][0]
                new_w = F.execute(stimulus)[1][0]
                # ** uncomment the lines below if you want to view the plot:
                # plot_v_list.append(new_v)
                # plot_w_list.append(new_w)
            expected_v_list.append(new_v)
            expected_w_list.append(new_w)

        # ** uncomment the lines below if you want to view the plot:
        # import matplotlib.pyplot as plt
        # plt.plot(plot_v_list)
        # plt.plot(plot_w_list)
        # plt.show()

        np.testing.assert_allclose(expected_v_list, [1.9861589924245777, 1.9184159304279109, 1.7920107368145777,
                                                     1.6651158106802393, 1.5360917598075965, 1.4019128309448776,
                                                     1.2568420252868404, 1.08773745582042, 0.8541804646541804,
                                                     0.34785588139530099])
        np.testing.assert_allclose(expected_w_list, [0.28713219302304327, 0.65355262255707869, 0.9581082373550347,
                                                     1.2070585850028435, 1.4068978270680454, 1.5629844531368104,
                                                     1.6793901854329185, 1.7583410650743645, 1.7981128658110572,
                                                     1.7817328532815251])

    # def test_FHN_gilzenrat(self):
    #
    #     F = IntegratorMechanism(
    #         name='IntegratorMech-FHNFunction',
    #         function=FHNIntegrator(
    #             time_step_size=0.1,
    #             initial_v=0.2,
    #             initial_w=0.0,
    #             t_0=0.0,
    #             time_constant_v=1.0,
    #             a_v=-1.0,
    #             b_v=1.5,
    #             c_v=-0.5,
    #             d_v=0.0,
    #             e_v=-1.0,
    #             f_v=0.0,
    #             time_constant_w=100.0,
    #             a_w=1.0,
    #             b_w=-0.5,
    #             c_w=0.0
    #         )
    #     )
    #     plot_v_list = []
    #     plot_w_list = []
    #
    #     expected_v_list = []
    #     expected_w_list = []
    #     stimulus = 0.0
    #     for i in range(10):
    #
    #         for j in range(50):
    #             new_v = F.execute(stimulus)[0][0]
    #             new_w = F.execute(stimulus)[1][0]
    #             # ** uncomment the lines below if you want to view the plot:
    #             plot_v_list.append(new_v)
    #             plot_w_list.append(new_w)
    #         expected_v_list.append(new_v)
    #         expected_w_list.append(new_w)
    #     # print(plot_v_list)
    #     # print(plot_w_list)
    #     # ** uncomment the lines below if you want to view the plot:
    #     import matplotlib.pyplot as plt
    #     plt.plot(plot_v_list)
    #     plt.plot(plot_w_list)
    #     plt.show()
    #
    #     # np.testing.assert_allclose(expected_v_list, [1.9861589924245777, 1.9184159304279109, 1.7920107368145777,
    #     #                                              1.6651158106802393, 1.5360917598075965, 1.4019128309448776,
    #     #                                              1.2568420252868404, 1.08773745582042, 0.8541804646541804,
    #     #                                              0.34785588139530099])
    #     # np.testing.assert_allclose(expected_w_list, [0.28713219302304327, 0.65355262255707869, 0.9581082373550347,
    #     #                                              1.2070585850028435, 1.4068978270680454, 1.5629844531368104,
    #     #                                              1.6793901854329185, 1.7583410650743645, 1.7981128658110572,
    #     #                                              1.7817328532815251])
    #     #
    #
    # def test_FHN_gilzenrat_low_electrotonic_coupling(self):
    #
    #     F = IntegratorMechanism(
    #         name='IntegratorMech-FHNFunction',
    #         function=FHNIntegrator(
    #             time_step_size=0.1,
    #             initial_v=0.2,
    #             initial_w=0.0,
    #             t_0=0.0,
    #             time_constant_v=1.0,
    #             a_v=-1.0,
    #             b_v=0.5,
    #             c_v=0.5,
    #             d_v=0.0,
    #             e_v=-1.0,
    #             f_v=0.0,
    #             electrotonic_coupling=0.55,
    #             time_constant_w=100.0,
    #             a_w=1.0,
    #             b_w=-0.5,
    #             c_w=0.0
    #         )
    #     )
    #     plot_v_list = []
    #     plot_w_list = []
    #
    #     expected_v_list = []
    #     expected_w_list = []
    #     stimulus = 0.0
    #     for i in range(10):
    #
    #         for j in range(600):
    #             new_v = F.execute(stimulus)[0][0]
    #             new_w = F.execute(stimulus)[1][0]
    #             # ** uncomment the lines below if you want to view the plot:
    #             plot_v_list.append(new_v)
    #             plot_w_list.append(new_w)
    #         expected_v_list.append(new_v)
    #         expected_w_list.append(new_w)
    #     # print(plot_v_list)
    #     # print(plot_w_list)
    #     # ** uncomment the lines below if you want to view the plot:
    #     import matplotlib.pyplot as plt
    #     plt.plot(plot_v_list)
    #     plt.plot(plot_w_list)
    #     plt.show()
    #
    #     # np.testing.assert_allclose(expected_v_list, [1.9861589924245777, 1.9184159304279109, 1.7920107368145777,
    #     #                                              1.6651158106802393, 1.5360917598075965, 1.4019128309448776,
    #     #                                              1.2568420252868404, 1.08773745582042, 0.8541804646541804,
    #     #                                              0.34785588139530099])
    #     # np.testing.assert_allclose(expected_w_list, [0.28713219302304327, 0.65355262255707869, 0.9581082373550347,
    #     #                                              1.2070585850028435, 1.4068978270680454, 1.5629844531368104,
    #     #                                              1.6793901854329185, 1.7583410650743645, 1.7981128658110572,
    #     #                                              1.7817328532815251])
    #     #
