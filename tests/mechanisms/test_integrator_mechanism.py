import numpy as np
import pytest

from PsyNeuLink.Components.Functions.Function import AccumulatorIntegrator, ConstantIntegrator, NormalDist, \
    SimpleIntegrator, FHNIntegrator
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, DriftDiffusionIntegrator, \
    OrnsteinUhlenbeckIntegrator, AGTUtilityIntegrator
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
                decay=0.5,
                initializer=10.0,
                rate=0.25,
                time_step_size=0.5,
                noise = 0.0,
                offset= 1.0
            )
        )
        # P = process(pathway=[I])
        # value = previous_value + decay * (previous_value -  rate * new_value) * time_step_size + np.sqrt(
        # time_step_size * noise) * np.random.normal()
        # step 1:

        val = I.execute(1)
        # value = 10 + 0.5 * ( 10.0 - 0.25*1.0) * 0.5 + sqrt(0.25*0)*random_sample
        #       = 10 + 0.5*9.75*0.5
        #       = 12.4375
        # adjusted_value = 12.4375 + 1.0
        # previous_value = 13.4375
        # RETURN 13.4375

        # step 2:
        val2 = I.execute(1)
        # value = 13.4375 + 0.5 * ( 13.4375 - 0.25*1.0) * 0.5
        #       = 13.4375 + 3.296875
        # adjusted_value = 16.734375 + 1.0
        # previous_value = 17.734375
        # RETURN 31

        assert (val, val2) == (13.4375, 17.734375)

    def test_ornstein_uhlenbeck_integrator_time(self):
        OU = IntegratorMechanism(
            function=OrnsteinUhlenbeckIntegrator(
                initializer=10.0,
                rate=10,
                time_step_size=0.2,
                t0=0.5,
                decay=0.1,
                offset=10,
            )
        )
        time_0 = OU.function_object.previous_time  # t_0  = 0.5
        np.testing.assert_allclose(time_0, [0.5], atol=1e-08)

        OU.execute(10)
        time_1 = OU.function_object.previous_time  # t_1  = 0.5 + 0.2 = 0.7
        np.testing.assert_allclose(time_1, [0.7], atol=1e-08)

        for i in range(11):  # t_11 = 0.7 + 10*0.2 = 2.7
            OU.execute(10)
        time_12 = OU.function_object.previous_time # t_12 = 2.7 + 0.2 = 2.9
        np.testing.assert_allclose(time_12, [2.9], atol=1e-08)

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
        np.testing.assert_allclose(val, [0.12167502, 0.44386323, 0.33367433, 1.49407907])

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
                noise=2.0,
                decay=0.5,
                initializer=1.0,
                rate=0.25
            ),
            time_scale=TimeScale.TIME_STEP
        )

        # val = 1.0 + 0.5 * (1.0 - 0.25 * 2.5) * 1.0 + np.sqrt(1.0 * 2.0) * np.random.normal()


        val = float(I.execute(2.5))

        np.testing.assert_allclose(val, 4.356601554140335)

class TestFHN:


    # def test_FHN_defaults(self):
    #     F = IntegratorMechanism(
    #         name='IntegratorMech-FHNFunction',
    #         function=FHNIntegrator(
    #             initial_v=0.2,
    #             initial_w=0.0,
    #             time_step_size=0.01
    #         )
    #     )
    #     plot_v_list = [0.2]
    #     plot_w_list = [0.0]
    #     plot_t_list = [0.0]
    #
    #     expected_v_list = []
    #     expected_w_list = []
    #     stimulus = 0.08
    #     for i in range(80):
    #         results = F.execute(stimulus)
    #         new_v = results[0][0]
    #         new_w = results[1][0]
    #         new_t = results[2]
    #
    #         # ** uncomment the lines below if you want to view the plot:
    #         plot_v_list.append(new_v)
    #         plot_w_list.append(new_w)
    #         plot_t_list.append(new_t)
    #     # ** uncomment the lines below if you want to view the plot:
    #     from matplotlib import pyplot as plt
    #     plt.plot(plot_v_list)
    #     plt.plot(plot_w_list)
    #     plt.show()
    #
    #     print(plot_v_list)
    #     print(plot_w_list)
    #     print(plot_t_list)
    #
    #     # np.testing.assert_allclose(expected_v_list, [1.9861589924245777, 1.9184159304279109, 1.7920107368145777,
    #     #                                              1.6651158106802393, 1.5360917598075965, 1.4019128309448776,
    #     #                                              1.2568420252868404, 1.08773745582042, 0.8541804646541804,
    #     #                                              0.34785588139530099])
    #     # np.testing.assert_allclose(expected_w_list, [0.28713219302304327, 0.65355262255707869, 0.9581082373550347,
    #     #                                              1.2070585850028435, 1.4068978270680454, 1.5629844531368104,
    #     #                                              1.6793901854329185, 1.7583410650743645, 1.7981128658110572,
    #     #                                              1.7817328532815251])

    def test_Gilzenrat_Figure_2(self):

        initial_v = 0.2
        initial_w = 0.0

        F = IntegratorMechanism(
            name='IntegratorMech-FHNFunction',
            function=FHNIntegrator(
                initial_v=initial_v,
                initial_w=initial_w,
                time_step_size=0.01,
                time_constant_w=1.0,
                time_constant_v=0.01,
                a_v=-1.0,
                b_v=1.0,
                c_v=1.0,
                d_v=0.0,
                e_v=-1.0,
                f_v=1.0,
                threshold=0.5,
                mode=1.0,
                uncorrelated_activity=0.0,
                a_w=1.0,
                b_w=-1.0,
                c_w=0.0

            )
        )
        plot_v_list = [initial_v]
        plot_w_list = [initial_w]

        # found this stimulus by guess and check b/c one was not provided with Figure 2 params
        stimulus = 0.073
        # increase range to 200 to match Figure 2 in Gilzenrat
        for i in range(10):
            results = F.execute(stimulus)
            plot_v_list.append(results[0][0][0])
            plot_w_list.append(results[1][0][0])

        # ** uncomment the lines below if you want to view the plot:
        # from matplotlib import pyplot as plt
        # plt.plot(plot_v_list)
        # plt.plot(plot_w_list)
        # plt.show()

        np.testing.assert_allclose(plot_v_list, [0.2, 0.22493312915681499, 0.24844236992931412, 0.27113468959297515,
                                                 0.29350254152625221, 0.31599112332052792, 0.33904651470437225,
                                                 0.36315614063656521, 0.38888742632665502, 0.41692645840176923,
                                                 0.44811281741549686]
)
        np.testing.assert_allclose(plot_w_list, [0.0, 0.0019518690642000148, 0.0041351416812363193,
                                                 0.0065323063637677276, 0.0091322677555586273, 0.011929028036111457,
                                                 0.014921084302726394, 0.018111324713170868, 0.021507331976846619,
                                                 0.025122069034563425, 0.028974949616469712]
)


class TestAGTUtilityIntegrator:

    def test_utility_integrator_default(self):
        # default params:
        # initial_short_term_utility = 0.0
        # initial_long_term_utility = 0.0
        # short_term_rate = 1.0
        # long_term_rate = 1.0

        U = IntegratorMechanism(
            name = "AGTUtilityIntegrator",
            function=AGTUtilityIntegrator(
            )

        )

        engagement = []
        short_term_util = []
        long_term_util = []
        for i in range(50):
            engagement.append(U.execute([1])[0][0])
            short_term_util.append(U.function_object.short_term_utility_logistic[0])
            long_term_util.append(U.function_object.long_term_utility_logistic[0])
        print("engagement = ", engagement)
        print("short_term_util = ", short_term_util)
        print("long_term_util = ", long_term_util)

    def test_utility_integrator_short_minus_long(self):
        # default params:
        # initial_short_term_utility = 0.0
        # initial_long_term_utility = 0.0
        # short_term_rate = 1.0
        # long_term_rate = 1.0

        U = IntegratorMechanism(
            name = "AGTUtilityIntegrator",
            function=AGTUtilityIntegrator(
                operation="s-l"
            )

        )

        engagement = []
        short_term_util = []
        long_term_util = []
        for i in range(50):
            engagement.append(U.execute([1])[0][0])
            short_term_util.append(U.function_object.short_term_utility_logistic[0])
            long_term_util.append(U.function_object.long_term_utility_logistic[0])
        print("engagement = ", engagement)
        print("short_term_util = ", short_term_util)
        print("long_term_util = ", long_term_util)

    def test_utility_integrator_short_plus_long(self):
        # default params:
        # initial_short_term_utility = 0.0
        # initial_long_term_utility = 0.0
        # short_term_rate = 1.0
        # long_term_rate = 1.0

        U = IntegratorMechanism(
            name = "AGTUtilityIntegrator",
            function=AGTUtilityIntegrator(
                operation="s+l"
            )

        )

        engagement = []
        short_term_util = []
        long_term_util = []
        for i in range(50):
            engagement.append(U.execute([1])[0][0])
            short_term_util.append(U.function_object.short_term_utility_logistic[0])
            long_term_util.append(U.function_object.long_term_utility_logistic[0])
        print("engagement = ", engagement)
        print("short_term_util = ", short_term_util)
        print("long_term_util = ", long_term_util)

    # def test_plot_utility_integrator(self):
        # from matplotlib import pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # from matplotlib import cm
        # import numpy as np
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        #
        # def logistic(variable):
        #
        #     try:
        #         return_val = 1 / (1 + np.exp(-variable))
        #     except (Warning):
        #         # handle RuntimeWarning: overflow in exp
        #         return_val = 0
        #
        #     return return_val
        #
        # short = np.linspace(0, 1, 20)
        # long = np.linspace(0, 1, 20)
        # short_grid, long_grid = np.meshgrid(short, long, sparse=True)
        #
        # short_logistic = logistic(short)
        # long_logistic = logistic(long)
        # short_grid_logistic, long_grid_logistic = np.meshgrid(short_grid, long_grid, sparse=True)
        #
        # z = (1-short_grid_logistic)*long_grid_logistic
        # surf = ax.plot_surface(short_grid_logistic, long_grid_logistic, z,
        #                        cmap=cm.gray,
        #                        linewidth=0,
        #                        # antialiased=False,
        #
        #                        )
        # # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

        # def logistic(x):
        #
        #     try:
        #         return_val = 1 / (1 + np.exp(-x))
        #     except (Warning):
        #         # handle RuntimeWarning: overflow in exp
        #         return_val = 0
        #
        #     return return_val
        #
        # def combine(s, l):
        #     return (1-s)*l
        #
        # import numpy as np
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # import random
        #
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # s = l = np.arange(0, 1.0, 0.05)
        # s = list(map(logistic, s))
        # l = list(map(logistic, l))
        # S, L = np.meshgrid(s, l)
        # zs = np.array([combine(s, l) for s, l in zip(np.ravel(S), np.ravel(L))])
        # Z = zs.reshape(S.shape)
        #
        # ax.plot_surface(S, L, Z)
        #
        # ax.set_xlabel('short term')
        # ax.set_ylabel('long term')
        # ax.set_zlabel('engagement')
        #
        # plt.show()


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
