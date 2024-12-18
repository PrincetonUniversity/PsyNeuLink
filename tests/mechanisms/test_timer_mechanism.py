import pytest
import numpy as np
import psyneulink as pnl
from psyneulink.library.components.mechanisms.processing.integrator.timermechanism import TimerMechanism
from psyneulink.core.components.functions.nonstateful.timerfunctions import (
    LinearTimer, AcceleratingTimer, DeceleratingTimer, AsymptoticTimer)



# class TestReset:
#     #                        reset_default_value     modulation
#     test_args = [('default',       1,                   None),
#                  ('OVERRIDE',     None,             pnl.OVERRIDE)]
#     @pytest.mark.parametrize('test_name, reset_default, modulation', test_args, ids=[x[0] for x in test_args])
#     def test_reset_integrator_mechanism(self, test_name, reset_default, modulation):
#         input = pnl.ProcessingMechanism(name='INPUT')
#         counter = IntegratorMechanism(function=SimpleIntegrator,
#                                       default_variable=1,
#                                       reset_default=reset_default,
#                                       name='COUNTER')
#         ctl = pnl.ControlMechanism(monitor_for_control=input,
#                                    modulation=modulation,
#                                    control=('reset', counter))
#         c = Composition(nodes=[input, ctl, counter])
#         c.run(inputs={input:[0,0,1,0,0]})
#         expected = [[[0.],[1.]], [[0.],[2.]], [[1.],[0.]], [[0.],[1.]], [[0.],[2.]]]
#         np.testing.assert_allclose(c.results, expected)
#
#     def test_FitzHughNagumo_valid(self):
#         I = IntegratorMechanism(name="I",
#                                 function=FitzHughNagumoIntegrator())
#         I.reset_stateful_function_when = Never()
#         I.execute(1.0)
#
#         np.testing.assert_allclose([[0.05127053]], I.value[0], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose([[0.00279552]], I.value[1], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose([[0.05]], I.value[2], rtol=1e-5, atol=1e-8)
#
#         I.function.reset(0.01, 0.02, 0.03)
#
#         np.testing.assert_allclose(0.01, I.function.value[0], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(0.02, I.function.value[1], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(0.03, I.function.value[2], rtol=1e-5, atol=1e-8)
#
#         np.testing.assert_allclose([[0.05127053]], I.value[0], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose([[0.00279552]], I.value[1], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose([[0.05]], I.value[2], rtol=1e-5, atol=1e-8)
#
#         np.testing.assert_allclose([[0.05127053]], I.output_ports[0].value, rtol=1e-5, atol=1e-8)
#
#         I.execute(1.0)
#
#         np.testing.assert_allclose([[0.06075727]], I.value[0], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose([[0.02277156]], I.value[1], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose([[0.08]], I.value[2], rtol=1e-5, atol=1e-8)
#
#         np.testing.assert_allclose([[0.06075727]], I.output_ports[0].value, rtol=1e-5, atol=1e-8)
#
#         # I.reset(new_previous_v=0.01, new_previous_w=0.02, new_previous_time=0.03)
#         I.reset(0.01, 0.02, 0.03)
#
#         np.testing.assert_allclose(0.01, I.value[0], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(0.02, I.value[1], rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(0.03, I.value[2], rtol=1e-5, atol=1e-8)
#
#         np.testing.assert_allclose(0.01, I.output_ports[0].value, rtol=1e-5, atol=1e-8)
#         # np.testing.assert_allclose(0.01, I.output_port.value[0], rtol=1e-5, atol=1e-8)
#         # np.testing.assert_allclose(0.02, I.output_port.value[1], rtol=1e-5, atol=1e-8)
#         # np.testing.assert_allclose(0.03, I.output_port.value[2], rtol=1e-5, atol=1e-8)
#
#     def test_AGTUtility_valid(self):
#         I = IntegratorMechanism(name="I",
#                                 function=DualAdaptiveIntegrator())
#         I.reset_stateful_function_when = Never()
#         np.testing.assert_allclose([0.0], I.function.previous_short_term_avg)
#         np.testing.assert_allclose([0.0], I.function.previous_long_term_avg)
#
#         I.function.reset(0.2, 0.8)
#
#         np.testing.assert_allclose([0.2], I.function.previous_short_term_avg)
#         np.testing.assert_allclose([0.8], I.function.previous_long_term_avg)
#
#         I.function.reset()
#
#         np.testing.assert_allclose([0.0], I.function.previous_short_term_avg)
#         np.testing.assert_allclose([0.0], I.function.previous_long_term_avg)
#
#         I.reset(0.3, 0.7)
#
#         np.testing.assert_allclose([0.3], I.function.previous_short_term_avg)
#         np.testing.assert_allclose([0.7], I.function.previous_long_term_avg)
#         context = Context(execution_id=None)
#         print(I.value)
#         print(I.function._combine_terms(0.3, 0.7, context))
#         np.testing.assert_allclose([I.function._combine_terms(0.3, 0.7, context)], I.value)
#
#         I.reset()
#
#         np.testing.assert_allclose([0.0], I.function.previous_short_term_avg)
#         np.testing.assert_allclose([0.0], I.function.previous_long_term_avg)
#         np.testing.assert_allclose([I.function._combine_terms(0.0, 0.0, context)], I.value)
#
#     def test_Simple_valid(self):
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=SimpleIntegrator(
#             ),
#         )
#         I.reset_stateful_function_when = Never()
#
#         #  returns previous_value + rate*variable + noise
#         # so in this case, returns 10.0
#         I.execute(10)
#         np.testing.assert_allclose(I.value, 10.0)
#         np.testing.assert_allclose(I.output_port.value, 10.0)
#
#         # reset function
#         I.function.reset(5.0)
#         np.testing.assert_allclose(I.function.value, 5.0)
#         np.testing.assert_allclose(I.value, 10.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 10.0)
#
#         # reset function without value spec
#         I.function.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 10.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 10.0)
#
#         # reset mechanism
#         I.reset(4.0)
#         np.testing.assert_allclose(I.function.value, 4.0)
#         np.testing.assert_allclose(I.value, 4.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 4.0)
#
#         I.execute(1)
#         np.testing.assert_allclose(I.value, 5.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 5.0)
#
#         # reset mechanism without value spec
#         I.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.0)
#
#     def test_Adaptive_valid(self):
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=AdaptiveIntegrator(
#                 rate=0.5
#             ),
#         )
#
#         #  returns (1-rate)*previous_value + rate*variable + noise
#         # so in this case, returns 0.5*0 + 0.5*10 + 0 = 5.0
#         I.execute(10)
#         np.testing.assert_allclose(I.value, 5.0)
#         np.testing.assert_allclose(I.output_port.value, 5.0)
#
#         # reset function
#         I.function.reset(1.0)
#         np.testing.assert_allclose(I.function.value, 1.0)
#         np.testing.assert_allclose(I.value, 5.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 5.0)
#
#         # reset function without value spec
#         I.function.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 5.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 5.0)
#
#         # reset mechanism
#         I.reset(2.0)
#         np.testing.assert_allclose(I.function.value, 2.0)
#         np.testing.assert_allclose(I.value, 2.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 2.0)
#
#         I.execute(1.0)
#         #  (1-0.5)*2.0 + 0.5*1.0 + 0 = 1.5
#         np.testing.assert_allclose(I.value, 1.5)
#         np.testing.assert_allclose(I.output_ports[0].value, 1.5)
#
#         # reset mechanism without value spec
#         I.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.0)
#
#     def test_Accumulator_warning(self):
#         regexp = r"AccumulatorIntegrator does not use its variable;  value passed .* will be ignored"
#         with pytest.warns(UserWarning, match=regexp):
#             I = AccumulatorIntegrator()
#             I(1000)
#
#     def test_Accumulator_valid(self):
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=AccumulatorIntegrator(increment=1.0),
#         )
#
#         #  returns previous_value + rate + noise
#         # so in this case, returns 0.0 + 1.0
#         I.execute(1000)
#         np.testing.assert_allclose(I.value, 1.0)
#         np.testing.assert_allclose(I.output_port.value, 1.0)
#
#         # reset function
#         I.function.reset(2.0)
#         np.testing.assert_allclose(I.function.value, 2.0)
#         np.testing.assert_allclose(I.value, 1.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 1.0)
#
#         # reset function without value spec
#         I.function.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 1.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 1.0)
#
#         # reset mechanism
#         I.reset(2.0)
#         np.testing.assert_allclose(I.function.value, 2.0)
#         np.testing.assert_allclose(I.value, 2.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 2.0)
#
#         I.execute(1000)
#         #  2.0 + 1.0 = 3.0
#         np.testing.assert_allclose(I.value, 3.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 3.0)
#
#         # reset mechanism without value spec
#         I.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.0)
#
#     def test_OU_valid(self):
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=OrnsteinUhlenbeckIntegrator(),
#         )
#
#         # previous_value + (decay * previous_value - rate * variable) * time_step_size + noise
#         # decay=1.0, initializer=0.0, rate=1.0, time_step_size=1.0, noise=0.0
#         # returns 0.0 + (1.0*0.0 - 1.0*10.0*1.0) + 0.0 = -10.0
#         I.execute(2.0)
#         np.testing.assert_allclose(I.value[0], -2.0)
#         np.testing.assert_allclose(I.output_port.value, -2.0)
#
#         # reset function
#         I.function.reset(5.0, 0.0)
#         np.testing.assert_allclose(I.function.value[0], 5.0)
#         np.testing.assert_allclose(I.value[0], -2.0)
#         np.testing.assert_allclose(I.output_ports[0].value, -2.0)
#
#         # reset function without value spec
#         I.function.reset()
#         np.testing.assert_allclose(I.function.value[0], 0.0)
#         np.testing.assert_allclose(I.value[0], -2.0)
#         np.testing.assert_allclose(I.output_ports[0].value, -2.0)
#
#         # reset mechanism
#         I.reset(4.0, 0.0)
#         np.testing.assert_allclose(I.function.value[0], 4.0)
#         np.testing.assert_allclose(I.value[0], 4.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 4.0)
#
#         I.execute(1.0)
#         # 4.0 + (1.0 * 4.0 - 1.0 * 1.0) * 1.0 = 4 + 3 = 7
#         np.testing.assert_allclose(I.value[0], 7.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 7.0)
#
#         # reset mechanism without value spec
#         I.reset()
#         np.testing.assert_allclose(I.function.value[0], 0.0)
#         np.testing.assert_allclose(I.value[0][0], 0.0)
#         np.testing.assert_allclose(I.output_ports[0].value[0], 0.0)
#
#     def test_Accumulator_valid(self):
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=AccumulatorIntegrator(increment=0.1),
#         )
#
#         #  returns previous_value * rate + noise + increment
#         # initializer = 0.0, rate = 1.0, noise = 0.0, increment = 0.1
#         # returns 0.0*1.0 + 0.0 + 0.1 = 0.1
#         I.execute(10000)
#         np.testing.assert_allclose(I.value, 0.1)
#         np.testing.assert_allclose(I.output_port.value, 0.1)
#
#         # reset function
#         I.function.reset(2.0)
#         np.testing.assert_allclose(I.function.value, 2.0)
#         np.testing.assert_allclose(I.value, 0.1)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.1)
#
#         # reset function without value spec
#         I.function.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.1)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.1)
#
#         # reset mechanism
#         I.reset(5.0)
#         np.testing.assert_allclose(I.function.value, 5.0)
#         np.testing.assert_allclose(I.value, 5.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 5.0)
#
#         I.execute(10000)
#         #  5.0 * 1.0 + 0.0 + 0.1
#         np.testing.assert_allclose(I.value, 5.1)
#         np.testing.assert_allclose(I.output_ports[0].value, 5.1)
#
#         # reset mechanism without value spec
#         I.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.0)
#
#     def test_LCIIntegrator_valid(self):
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=LeakyCompetingIntegrator(leak=1),
#         )
#
#         # previous_value + (new_value - rate*previous_value)*time_step_size + noise
#         # initializer=0.0, rate=1.0, time_step_size=0.1, noise=0.0
#         # returns 0.0 + (1.0*0.0 + 2.0)*0.1 = 2.0
#         I.execute(2.0)
#         np.testing.assert_allclose(I.value, 0.2)
#         np.testing.assert_allclose(I.output_port.value, 0.2)
#
#         # reset function
#         I.function.reset(5.0)
#         np.testing.assert_allclose(I.function.value, 5.0)
#         np.testing.assert_allclose(I.value, 0.2)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.2)
#
#         # reset function without value spec
#         I.function.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.2)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.2)
#
#         # reset mechanism
#         I.reset(4.0)
#         np.testing.assert_allclose(I.function.value, 4.0)
#         np.testing.assert_allclose(I.value, 4.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 4.0)
#
#         I.execute(1.0)
#         # 4.0 + (1.0*4.0 + 1.0)*0.1 + 0.0
#         np.testing.assert_allclose(I.value, 3.7)
#         np.testing.assert_allclose(I.output_ports[0].value, 3.7)
#
#         # reset mechanism without value spec
#         I.reset()
#         np.testing.assert_allclose(I.function.value, 0.0)
#         np.testing.assert_allclose(I.value, 0.0)
#         np.testing.assert_allclose(I.output_ports[0].value, 0.0)
#
#     def test_reset_not_integrator(self):
#
#         with pytest.raises(MechanismError) as err_txt:
#             I_not_integrator = IntegratorMechanism(function=Linear)
#             I_not_integrator.execute(1.0)
#             I_not_integrator.reset(0.0)
#         assert "not allowed because this Mechanism is not stateful;" in str(err_txt.value)
#         assert "it does not have an accumulator to reset" in str(err_txt.value)


class TestTimerFunctions:


    arg_names = ('timer_function', 'start','end','duration','increment','expected')
    timer_test_data = [
        (LinearTimer, .1, .4, .3, .1, (.2, .3, .4, .4)),
        (LinearTimer, 1, 4, 3, 1, (2, 3, 4, 4)),
        (AcceleratingTimer, .1, .3, .3, .1, (0.13422781, 0.19553751, .3, .3)),
        (AcceleratingTimer, 1, 3, 3, 1, (1.3422781, 1.9553751, 3, 3)),
        (DeceleratingTimer, .1, .3, .3, .1, (0.17075676534276596, 0.2373414308173889,
                                             0.30000000000000004, 0.30000000000000004)),
        (DeceleratingTimer, 1, 3, 3, 1, (1.919916176948096, 2.5577504296925917, 3.0, 3.0)),
        (AsymptoticTimer, .1, .3, .3, .1, (0.25691130619936233, 0.29071682233277443, 0.298, 0.298, 0.298)),
        (AsymptoticTimer, 1, 3, 3, 1, (2.5691130619936233, 2.9071682233277443, 2.98, 2.98, 2.98))
    ]

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.parametrize(arg_names,
                             timer_test_data,
                             ids=[x[0] for x in timer_test_data]
                             )
    def test_AcceleratingTimer(self, timer_function, start, end, duration, increment, expected):
        timer = TimerMechanism(trajectory=timer_function, start=start,
                               end=end,
                               increment=increment,
                               duration=duration)
        for i in range(len(expected)-1):
            np.testing.assert_allclose(timer.execute(1), expected[i])
