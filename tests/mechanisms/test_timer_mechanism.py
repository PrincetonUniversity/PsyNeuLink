import pytest
import numpy as np
import psyneulink as pnl
from psyneulink.library.components.mechanisms.processing.integrator.timermechanism import TimerMechanism
from psyneulink.core.components.functions.nonstateful.timerfunctions import (
    LinearTimer, AcceleratingTimer, DeceleratingTimer, AsymptoticTimer)


class TestTimer:

    arg_names = ('timer_function', 'start','end','duration','increment','expected_no_input', 'expected_w_input')
    # Note: the values below have been independently verified against those in the Desmos graphing calculator
    timer_test_data = [
        (LinearTimer, .1, .4, .3, .1, (.2, .3, .4, .4),
         (.35, .45)),
        (LinearTimer, -1, -4, 3, 1, (-2, -3, -4, -4),
         (-3.5, -4.5)),
        (AcceleratingTimer, .1, .3, .3, .1, (0.13422781, 0.19553751, .3, .3),
         (0.24108029, 0.37565076)),
        (AcceleratingTimer, -1, -3, 3, 1, (-1.3422781, -1.9553751, -3, -3),
         (-2.41080287, -3.7565076)),
        (DeceleratingTimer, -.1, -.3, .3, .1, (-0.17075676534276596, -0.2373414308173889,
                                             -0.30000000000000004, -0.30000000000000004),
         (-0.26914668, -0.32992988)),
        (DeceleratingTimer, 1, 3, 3, 1, (1.919916176948096, 2.5577504296925917, 3.0, 3.0),
         (2.79906304, 3.16731682)),
        (AsymptoticTimer, .1, .3, .3, .1, (0.25691130619936233, 0.29071682233277443, 0.298, 0.298, 0.298),
         (0.29569113, 0.29907168)),
        (AsymptoticTimer, -1, -3, 3, 1, (-2.5691130619936233, -2.9071682233277443, -2.98, -2.98, -2.98),
         (-2.95691131, -2.9907168))
    ]

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.parametrize(arg_names,
                             timer_test_data,
                             ids=[x[0] for x in timer_test_data]
                             )
    def test_timer_functions(self, timer_function, start, end, duration, increment,
                               expected_no_input, expected_w_input):
        timer = TimerMechanism(trajectory=timer_function, start=start,
                               end=end,
                               increment=increment,
                               duration=duration)
        # Test without input:
        for i in range(len(expected_no_input) - 1):
            np.testing.assert_allclose(timer.execute(), expected_no_input[i])

        # Test with input on 2nd execution:
        assert timer.value != 0
        timer.reset()
        assert timer.value == 0
        np.testing.assert_allclose(timer.execute(), expected_no_input[0])
        np.testing.assert_allclose(timer.execute(1.5), expected_w_input[0])
        np.testing.assert_allclose(timer.execute(), expected_w_input[1])

    def test_timer_in_composition(self):
        for with_input in {True, False}:
            input_mech = pnl.ProcessingMechanism()
            if with_input:
                timer = TimerMechanism(trajectory=LinearTimer, start=1, end=3, increment=1, duration=2)
                inputs = {input_mech: 2.5, timer: 1.7}
                expected = [[2.5], [2.7]]
            else:
                timer = TimerMechanism(trajectory=LinearTimer,
                               start=1, end=3, increment=1, duration=2,
                               input_ports={pnl.NAME: 'STEPS',
                                            pnl.DEFAULT_INPUT: pnl.DEFAULT_VARIABLE})
                inputs = {input_mech: 2.5}
                expected = [[2.5], [2]]
            comp = pnl.Composition(nodes=[input_mech, timer])
            result = comp.run(inputs=inputs)
            np.testing.assert_allclose(result, expected)
