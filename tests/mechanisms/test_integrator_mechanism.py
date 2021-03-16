import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.llvm as pnlvm

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.distributionfunctions import NormalDist
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import \
    SimpleIntegrator, AdaptiveIntegrator, DriftDiffusionIntegrator, OrnsteinUhlenbeckIntegrator, \
    FitzHughNagumoIntegrator, AccumulatorIntegrator, LeakyCompetingIntegrator, DualAdaptiveIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.components.mechanisms.processing.integratormechanism import \
    IntegratorMechanism, IntegratorMechanismError
from psyneulink.core.globals.context import Context
from psyneulink.core.scheduling.condition import AtTrial
from psyneulink.core.scheduling.condition import Never


class TestReset:
    def test_FitzHughNagumo_valid(self):
        I = IntegratorMechanism(name="I",
                                function=FitzHughNagumoIntegrator())
        I.reset_stateful_function_when = Never()
        I.execute(1.0)

        assert np.allclose([[0.05127053]], I.value[0])
        assert np.allclose([[0.00279552]], I.value[1])
        assert np.allclose([[0.05]], I.value[2])

        I.function.reset(0.01, 0.02, 0.03)

        assert np.allclose(0.01, I.function.value[0])
        assert np.allclose(0.02, I.function.value[1])
        assert np.allclose(0.03, I.function.value[2])

        assert np.allclose([[0.05127053]], I.value[0])
        assert np.allclose([[0.00279552]], I.value[1])
        assert np.allclose([[0.05]], I.value[2])

        assert np.allclose([[0.05127053]], I.output_ports[0].value)

        I.execute(1.0)

        assert np.allclose([[0.06075727]], I.value[0])
        assert np.allclose([[0.02277156]], I.value[1])
        assert np.allclose([[0.08]], I.value[2])

        assert np.allclose([[0.06075727]], I.output_ports[0].value)

        # I.reset(new_previous_v=0.01, new_previous_w=0.02, new_previous_time=0.03)
        I.reset(0.01, 0.02, 0.03)

        assert np.allclose(0.01, I.value[0])
        assert np.allclose(0.02, I.value[1])
        assert np.allclose(0.03, I.value[2])

        assert np.allclose(0.01, I.output_ports[0].value)
        # assert np.allclose(0.01, I.output_port.value[0])
        # assert np.allclose(0.02, I.output_port.value[1])
        # assert np.allclose(0.03, I.output_port.value[2])

    def test_AGTUtility_valid(self):
        I = IntegratorMechanism(name="I",
                                function=DualAdaptiveIntegrator())
        I.reset_stateful_function_when = Never()
        assert np.allclose([[0.0]], I.function.previous_short_term_avg)
        assert np.allclose([[0.0]], I.function.previous_long_term_avg)

        I.function.reset(0.2, 0.8)

        assert np.allclose([[0.2]], I.function.previous_short_term_avg)
        assert np.allclose([[0.8]], I.function.previous_long_term_avg)

        I.function.reset()

        assert np.allclose([[0.0]], I.function.previous_short_term_avg)
        assert np.allclose([[0.0]], I.function.previous_long_term_avg)

        I.reset(0.3, 0.7)

        assert np.allclose([[0.3]], I.function.previous_short_term_avg)
        assert np.allclose([[0.7]], I.function.previous_long_term_avg)
        context = Context(execution_id=None)
        print(I.value)
        print(I.function._combine_terms(0.3, 0.7, context))
        assert np.allclose(I.function._combine_terms(0.3, 0.7, context), I.value)

        I.reset()

        assert np.allclose([[0.0]], I.function.previous_short_term_avg)
        assert np.allclose([[0.0]], I.function.previous_long_term_avg)
        assert np.allclose(I.function._combine_terms(0.0, 0.0, context), I.value)

    def test_Simple_valid(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
            ),
        )
        I.reset_stateful_function_when = Never()

        #  returns previous_value + rate*variable + noise
        # so in this case, returns 10.0
        I.execute(10)
        assert np.allclose(I.value, 10.0)
        assert np.allclose(I.output_port.value, 10.0)

        # reset function
        I.function.reset(5.0)
        assert np.allclose(I.function.value, 5.0)
        assert np.allclose(I.value, 10.0)
        assert np.allclose(I.output_ports[0].value, 10.0)

        # reset function without value spec
        I.function.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 10.0)
        assert np.allclose(I.output_ports[0].value, 10.0)

        # reset mechanism
        I.reset(4.0)
        assert np.allclose(I.function.value, 4.0)
        assert np.allclose(I.value, 4.0)
        assert np.allclose(I.output_ports[0].value, 4.0)

        I.execute(1)
        assert np.allclose(I.value, 5.0)
        assert np.allclose(I.output_ports[0].value, 5.0)

        # reset mechanism without value spec
        I.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.0)
        assert np.allclose(I.output_ports[0].value, 0.0)

    def test_Adaptive_valid(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=0.5
            ),
        )

        #  returns (1-rate)*previous_value + rate*variable + noise
        # so in this case, returns 0.5*0 + 0.5*10 + 0 = 5.0
        I.execute(10)
        assert np.allclose(I.value, 5.0)
        assert np.allclose(I.output_port.value, 5.0)

        # reset function
        I.function.reset(1.0)
        assert np.allclose(I.function.value, 1.0)
        assert np.allclose(I.value, 5.0)
        assert np.allclose(I.output_ports[0].value, 5.0)

        # reset function without value spec
        I.function.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 5.0)
        assert np.allclose(I.output_ports[0].value, 5.0)

        # reset mechanism
        I.reset(2.0)
        assert np.allclose(I.function.value, 2.0)
        assert np.allclose(I.value, 2.0)
        assert np.allclose(I.output_ports[0].value, 2.0)

        I.execute(1.0)
        #  (1-0.5)*2.0 + 0.5*1.0 + 0 = 1.5
        assert np.allclose(I.value, 1.5)
        assert np.allclose(I.output_ports[0].value, 1.5)

        # reset mechanism without value spec
        I.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.0)
        assert np.allclose(I.output_ports[0].value, 0.0)

    def test_Accumulator_warning(self):
        regexp = r"AccumulatorIntegrator does not use its variable;  value passed .* will be ignored"
        with pytest.warns(UserWarning, match=regexp):
            I = AccumulatorIntegrator()
            I(1000)

    def test_Accumulator_valid(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(increment=1.0),
        )

        #  returns previous_value + rate + noise
        # so in this case, returns 0.0 + 1.0
        I.execute(1000)
        assert np.allclose(I.value, 1.0)
        assert np.allclose(I.output_port.value, 1.0)

        # reset function
        I.function.reset(2.0)
        assert np.allclose(I.function.value, 2.0)
        assert np.allclose(I.value, 1.0)
        assert np.allclose(I.output_ports[0].value, 1.0)

        # reset function without value spec
        I.function.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 1.0)
        assert np.allclose(I.output_ports[0].value, 1.0)

        # reset mechanism
        I.reset(2.0)
        assert np.allclose(I.function.value, 2.0)
        assert np.allclose(I.value, 2.0)
        assert np.allclose(I.output_ports[0].value, 2.0)

        I.execute(1000)
        #  2.0 + 1.0 = 3.0
        assert np.allclose(I.value, 3.0)
        assert np.allclose(I.output_ports[0].value, 3.0)

        # reset mechanism without value spec
        I.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.0)
        assert np.allclose(I.output_ports[0].value, 0.0)

    def test_OU_valid(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=OrnsteinUhlenbeckIntegrator(),
        )

        # previous_value + (decay * previous_value - rate * variable) * time_step_size + noise
        # decay=1.0, initializer=0.0, rate=1.0, time_step_size=1.0, noise=0.0
        # returns 0.0 + (1.0*0.0 - 1.0*10.0*1.0) + 0.0 = -10.0
        I.execute(2.0)
        assert np.allclose(I.value[0], -2.0)
        assert np.allclose(I.output_port.value, -2.0)

        # reset function
        I.function.reset(5.0, 0.0)
        assert np.allclose(I.function.value[0], 5.0)
        assert np.allclose(I.value[0], -2.0)
        assert np.allclose(I.output_ports[0].value, -2.0)

        # reset function without value spec
        I.function.reset()
        assert np.allclose(I.function.value[0], 0.0)
        assert np.allclose(I.value[0], -2.0)
        assert np.allclose(I.output_ports[0].value, -2.0)

        # reset mechanism
        I.reset(4.0, 0.0)
        assert np.allclose(I.function.value[0], 4.0)
        assert np.allclose(I.value[0], 4.0)
        assert np.allclose(I.output_ports[0].value, 4.0)

        I.execute(1.0)
        # 4.0 + (1.0 * 4.0 - 1.0 * 1.0) * 1.0 = 4 + 3 = 7
        assert np.allclose(I.value[0], 7.0)
        assert np.allclose(I.output_ports[0].value, 7.0)

        # reset mechanism without value spec
        I.reset()
        assert np.allclose(I.function.value[0], 0.0)
        assert np.allclose(I.value[0][0], 0.0)
        assert np.allclose(I.output_ports[0].value[0], 0.0)

    def test_Accumulator_valid(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(increment=0.1),
        )

        #  returns previous_value * rate + noise + increment
        # initializer = 0.0, rate = 1.0, noise = 0.0, increment = 0.1
        # returns 0.0*1.0 + 0.0 + 0.1 = 0.1
        I.execute(10000)
        assert np.allclose(I.value, 0.1)
        assert np.allclose(I.output_port.value, 0.1)

        # reset function
        I.function.reset(2.0)
        assert np.allclose(I.function.value, 2.0)
        assert np.allclose(I.value, 0.1)
        assert np.allclose(I.output_ports[0].value, 0.1)

        # reset function without value spec
        I.function.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.1)
        assert np.allclose(I.output_ports[0].value, 0.1)

        # reset mechanism
        I.reset(5.0)
        assert np.allclose(I.function.value, 5.0)
        assert np.allclose(I.value, 5.0)
        assert np.allclose(I.output_ports[0].value, 5.0)

        I.execute(10000)
        #  5.0 * 1.0 + 0.0 + 0.1
        assert np.allclose(I.value, 5.1)
        assert np.allclose(I.output_ports[0].value, 5.1)

        # reset mechanism without value spec
        I.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.0)
        assert np.allclose(I.output_ports[0].value, 0.0)

    def test_LCIIntegrator_valid(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=LeakyCompetingIntegrator(leak=1),
        )

        # previous_value + (new_value - rate*previous_value)*time_step_size + noise
        # initializer=0.0, rate=1.0, time_step_size=0.1, noise=0.0
        # returns 0.0 + (1.0*0.0 + 2.0)*0.1 = 2.0
        I.execute(2.0)
        assert np.allclose(I.value, 0.2)
        assert np.allclose(I.output_port.value, 0.2)

        # reset function
        I.function.reset(5.0)
        assert np.allclose(I.function.value, 5.0)
        assert np.allclose(I.value, 0.2)
        assert np.allclose(I.output_ports[0].value, 0.2)

        # reset function without value spec
        I.function.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.2)
        assert np.allclose(I.output_ports[0].value, 0.2)

        # reset mechanism
        I.reset(4.0)
        assert np.allclose(I.function.value, 4.0)
        assert np.allclose(I.value, 4.0)
        assert np.allclose(I.output_ports[0].value, 4.0)

        I.execute(1.0)
        # 4.0 + (1.0*4.0 + 1.0)*0.1 + 0.0
        assert np.allclose(I.value, 3.7)
        assert np.allclose(I.output_ports[0].value, 3.7)

        # reset mechanism without value spec
        I.reset()
        assert np.allclose(I.function.value, 0.0)
        assert np.allclose(I.value, 0.0)
        assert np.allclose(I.output_ports[0].value, 0.0)

    def test_reset_not_integrator(self):

        with pytest.raises(MechanismError) as err_txt:
            I_not_integrator = IntegratorMechanism(function=Linear)
            I_not_integrator.execute(1.0)
            I_not_integrator.reset(0.0)
        assert "not allowed because this Mechanism is not stateful;" in str(err_txt.value)
        assert "it does not have an accumulator to reset" in str(err_txt.value)


VECTOR_SIZE=4

def _get_mechanism_execution(mech, mode):
    if mode == 'Python':
        def ex(variable):
            mech.execute(variable)
            return mech.output_values
        return ex
    elif mode == 'LLVM':
        e = pnlvm.execution.MechExecution(mech)
        return e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.MechExecution(mech)
        return e.cuda_execute
    assert False, "Unknown execution mode: {}".format(mode)

class TestIntegratorFunctions:

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_simple_integrator(self):
        I = IntegratorMechanism(
            function=SimpleIntegrator(
                initializer=10.0,
                rate=5.0,
                offset=10,
            )
        )
        # P = Process(pathway=[I])
        val = I.execute(1)
        assert val == 25

    @pytest.mark.mimo
    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_integrator_multiple_input(self, benchmark, mode):
        I = IntegratorMechanism(
            function=Linear(slope=2.0, intercept=1.0),
            default_variable=[[1], [2]],
            input_ports=['a', 'b'],
        )
        ex = _get_mechanism_execution(I, mode)

        val = ex([[1], [2]])
        assert np.allclose(val, [[3]])
        if benchmark.enabled:
            benchmark(ex, [[1], [2]])

    @pytest.mark.mimo
    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_integrator_multiple_output(self, benchmark, mode):
        I = IntegratorMechanism(
            default_variable=[5],
            output_ports=[{pnl.VARIABLE: (pnl.OWNER_VALUE, 0)}, 'c'],
        )
        ex = _get_mechanism_execution(I, mode)

        val = ex([5])
        assert np.allclose(val, [[2.5], [2.5]])
        if benchmark.enabled:
            benchmark(ex, [5])

    @pytest.mark.mimo
    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_integrator_multiple_input_output(self, benchmark, mode):
        I = IntegratorMechanism(
            function=Linear(slope=2.0, intercept=1.0),
            default_variable=[[1], [2]],
            input_ports=['a', 'b'],
            output_ports=[{pnl.VARIABLE: (pnl.OWNER_VALUE, 1)},
                          {pnl.VARIABLE: (pnl.OWNER_VALUE, 0)}],
        )
        ex = _get_mechanism_execution(I, mode)

        val = ex([[1], [2]])
        assert np.allclose(val, [[5], [3]])
        if benchmark.enabled:
            benchmark(ex, [[1], [2]])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_FitzHughNagumo_simple_scalar(self, benchmark, mode):
        var = [1.0]
        I = IntegratorMechanism(name="I",
                                default_variable=[var],
                                function=FitzHughNagumoIntegrator())
        ex = _get_mechanism_execution(I, mode)

        val = ex(var)
        assert np.allclose(val[0], [0.05127053])
        if benchmark.enabled:
            benchmark(ex, var)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_FitzHughNagumo_simple_vector(self, benchmark, mode):
        var = [1.0, 3.0]
        I = IntegratorMechanism(name="I",
                                default_variable=var,
                                function=FitzHughNagumoIntegrator)
        ex = _get_mechanism_execution(I, mode)

        val = ex(var)
        assert np.allclose(val[0], [0.05127053, 0.15379818])
        if benchmark.enabled:
            benchmark(ex, var)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_integrator(self, benchmark, mode):
        I = IntegratorMechanism(
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(slope=5.0))
        ex = _get_mechanism_execution(I, mode)

        val = benchmark(ex, [1.0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[5.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_accumulator_integrator(self):
        I = IntegratorMechanism(
            function=AccumulatorIntegrator(
                initializer=10.0,
                increment=15.0,
            )
        )
        # P = Process(pathway=[I])
        # accumulator integrator does not use input value (variable)

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

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_adaptive_integrator(self):
        I = IntegratorMechanism(
            function=AdaptiveIntegrator(
                initializer=10.0,
                rate=0.5,
                offset=10,
            )
        )
        # P = Process(pathway=[I])
        # 10*0.5 + 1*0.5 + 10
        val = I.execute(1)
        assert val == 15.5

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_drift_diffusion_integrator(self):
        I = IntegratorMechanism(
            function=DriftDiffusionIntegrator(
                initializer=10.0,
                rate=10,
                time_step_size=0.5,
                offset=10,
            )
        )
        # P = Process(pathway=[I])
        # 10 + 10*0.5 + 0 + 10 = 25
        val = I.execute(1)
        assert np.allclose([[[25.]], [[0.5]]], val)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
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
        # P = Process(pathway=[I])
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

# COMMENTED OUT UNTIL OU INTEGRATOR IS VALIDATED
        # assert (val, val2) == (13.4375, 17.734375)

# COMMENTED OUT UNTIL OU INTEGRATOR IS VALIDATED
    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_ornstein_uhlenbeck_integrator_time(self):
        OU = IntegratorMechanism(
            function=OrnsteinUhlenbeckIntegrator(
                initializer=10.0,
                rate=10,
                time_step_size=0.2,
                starting_point=0.5,
                decay=0.1,
                offset=10,
            )
        )
        time_0 = OU.function.previous_time  # t_0  = 0.5
        # np.testing.assert_allclose(time_0, [0.5], atol=1e-08)

        OU.execute(10)
        time_1 = OU.function.previous_time  # t_1  = 0.5 + 0.2 = 0.7
        # np.testing.assert_allclose(time_1, [0.7], atol=1e-08)

        for i in range(11):  # t_11 = 0.7 + 10*0.2 = 2.7
            OU.execute(10)
        time_12 = OU.function.previous_time # t_12 = 2.7 + 0.2 = 2.9
        # np.testing.assert_allclose(time_12, [2.9], atol=1e-08)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.benchmark(group="IntegratorMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_integrator_no_function(self, benchmark, mode):
        I = IntegratorMechanism()
        ex = _get_mechanism_execution(I, mode)

        val = ex([10])
        assert np.allclose(val, [[5.0]])
        if benchmark.enabled:
            benchmark(ex, [10])

class TestIntegratorInputs:
    # Part 1: VALID INPUT:

    # input = float

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_input_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
            )
        )
        # P = Process(pathway=[I])
        val = float(I.execute(10.0))
        assert val == 10.0

    # input = list of length 1

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_input_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
            )
        )
        # P = Process(pathway=[I])
        val = float(I.execute([10.0]))
        assert val == 10.0

    # input = list of length 5

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_input_list_len_5(self):

        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0, 0],
            function=SimpleIntegrator(
            )
        )
        # P = Process(pathway=[I])
        val = I.execute([10.0, 5.0, 2.0, 1.0, 0.0])[0]
        expected_output = [10.0, 5.0, 2.0, 1.0, 0.0]

        for i in range(len(expected_output)):
            v = val[i]
            e = expected_output[i]
            np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    # input = numpy array of length 5

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_input_array_len_5(self):

        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0, 0],
            function=SimpleIntegrator(
            )
        )
        # P = Process(pathway=[I])
        input_array = np.array([10.0, 5.0, 2.0, 1.0, 0.0])
        val = I.execute(input_array)[0]
        expected_output = [10.0, 5.0, 2.0, 1.0, 0.0]

        for i in range(len(expected_output)):
            v = val[i]
            e = expected_output[i]
            np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    # Part 2: INVALID INPUT

    # input = list of length > default length

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_input_array_greater_than_default(self):

        with pytest.raises(MechanismError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                default_variable=[0, 0, 0]
            )
            # P = Process(pathway=[I])
            I.execute([10.0, 5.0, 2.0, 1.0, 0.0])
        assert "does not match required length" in str(error_text.value)

    # input = list of length < default length

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_input_array_less_than_default(self):

        with pytest.raises(MechanismError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                default_variable=[0, 0, 0, 0, 0]
            )
            # P = Process(pathway=[I])
            I.execute([10.0, 5.0, 2.0])
        assert "does not match required length" in str(error_text.value)


# ======================================= RATE TESTS ============================================
class TestIntegratorRate:
    # VALID RATE:

    # rate = float, integration_type = simple

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_simple_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                rate=5.0
            )
        )
        # P = Process(pathway=[I])
        val = float(I.execute(10.0))
        assert val == 50.0

    # rate = float, increment = float, integration_type = accumulator

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_accumulator_rate_and_increment_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(
                rate=2.0,
                increment=3.0
            )
        )
        # P = Process(pathway=[I])
        float(I.execute())
        val = float(I.execute())
        assert val == 9.0

    # rate = float, integration_type = diffusion

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_diffusion_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=DriftDiffusionIntegrator(
                rate=5.0
            )
        )
        # P = Process(pathway=[I])
        val = I.execute(10.0)
        assert np.allclose([[[50.0]], [[1.0]]], val)

    # rate = list, integration_type = simple

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_simple_rate_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0],
            function=SimpleIntegrator(
                rate=[5.0, 5.0, 5.0]
            )
        )
        # P = Process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [50.0, 50.0, 50.0]

    # rate = float, increment = list, integration_type = accumulator

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_accumulator_rate_float_increment_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(
                rate = 2.0,
                increment=[4.0, 5.0, 6.0]
            )
        )
        # P = Process(pathway=[I])
        list(I.execute([10.0, 10.0, 10.0])[0])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [12.0, 15.0, 18.0]

    # rate = float, increment = list, integration_type = accumulator

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_accumulator_rate_list_increment_float(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(
                rate = [2.0, 3.0, 4.0],
                increment=5.0
            )
        )
        # P = Process(pathway=[I])
        list(I.execute([10.0, 10.0, 10.0])[0])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [15.0, 20.0, 25.0]

    # rate = list, increment = list, integration_type = accumulator

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_accumulator_rate_and_increment_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(
                rate = [1.0, 2.0, 3.0],
                increment=[4.0, 5.0, 6.0]
            )
        )
        # P = Process(pathway=[I])
        list(I.execute([10.0, 10.0, 10.0])[0])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [8.0, 15.0, 24.0]


    # rate = list, integration_type = diffusion

    # def test_integrator_type_diffusion_rate_list(self):
    #     I = IntegratorMechanism(
    #         default_variable=[0, 0, 0],
    #         name='IntegratorMechanism',
    #         function=DriftDiffusionIntegrator(
    #             rate=[5.0, 5.0, 5.0],
    #             threshold=[100.0, 100.0, 100.0]
    #         )
    #     )
    #     # P = Process(pathway=[I])
    #     val = list(I.execute([10.0, 10.0, 10.0])[0])
    #     assert val == [50.0, 50.0, 50.0]

    # rate = list, integration_type = diffusion

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_adaptive_rate_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=[0.5, 0.5, 0.5]
            )
        )
        # P = Process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [5.0, 5.0, 5.0]

    # rate = float, integration_type = modulatory

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_adaptive_rate_float_input_list(self):
        I = IntegratorMechanism(
            default_variable=[0, 0, 0],
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=0.5
            )
        )
        # P = Process(pathway=[I])
        val = list(I.execute([10.0, 10.0, 10.0])[0])
        assert val == [5.0, 5.0, 5.0]

    # rate = float, integration_type = modulatory

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_adaptive_rate_float(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                rate=0.5
            )
        )
        # P = Process(pathway=[I])
        val = list(I.execute(10.0))
        assert val == [5.0]

    # INVALID RATE:

    # rate = list, execute float, integration_type = simple

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_adaptive_variable_and_rate_conflict(self):
        with pytest.raises(IntegratorMechanismError) as error_text:
            I = IntegratorMechanism(
                    name='IntegratorMechanism',
                    default_variable=[0],
                    function=AdaptiveIntegrator(rate=[0.5, 0.6])
            )
        error_msg_a = "Shape of 'variable' for function specified for IntegratorMechanism (AdaptiveIntegrator Function"
        error_msg_b = "(2,)) does not match the shape of the 'default_variable' specified for the 'Mechanism'."
        assert error_msg_a in str(error_text.value)
        assert error_msg_b in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_simple_rate_list_input_float(self):
        with pytest.raises(MechanismError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                function=SimpleIntegrator(
                    rate=[5.0, 5.0, 5.0]
                )
            )
            result = I.execute(10.0)
            float(result)
        returned_error_msg = str(error_text.value)
        # Need to break up error message because length value returned ([10.0] vs. [ 10.0]) may differ by Python vers.
        error_msg_a1 = 'Length (1) of input'
        error_msg_a2 = 'does not match required length (3) for input to '
        error_msg_b = "InputPort 'InputPort-0' of IntegratorMechanism"
        assert [msg in returned_error_msg for msg in {error_msg_a1, error_msg_a2, '10'}]
        assert error_msg_b in str(error_text.value)

    # rate = list len 2, incrment = list len 3, integration_type = accumulator

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_type_accumulator_increment_list_input_float(self):
        with pytest.raises(FunctionError) as error_text:
            I = IntegratorMechanism(
                name='IntegratorMechanism',
                function=AccumulatorIntegrator(
                    rate=[1.0, 2.0],
                    increment=[3.0, 4.0, 5.0]
                ))

        error_msg_a = "The parameters with len>1 specified for AccumulatorIntegrator Function-0 "
        error_msg_b = "(['rate', 'increment']) don't all have the same length"
        assert error_msg_a in str(error_text.value)
        assert error_msg_b in str(error_text.value)


    # @pytest.mark.mechanism
    # @pytest.mark.integrator_mechanism
    # def test_integrator_type_diffusion_rate_list_input_float(self):
    #     with pytest.raises(FunctionError) as error_text:
    #         I = IntegratorMechanism(
    #             name='IntegratorMechanism',
    #             function=DriftDiffusionIntegrator(
    #                 rate=[5.0, 5.0, 5.0]
    #             )
    #         )
    #         # P = Process(pathway=[I])
    #         float(I.execute(10.0))
    #     assert "array specified for the rate parameter" in str(error_text.value)
    #     assert "must match the length" in str(error_text.value)
    #     assert "of the default input" in str(error_text.value)

    # @pytest.mark.mechanism
    # @pytest.mark.integrator_mechanism
    # def test_accumulator_integrator(self):
    #     I = IntegratorMechanism(
    #             function = AccumulatorIntegrator(
    #                 initializer = 10.0,
    #                 rate = 5.0,
    #                 increment= 1.0
    #             )
    #         )
    # #     P = Process(pathway=[I])

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


# ======================================= NOISE TESTS ============================================
class TestIntegratorNoise:

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_simple_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                noise=NormalDist()
            ),
        )

        val = float(I.execute(10))

        I.function.reset(5.0)

        val2 = float(I.execute(0))

        np.testing.assert_allclose(val, 11.00018002983055)
        np.testing.assert_allclose(val2, 7.549690404329112)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_simple_noise_fn_noise_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                noise=[NormalDist()]
            ),
        )
        val = float(I.execute(10))

        np.testing.assert_allclose(val, 10.302846)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_simple_noise_fn_noise_list_squeezed(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                default_variable=[[0, 0, 0]],
                noise=[NormalDist(seed=0), NormalDist(seed=0), NormalDist(seed=0)], # seed to check elementwise
            ),
        )
        val = I.execute([10, 10, 10])

        np.testing.assert_allclose(val, [[10.302846, 10.302846, 10.302846]])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_simple_noise_fn_noise_shaped(self):
        I = IntegratorMechanism(
            variable=[[0], [0], [0]],
            name='IntegratorMechanism',
            function=SimpleIntegrator(
                default_variable=[[0], [0], [0]],
                noise=NormalDist([[0], [0], [0]]),
            ),
        )
        val = I.execute([[10], [10], [10]])

        np.testing.assert_allclose(val, [[10.660535], [11.108879], [ 9.084011]])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_simple_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=SimpleIntegrator(
                noise=NormalDist(),
            ),
        )

        val = I.execute([10, 10, 10, 10])[0]

        np.testing.assert_allclose(val, [11.10887925, 9.0840107, 10.30157835, 10.65375815])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_accumulator_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AccumulatorIntegrator(
                noise=NormalDist()
            ),
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 1.00018)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_accumulator_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=AccumulatorIntegrator(
                noise=NormalDist(),
            ),
        )

        val = I.execute([10, 10, 10, 10])[0]
        np.testing.assert_allclose(val, [1.10887925, -0.9159893, 0.30157835, 0.65375815])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_adaptive_noise_fn(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=AdaptiveIntegrator(
                noise=NormalDist()
            ),
        )

        val = float(I.execute(10))

        np.testing.assert_allclose(val, 11.00018002983055)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_adaptive_noise_fn_var_list(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_variable=[0, 0, 0, 0],
            function=AdaptiveIntegrator(
                noise=NormalDist(),
            ),
        )

        val = I.execute([10, 10, 10, 10])[0]

        np.testing.assert_allclose(val, [11.10887925, 9.0840107, 10.30157835, 10.65375815])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_drift_diffusion_noise_val(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=DriftDiffusionIntegrator(
                noise=5.0,
            ),
        )

        val = I.execute(10.0)
        assert np.allclose(val, [[[4.29013944]], [[ 1.        ]]])

# COMMENTED OUT UNTIL OU INTEGRATOR IS VALIDATED
    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_integrator_ornstein_uhlenbeck_noise_val(self):
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            function=OrnsteinUhlenbeckIntegrator(
                noise=2.0,
                decay=0.5,
                initializer=1.0,
                rate=0.25
            ),
        )

        # val = 1.0 + 0.5 * (1.0 - 0.25 * 2.5) * 1.0 + np.sqrt(1.0 * 2.0) * np.random.normal()


        val = I.execute(2.5)

        # np.testing.assert_allclose(val, 4.356601554140335)

class TestStatefulness:

    def test_has_initializers(self):
        I = IntegratorMechanism()
        assert I.has_initializers
        assert hasattr(I, "reset_stateful_function_when")

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    @pytest.mark.parametrize('cond0, cond1, expected', [
        (pnl.Never(), pnl.AtTrial(2),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.75]), np.array([0.75])],
          [np.array([0.875]), np.array([0.5])],   # I2 resets at Trial 2
          [np.array([0.9375]), np.array([0.75])],
          [np.array([0.96875]), np.array([0.875])],
          [np.array([0.984375]), np.array([0.9375])],
          [np.array([0.9921875]), np.array([0.96875])]]),
        (pnl.Never(), pnl.AtTrialStart(),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.75]), np.array([0.5])],
          [np.array([0.875]), np.array([0.5])],
          [np.array([0.9375]), np.array([0.5])],
          [np.array([0.96875]), np.array([0.5])],
          [np.array([0.984375]), np.array([0.5])],
          [np.array([0.9921875]), np.array([0.5])]]),
        (pnl.AtPass(0), pnl.AtTrial(2),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.5]), np.array([0.75])],
          [np.array([0.5]), np.array([0.5])],   # I2 resets at Trial 2
          [np.array([0.5]), np.array([0.75])],
          [np.array([0.5]), np.array([0.875])],
          [np.array([0.5]), np.array([0.9375])],
          [np.array([0.5]), np.array([0.96875])]]),
        ], ids=lambda x: str(x) if isinstance(x, pnl.Condition) else "")
    def test_reset_stateful_function_when_composition(self, mode, cond0, cond1, expected):
        I1 = pnl.IntegratorMechanism()
        I2 = pnl.IntegratorMechanism()
        I1.reset_stateful_function_when = cond0
        I2.reset_stateful_function_when = cond1
        C = pnl.Composition()
        C.add_node(I1)
        C.add_node(I2)

        C.run(inputs={I1: [[1.0]], I2: [[1.0]]}, num_trials=7, bin_execute=mode)

        assert np.allclose(expected, C.results)

    def test_reset_stateful_function_when(self):
        I1 = IntegratorMechanism()
        I2 = IntegratorMechanism()
        I2.reset_stateful_function_when = AtTrial(2)
        C = Composition(pathways=[[I1], [I2]])

        C.run(inputs={I1: [[1.0]],
                      I2: [[1.0]]},
              num_trials=7,
              reset_stateful_functions_when=AtTrial(3))

        expected_results = [[np.array([0.5]), np.array([0.5])],
                            [np.array([0.75]), np.array([0.75])],
                            [np.array([0.875]), np.array([0.5])],   # I2 resets at Trial 2
                            [np.array([0.5]), np.array([0.75])],    # I1 resets at Trial 3
                            [np.array([0.75]), np.array([0.875])],
                            [np.array([0.875]), np.array([0.9375])],
                            [np.array([0.9375]), np.array([0.96875])]]

        assert np.allclose(expected_results, C.results)


class TestDualAdaptiveIntegrator:

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_utility_integrator_default(self):
        # default params:
        # initial_short_term_avg = 0.0
        # initial_long_term_avg = 0.0
        # short_term_rate = 1.0
        # long_term_rate = 1.0

        U = IntegratorMechanism(
            name = "DualAdaptiveIntegrator",
            function=DualAdaptiveIntegrator(
            )

        )

        engagement = []
        short_term_util = []
        long_term_util = []
        for i in range(50):
            engagement.append(U.execute([1])[0][0])
            short_term_util.append(U.function.short_term_logistic[0])
            long_term_util.append(U.function.long_term_logistic[0])
        print("engagement = ", engagement)
        print("short_term_util = ", short_term_util)
        print("long_term_util = ", long_term_util)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_utility_integrator_short_minus_long(self):
        # default params:
        # initial_short_term_avg = 0.0
        # initial_long_term_avg = 0.0
        # short_term_rate = 1.0
        # long_term_rate = 1.0

        U = IntegratorMechanism(
            name = "DualAdaptiveIntegrator",
            function=DualAdaptiveIntegrator(
                operation=pnl.S_MINUS_L
            )

        )

        engagement = []
        short_term_util = []
        long_term_util = []
        for i in range(50):
            engagement.append(U.execute([1])[0][0])
            short_term_util.append(U.function.short_term_logistic[0])
            long_term_util.append(U.function.long_term_logistic[0])
        print("engagement = ", engagement)
        print("short_term_util = ", short_term_util)
        print("long_term_util = ", long_term_util)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    def test_utility_integrator_short_plus_long(self):
        # default params:
        # initial_short_term_avg = 0.0
        # initial_long_term_avg = 0.0
        # short_term_rate = 1.0
        # long_term_rate = 1.0

        U = IntegratorMechanism(
            name = "DualAdaptiveIntegrator",
            function=DualAdaptiveIntegrator(
                operation=pnl.SUM
            )

        )

        engagement = []
        short_term_util = []
        long_term_util = []
        for i in range(50):
            engagement.append(U.execute([1])[0][0])
            short_term_util.append(U.function.short_term_logistic[0])
            long_term_util.append(U.function.long_term_logistic[0])
        print("engagement = ", engagement)
        print("short_term_util = ", short_term_util)
        print("long_term_util = ", long_term_util)

    # @pytest.mark.mechanism
    # @pytest.mark.integrator_mechanism
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


    # @pytest.mark.mechanism
    # @pytest.mark.integrator_mechanism
    # def test_FitzHughNagumo_gilzenrat(self):
    #
    #     F = IntegratorMechanism(
    #         name='IntegratorMech-FitzHughNagumoFunction',
    #         function=FitzHughNagumoIntegrator(
    #             time_step_size=0.1,
    #             initial_v=0.2,
    #             initial_w=0.0,
    #             t_0=0.0,
    #             time_accumulator_v=1.0,
    #             a_v=-1.0,
    #             b_v=1.5,
    #             c_v=-0.5,
    #             d_v=0.0,
    #             e_v=-1.0,
    #             f_v=0.0,
    #             time_accumulator_w=100.0,
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
    # def test_FitzHughNagumo_gilzenrat_low_electrotonic_coupling(self):
    #
    #     F = IntegratorMechanism(
    #         name='IntegratorMech-FitzHughNagumoFunction',
    #         function=FitzHughNagumoIntegrator(
    #             time_step_size=0.1,
    #             initial_v=0.2,
    #             initial_w=0.0,
    #             t_0=0.0,
    #             time_accumulator_v=1.0,
    #             a_v=-1.0,
    #             b_v=0.5,
    #             c_v=0.5,
    #             d_v=0.0,
    #             e_v=-1.0,
    #             f_v=0.0,
    #             electrotonic_coupling=0.55,
    #             time_accumulator_w=100.0,
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
