import numpy as np
import pytest

import psyneulink as pnl

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.nonstateful.distributionfunctions import DriftDiffusionAnalytical, NormalDist
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.stateful.integratorfunctions import DriftDiffusionIntegrator
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.condition import Never, WhenFinished
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.globals.keywords import IDENTITY_MATRIX, FULL_CONNECTIVITY_MATRIX
from psyneulink.core.globals.utilities import _SeededPhilox
from psyneulink.library.components.mechanisms.processing.integrator.ddm import \
    ARRAY, DDM, DDMError, DECISION_VARIABLE_ARRAY, SELECTED_INPUT_ARRAY, DECISION_OUTCOME

class TestReset:

    def test_valid(self):
        D = DDM(
            name='DDM',
            function=DriftDiffusionIntegrator(seed=0, time_step_size=1.0),
            execute_until_finished=False,
        )

        #  returns previous_value + rate * variable * time_step_size  + noise
        #  0.0 + 1.0 * 1.0 * 1.0 + 0.0
        D.execute(1.0)
        np.testing.assert_allclose(np.asfarray(D.value),  [[1.0], [1.0]])
        np.testing.assert_allclose(D.output_ports[0].value[0], 1.0)
        np.testing.assert_allclose(D.output_ports[1].value[0], 1.0)

        # reset function
        D.function.reset(2.0, 0.1)
        np.testing.assert_allclose(D.function.value[0], 2.0)
        np.testing.assert_allclose(D.function.parameters.previous_value.get(), 2.0)
        np.testing.assert_allclose(D.function.previous_time, 0.1)
        np.testing.assert_allclose(np.asfarray(D.value),  [[1.0], [1.0]])
        np.testing.assert_allclose(D.output_ports[0].value[0], 1.0)
        np.testing.assert_allclose(D.output_ports[1].value[0], 1.0)

        # reset function without value spec
        D.function.reset()
        np.testing.assert_allclose(D.function.value[0], 0.0)
        np.testing.assert_allclose(D.function.parameters.previous_value.get(), 0.0)
        np.testing.assert_allclose(D.function.previous_time, 0.0)
        np.testing.assert_allclose(np.asfarray(D.value), [[1.0], [1.0]])
        np.testing.assert_allclose(D.output_ports[0].value[0], 1.0)
        np.testing.assert_allclose(D.output_ports[1].value[0], 1.0)

        # reset mechanism
        D.reset(2.0, 0.1)
        np.testing.assert_allclose(D.function.value[0], 2.0)
        np.testing.assert_allclose(D.function.parameters.previous_value.get(), 2.0)
        np.testing.assert_allclose(D.function.previous_time, 0.1)
        np.testing.assert_allclose(np.asfarray(D.value), [[2.0], [0.1]])
        np.testing.assert_allclose(D.output_ports[0].value, 2.0)
        np.testing.assert_allclose(D.output_ports[1].value, 0.1)

        D.execute(1.0)
        #  2.0 + 1.0 = 3.0 ; 0.1 + 1.0 = 1.1
        np.testing.assert_allclose(np.asfarray(D.value), [[3.0], [1.1]])
        np.testing.assert_allclose(D.output_ports[0].value[0], 3.0)
        np.testing.assert_allclose(D.output_ports[1].value[0], 1.1)

        # reset mechanism without value spec
        D.reset()
        np.testing.assert_allclose(D.function.value[0], 0.0)
        np.testing.assert_allclose(D.function.parameters.previous_value.get(), 0.0)
        np.testing.assert_allclose(D.function.previous_time, 0.0)
        np.testing.assert_allclose(D.output_ports[0].value[0], 0.0)
        np.testing.assert_allclose(D.output_ports[1].value[0], 0.0)

        # reset only decision variable
        D.function.initializer = 1.0
        D.function.non_decision_time.base = 0.0
        D.reset()
        np.testing.assert_allclose(D.function.value[0], 1.0)
        np.testing.assert_allclose(D.function.parameters.previous_value.get(), 1.0)
        np.testing.assert_allclose(D.function.previous_time, 0.0)
        np.testing.assert_allclose(D.output_ports[0].value[0], 1.0)
        np.testing.assert_allclose(D.output_ports[1].value[0], 0.0)


class TestThreshold:
    def test_threshold_param(self):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=10.0, time_step_size=1.0))

        assert D.function.threshold.base == 10.0

        D.function.threshold.base = 5.0
        assert D.function.threshold.base == 5.0

    def test_threshold_sets_is_finished(self):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=5.0, time_step_size=1.0),
                execute_until_finished=False,
                reset_stateful_function_when=Never())
        D.execute(2.0)  # 2.0 < 5.0
        assert not D.is_finished()

        D.execute(2.0)  # 4.0 < 5.0
        assert not D.is_finished()

        D.execute(2.0)   # 5.0 = threshold
        assert D.is_finished()

    @pytest.mark.ddm_mechanism
    @pytest.mark.mechanism
    @pytest.mark.benchmark(group="DDM")
    @pytest.mark.parametrize("variable, expected", [
        (2., [2.0, 4.0, 5.0, 5.0, 5.0]),
        (-2., [-2.0, -4.0, -5.0, -5.0, -5.0]),
        ], ids=["POSITIVE", "NEGATIVE"])
    def test_threshold_stops_accumulation(self, mech_mode, variable, expected, benchmark):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=5.0, time_step_size=1.0),
                execute_until_finished=False)
        ex = pytest.helpers.get_mech_execution(D, mech_mode)

        decision_variables = []
        time_points = []
        results = []
        for i in range(4):
            results.append(ex([variable]))

        results.append(benchmark(ex,[variable]))

        # decision variable accumulation stops
        # time accumulation does not stop
        np.testing.assert_allclose(results, [[[b], [a + 1.0]] for a,b in enumerate(expected)])

    # def test_threshold_stops_accumulation_multiple_variables(self):
    #     D = IntegratorMechanism(name='DDM',
    #                             default_variable=[[0,0,0]],
    #                             function=DriftDiffusionIntegrator(threshold=[5.0, 5.0, 10.0],
    #                                                               initializer=[[0.0, 0.0, 0.0]],
    #                                                               rate=[2.0, -2.0, -2.0 ]))
    #     decision_variables_a = []
    #     decision_variables_b = []
    #     decision_variables_c = []
    #     for i in range(5):
    #         output = D.execute([2.0, 2.0, 2.0])
    #         print(output)
    #         decision_variables_a.append(output[0][0])
    #         decision_variables_b.append(output[0][1])
    #         decision_variables_c.append(output[0][2])
    #
    #     # decision variable accumulation stops
    #     np.testing.assert_allclose(decision_variables_a, [2.0, 4.0, 5.0, 5.0, 5.0])


    @pytest.mark.composition
    def test_is_finished_stops_composition(self):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=10.0, time_step_size=1.0),
                execute_until_finished=False,
                reset_stateful_function_when=Never())
        C = Composition(pathways=[D], reset_stateful_function_when=Never())
        C.run(inputs={D: 2.0},
              termination_processing={TimeScale.TRIAL: WhenFinished(D)})

        # decision variable's value should match threshold
        assert D.parameters.value.get(C)[0] == 10.0
        # it should have taken 5 executions (and time_step_size = 1.0)
        assert D.parameters.value.get(C)[1] == 5.0

@pytest.mark.composition
class TestInputPorts:

    def test_regular_input_mode(self):
        input_mech = ProcessingMechanism(input_shapes=2)
        ddm = DDM(
            function=DriftDiffusionAnalytical(),
            output_ports=[SELECTED_INPUT_ARRAY, DECISION_VARIABLE_ARRAY],
            name='DDM'
        )
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[input_mech, [[1],[-1]], ddm])
        result = comp.run(inputs={input_mech:[1,0]})
        np.testing.assert_allclose(ddm.output_ports[0].value, [1])
        np.testing.assert_allclose(ddm.output_ports[1].value, [1])
        np.testing.assert_allclose(ddm.value,
                           [[1.00000000e+00], [1.19932930e+00], [9.99664650e-01], [3.35350130e-04],
                            [1.19932930e+00], [2.48491374e-01], [1.48291009e+00], [1.19932930e+00],
                            [2.48491374e-01], [1.48291009e+00]])
        np.testing.assert_allclose(result, [[1.0], [1.0]])

    def test_array_mode(self):
        input_mech = ProcessingMechanism(input_shapes=2)
        ddm = DDM(
            input_format=ARRAY,
            function=DriftDiffusionAnalytical(),
            output_ports=[SELECTED_INPUT_ARRAY, DECISION_VARIABLE_ARRAY],
            name='DDM'
        )
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[input_mech, ddm])
        result = comp.run(inputs={input_mech:[1,0]})
        np.testing.assert_allclose(ddm.output_ports[0].value, [1,0])
        np.testing.assert_allclose(ddm.output_ports[1].value, [1,0])
        np.testing.assert_allclose(ddm.value,
                           [[1.00000000e+00], [1.19932930e+00], [9.99664650e-01], [3.35350130e-04],
                            [1.19932930e+00], [2.48491374e-01], [1.48291009e+00], [1.19932930e+00],
                            [2.48491374e-01], [1.48291009e+00]])
        np.testing.assert_allclose(result, [[1., 0.], [1.0, 0.0]])

class TestOutputPorts:

    def test_selected_input_array(self):
        action_selection = DDM(
            input_format=ARRAY,
            function=DriftDiffusionAnalytical(
            ),
            output_ports=[SELECTED_INPUT_ARRAY],
            name='DDM'
        )
        with pytest.raises(MechanismError) as error:
            action_selection.execute([1.0])
        assert ("Shape ((1,)) of input ([1.]) does not match required shape ((1, 2)) "
                "for input to InputPort 'ARRAY' of DDM.") in str(error.value)
        action_selection.execute([1.0, 0.0])

    def test_decision_outcome_integrator(self):
        ddm = DDM(
            function=DriftDiffusionIntegrator(rate=0.5, threshold=0.5, non_decision_time=0.0, noise=0.0),
            output_ports=[DECISION_OUTCOME],
            name='DDM'
        )
        assert np.allclose(ddm.execute([10.0]), [[0.5], [1]]) and ddm.output_ports[0].value == [1.0]
        assert np.allclose(ddm.execute([-10.0]), [[-0.5], [2]]) and ddm.output_ports[0].value == [0.0]

    def test_decision_outcome_analytical(self):
        ddm = DDM(
            function=DriftDiffusionAnalytical(drift_rate=0.5, threshold=0.5, non_decision_time=0.0, noise=0.0001),
            output_ports=[DECISION_OUTCOME],
            name='DDM'
        )
        ddm.execute([10.0])
        assert ddm.output_ports[0].value == [1.0]
        ddm.execute([-10.0])
        assert ddm.output_ports[0].value == [0.0]


# ------------------------------------------------------------------------------------------------
# TEST 2
# function = Bogacz

@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark
@pytest.mark.parametrize('prng', ['Default', 'Philox'])
def test_DDM_Integrator_Bogacz(benchmark, mech_mode, prng):
    stim = 10
    T = DDM(
        name='DDM',
        function=DriftDiffusionAnalytical()
    )
    if prng == 'Philox':
        T.parameters.random_state.set(_SeededPhilox([0]))
    ex = pytest.helpers.get_mech_execution(T, mech_mode)

    ex(stim)
    val = benchmark(ex, stim)
    np.testing.assert_allclose(val, [[1.0], [0.3]])

# ------------------------------------------------------------------------------------------------
# # TEST 3
# # function = Navarro

# ******
# Requires matlab import
# ******


# def test_DDM_Integrator():
#     stim = 10
#     T = DDM(
#         name='DDM',
#         function=NavarroAndFuss()
#     )
#     val = T.execute(stim)
#     np.testing.assert_array_equal(val, [[10]])


# ======================================= NOISE TESTS ============================================

# VALID NOISE:

# ------------------------------------------------------------------------------------------------
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM")
@pytest.mark.parametrize("noise, expected", [
    (0., 20),
    (np.sqrt(0.5), 18.40852795454561),
    (np.sqrt(2.0), 16.817055909091223),
    ], ids=["0", "0.5", "2.0"])
def test_DDM_noise(mech_mode, benchmark, noise, expected):
    T = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=noise,
            rate=1.0,
            time_step_size=1.0
        ),
        execute_until_finished=False
    )
    ex = pytest.helpers.get_mech_execution(T, mech_mode)

    ex([10])
    val = benchmark(ex, [10])
    np.testing.assert_allclose(val[0][0], expected)

# ------------------------------------------------------------------------------------------------

# INVALID NOISE:

# ------------------------------------------------------------------------------------------------
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.parametrize("noise", [
    2, NormalDist(),
    ], ids=["int", "functions"])
def test_DDM_noise_invalid(noise):
    with pytest.raises(FunctionError) as error_text:
        stim = 10
        T = DDM(
            name='DDM',
            function=DriftDiffusionIntegrator(

                noise=noise,
                rate=1.0,
                time_step_size=1.0
            ),
        )
        T.execute(stim)
    assert "DriftDiffusionIntegrator requires noise parameter to be a float" in str(error_text.value)

# ======================================= INPUT TESTS ============================================

# VALID INPUTS:

# ------------------------------------------------------------------------------------------------
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.parametrize("stim", [
    10, 10.0, [10],
    ], ids=["int", "float", "list"])
def test_DDM_input(stim):
    T = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=1.0,
            time_step_size=1.0
        ),
        execute_until_finished=False,
    )
    val = T.execute(stim)
    np.testing.assert_array_equal(val, [[10], [1]])

# ------------------------------------------------------------------------------------------------

# INVALID INPUTS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# input = List len 2


def test_DDM_input_list_len_2():
    with pytest.raises(DDMError) as error_text:
        stim = [10, 10]
        T = DDM(
            name='DDM',
            default_variable=[0, 0],
            function=DriftDiffusionIntegrator(

                noise=0.0,
                rate=1.0,
                time_step_size=1.0
            ),
            execute_until_finished=False,
        )
        T.execute(stim)
    assert "single numeric item" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# input = Fn

# ******
# Should functions be caught in validation, rather than with TypeError [just check callable()]?
# So that functions cause the same error as lists (see TEST 1 above)
# ******


def test_DDM_input_fn():
    with pytest.raises(MechanismError) as error_text:
        stim = NormalDist()
        T = DDM(
            name='DDM',
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=1.0,
                time_step_size=1.0
            ),
            execute_until_finished=False,
        )
        T.execute(stim)
    assert 'Input to \'DDM\' ([(NormalDist Normal Distribution Function' in str(error_text.value)
    assert 'is incompatible with its corresponding InputPort (DDM[InputPort-0]): ' \
           '\'unsupported operand type(s) for *: \'NormalDist\' and \'float\'.\'' in str(error_text.value)

# ======================================= RATE TESTS ============================================

# VALID RATES:

@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM")
@pytest.mark.parametrize("rate, expected", [
    (5, 100), (5., 100), ([5], 100), (-5.0, -100),
    ], ids=["int", "float", "list", "negative"])
# ******
# Should negative pass?
# ******
def test_DDM_rate(benchmark, rate, expected, mech_mode):
    stim = [10]
    T = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=rate,
            time_step_size=1.0
        ),
        execute_until_finished=False,
    )
    ex = pytest.helpers.get_mech_execution(T, mech_mode)

    ex(stim)
    val = benchmark(ex, stim)
    np.testing.assert_array_equal(val, [[expected], [2]])

# ------------------------------------------------------------------------------------------------
# INVALID RATES:

# ------------------------------------------------------------------------------------------------
# TEST 1
# rate = fn

# ******
# Should this pass? (Build in function logic for rate, similar to noise)?
# Should it fail with a DDM error in validate_params()?
# ******


def test_DDM_rate_fn():
    with pytest.raises(ValueError) as error_text:
        stim = [10]
        T = DDM(
            name='DDM',
            default_variable=[0],
            function=DriftDiffusionIntegrator(

                noise=0.0,
                rate=NormalDist().function,
                time_step_size=1.0
            ),
            execute_until_finished=False,
        )
        T.execute(stim)
    assert "incompatible value" in str(error_text.value)

# ------------------------------------------------------------------------------------------------


# ======================================= SIZE_INITIALIZATION TESTS ============================================

# VALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# input_shapes = int, check if variable is an array of zeros


def test_DDM_size_int_check_var():
    T = DDM(
        name='DDM',
        input_shapes=1,
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=-5.0,
            time_step_size=1.0
        ),
        execute_until_finished=False,
    )
    assert len(T.defaults.variable) == 1 and T.defaults.variable[0][0] == 0

# ------------------------------------------------------------------------------------------------
# TEST 2
# input_shapes = float, variable = [.4], check output after execution

def test_DDM_size_int_inputs():

    T = DDM(
        name='DDM',
        input_shapes=1,
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=-5.0,
            time_step_size=1.0
        ),
        execute_until_finished=False,
    )
    val = T.execute([.4])
    decision_variable = val[0][0]
    time = val[1][0]
    assert decision_variable == -2.0
    assert time == 1.0

# ------------------------------------------------------------------------------------------------

# INVALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 2
# input_shapes = -1.0, check less-than-one error


def test_DDM_mech_size_negative_one():
    with pytest.raises(ComponentError) as error_text:
        T = DDM(
            name='DDM',
            input_shapes=-1,
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
            execute_until_finished=False,
        )
    assert "negative dimensions" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# input_shapes = 3.0, check input_shapes-too-large error


def test_DDM_size_too_large():
    with pytest.raises(DDMError) as error_text:
        T = DDM(
            name='DDM',
            input_shapes=3,
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
            execute_until_finished=False,
        )
    assert "single numeric item" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# input_shapes = [1,1], check too-many-input-ports error


def test_DDM_size_too_long():
    with pytest.raises(DDMError) as error_text:
        T = DDM(
            name='DDM',
            input_shapes=[1, 1],
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
            execute_until_finished=False,
        )
    assert "is greater than 1, implying there are" in str(error_text.value)


def test_DDM_time():

    D = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=-5.0,
            time_step_size=0.2,
            non_decision_time=0.5
        ),
        execute_until_finished=False,
    )

    time_0 = D.function.previous_time   # t_0  = 0.5
    np.testing.assert_allclose(time_0, 0.5, atol=1e-08)

    time_1 = D.execute(10)[1][0]   # t_1  = 0.5 + 0.2 = 0.7
    np.testing.assert_allclose(time_1, 0.7, atol=1e-08)

    for i in range(10):                                     # t_11 = 0.7 + 10*0.2 = 2.7
        D.execute(10)
    time_12 = D.execute(10)[1][0]                              # t_12 = 2.7 + 0.2 = 2.9
    np.testing.assert_allclose(time_12, 2.9, atol=1e-08)


def test_WhenFinished_DDM_Analytical():
    D = DDM(function=DriftDiffusionAnalytical)
    c = WhenFinished(D)
    c.is_satisfied()


@pytest.mark.composition
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM-comp")
def test_DDM_in_composition(benchmark, comp_mode):
    M = pnl.DDM(
        name='DDM',
        function=pnl.DriftDiffusionIntegrator(
            rate=1,
            noise=0.0,
            offset=0.0,
            non_decision_time=0.0,
            time_step_size=0.1,
        ),
        execute_until_finished=False,
        reset_stateful_function_when=Never()
    )
    C = pnl.Composition()
    C.add_linear_processing_pathway([M])
    inputs = {M: [10]}
    val = benchmark(C.run, inputs, num_trials=2, execution_mode=comp_mode)

    # FIXME: Python version returns dtype=object
    val = np.asfarray(val)
    np.testing.assert_allclose(val[0], [2.0])
    np.testing.assert_allclose(val[1], [0.2])


@pytest.mark.composition
@pytest.mark.ddm_mechanism
def test_DDM_threshold_modulation_analytical(comp_mode):
    M = pnl.DDM(name='DDM',
                function=pnl.DriftDiffusionAnalytical(
                    threshold=20.0,
                ),
               )

    control = pnl.ControlMechanism(control_signals=[(pnl.THRESHOLD, M)])

    C = pnl.Composition()
    C.add_node(M, required_roles=[pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT])
    C.add_node(control)
    inputs = {M:[1], control:[3]}
    val = C.run(inputs, num_trials=1, execution_mode=comp_mode)

    # Default modulation is 'multiplicative so the threshold is 20 * 3
    np.testing.assert_allclose(val[0], [60.0])
    np.testing.assert_allclose(val[1], [60.2])


@pytest.mark.composition
@pytest.mark.ddm_mechanism
def test_DDM_threshold_modulation_integrator(comp_mode):
    M = pnl.DDM(name='DDM',
                execute_until_finished=True,
                function=pnl.DriftDiffusionIntegrator(threshold=20),
               )

    control = pnl.ControlMechanism(
            control_signals=[(pnl.THRESHOLD, M)])

    C = pnl.Composition()
    C.add_node(M, required_roles=[pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT])
    C.add_node(control)
    inputs = {M:[1], control:[3]}
    val = C.run(inputs, num_trials=1, execution_mode=comp_mode)

    np.testing.assert_allclose(val[0], [60.0])
    np.testing.assert_allclose(val[1], [60.0])


@pytest.mark.composition
@pytest.mark.parametrize(["noise", "threshold", "expected_results"],[
                            (1.0, 0.0, [[0.0], [1.0]]),
                            (1.5, 2, [[-2.0], [1.0]]),
                            (10.0, 10.0, [[10.0], [29.0]]),
                            (100.0, 100.0, [[100.0], [76.0]]),
                        ])
# 3/5/2021 - DDM' default behaviour now requires resetting stateful
# functions after each trial. This is not supported in LLVM execution mode.
# See: https://github.com/PrincetonUniversity/PsyNeuLink/issues/1935
@pytest.mark.usefixtures("comp_mode_no_llvm")
def test_ddm_is_finished(comp_mode, noise, threshold, expected_results):

    comp = Composition()
    ddm = DDM(function=DriftDiffusionIntegrator(threshold=threshold, noise=np.sqrt(noise), time_step_size=1.0),
              execute_until_finished=True)
    comp.add_node(ddm)

    results = comp.run([0], execution_mode=comp_mode)

    np.testing.assert_array_equal(results, expected_results)

@pytest.mark.composition
@pytest.mark.parametrize("until_finished", ["until_finished", "not_until_finished"])
@pytest.mark.parametrize("threshold_mod", ["threshold_modulated", "threshold_not_modulated"])
# 3/5/2021 - DDM' default behaviour now requires resetting stateful
# functions after each trial. This is not supported in LLVM execution mode.
# See: https://github.com/PrincetonUniversity/PsyNeuLink/issues/1935
# Moreover, evaluating scheduler conditions in Python is not supported
# for compiled execution
@pytest.mark.usefixtures("comp_mode_no_llvm")
def test_ddm_is_finished_with_dependency(comp_mode, until_finished, threshold_mod):

    comp = Composition()
    ddm = DDM(function=DriftDiffusionIntegrator(),
              # Use only the decision variable in this test
              output_ports=[pnl.DECISION_VARIABLE],
              execute_until_finished=until_finished == "until_finished")
    dep = pnl.ProcessingMechanism()
    comp.add_linear_processing_pathway([ddm, dep])
    comp.scheduler.add_condition(dep, pnl.WhenFinished(ddm))

    inputs = {ddm: [4]}
    expected_results = [[100]]

    if threshold_mod == "threshold_modulated":
        control = pnl.ControlMechanism(control_signals=[(pnl.THRESHOLD, ddm)])
        comp.add_node(control)

        # reduce the threshold by half
        inputs[control] = 0.5
        expected_results = [[50]]

    results = comp.run(inputs, execution_mode=comp_mode)

    np.testing.assert_array_equal(results, expected_results)

def test_sequence_of_DDM_mechs_in_Composition_Pathway():
    myMechanism = DDM(
        function=DriftDiffusionAnalytical(
            drift_rate=(1.0),
            threshold=(10.0),
            starting_value=0.0,
        ),
        name='My_DDM',
    )

    myMechanism_2 = DDM(
        function=DriftDiffusionAnalytical(
            drift_rate=2.0,
            threshold=20.0),
        name='My_DDM_2'
    )

    myMechanism_3 = DDM(
        function=DriftDiffusionAnalytical(
            drift_rate=3.0,
            threshold=30.0
        ),
        name='My_DDM_3',
    )

    z = Composition(
        # default_variable=[[30], [10]],
        pathways=[[
            myMechanism,
            (IDENTITY_MATRIX),
            myMechanism_2,
            (FULL_CONNECTIVITY_MATRIX),
            myMechanism_3
        ]],
    )

    result = z.execute(inputs={myMechanism:[40]})

    expected_output = [
        (myMechanism.input_ports[0].parameters.value.get(z), np.array([40.])),
        (myMechanism.output_ports[0].parameters.value.get(z), np.array([10.])),
        (myMechanism_2.input_ports[0].parameters.value.get(z), np.array([10.])),
        (myMechanism_2.output_ports[0].parameters.value.get(z), np.array([20.])),
        (myMechanism_3.input_ports[0].parameters.value.get(z), np.array([20.])),
        (myMechanism_3.output_ports[0].parameters.value.get(z), np.array([30.])),
        (result[0], np.array([30.])),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))


@pytest.mark.composition
@pytest.mark.ddm_mechanism
# 3/5/2021 - DDM' default behaviour now requires resetting stateful
# functions after each trial. This is not supported in LLVM execution mode.
# See: https://github.com/PrincetonUniversity/PsyNeuLink/issues/1935
@pytest.mark.usefixtures("comp_mode_no_llvm")
def test_DDMMechanism_LCA_equivalent(comp_mode):

    ddm = DDM(default_variable=[0],
              function=DriftDiffusionIntegrator(rate=1, time_step_size=0.1),
              execute_until_finished=False)
    comp2 = Composition()
    comp2.add_node(ddm)
    result2 = comp2.run(inputs={ddm:[1]}, execution_mode=comp_mode)
    np.testing.assert_allclose(np.asfarray(result2[0]), [0.1])
    np.testing.assert_allclose(np.asfarray(result2[1]), [0.1])
