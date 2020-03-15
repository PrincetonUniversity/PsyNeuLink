import numpy as np
import pytest
import typecheck

import psyneulink as pnl
import psyneulink.core.llvm as pnlvm

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.distributionfunctions import DriftDiffusionAnalytical, NormalDist
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import DriftDiffusionIntegrator
from psyneulink.core.components.process import Process
from psyneulink.core.components.system import System
from psyneulink.core.scheduling.condition import Never, WhenFinished
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.processing.integrator.ddm import ARRAY, DDM, DDMError, SELECTED_INPUT_ARRAY

class TestReinitialize:

    def test_valid(self):
        D = DDM(
            name='DDM',
            function=DriftDiffusionIntegrator(seed=0),
        )

        #  returns previous_value + rate * variable * time_step_size  + noise
        #  0.0 + 1.0 * 1.0 * 1.0 + 0.0
        D.execute(1.0)
        assert np.allclose(np.asfarray(D.value),  [[1.0], [1.0]])
        assert np.allclose(D.output_ports[0].value[0][0], 1.0)
        assert np.allclose(D.output_ports[1].value[0][0], 1.0)

        # reinitialize function
        D.function.reinitialize(2.0, 0.1, 0)
        assert np.allclose(D.function.value[0], 2.0)
        assert np.allclose(D.function.previous_value, 2.0)
        assert np.allclose(D.function.previous_time, 0.1)
        assert np.allclose(np.asfarray(D.value),  [[1.0], [1.0]])
        assert np.allclose(D.output_ports[0].value[0][0], 1.0)
        assert np.allclose(D.output_ports[1].value[0][0], 1.0)

        # reinitialize function without value spec
        D.function.reinitialize()
        assert np.allclose(D.function.value[0], 0.0)
        assert np.allclose(D.function.previous_value, 0.0)
        assert np.allclose(D.function.previous_time, 0.0)
        assert np.allclose(np.asfarray(D.value), [[1.0], [1.0]])
        assert np.allclose(D.output_ports[0].value[0][0], 1.0)
        assert np.allclose(D.output_ports[1].value[0][0], 1.0)

        # reinitialize mechanism
        D.reinitialize(2.0, 0.1, 0)
        assert np.allclose(D.function.value[0], 2.0)
        assert np.allclose(D.function.previous_value, 2.0)
        assert np.allclose(D.function.previous_time, 0.1)
        assert np.allclose(np.asfarray(D.value), [[2.0], [0.1]])
        assert np.allclose(D.output_ports[0].value, 2.0)
        assert np.allclose(D.output_ports[1].value, 0.1)

        D.execute(1.0)
        #  2.0 + 1.0 = 3.0 ; 0.1 + 1.0 = 1.1
        assert np.allclose(np.asfarray(D.value), [[[3.0]], [[1.1]]])
        assert np.allclose(D.output_ports[0].value[0][0], 3.0)
        assert np.allclose(D.output_ports[1].value[0][0], 1.1)

        # reinitialize mechanism without value spec
        D.reinitialize()
        assert np.allclose(D.function.value[0], 0.0)
        assert np.allclose(D.function.previous_value, 0.0)
        assert np.allclose(D.function.previous_time, 0.0)
        assert np.allclose(D.output_ports[0].value[0], 0.0)
        assert np.allclose(D.output_ports[1].value[0], 0.0)

        # reinitialize only decision variable
        D.function.initializer = 1.0
        D.function.starting_point = 0.0
        D.reinitialize()
        assert np.allclose(D.function.value[0], 1.0)
        assert np.allclose(D.function.previous_value, 1.0)
        assert np.allclose(D.function.previous_time, 0.0)
        assert np.allclose(D.output_ports[0].value[0], 1.0)
        assert np.allclose(D.output_ports[1].value[0], 0.0)


class TestThreshold:
    def test_threshold_param(self):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=10.0))

        assert D.function.threshold == 10.0

        D.function.threshold = 5.0
        assert D.function.threshold == 5.0

    def test_threshold_sets_is_finished(self):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=5.0))
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
    @pytest.mark.parametrize("mode", [
        "Python",
        pytest.param("LLVM", marks=pytest.mark.llvm),
        pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda]),
    ])
    def test_threshold_stops_accumulation(self, mode, variable, expected, benchmark):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=5.0))
        if mode == "Python":
            ex = D.execute
        elif mode == "LLVM":
            ex = pnlvm.execution.MechExecution(D).execute
        elif mode == "PTX":
            ex = pnlvm.execution.MechExecution(D).cuda_execute

        decision_variables = []
        time_points = []
        for i in range(5):
            output = ex([variable])
            decision_variables.append(output[0][0][0])
            time_points.append(output[1][0][0])

        # decision variable accumulation stops
        assert np.allclose(decision_variables, expected)

        # time accumulation does not stop
        assert np.allclose(time_points, [1.0, 2.0, 3.0, 4.0, 5.0])
        benchmark(ex, [variable])

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
    #     assert np.allclose(decision_variables_a, [2.0, 4.0, 5.0, 5.0, 5.0])


    def test_is_finished_stops_system(self):
        D = DDM(name='DDM',
                function=DriftDiffusionIntegrator(threshold=10.0))
        P = Process(pathway=[D])
        S = System(processes=[P],
                   reinitialize_mechanisms_when=Never())
        S.run(inputs={D: 2.0},
              termination_processing={TimeScale.TRIAL: WhenFinished(D)})

        # decision variable's value should match threshold
        assert D.parameters.value.get(S)[0] == 10.0
        # it should have taken 5 executions (and time_step_size = 1.0)
        assert D.parameters.value.get(S)[1] == 5.0


    # def test_is_finished_stops_mechanism(self):
    #     D = DDM(name='DDM',
    #             function=DriftDiffusionIntegrator(threshold=10.0))
    #     T = TransferMechanism(function=Linear(slope=2.0))
    #     P = Process(pathway=[D, T])
    #     S = System(processes=[P])
    #
    #     sched = Scheduler(system=S)

class TestOutputPorts:

    def test_selected_input_array(self):
        action_selection = DDM(
            input_format=ARRAY,
            function=DriftDiffusionAnalytical(
            ),
            output_ports=[SELECTED_INPUT_ARRAY],
            name='DDM'
        )
        action_selection.execute([1.0])

# ------------------------------------------------------------------------------------------------
# TEST 2
# function = Bogacz


@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark
@pytest.mark.parametrize("mode", [
    "Python",
    pytest.param("LLVM", marks=pytest.mark.llvm),
    pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda]),
])
def test_DDM_Integrator_Bogacz(benchmark, mode):
    stim = 10
    T = DDM(
        name='DDM',
        function=DriftDiffusionAnalytical()
    )
    if mode == "Python":
        ex = T.execute
    elif mode == "LLVM":
        ex = pnlvm.execution.MechExecution(T).execute
    elif mode == "PTX":
        ex = pnlvm.execution.MechExecution(T).cuda_execute
    val = ex(stim)[0]
    assert np.allclose(val, [1.0])
    benchmark(ex, stim)

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
#     val = float(T.execute(stim)[0])
#     assert val == 10


# ======================================= NOISE TESTS ============================================

# VALID NOISE:

# ------------------------------------------------------------------------------------------------
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM")
@pytest.mark.parametrize("noise, expected", [
    (0., 10),
    (0.5, 8.194383551861414),
    (2., 6.388767103722829),
    ], ids=["0", "0.5", "2.0"])
@pytest.mark.parametrize("mode", [
    "Python",
    pytest.param("LLVM", marks=pytest.mark.llvm),
    pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda]),
])
def test_DDM_noise(mode, benchmark, noise, expected):
    T = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=noise,
            rate=1.0,
            time_step_size=1.0
        )
    )
    if mode == "Python":
        ex = T.execute
    elif mode == "LLVM":
        ex = pnlvm.execution.MechExecution(T).execute
    elif mode == "PTX":
        ex = pnlvm.execution.MechExecution(T).cuda_execute

    val = ex([10])
    assert np.allclose(val[0][0][0], expected)
    benchmark(ex, [10])

# ------------------------------------------------------------------------------------------------

# INVALID NOISE:

# ------------------------------------------------------------------------------------------------
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM")
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
        float(T.execute(stim)[0])
    assert "DriftDiffusionIntegrator requires noise parameter to be a float" in str(error_text.value)

# ======================================= INPUT TESTS ============================================

# VALID INPUTS:

# ------------------------------------------------------------------------------------------------
@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM")
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
    )
    val = float(T.execute(stim)[0])
    assert val == 10

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
        )
        float(T.execute(stim)[0])
    assert "single numeric item" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# input = Fn

# ******
# Should functions be caught in validation, rather than with TypeError [just check callable()]?
# So that functions cause the same error as lists (see TEST 1 above)
# ******


def test_DDM_input_fn():
    with pytest.raises(TypeError) as error_text:
        stim = NormalDist()
        T = DDM(
            name='DDM',
            function=DriftDiffusionIntegrator(

                noise=0.0,
                rate=1.0,
                time_step_size=1.0
            ),
        )
        float(T.execute(stim))
    assert "not supported for the input types" in str(error_text.value)


# ======================================= RATE TESTS ============================================

# VALID RATES:

@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM")
@pytest.mark.parametrize("rate, expected", [
    (5, 50), (5., 50), ([5], 50), (-5.0, -50),
    ], ids=["int", "float", "list", "negative"])
@pytest.mark.parametrize("mode", [
    "Python",
    pytest.param("LLVM", marks=pytest.mark.llvm),
    pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda]),
])
# ******
# Should negative pass?
# ******
def test_DDM_rate(benchmark, rate, expected, mode):
    stim = [10]
    T = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=rate,
            time_step_size=1.0
        ),
    )
    if mode == "Python":
        ex = T.execute
    elif mode == "LLVM":
        ex = pnlvm.execution.MechExecution(T).execute
    elif mode == "PTX":
        ex = pnlvm.execution.MechExecution(T).cuda_execute
    val = float(ex(stim)[0][0][0])
    assert val == expected
    benchmark(ex, stim)

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
    with pytest.raises(typecheck.framework.InputParameterError) as error_text:
        stim = [10]
        T = DDM(
            name='DDM',
            default_variable=[0],
            function=DriftDiffusionIntegrator(

                noise=0.0,
                rate=NormalDist().function,
                time_step_size=1.0
            ),
        )
        float(T.execute(stim)[0])
    assert "incompatible value" in str(error_text.value)

# ------------------------------------------------------------------------------------------------


# ======================================= SIZE_INITIALIZATION TESTS ============================================

# VALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# size = int, check if variable is an array of zeros


def test_DDM_size_int_check_var():
    T = DDM(
        name='DDM',
        size=1,
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=-5.0,
            time_step_size=1.0
        ),
    )
    assert len(T.defaults.variable) == 1 and T.defaults.variable[0][0] == 0

# ------------------------------------------------------------------------------------------------
# TEST 2
# size = float, variable = [.4], check output after execution

def test_DDM_size_int_inputs():

    T = DDM(
        name='DDM',
        size=1,
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=-5.0,
            time_step_size=1.0
        ),
    )
    val = T.execute([.4])
    decision_variable = val[0][0]
    time = val[1][0]
    assert decision_variable == -2.0
    assert time == 1.0

# ------------------------------------------------------------------------------------------------

# INVALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# size = 0, check less-than-one error


def test_DDM_mech_size_zero():
    with pytest.raises(ComponentError) as error_text:
        T = DDM(
            name='DDM',
            size=0,
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
        )
    assert "is not a positive number" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# size = -1.0, check less-than-one error


def test_DDM_mech_size_negative_one():
    with pytest.raises(ComponentError) as error_text:
        T = DDM(
            name='DDM',
            size=-1.0,
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
        )
    assert "is not a positive number" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# size = 3.0, check size-too-large error


def test_DDM_size_too_large():
    with pytest.raises(DDMError) as error_text:
        T = DDM(
            name='DDM',
            size=3.0,
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
        )
    assert "single numeric item" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# size = [1,1], check too-many-input-ports error


def test_DDM_size_too_long():
    with pytest.raises(DDMError) as error_text:
        T = DDM(
            name='DDM',
            size=[1, 1],
            function=DriftDiffusionIntegrator(
                noise=0.0,
                rate=-5.0,
                time_step_size=1.0
            ),
        )
    assert "is greater than 1, implying there are" in str(error_text.value)


def test_DDM_time():

    D = DDM(
        name='DDM',
        function=DriftDiffusionIntegrator(
            noise=0.0,
            rate=-5.0,
            time_step_size=0.2,
            starting_point=0.5
        )
    )

    time_0 = D.function.previous_time   # t_0  = 0.5
    np.testing.assert_allclose(time_0, 0.5, atol=1e-08)

    time_1 = D.execute(10)[1][0]   # t_1  = 0.5 + 0.2 = 0.7
    np.testing.assert_allclose(time_1[0], 0.7, atol=1e-08)

    for i in range(10):                                     # t_11 = 0.7 + 10*0.2 = 2.7
        D.execute(10)
    time_12 = D.execute(10)[1][0]                              # t_12 = 2.7 + 0.2 = 2.9
    np.testing.assert_allclose(time_12[0], 2.9, atol=1e-08)


def test_WhenFinished_DDM_Analytical():
    D = DDM(function=DriftDiffusionAnalytical)
    c = WhenFinished(D)
    c.is_satisfied()

@pytest.mark.ddm_mechanism
@pytest.mark.mechanism
@pytest.mark.benchmark(group="DDM-comp")
@pytest.mark.parametrize("mode", [
    "Python",
    pytest.param("LLVM", marks=pytest.mark.llvm),
    pytest.param("LLVMExec", marks=pytest.mark.llvm),
    pytest.param("LLVMRun", marks=pytest.mark.llvm),
    pytest.param("PTXExec", marks=[pytest.mark.llvm, pytest.mark.cuda]),
    pytest.param("PTXRun", marks=[pytest.mark.llvm, pytest.mark.cuda]),
])
def test_DDM_in_composition(benchmark, mode):
    M = pnl.DDM(
        name='DDM',
        function=pnl.DriftDiffusionIntegrator(
            rate=1,
            noise=0.0,
            offset=0.0,
            starting_point=0.0,
            time_step_size=0.1,
        ),
    )
    C = pnl.Composition()
    C.add_linear_processing_pathway([M])
    inputs = {M: [10]}
    val = C.run(inputs, num_trials=2, bin_execute=mode)
    # FIXME: Python version returns dtype=object
    val = np.asfarray(val)
    assert np.allclose(val[0], [2.0])
    assert np.allclose(val[1], [0.2])
    benchmark(C.run, inputs, num_trials=2, bin_execute=mode)
