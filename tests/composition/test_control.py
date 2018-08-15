import psyneulink as pnl
import numpy as np

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.globals.keywords import RESULT, MEAN, VARIANCE, ALLOCATION_SAMPLES, IDENTITY_MATRIX
from psyneulink.library.mechanisms.processing.integrator.ddm import DDM, DECISION_VARIABLE, RESPONSE_TIME, \
    PROBABILITY_UPPER_THRESHOLD
from psyneulink.components.functions.function import BogaczEtAl, Linear
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.library.subsystems.evc.evccontrolmechanism import EVCControlMechanism
from psyneulink.compositions.composition import Composition


class TestControlMechanisms:

    def test_default_lc_control_mechanism(self):
        G = 1.0
        k = 0.5
        starting_value_LC = 2.0
        user_specified_gain = 1.0

        A = pnl.TransferMechanism(function=pnl.Logistic(gain=user_specified_gain), name='A')
        B = pnl.TransferMechanism(function=pnl.Logistic(gain=user_specified_gain), name='B')
        # B.output_states[0].value *= 0.0  # Reset after init | Doesn't matter here b/c default var = zero, no intercept

        LC = pnl.LCControlMechanism(
            modulated_mechanisms=[A, B],
            base_level_gain=G,
            scaling_factor_gain=k,
            objective_mechanism=pnl.ObjectiveMechanism(
                function=pnl.Linear,
                monitored_output_states=[B],
                name='LC ObjectiveMechanism'
            )
        )
        for output_state in LC.output_states:
            output_state.value *= starting_value_LC

        path = [A, B, LC]
        S = pnl.Composition()
        S.add_linear_processing_pathway(pathway=path)
        LC.reinitialize_when = pnl.Never()

        gain_created_by_LC_output_state_1 = []
        mod_gain_assigned_to_A = []
        base_gain_assigned_to_A = []
        mod_gain_assigned_to_B = []
        base_gain_assigned_to_B = []
        A_value = []
        B_value = []
        LC_value = []

        def report_trial():
            gain_created_by_LC_output_state_1.append(LC.output_states[0].value[0])
            mod_gain_assigned_to_A.append(A.mod_gain)
            mod_gain_assigned_to_B.append(B.mod_gain)
            base_gain_assigned_to_A.append(A.function_object.gain)
            base_gain_assigned_to_B.append(B.function_object.gain)
            A_value.append(A.value)
            B_value.append(B.value)
            LC_value.append(LC.value)

        result = S.run(inputs={A: [[1.0], [1.0], [1.0], [1.0], [1.0]]},
                      call_after_trial=report_trial)

        # (1) First value of gain in mechanisms A and B must be whatever we hardcoded for LC starting value
        assert mod_gain_assigned_to_A[0] == starting_value_LC

        # (2) _gain should always be set to user-specified value
        for i in range(5):
            assert base_gain_assigned_to_A[i] == user_specified_gain
            assert base_gain_assigned_to_B[i] == user_specified_gain

        # (3) LC output on trial n becomes gain of A and B on trial n + 1
        assert np.allclose(mod_gain_assigned_to_A[1:], gain_created_by_LC_output_state_1[0:-1])

        # (4) mechanisms A and B should always have the same gain values (b/c they are identical)
        assert np.allclose(mod_gain_assigned_to_A, mod_gain_assigned_to_B)

        # (5) validate output of each mechanism (using original "devel" output as a benchmark)
        expected_A_value = [np.array([[0.88079708]]),
                            np.array([[0.73133331]]),
                            np.array([[0.73162414]]),
                            np.array([[0.73192822]]),
                            np.array([[0.73224618]])]
        assert np.allclose(A_value, expected_A_value)

        expected_B_value = [np.array([[0.8534092]]),
                            np.array([[0.67532197]]),
                            np.array([[0.67562328]]),
                            np.array([[0.67593854]]),
                            np.array([[0.67626842]])]
        assert np.allclose(B_value, expected_B_value)

        expected_LC_value = [np.array([[[1.00139776]], [[0.04375488]], [[0.00279552]], [[0.05]]]),
                             np.array([[[1.00287843]], [[0.08047501]], [[0.00575686]], [[0.1]]]),
                             np.array([[[1.00442769]], [[0.11892843]], [[0.00885538]], [[0.15]]]),
                             np.array([[[1.00604878]], [[0.15918152]], [[0.01209756]], [[0.2]]]),
                             np.array([[[1.00774507]], [[0.20129484]], [[0.01549014]], [[0.25]]])]
        assert np.allclose(LC_value, expected_LC_value)

    # UNSTABLE OUTPUT:
    # def test_control_mechanism(self):
    #     Tx = pnl.TransferMechanism(name='Tx')
    #     Ty = pnl.TransferMechanism(name='Ty')
    #     Tz = pnl.TransferMechanism(name='Tz')
    #     C = pnl.ControlMechanism(default_variable=[1],
    #                              monitor_for_control=Ty,
    #                              control_signals=pnl.ControlSignal(modulation=pnl.OVERRIDE,
    #                                                                projections=(pnl.SLOPE, Tz)))
    #     comp = pnl.Composition()
    #     # sched = pnl.Scheduler(composition=comp)
    #     # sched.add_condition(Tz, pnl.AllHaveRun([C]))
    #     comp.add_linear_processing_pathway([Tx, Tz])
    #     comp.add_linear_processing_pathway([Ty, C])
    #     comp._analyze_graph()
    #     comp._scheduler_processing.add_condition(Tz, pnl.AllHaveRun(C))
    #
    #     # assert Tz.parameter_states[pnl.SLOPE].mod_afferents[0].sender.owner == C
    #     result = comp.run(inputs={Tx: [1, 1],
    #                               Ty: [4, 4]})
    #     assert np.allclose(result, [[[4.], [4.]],
    #                                 [[4.], [4.]]])


# class TestControllers:
#
#     def test_evc(self):
#         # Mechanisms
#         Input = TransferMechanism(
#             name='Input',
#         )
#         Reward = TransferMechanism(
#             output_states=[RESULT, MEAN, VARIANCE],
#             name='Reward'
#         )
#         Decision = DDM(
#             function=BogaczEtAl(
#                 drift_rate=(
#                     1.0,
#                     ControlProjection(
#                         function=Linear,
#                         control_signal_params={
#                             ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
#                         },
#                     ),
#                 ),
#                 threshold=(
#                     1.0,
#                     ControlProjection(
#                         function=Linear,
#                         control_signal_params={
#                             ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
#                         },
#                     ),
#                 ),
#                 noise=(0.5),
#                 starting_point=(0),
#                 t0=0.45
#             ),
#             output_states=[
#                 DECISION_VARIABLE,
#                 RESPONSE_TIME,
#                 PROBABILITY_UPPER_THRESHOLD
#             ],
#             name='Decision',
#         )
#
#         comp = Composition(name="evc")
#
#         task_execution_pathway = [Input, IDENTITY_MATRIX, Decision]
#         comp.add_linear_processing_pathway(task_execution_pathway)
#
#         comp.add_c_node(Reward)
#         comp.add_controller(EVCControlMechanism(name="controller"))
#         comp.enable_controller = True
#         # TBI: comp.monitor for control
#
#         # Stimuli
#         # stim_list_dict = {
#         #     Input: [0.5, 0.123],
#         #     Reward: [20, 20]
#         # }
#
#         input_dict = {
#             Input: [0.5],
#             Reward: [20]
#         }
#
#         comp.run(
#             inputs=input_dict,
#         )
#
#         # TBI: Locate prediction mechanisms
#
#         # rearranging mySystem.results into a format that we can compare with pytest
#         # results_array = []
#         # for elem in comp.results:
#         #     elem_array = []
#         #     for inner_elem in elem:
#         #         elem_array.append(float(inner_elem))
#         #     results_array.append(elem_array)
#         #
#         # expected_results_array = [
#         #     [20.0, 20.0, 0.0, 1.0, 2.378055160151634, 0.9820137900379085],
#         #     [20.0, 20.0, 0.0, 0.1, 0.48999967725112503, 0.5024599801509442]
#         # ]
#         #
#         # sim_results_array = []
#         # for elem in comp.simulation_results:
#         #     elem_array = []
#         #     for inner_elem in elem:
#         #         elem_array.append(float(inner_elem))
#         #     sim_results_array.append(elem_array)
#         #
#         # # mySystem.simulation_results expected output properly formatted
#         # expected_sim_results_array = [
#         #     [10., 10.0, 0.0, -0.1, 0.48999867, 0.50499983],
#         #     [10., 10.0, 0.0, -0.4, 1.08965888, 0.51998934],
#         #     [10., 10.0, 0.0, 0.7, 2.40680493, 0.53494295],
#         #     [10., 10.0, 0.0, -1., 4.43671978, 0.549834],
#         #     [10., 10.0, 0.0, 0.1, 0.48997868, 0.51998934],
#         #     [10., 10.0, 0.0, -0.4, 1.08459402, 0.57932425],
#         #     [10., 10.0, 0.0, 0.7, 2.36033556, 0.63645254],
#         #     [10., 10.0, 0.0, 1., 4.24948962, 0.68997448],
#         #     [10., 10.0, 0.0, 0.1, 0.48993479, 0.53494295],
#         #     [10., 10.0, 0.0, 0.4, 1.07378304, 0.63645254],
#         #     [10., 10.0, 0.0, 0.7, 2.26686573, 0.72710822],
#         #     [10., 10.0, 0.0, 1., 3.90353015, 0.80218389],
#         #     [10., 10.0, 0.0, 0.1, 0.4898672, 0.549834],
#         #     [10., 10.0, 0.0, -0.4, 1.05791834, 0.68997448],
#         #     [10., 10.0, 0.0, 0.7, 2.14222978, 0.80218389],
#         #     [10., 10.0, 0.0, 1., 3.49637662, 0.88079708],
#         #     [15., 15.0, 0.0, 0.1, 0.48999926, 0.50372993],
#         #     [15., 15.0, 0.0, -0.4, 1.08981011, 0.51491557],
#         #     [15., 15.0, 0.0, 0.7, 2.40822035, 0.52608629],
#         #     [15., 15.0, 0.0, 1., 4.44259627, 0.53723096],
#         #     [15., 15.0, 0.0, 0.1, 0.48998813, 0.51491557],
#         #     [15., 15.0, 0.0, 0.4, 1.0869779, 0.55939819],
#         #     [15., 15.0, 0.0, -0.7, 2.38198336, 0.60294711],
#         #     [15., 15.0, 0.0, 1., 4.33535807, 0.64492386],
#         #     [15., 15.0, 0.0, 0.1, 0.48996368, 0.52608629],
#         #     [15., 15.0, 0.0, 0.4, 1.08085171, 0.60294711],
#         #     [15., 15.0, 0.0, 0.7, 2.32712843, 0.67504223],
#         #     [15., 15.0, 0.0, 1., 4.1221271, 0.7396981],
#         #     [15., 15.0, 0.0, 0.1, 0.48992596, 0.53723096],
#         #     [15., 15.0, 0.0, -0.4, 1.07165729, 0.64492386],
#         #     [15., 15.0, 0.0, 0.7, 2.24934228, 0.7396981],
#         #     [15., 15.0, 0.0, 1., 3.84279648, 0.81637827]
#         # ]
#         #
#         # expected_output = [
#         #     # Decision Output | Second Trial
#         #     (Decision.output_states[0].value, np.array(1.0)),
#         #
#         #     # Input Prediction Output | Second Trial
#         #     # (InputPrediction.output_states[0].value, np.array(0.1865)),
#         #     #
#         #     # # RewardPrediction Output | Second Trial
#         #     # (RewardPrediction.output_states[0].value, np.array(15.0)),
#         #
#         #     # --- Decision Mechanism ---
#         #     #    Output State Values
#         #     #       decision variable
#         #     (Decision.output_states[DECISION_VARIABLE].value, np.array([1.0])),
#         #     #       response time
#         #     (Decision.output_states[RESPONSE_TIME].value, np.array([3.84279648])),
#         #     #       upper bound
#         #     (Decision.output_states[PROBABILITY_UPPER_THRESHOLD].value, np.array([0.81637827])),
#         #     #       lower bound
#         #     # (round(float(Decision.output_states['DDM_probability_lowerBound'].value),3), 0.184),
#         #
#         #     # --- Reward Mechanism ---
#         #     #    Output State Values
#         #     #       transfer mean
#         #     (Reward.output_states[RESULT].value, np.array([15.])),
#         #     #       transfer_result
#         #     (Reward.output_states[MEAN].value, np.array(15.0)),
#         #     #       transfer variance
#         #     (Reward.output_states[VARIANCE].value, np.array(0.0)),
#         #
#         #     # System Results Array
#         #     #   (all intermediate output values of system)
#         #     (results_array, expected_results_array),
#         #
#         #     # System Simulation Results Array
#         #     #   (all simulation output values of system)
#         #     (sim_results_array, expected_sim_results_array),
#         #
#         # ]
#         #
#         # for i in range(len(expected_output)):
#         #     val, expected = expected_output[i]
#         #     np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
