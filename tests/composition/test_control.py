import functools
import numpy as np
import pytest
import psyneulink as pnl
import psyneulink.core.components.functions.distributionfunctions
from psyneulink.core.components.functions.optimizationfunctions import SampleIterator, SampleSpec, OptimizationFunctionError
class TestControlMechanisms:

    def test_lvoc(self):
        m1 = pnl.TransferMechanism(input_states=["InputState A", "InputState B"])
        m2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_c_node(m1, required_roles=pnl.CNodeRole.ORIGIN)
        c.add_c_node(m2, required_roles=pnl.CNodeRole.ORIGIN)
        c._analyze_graph()
        lvoc = pnl.OptimizationControlMechanism(agent_rep=pnl.RegressionCFA,
                                                features=[{pnl.SHADOW_EXTERNAL_INPUTS: [m1, m2]}],
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                    monitored_output_states=[m1, m2]),
                                                terminal_objective_mechanism=True,
                                                function=pnl.GridSearch(max_iterations=1),
                                                control_signals=[(pnl.SLOPE, m1), (pnl.SLOPE, m2)])
        c.add_c_node(lvoc)
        input_dict = {m1: [[1], [1]], m2: [1]}

        c.run(inputs=input_dict)

        assert len(lvoc.input_states) == 4

    def test_lvoc_both_predictors_specs(self):
        m1 = pnl.TransferMechanism(input_states=["InputState A", "InputState B"])
        m2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_c_node(m1, required_roles=pnl.CNodeRole.ORIGIN)
        c.add_c_node(m2, required_roles=pnl.CNodeRole.ORIGIN)
        c._analyze_graph()
        lvoc = pnl.OptimizationControlMechanism(agent_rep=pnl.RegressionCFA,
                                                features=[{pnl.SHADOW_EXTERNAL_INPUTS: [m1, m2]}, m2],
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                    monitored_output_states=[m1, m2]),
                                                terminal_objective_mechanism=True,
                                                function=pnl.GridSearch(max_iterations=1),
                                                control_signals=[(pnl.SLOPE, m1), (pnl.SLOPE, m2)])
        c.add_c_node(lvoc)
        input_dict = {m1: [[1], [1]], m2: [1]}


        c.run(inputs=input_dict)

        assert len(lvoc.input_states) == 5

    def test_lvoc_features_function(self):
        m1 = pnl.TransferMechanism(input_states=["InputState A", "InputState B"])
        m2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_c_node(m1, required_roles=pnl.CNodeRole.ORIGIN)
        c.add_c_node(m2, required_roles=pnl.CNodeRole.ORIGIN)
        c._analyze_graph()
        lvoc = pnl.OptimizationControlMechanism(agent_rep=pnl.RegressionCFA,
                                                features=[{pnl.SHADOW_EXTERNAL_INPUTS: [m1, m2]}, m2],
                                                feature_function=pnl.LinearCombination(offset=10.0),
                                                objective_mechanism=pnl.ObjectiveMechanism(
                                                    monitored_output_states=[m1, m2]),
                                                terminal_objective_mechanism=True,
                                                function=pnl.GradientOptimization(max_iterations=1),
                                                control_signals=[(pnl.SLOPE, m1), (pnl.SLOPE, m2)])
        c.add_c_node(lvoc)
        input_dict = {m1: [[1], [1]], m2: [1]}

        c.run(inputs=input_dict)

        assert len(lvoc.input_states) == 5

        for i in range(1,5):
            assert lvoc.input_states[i].function.offset == 10.0

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
        S.add_c_node(A, required_roles=pnl.CNodeRole.ORIGIN)
        S.add_linear_processing_pathway(pathway=path)
        S.add_c_node(LC, required_roles=pnl.CNodeRole.TERMINAL)
        LC.reinitialize_when = pnl.Never()

        gain_created_by_LC_output_state_1 = []
        mod_gain_assigned_to_A = []
        base_gain_assigned_to_A = []
        mod_gain_assigned_to_B = []
        base_gain_assigned_to_B = []
        A_value = []
        B_value = []
        LC_value = []

        def report_trial(system):
            gain_created_by_LC_output_state_1.append(LC.output_state.parameters.value.get(system)[0])
            mod_gain_assigned_to_A.append(A.get_mod_gain(system))
            mod_gain_assigned_to_B.append(B.get_mod_gain(system))
            base_gain_assigned_to_A.append(A.function.parameters.gain.get())
            base_gain_assigned_to_B.append(B.function.parameters.gain.get())
            A_value.append(A.parameters.value.get(system))
            B_value.append(B.parameters.value.get(system))
            LC_value.append(LC.parameters.value.get(system))

        result = S.run(inputs={A: [[1.0], [1.0], [1.0], [1.0], [1.0]]},
                      call_after_trial=functools.partial(report_trial, S))

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

        # # (5) validate output of each mechanism (using original "devel" output as a benchmark)
        # expected_A_value = [np.array([[0.88079708]]),
        #                     np.array([[0.73133331]]),
        #                     np.array([[0.73162414]]),
        #                     np.array([[0.73192822]]),
        #                     np.array([[0.73224618]])]
        #
        # assert np.allclose(A_value, expected_A_value)
        #
        # expected_B_value = [np.array([[0.8534092]]),
        #                     np.array([[0.67532197]]),
        #                     np.array([[0.67562328]]),
        #                     np.array([[0.67593854]]),
        #                     np.array([[0.67626842]])]
        # assert np.allclose(B_value, expected_B_value)
        #
        # expected_LC_value = [np.array([[[1.00139776]], [[0.04375488]], [[0.00279552]], [[0.05]]]),
        #                      np.array([[[1.00287843]], [[0.08047501]], [[0.00575686]], [[0.1]]]),
        #                      np.array([[[1.00442769]], [[0.11892843]], [[0.00885538]], [[0.15]]]),
        #                      np.array([[[1.00604878]], [[0.15918152]], [[0.01209756]], [[0.2]]]),
        #                      np.array([[[1.00774507]], [[0.20129484]], [[0.01549014]], [[0.25]]])]
        # assert np.allclose(LC_value, expected_LC_value)

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
    #     # sched = pnl.Scheduler(omp)
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

# KAM will reintroduce objective mechanism role tests after INPUT and OUTPUT roles are implemented
# class TestObjectiveMechanismRoles:
#
#     def test_origin_objective_mechanism_false(self):
#         #  When False, even if the ObjectiveMechanism is an origin node according to the structure of the graph, the
#         #  ObjectiveMechanism is not marked as origin
#         #  If the ObjectiveMechanism was the only origin node, then the user must use required_roles to assign the
#         #  origin role to another node.
#
#         c = pnl.Composition()
#
#         A = pnl.TransferMechanism()
#         B = pnl.TransferMechanism()
#         lvoc = pnl.ControlMechanism()
#
#         c.add_linear_processing_pathway([lvoc, A])
#
#         c.show_graph()
#
#         assert lvoc.objective_mechanism not in c.get_c_nodes_by_role(pnl.CNodeRole.ORIGIN)
#
#     def test_origin_objective_mechanism_true_origin(self):
#         # When True, if the ObjectiveMechanism is an origin node according to the structure of the graph, it is treated
#         # normally.
#         c = pnl.Composition()
#
#         A = pnl.TransferMechanism()
#         lvoc = pnl.ControlMechanism(
#                                         origin_objective_mechanism=True)
#         B = pnl.TransferMechanism()
#
#         c.add_linear_processing_pathway([lvoc, A])
#         c.add_c_node(B)
#
#         c._analyze_graph()
#
#         assert lvoc.objective_mechanism in c.get_c_nodes_by_role(pnl.CNodeRole.ORIGIN) and \
#                B in c.get_c_nodes_by_role(pnl.CNodeRole.ORIGIN)
#
#     def test_origin_objective_mechanism_true_not_origin(self):
#         # If the ObjectiveMechanism is not an origin node according to the structure of the graph, then it
#         # takes on origin as a required role.
#         c = pnl.Composition()
#
#         A = pnl.TransferMechanism()
#         lvoc = pnl.ControlMechanism(
#                                         origin_objective_mechanism=True)
#         B = pnl.TransferMechanism()
#
#         c.add_linear_processing_pathway([lvoc, A])
#         c.add_linear_processing_pathway([B, lvoc.objective_mechanism])
#
#         c._analyze_graph()
#
#         assert lvoc.objective_mechanism in c.get_c_nodes_by_role(pnl.CNodeRole.ORIGIN) and \
#                B in c.get_c_nodes_by_role(pnl.CNodeRole.ORIGIN)

#     def test_evc(self):
#         # Mechanisms
#         Input = TransferMechanism(
#             name='Input',
#         )
#         reward = TransferMechanism(
#             output_states=[RESULT, OUTPUT_MEAN, OUTPUT_VARIANCE],
#             name='reward'
#         )
#         Decision = DDM(
#             function=DriftDiffusionAnalytical(
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
#     def test_terminal_objective_mechanism_false(self):
#         # When False, even if the ObjectiveMechanism is a terminal node according to the structure of the graph, the
#         # ObjectiveMechanism is not marked as terminal. If the ObjectiveMechanism was the only terminal node, then the
#         # user must use required_roles to assign the terminal role to another node.
#
#         c = pnl.Composition()
#
#         A = pnl.TransferMechanism()
#         lvoc = pnl.ControlMechanism()
#         B = pnl.TransferMechanism()
#
#         c.add_linear_processing_pathway([lvoc, A])
#         c.add_linear_processing_pathway([B, lvoc.objective_mechanism])
#
#         c._analyze_graph()
#         assert lvoc.objective_mechanism not in c.get_c_nodes_by_role(pnl.CNodeRole.TERMINAL)
#
#     def test_terminal_objective_mechanism_true_terminal(self):
#         # When True, if the ObjectiveMechanism is a terminal node according to the structure of the graph, it is treated
#         # normally.
#
#         c = pnl.Composition()
#
#         A = pnl.TransferMechanism()
#         lvoc = pnl.ControlMechanism(
#                                         terminal_objective_mechanism=True)
#         B = pnl.TransferMechanism()
#
#         c.add_linear_processing_pathway([lvoc, A])
#         c.add_linear_processing_pathway([B, lvoc.objective_mechanism])
#
#         c._analyze_graph()
#         assert lvoc.objective_mechanism in c.get_c_nodes_by_role(pnl.CNodeRole.TERMINAL)
#
#     def test_terminal_objective_mechanism_true_not_terminal(self):
#         # If the ObjectiveMechanism is not a terminal node according to the structure of the graph, then it
#         # takes on terminal as a required role.
#         c = pnl.Composition()
#
#         A = pnl.TransferMechanism()
#         lvoc = pnl.ControlMechanism(
#                                         terminal_objective_mechanism=True
#                                         )
#         B = pnl.TransferMechanism()
#         C = pnl.TransferMechanism()
#
#         c.add_linear_processing_pathway([lvoc, A])
#         c.add_linear_processing_pathway([B, lvoc.objective_mechanism, C])
#
#         c._analyze_graph()
#
#         assert lvoc.objective_mechanism in c.get_c_nodes_by_role(pnl.CNodeRole.TERMINAL)

class TestModelBasedOptimizationControlMechanisms:

    def test_evc(self):
        # Mechanisms
        Input = pnl.TransferMechanism(name='Input')
        reward = pnl.TransferMechanism(output_states=[pnl.RESULT, pnl.OUTPUT_MEAN, pnl.OUTPUT_VARIANCE],
                                       name='reward')
        Decision = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0,
                                                                             pnl.ControlProjection(function=pnl.Linear,
                                                                                                   control_signal_params={pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})),
                                                                 threshold=(1.0,
                                                                            pnl.ControlProjection(function=pnl.Linear,
                                                                                                  control_signal_params={pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})),
                                                                 noise=0.5,
                                                                 starting_point=0,
                                                                 t0=0.45),
                           output_states=[pnl.DECISION_VARIABLE,
                                        pnl.RESPONSE_TIME,
                                        pnl.PROBABILITY_UPPER_THRESHOLD],
                           name='Decision')

        comp = pnl.Composition(name="evc")
        comp.add_c_node(reward, required_roles=[pnl.CNodeRole.TERMINAL])
        comp.add_c_node(Decision, required_roles=[pnl.CNodeRole.TERMINAL])
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
        comp.add_linear_processing_pathway(task_execution_pathway)

        comp.add_model_based_optimizer(optimizer=pnl.OptimizationControlMechanism(agent_rep=comp,
                                                                                  features={pnl.SHADOW_EXTERNAL_INPUTS: [Input, reward]},
                                                                                  feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                                                                                  objective_mechanism=pnl.ObjectiveMechanism(function=pnl.LinearCombination(operation=pnl.PRODUCT),
                                                                                                                             monitored_output_states=[reward,
                                                                                                                                                      Decision.output_states[pnl.PROBABILITY_UPPER_THRESHOLD],
                                                                                                                                                      (Decision.output_states[pnl.RESPONSE_TIME], -1, 1)]),
                                                                                  function=pnl.GridSearch(),
                                                                                  control_signals=[("drift_rate", Decision),
                                                                                                   ("threshold", Decision)])
                                       )

        comp.enable_model_based_optimizer = True

        comp._analyze_graph()

        stim_list_dict = {
            Input: [0.5, 0.123],
            reward: [20, 20]
        }

        comp.run(inputs=stim_list_dict)

        # Note: Removed decision variable OutputState from simulation results because sign is chosen randomly
        expected_sim_results_array = [
            [[10.], [10.0], [0.0], [0.48999867], [0.50499983]],
            [[10.], [10.0], [0.0], [1.08965888], [0.51998934]],
            [[10.], [10.0], [0.0], [2.40680493], [0.53494295]],
            [[10.], [10.0], [0.0], [4.43671978], [0.549834]],
            [[10.], [10.0], [0.0], [0.48997868], [0.51998934]],
            [[10.], [10.0], [0.0], [1.08459402], [0.57932425]],
            [[10.], [10.0], [0.0], [2.36033556], [0.63645254]],
            [[10.], [10.0], [0.0], [4.24948962], [0.68997448]],
            [[10.], [10.0], [0.0], [0.48993479], [0.53494295]],
            [[10.], [10.0], [0.0], [1.07378304], [0.63645254]],
            [[10.], [10.0], [0.0], [2.26686573], [0.72710822]],
            [[10.], [10.0], [0.0], [3.90353015], [0.80218389]],
            [[10.], [10.0], [0.0], [0.4898672], [0.549834]],
            [[10.], [10.0], [0.0], [1.05791834], [0.68997448]],
            [[10.], [10.0], [0.0], [2.14222978], [0.80218389]],
            [[10.], [10.0], [0.0], [3.49637662], [0.88079708]],
            [[15.], [15.0], [0.0], [0.48999926], [0.50372993]],
            [[15.], [15.0], [0.0], [1.08981011], [0.51491557]],
            [[15.], [15.0], [0.0], [2.40822035], [0.52608629]],
            [[15.], [15.0], [0.0], [4.44259627], [0.53723096]],
            [[15.], [15.0], [0.0], [0.48998813], [0.51491557]],
            [[15.], [15.0], [0.0], [1.0869779], [0.55939819]],
            [[15.], [15.0], [0.0], [2.38198336], [0.60294711]],
            [[15.], [15.0], [0.0], [4.33535807], [0.64492386]],
            [[15.], [15.0], [0.0], [0.48996368], [0.52608629]],
            [[15.], [15.0], [0.0], [1.08085171], [0.60294711]],
            [[15.], [15.0], [0.0], [2.32712843], [0.67504223]],
            [[15.], [15.0], [0.0], [4.1221271], [0.7396981]],
            [[15.], [15.0], [0.0], [0.48992596], [0.53723096]],
            [[15.], [15.0], [0.0], [1.07165729], [0.64492386]],
            [[15.], [15.0], [0.0], [2.24934228], [0.7396981]],
            [[15.], [15.0], [0.0], [3.84279648], [0.81637827]]
        ]

        for simulation in range(len(expected_sim_results_array)):
            assert np.allclose(expected_sim_results_array[simulation],
                               # Note: Skip decision variable OutputState
                               comp.simulation_results[simulation][0:3] + comp.simulation_results[simulation][4:6])

        expected_results_array = [
            [[20.0], [20.0], [0.0], [1.0], [2.378055160151634], [0.9820137900379085]],
            [[20.0], [20.0], [0.0], [0.1], [0.48999967725112503], [0.5024599801509442]]
        ]

        for trial in range(len(expected_results_array)):
            np.testing.assert_allclose(comp.results[trial], expected_results_array[trial], atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(trial))

    def test_evc_gratton(self):
        # Stimulus Mechanisms
        target_stim = pnl.TransferMechanism(name='Target Stimulus',
                                            function=pnl.Linear(slope=0.3324))
        flanker_stim = pnl.TransferMechanism(name='Flanker Stimulus',
                                             function=pnl.Linear(slope=0.3545221843))

        # Processing Mechanisms (Control)
        Target_Rep = pnl.TransferMechanism(name='Target Representation')
        Flanker_Rep = pnl.TransferMechanism(name='Flanker Representation')

        # Processing Mechanism (Automatic)
        Automatic_Component = pnl.TransferMechanism(name='Automatic Component')

        # Decision Mechanism
        Decision = pnl.DDM(name='Decision',
                           function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0),
                                                                 threshold=(0.2645),
                                                                 noise=(0.5),
                                                                 starting_point=(0),
                                                                 t0=0.15),
                           output_states=[pnl.DECISION_VARIABLE,
                                          pnl.RESPONSE_TIME,
                                          pnl.PROBABILITY_UPPER_THRESHOLD]
                           )

        # Outcome Mechanism
        reward = pnl.TransferMechanism(name='reward')

        # Pathways
        target_control_pathway = [target_stim, Target_Rep, Decision]
        flanker_control_pathway = [flanker_stim, Flanker_Rep, Decision]
        target_automatic_pathway = [target_stim, Automatic_Component, Decision]
        flanker_automatic_pathway = [flanker_stim, Automatic_Component, Decision]
        pathways = [target_control_pathway, flanker_control_pathway, target_automatic_pathway,
                    flanker_automatic_pathway]

        # Composition
        evc_gratton = pnl.Composition(name="EVCGratton")
        evc_gratton.add_c_node(Decision, required_roles=pnl.CNodeRole.TERMINAL)
        for path in pathways:
            evc_gratton.add_linear_processing_pathway(path)
        evc_gratton.add_c_node(reward, required_roles=pnl.CNodeRole.TERMINAL)

        # Control Signals
        signalSearchRange = pnl.SampleSpec(start=1.0, stop=1.8, step=0.2)

        target_rep_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, Target_Rep)],
                                                      function=pnl.Linear,
                                                      variable=1.0,
                                                      intensity_cost_function=pnl.Exponential(rate=0.8046),
                                                      allocation_samples=signalSearchRange)

        flanker_rep_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, Flanker_Rep)],
                                                       function=pnl.Linear,
                                                       variable=1.0,
                                                       intensity_cost_function=pnl.Exponential(rate=0.8046),
                                                       allocation_samples=signalSearchRange)

        objective_mech = pnl.ObjectiveMechanism(function=pnl.LinearCombination(operation=pnl.PRODUCT),
                                                monitored_output_states=[reward,
                                                                         (Decision.output_states[
                                                                              pnl.PROBABILITY_UPPER_THRESHOLD], 1, -1)])
        # Model Based OCM (formerly controller)
        evc_gratton.add_model_based_optimizer(optimizer=pnl.OptimizationControlMechanism(agent_rep=evc_gratton,
                                                                                         features={
                                                                                             pnl.SHADOW_EXTERNAL_INPUTS: [
                                                                                                 target_stim,
                                                                                                 flanker_stim, reward]},
                                                                                         feature_function=pnl.AdaptiveIntegrator(
                                                                                             rate=1.0),
                                                                                         objective_mechanism=objective_mech,
                                                                                         function=pnl.GridSearch(),
                                                                                         control_signals=[
                                                                                             target_rep_control_signal,
                                                                                             flanker_rep_control_signal]))
        evc_gratton.enable_model_based_optimizer = True

        targetFeatures = [1, 1, 1]
        flankerFeatures = [1, -1, 1]
        rewardValues = [100, 100, 100]

        stim_list_dict = {target_stim: targetFeatures,
                          flanker_stim: flankerFeatures,
                          reward: rewardValues}

        evc_gratton.run(inputs=stim_list_dict)

        expected_results_array = [[[0.32257752863413636], [0.9481940753514433], [100.]],
                                  [[0.42963678062444666], [0.47661180945923376], [100.]],
                                  [[0.300291026852769], [0.97089165101931], [100.]]]

        expected_sim_results_array = [
            [[0.32257753], [0.94819408], [100.]],
            [[0.31663196], [0.95508757], [100.]],
            [[0.31093566], [0.96110142], [100.]],
            [[0.30548947], [0.96633839], [100.]],
            [[0.30029103], [0.97089165], [100.]],
            [[0.3169957], [0.95468427], [100.]],
            [[0.31128378], [0.9607499], [100.]],
            [[0.30582202], [0.96603252], [100.]],
            [[0.30060824], [0.9706259], [100.]],
            [[0.29563774], [0.97461444], [100.]],
            [[0.31163288], [0.96039533], [100.]],
            [[0.30615555], [0.96572397], [100.]],
            [[0.30092641], [0.97035779], [100.]],
            [[0.2959409], [0.97438178], [100.]],
            [[0.29119255], [0.97787196], [100.]],
            [[0.30649004], [0.96541272], [100.]],
            [[0.30124552], [0.97008732], [100.]],
            [[0.29624499], [0.97414704], [100.]],
            [[0.29148205], [0.97766847], [100.]],
            [[0.28694892], [0.98071974], [100.]],
            [[0.30156558], [0.96981445], [100.]],
            [[0.29654999], [0.97391021], [100.]],
            [[0.29177245], [0.97746315], [100.]],
            [[0.28722523], [0.98054192], [100.]],
            [[0.28289958], [0.98320731], [100.]],
            [[0.42963678], [0.47661181], [100.]],
            [[0.42846471], [0.43938586], [100.]],
            [[0.42628176], [0.40282965], [100.]],
            [[0.42314468], [0.36732207], [100.]],
            [[0.41913221], [0.333198], [100.]],
            [[0.42978939], [0.51176048], [100.]],
            [[0.42959394], [0.47427693], [100.]],
            [[0.4283576], [0.43708106], [100.]],
            [[0.4261132], [0.40057958], [100.]],
            [[0.422919], [0.36514906], [100.]],
            [[0.42902209], [0.54679323], [100.]],
            [[0.42980788], [0.50942101], [100.]],
            [[0.42954704], [0.47194318], [100.]],
            [[0.42824656], [0.43477897], [100.]],
            [[0.42594094], [0.3983337], [100.]],
            [[0.42735293], [0.58136855], [100.]],
            [[0.42910149], [0.54447221], [100.]],
            [[0.42982229], [0.50708112], [100.]],
            [[0.42949608], [0.46961065], [100.]],
            [[0.42813159], [0.43247968], [100.]],
            [[0.42482049], [0.61516258], [100.]],
            [[0.42749136], [0.57908829], [100.]],
            [[0.42917687], [0.54214925], [100.]],
            [[0.42983261], [0.50474093], [100.]],
            [[0.42944107], [0.46727945], [100.]],
            [[0.32257753], [0.94819408], [100.]],
            [[0.31663196], [0.95508757], [100.]],
            [[0.31093566], [0.96110142], [100.]],
            [[0.30548947], [0.96633839], [100.]],
            [[0.30029103], [0.97089165], [100.]],
            [[0.3169957], [0.95468427], [100.]],
            [[0.31128378], [0.9607499], [100.]],
            [[0.30582202], [0.96603252], [100.]],
            [[0.30060824], [0.9706259], [100.]],
            [[0.29563774], [0.97461444], [100.]],
            [[0.31163288], [0.96039533], [100.]],
            [[0.30615555], [0.96572397], [100.]],
            [[0.30092641], [0.97035779], [100.]],
            [[0.2959409], [0.97438178], [100.]],
            [[0.29119255], [0.97787196], [100.]],
            [[0.30649004], [0.96541272], [100.]],
            [[0.30124552], [0.97008732], [100.]],
            [[0.29624499], [0.97414704], [100.]],
            [[0.29148205], [0.97766847], [100.]],
            [[0.28694892], [0.98071974], [100.]],
            [[0.30156558], [0.96981445], [100.]],
            [[0.29654999], [0.97391021], [100.]],
            [[0.29177245], [0.97746315], [100.]],
            [[0.28722523], [0.98054192], [100.]],
            [[0.28289958], [0.98320731], [100.]],
        ]

        for trial in range(len(evc_gratton.results)):
            assert np.allclose(expected_results_array[trial],
                               # Note: Skip decision variable OutputState
                               evc_gratton.results[trial][1:])
        for simulation in range(len(evc_gratton.simulation_results)):
            assert np.allclose(expected_sim_results_array[simulation],
                               # Note: Skip decision variable OutputState
                               evc_gratton.simulation_results[simulation][1:])

class TestSampleIterator:

    def test_int_step(self):
        spec = SampleSpec(step=2,
                          start=0,
                          stop=10)
        sample_iterator = SampleIterator(specification=spec)

        expected = [0, 2, 4, 6, 8, 10]

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_int_num(self):
        spec = SampleSpec(num=6,
                          start=0,
                          stop=10)
        sample_iterator = SampleIterator(specification=spec)

        expected = [0, 2, 4, 6, 8, 10]

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(6):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_neither_num_nor_step(self):
        with pytest.raises(OptimizationFunctionError) as error_text:
            SampleSpec(start=0,
                       stop=10)
        assert "Must specify one of 'step', 'num' or 'function'" in str(error_text.value)

    def test_float_step(self):
        # Need to decide whether stop should be exclusive
        spec = SampleSpec(step=2.79,
                          start=0.65,
                          stop=10.25)
        sample_iterator = SampleIterator(specification=spec)

        expected = [0.65, 3.44, 6.23, 9.02]

        for i in range(4):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(4):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_function(self):
        fun = pnl.NormalDist(mean=5.0).function
        spec = SampleSpec(function=fun)
        sample_iterator = SampleIterator(specification=spec)

        expected = [5.400157208367223, 5.978737984105739, 7.240893199201458, 6.867557990149967, 4.022722120123589]

        for i in range(5):
            assert np.allclose(next(sample_iterator), expected[i])

    def test_function_with_num(self):
        fun = pnl.NormalDist(mean=5.0).function
        spec = SampleSpec(function=fun,
                          num=4)
        sample_iterator = SampleIterator(specification=spec)

        expected = [5.400157208367223, 5.978737984105739, 7.240893199201458, 6.867557990149967]

        for i in range(4):
            assert np.allclose(next(sample_iterator), expected[i])

        assert next(sample_iterator, None) is None

    def test_list(self):
        sample_list = [1, 2.0, 3.456, 7.8]
        sample_iterator = SampleIterator(specification=sample_list)

        for i in range(len(sample_list)):
            assert np.allclose(next(sample_iterator), sample_list[i])

        assert next(sample_iterator, None) is None

        sample_iterator.reset()

        for i in range(len(sample_list)):
            assert np.allclose(next(sample_iterator), sample_list[i])

        assert next(sample_iterator, None) is None

        assert sample_iterator.start == 1
        assert sample_iterator.stop == 7.8
        assert sample_iterator.num == len(sample_list)
