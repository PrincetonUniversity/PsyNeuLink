import psyneulink as pnl
import numpy as np

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
    def test_control_mechanism(self):
        Tx = pnl.TransferMechanism(name='Tx')
        Ty = pnl.TransferMechanism(name='Ty')
        Tz = pnl.TransferMechanism(name='Tz')
        C = pnl.ControlMechanism(default_variable=[1],
                                 monitor_for_control=Ty,
                                 control_signals=pnl.ControlSignal(modulation=pnl.OVERRIDE,
                                                                   projections=(pnl.SLOPE, Tz)))
        comp = pnl.Composition()
        comp.add_linear_processing_pathway([Tx, Tz])
        comp.add_linear_processing_pathway([Ty, C])

        assert Tz.parameter_states[pnl.SLOPE].mod_afferents[0].sender.owner == C
        result = comp.run(inputs={Tx: [1, 1],
                                  Ty: [4, 4]})
        assert np.allclose(result, [[[4.], [4.]],
                                    [[4.], [4.]]])