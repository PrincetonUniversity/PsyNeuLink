import numpy as np
import psyneulink as pnl
import pytest


class TestLCControlMechanism:

    @pytest.mark.mechanism
    @pytest.mark.control_mechanism
    @pytest.mark.benchmark(group="LCControlMechanism Default")
    @pytest.mark.parametrize("mode", ['Python'])
    def test_default_lc_control_mechanism(self, benchmark, mode):
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

        P = pnl.Process(pathway=[A, B, LC])
        S = pnl.System(processes=[P])
        LC.reinitialize_when = pnl.Never()
        # THIS CURRENTLY DOES NOT WORK:
        # P = pnl.Process(pathway=[A, B])
        # P2 = pnl.Process(pathway=[LC])
        # S = pnl.System(processes=[P, P2])
        # S.show_graph()

        gain_created_by_LC_output_state_1 = []
        mod_gain_assigned_to_A = []
        base_gain_assigned_to_A = []
        mod_gain_assigned_to_B = []
        base_gain_assigned_to_B = []

        def report_trial():
            gain_created_by_LC_output_state_1.append(LC.output_states[0].value[0])
            mod_gain_assigned_to_A.append(A.mod_gain)
            mod_gain_assigned_to_B.append(B.mod_gain)
            base_gain_assigned_to_A.append(A.function_object.gain)
            base_gain_assigned_to_B.append(B.function_object.gain)

        benchmark(S.run, inputs={A: [[1.0], [1.0], [1.0], [1.0], [1.0]]},
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


    @pytest.mark.mechanism
    @pytest.mark.control_mechanism
    @pytest.mark.benchmark(group="LCControlMechanism Basic")
    @pytest.mark.parametrize("mode", ['Python'])
    def test_lc_control_mech_basic(self, benchmark, mode):

        LC = pnl.LCControlMechanism(
            base_level_gain=3.0,
            scaling_factor_gain=0.5
        )
        val = LC.execute([[10.0]])
        assert np.allclose(np.asfarray(val).flatten(), [3.00139776,  0.512152259, .00279552477, 0.05000])
        val = benchmark(LC.execute, [[10.0]])

    def test_lc_control_modulated_mechanisms_all(self):

        T_1 = pnl.TransferMechanism(name='T_1')
        T_2 = pnl.TransferMechanism(name='T_2')

        LC = pnl.LCControlMechanism(monitor_for_control=[T_1, T_2],
                                    modulated_mechanisms=pnl.ALL
                                    )
        S = pnl.System(processes=[pnl.proc(T_1, T_2, LC)])
        assert len(LC.control_signals)==1
        assert len(LC.control_signals[0].efferents)==2
        assert T_1.parameter_states[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents
        assert T_2.parameter_states[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents


    def test_control_modulation(self):
        Tx = pnl.TransferMechanism(name='Tx')
        Ty = pnl.TransferMechanism(name='Ty')
        Tz = pnl.TransferMechanism(name='Tz')
        C =  pnl.ControlMechanism(
                # function=pnl.Linear,
                default_variable=[1],
                monitor_for_control=Ty,
                control_signals=pnl.ControlSignal(modulation=pnl.OVERRIDE,
                                                  projections=(pnl.SLOPE,Tz)))
        P1=pnl.Process(pathway=[Tx,Tz])
        P2=pnl.Process(pathway=[Ty, C])
        S=pnl.System(processes=[P1, P2])

        assert Tz.parameter_states[pnl.SLOPE].mod_afferents[0].sender.owner == C
        result = S.run(inputs={Tx:[1,1], Ty:[4,4]})
        assert result == [[[4.], [4.]], [[4.], [4.]]]
