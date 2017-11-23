import pytest

import psyneulink as pnl


@pytest.fixture(scope='module')
def clear_registry():
    # Clear Registry to have a stable reference for indexed suffixes of default names
    from psyneulink.components.component import DeferredInitRegistry
    from psyneulink.components.mechanisms.mechanism import MechanismRegistry
    from psyneulink.components.projections.projection import ProjectionRegistry
    pnl.clear_registry(DeferredInitRegistry)
    pnl.clear_registry(MechanismRegistry)
    pnl.clear_registry(ProjectionRegistry)


@pytest.mark.usefixtures('clear_registry')
class TestNaming:
    # ------------------------------------------------------------------------------------------------

    # NAMING CONVENTIONS

    # ------------------------------------------------------------------------------------------------
    # TEST 1
    # Test that Processes and Systems are given default names with incrementing index, starting at 0
    def test_process_and_system_default_names(self):

        T = pnl.TransferMechanism()
        P1 = pnl.Process(pathway=[T])
        P2 = pnl.Process(pathway=[T])
        assert P1.name == 'Process-0'
        assert P2.name == 'Process-1'
        S1 = pnl.System(processes=[P1])
        S2 = pnl.System(processes=[P1])
        assert S1.name == 'System-0'
        assert S2.name == 'System-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 2
    # Test that Processes and Systems assigned duplicate names are indexed starting at 1 (original is not indexed)
    def test_process_and_system_default_names(self):

        T = pnl.TransferMechanism(name='T0')
        P1 = pnl.Process(name='MY PROCESS', pathway=[T])
        P2 = pnl.Process(name='MY PROCESS', pathway=[T])
        assert P1.name == 'MY PROCESS'
        assert P2.name == 'MY PROCESS-1'
        S1 = pnl.System(name='MY SYSTEM', processes=[P1])
        S2 = pnl.System(name='MY SYSTEM', processes=[P1])
        assert S1.name == 'MY SYSTEM'
        assert S2.name == 'MY SYSTEM-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 3
    # Test that Mechanisms are given default names with incrementing index, starting at 0

    def test_default_mechanism_names(self):
        T1 = pnl.TransferMechanism()
        T2 = pnl.TransferMechanism()
        assert T1.name == 'TransferMechanism-0'
        assert T2.name == 'TransferMechanism-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 4
    # Test that Mechanism assigned a duplicate name is incremented starting at 1 (original is not indexed)

    def test_duplicate_assigned_mechanism_names(self):
        TN1 = pnl.TransferMechanism(name='MY TRANSFER MECHANISM')
        TN2 = pnl.TransferMechanism(name='MY TRANSFER MECHANISM')
        assert TN1.name == 'MY TRANSFER MECHANISM'
        assert TN2.name == 'MY TRANSFER MECHANISM-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 5
    # Test that default MappingProjections in deferred init are assigned indexed names

    def test_deferred_init_default_MappingProjection_names(self):
        P1 = pnl.MappingProjection()
        P2 = pnl.MappingProjection()
        assert P1.name == 'Deferred Init MappingProjection'
        assert P2.name == 'Deferred Init MappingProjection-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 6
    # Test that MappingProjections with assigned names in deferred init are assigned indexed names

    def test_deferred_init_assigned_MappingProjection_names(self):
        PN1 = pnl.MappingProjection(name='MY PROJECTION')
        PN2 = pnl.MappingProjection(name='MY PROJECTION')
        assert PN1.name == 'MY PROJECTION [Deferred Init]'
        assert PN2.name == 'MY PROJECTION [Deferred Init]-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 7
    # Test that default ModulatoryProjections in deferred init are assigned indexed names

    def test_deferred_init_default_ModulatoryProjection_names(self):
        LP1 = pnl.LearningProjection()
        LP2 = pnl.LearningProjection()
        assert LP1.name == 'Deferred Init LearningProjection'
        assert LP2.name == 'Deferred Init LearningProjection-1'

        CP1 = pnl.ControlProjection()
        CP2 = pnl.ControlProjection()
        assert CP1.name == 'Deferred Init ControlProjection'
        assert CP2.name == 'Deferred Init ControlProjection-1'

        GP1 = pnl.GatingProjection()
        GP2 = pnl.GatingProjection()
        assert GP1.name == 'Deferred Init GatingProjection'
        assert GP2.name == 'Deferred Init GatingProjection-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 8
    # Test that objects of different types can have the same name

    def test_different_object_types_with_same_names(self):
        T1 = pnl.TransferMechanism(name='MY NAME')
        T2 = pnl.TransferMechanism(name='MY NAME')
        P1 = pnl.MappingProjection(sender=T1, receiver=T2, name='MY NAME')
        assert T1.name == 'MY NAME'
        assert P1.name == 'MY NAME'

    # ------------------------------------------------------------------------------------------------
    # TEST 9
    # Test that InputStates and Projections constructed on their own and assigned are properly named

    def test_input_state_and_assigned_projection_names(self):
        T1 = pnl.TransferMechanism(name='T1')
        T2 = pnl.TransferMechanism(name='T2', input_states=[T1])
        I1 = pnl.InputState(owner=T2)
        I2 = pnl.InputState(projections=[T1])
        assert I2.name == 'Deferred Init InputState'
        T2.add_states([I2])
        assert I1.name == 'InputState-1'
        assert I2.name == 'InputState-2'
        assert T2.input_states[0].path_afferents[0].name == \
               'MappingProjection from T1[RESULTS] to T2[InputState-0]'
        assert T2.input_states[2].path_afferents[0].name == \
               'MappingProjection from T1[RESULTS] to T2[InputState-2]'

    # ------------------------------------------------------------------------------------------------
    # TEST 10
    # Test that OutputStates are properly named

        T1 = pnl.TransferMechanism(output_states=['MY OUTPUT_STATE',[0]])
        assert T1.output_states[0].name == 'MY OUTPUT_STATE'
        assert T1.output_states[1].name == 'OutputState-0'
        O = pnl.OutputState(owner=T1)
        assert T1.output_states[2].name == 'OutputState-1'
        O2 = pnl.OutputState()
        T1.add_states([O2])
        assert T1.output_states[3].name == 'OutputState-2'

    # ------------------------------------------------------------------------------------------------
    # TEST 11
    # Test that ControlSignals and ControlProjections are properly named

    def test_control_signal_and_control_projection_names(self):
        D1 = pnl.DDM(name='D1')
        D2 = pnl.DDM(name='D2')

        # ControlSignal with one ControlProjection
        C1 = pnl.ControlMechanism(control_signals=[D1.parameter_states[pnl.DRIFT_RATE]])
        assert C1.control_signals[0].name == 'D1[drift_rate] ControlSignal'
        assert C1.control_signals[0].efferents[0].name == 'ControlProjection for D1[drift_rate]'

        # ControlSignal with two ControlProjection to two parameters of same Mechanism
        C2 = pnl.ControlMechanism(control_signals=[{pnl.PROJECTIONS:[D1.parameter_states[pnl.DRIFT_RATE],
                                                                     D1.parameter_states[pnl.THRESHOLD]]}])
        assert C2.control_signals[0].name == 'D1[drift_rate, threshold] ControlSignal'
        assert C2.control_signals[0].efferents[0].name == 'ControlProjection for D1[drift_rate]'
        assert C2.control_signals[0].efferents[1].name == 'ControlProjection for D1[threshold]'

        # ControlSignal with two ControlProjection to two parameters of different Mechanisms
        C3 = pnl.ControlMechanism(control_signals=[{pnl.PROJECTIONS:[D1.parameter_states[pnl.DRIFT_RATE],
                                                                     D2.parameter_states[pnl.DRIFT_RATE]]}])
        assert C3.control_signals[0].name == 'ControlSignal-0 divergent ControlSignal'
        assert C3.control_signals[0].efferents[0].name == 'ControlProjection for D1[drift_rate]'
        assert C3.control_signals[0].efferents[1].name == 'ControlProjection for D2[drift_rate]'

    # ------------------------------------------------------------------------------------------------
    # TEST 12
    # Test that GatingSignals and GatingProjections are properly named

    def test_gating_signal_and_gating_projection_names(self):
        T3 = pnl.TransferMechanism(name='T3')
        T4 = pnl.TransferMechanism(name='T4', input_states=['First State','Second State'])

        # GatingSignal with one GatingProjection
        G1 = pnl.GatingMechanism(gating_signals=[T3])
        assert G1.gating_signals[0].name == 'T3[InputState-0] GatingSignal'
        assert G1.gating_signals[0].efferents[0].name == 'GatingProjection for T3[InputState-0]'

        # GatingSignal with two GatingProjections to two States of same Mechanism
        G2 = pnl.GatingMechanism(gating_signals=[{pnl.PROJECTIONS:[T4.input_states[0], T4.input_states[1]]}])
        assert G2.gating_signals[0].name == 'T4[First State, Second State] GatingSignal'
        assert G2.gating_signals[0].efferents[0].name == 'GatingProjection for T4[First State]'
        assert G2.gating_signals[0].efferents[1].name == 'GatingProjection for T4[Second State]'

        # GatingSignal with two GatingProjections to two States of different Mechanisms
        G3 = pnl.GatingMechanism(gating_signals=[{pnl.PROJECTIONS:[T3, T4]}])
        assert G3.gating_signals[0].name == 'GatingSignal-0 divergent GatingSignal'
        assert G3.gating_signals[0].efferents[0].name == 'GatingProjection for T3[InputState-0]'
        assert G3.gating_signals[0].efferents[1].name == 'GatingProjection for T4[First State]'

        # GatingProjections to ProcessingMechanism from GatingSignals of existing GatingMechanism
        T5 = pnl.TransferMechanism(name='T5',
                                   input_states=[T3.output_states[pnl.RESULTS],
                                                 G3.gating_signals['GatingSignal-0 divergent GatingSignal']],
                                   output_states=[G3.gating_signals['GatingSignal-0 divergent GatingSignal']])
