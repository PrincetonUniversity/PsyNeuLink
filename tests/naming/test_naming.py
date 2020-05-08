import pytest

import psyneulink as pnl
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.statefulfunctions.integratorfunctions


class TestNaming:
    # ------------------------------------------------------------------------------------------------

    # NAMING CONVENTIONS

# ------------------------------------------------------------------------------------------------
    # TEST 1
    # Test that Compositions are given default names with incrementing index, starting at 0

    def test_composition_names(self):

        T = pnl.TransferMechanism()
        C1 = pnl.Composition(pathways=[T])
        C2 = pnl.Composition(pathway=[T])
        assert C1.name == 'Composition-0'
        assert C2.name == 'Composition-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 2
    # Test that Processes and Systems assigned duplicate names are indexed starting at 1 (original is not indexed)

    def test_composition_default_names_2(self):

        T = pnl.TransferMechanism(name='T0')
        C1 = pnl.Composition(name='MY COMPOSITION', pathways=[T])
        C2 = pnl.Composition(name='MY COMPOSITION', pathways=[T])
        assert C1.name == 'MY COMPOSITION'
        assert C2.name == 'MY COMPOSITION-1'

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

    @pytest.mark.parametrize(
        'name, expected_list',
        [
            ('MY TRANSFER MECHANISM', ['MY TRANSFER MECHANISM', 'MY TRANSFER MECHANISM-1']),
            ('A-1', ['A-1', 'A-1-1', 'A-1-2']),
            ('A', ['A', 'A-1', 'A-2']),
        ]
    )
    def test_duplicate_assigned_mechanism_names(self, name, expected_list):
        for expected_name in expected_list:
            t = pnl.TransferMechanism(name=name)
            assert t.name == expected_name

    def test_duplicate_assigned_mechanism_names_2(self):
        pnl.TransferMechanism(name='A')
        pnl.TransferMechanism(name='A')  # A-1
        pnl.TransferMechanism(name='A')  # A-2
        t = pnl.TransferMechanism(name='A-1')

        assert t.name == 'A-3'

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
    # Test that InputPorts and Projections constructed on their own and assigned are properly named

    def test_input_port_and_assigned_projection_names(self):
        T1 = pnl.TransferMechanism(name='T1')
        T2 = pnl.TransferMechanism(name='T2', input_ports=[T1])
        I1 = pnl.InputPort(owner=T2)
        I2 = pnl.InputPort(projections=[T1])
        assert I2.name == 'Deferred Init InputPort'
        T2.add_ports([I2])
        assert I1.name == 'InputPort-1'
        assert I2.name == 'InputPort-2'
        assert T2.input_ports[0].path_afferents[0].name == \
               'MappingProjection from T1[RESULT] to T2[InputPort-0]'
        assert T2.input_ports[2].path_afferents[0].name == \
               'MappingProjection from T1[RESULT] to T2[InputPort-2]'

    # ------------------------------------------------------------------------------------------------
    # TEST 10
    # Test that OutputPorts are properly named

        T1 = pnl.TransferMechanism(output_ports=['MY OUTPUT_PORT',[0]])
        assert T1.output_ports[0].name == 'MY OUTPUT_PORT'
        assert T1.output_ports[1].name == 'OutputPort-0'
        O = pnl.OutputPort(owner=T1)
        assert T1.output_ports[2].name == 'OutputPort-1'
        O2 = pnl.OutputPort()
        T1.add_ports([O2])
        assert T1.output_ports[3].name == 'OutputPort-2'

    # ------------------------------------------------------------------------------------------------
    # TEST 11
    # Test that ControlSignals and ControlProjections are properly named

    def test_control_signal_and_control_projection_names(self):
        D1 = pnl.DDM(name='D1')
        D2 = pnl.DDM(name='D2')

        # ControlSignal with one ControlProjection
        C1 = pnl.ControlMechanism(control_signals=[D1.parameter_ports[
                                                       psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE]])
        assert C1.control_signals[0].name == 'D1[drift_rate] ControlSignal'
        assert C1.control_signals[0].efferents[0].name == 'ControlProjection for D1[drift_rate]'

        # ControlSignal with two ControlProjection to two parameters of same Mechanism
        C2 = pnl.ControlMechanism(control_signals=[{pnl.PROJECTIONS:[D1.parameter_ports[
                                                                         psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE],
                                                                     D1.parameter_ports[
                                                                         psyneulink.core.globals.keywords.THRESHOLD]]}])
        assert C2.control_signals[0].name == 'D1[drift_rate, threshold] ControlSignal'
        assert C2.control_signals[0].efferents[0].name == 'ControlProjection for D1[drift_rate]'
        assert C2.control_signals[0].efferents[1].name == 'ControlProjection for D1[threshold]'

        # ControlSignal with two ControlProjection to two parameters of different Mechanisms
        C3 = pnl.ControlMechanism(control_signals=[{pnl.PROJECTIONS:[D1.parameter_ports[
                                                                         psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE],
                                                                     D2.parameter_ports[
                                                                         psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE]]}])
        assert C3.control_signals[0].name == 'ControlSignal-0 divergent ControlSignal'
        assert C3.control_signals[0].efferents[0].name == 'ControlProjection for D1[drift_rate]'
        assert C3.control_signals[0].efferents[1].name == 'ControlProjection for D2[drift_rate]'

    # ------------------------------------------------------------------------------------------------
    # TEST 12
    # Test that GatingSignals and GatingProjections are properly named

    def test_gating_signal_and_gating_projection_names(self):
        T3 = pnl.TransferMechanism(name='T3')
        T4 = pnl.TransferMechanism(name='T4', input_ports=['First Port','Second Port'])

        # GatingSignal with one GatingProjection
        G1 = pnl.GatingMechanism(gating_signals=[T3])
        assert G1.gating_signals[0].name == 'T3[InputPort-0] GatingSignal'
        assert G1.gating_signals[0].efferents[0].name == 'GatingProjection for T3[InputPort-0]'

        # GatingSignal with two GatingProjections to two Ports of same Mechanism
        G2 = pnl.GatingMechanism(gating_signals=[{pnl.PROJECTIONS:[T4.input_ports[0], T4.input_ports[1]]}])
        assert G2.gating_signals[0].name == 'T4[First Port, Second Port] GatingSignal'
        assert G2.gating_signals[0].efferents[0].name == 'GatingProjection for T4[First Port]'
        assert G2.gating_signals[0].efferents[1].name == 'GatingProjection for T4[Second Port]'

        # GatingSignal with two GatingProjections to two Ports of different Mechanisms
        G3 = pnl.GatingMechanism(gating_signals=[{pnl.PROJECTIONS:[T3, T4]}])
        assert G3.gating_signals[0].name == 'GatingSignal-0 divergent GatingSignal'
        assert G3.gating_signals[0].efferents[0].name == 'GatingProjection for T3[InputPort-0]'
        assert G3.gating_signals[0].efferents[1].name == 'GatingProjection for T4[First Port]'

        # GatingProjections to ProcessingMechanism from GatingSignals of existing GatingMechanism
        T5 = pnl.TransferMechanism(name='T5',
                                   input_ports=[T3.output_ports[pnl.RESULT],
                                                 G3.gating_signals['GatingSignal-0 divergent GatingSignal']],
                                   output_ports=[G3.gating_signals['GatingSignal-0 divergent GatingSignal']])

    def test_composition_names(self):
        C1 = pnl.Composition()
        C2 = pnl.Composition()
        C3 = pnl.Composition()

        assert C1.name == 'Composition-0'
        assert C2.name == 'Composition-1'
        assert C3.name == 'Composition-2'
