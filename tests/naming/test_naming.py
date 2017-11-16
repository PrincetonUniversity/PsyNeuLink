import pytest

import psyneulink as pnl


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
        T1 = pnl.TransferMechanism(name='MY NAME')
        T2 = pnl.TransferMechanism(name='MY NAME')
        P1 = pnl.MappingProjection(sender=T1, receiver=T2, name='MY NAME')
        assert T1.name == 'MY NAME'
        assert P1.name == 'MY NAME'

    # ------------------------------------------------------------------------------------------------
    # TEST 9
    # Test that InputStates and Projections constructed on their own and assigned are properly named

        T1 = pnl.TransferMechanism(name='T1')
        T2 = pnl.TransferMechanism(name='T2', input_states=[T1])
        I1 = pnl.InputState(owner=T2)
        I2 = pnl.InputState(projections=[T1])
        assert I2.name == 'Deferred Init InputState'
        T2.add_states([I2])
        assert I1.name == 'InputState-1'
        assert I2.name == 'InputState-2'
        assert T2.input_states[0].path_afferents[0].name == \
               'MappingProjection from T1[RESULT] to T2[InputState-0]'
        assert T2.input_states[2].path_afferents[0].name == \
               'MappingProjection from T1[RESULT] to T2[InputState-2]'