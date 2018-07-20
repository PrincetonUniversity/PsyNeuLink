import psyneulink as pnl

class TestControlMechanisms:
    def test_lc_control_mechanism(self):
        T_1 = pnl.TransferMechanism(name='T_1')
        T_2 = pnl.TransferMechanism(name='T_2')

        LC = pnl.LCControlMechanism(monitor_for_control=[T_1, T_2],
                                    modulated_mechanisms=pnl.ALL
                                    )
        T_3 = pnl.TransferMechanism()
        comp = pnl.Composition()
        comp.add_linear_processing_pathway([T_1, T_2, LC, T_3])
        # comp.add_c_node(T_1)
        # comp.add_c_node(T_2)
        # comp.add_c_node(LC)
        # comp.add_projection(T_1, pnl.MappingProjection(), T_2)
        # comp.add_projection(T_2, pnl.MappingProjection(), LC)
        comp._analyze_graph()
        # comp._add_c_node_role(LC, pnl.CNodeRole.TERMINAL)

        assert len(LC.control_signals) == 1
        assert len(LC.control_signals[0].efferents) == 2
        assert T_1.parameter_states[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents
        assert T_2.parameter_states[pnl.SLOPE].mod_afferents[0] in LC.control_signals[0].efferents

    def test_control_mechanism(self):
        Tx = pnl.TransferMechanism(name='Tx')
        Ty = pnl.TransferMechanism(name='Ty')
        Tz = pnl.TransferMechanism(name='Tz')
        C =  pnl.ControlMechanism(
                # function=pnl.Linear,
                default_variable=[1],
                monitor_for_control=Ty,
                control_signals=pnl.ControlSignal(modulation=pnl.OVERRIDE,
                                                  projections=(pnl.SLOPE,Tz)))
        comp = pnl.Composition()
        comp.add_c_node(Tx)
        comp.add_c_node(Tz)
        comp.add_c_node(Ty)
        comp.add_c_node(C)
        comp.add_projection(Tx, pnl.MappingProjection(), Tz)
        comp.add_projection(Ty, pnl.MappingProjection(), C)
        # comp.add_linear_processing_pathway([Tx,Tz])
        # comp.add_linear_processing_pathway([Ty, C])
        # P1=pnl.Process(pathway=[Tx,Tz])
        # P2=pnl.Process(pathway=[Ty, C])
        # S=pnl.System(processes=[P1, P2])

        assert Tz.parameter_states[pnl.SLOPE].mod_afferents[0].sender.owner == C
        result = comp.run(inputs={Tx:[1,1], Ty:[4,4]})
        assert result == [[[4.], [4.]], [[4.], [4.]]]