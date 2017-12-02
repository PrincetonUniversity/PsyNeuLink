import psyneulink as pnl

class TestProjectionSpecificationFormats:

    def test_multiple_modulatory_projection_specs(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{pnl.PROJECTIONS: [M.parameter_states[pnl.DRIFT_RATE],
                                                                     M.parameter_states[pnl.THRESHOLD]]}])
        G = pnl.GatingMechanism(gating_signals=[{pnl.PROJECTIONS: [M.output_states[pnl.DECISION_VARIABLE],
                                                                     M.output_states[pnl.RESPONSE_TIME]]}])
        assert len(C.control_signals)==1
        assert len(C.control_signals[0].efferents)==2
        assert M.parameter_states[pnl.DRIFT_RATE].mod_afferents[0]==C.control_signals[0].efferents[0]
        assert M.parameter_states[pnl.THRESHOLD].mod_afferents[0]==C.control_signals[0].efferents[1]
        assert len(G.gating_signals)==1
        assert len(G.gating_signals[0].efferents)==2
        assert M.output_states[pnl.DECISION_VARIABLE].mod_afferents[0]==G.gating_signals[0].efferents[0]
        assert M.output_states[pnl.RESPONSE_TIME].mod_afferents[0]==G.gating_signals[0].efferents[1]

    def test_multiple_modulatory_projections_with_state_name(self):
        
        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{'DECISION_CONTROL':[M.parameter_states[pnl.DRIFT_RATE],
                                                                       M.parameter_states[pnl.THRESHOLD]]}])
        G = pnl.GatingMechanism(gating_signals=[{'DDM_OUTPUT_GATE':[M.output_states[pnl.DECISION_VARIABLE],
                                                                    M.output_states[pnl.RESPONSE_TIME]]}])
        assert len(C.control_signals)==1
        assert C.control_signals[0].name=='DECISION_CONTROL'
        assert len(C.control_signals[0].efferents)==2
        assert M.parameter_states[pnl.DRIFT_RATE].mod_afferents[0]==C.control_signals[0].efferents[0]
        assert M.parameter_states[pnl.THRESHOLD].mod_afferents[0]==C.control_signals[0].efferents[1]
        assert len(G.gating_signals)==1
        assert G.gating_signals[0].name=='DDM_OUTPUT_GATE'
        assert len(G.gating_signals[0].efferents)==2
        assert M.output_states[pnl.DECISION_VARIABLE].mod_afferents[0]==G.gating_signals[0].efferents[0]
        assert M.output_states[pnl.RESPONSE_TIME].mod_afferents[0]==G.gating_signals[0].efferents[1]

    def test_multiple_modulatory_projections_with_mech_and_state_name_specs(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{pnl.MECHANISM: M,
                                                   pnl.PARAMETER_STATES: [pnl.DRIFT_RATE, pnl.THRESHOLD]}])
        G = pnl.GatingMechanism(gating_signals=[{pnl.MECHANISM: M,
                                                 pnl.OUTPUT_STATES: [pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME]}])
        assert len(C.control_signals)==1
        assert len(C.control_signals[0].efferents)==2
        assert M.parameter_states[pnl.DRIFT_RATE].mod_afferents[0]==C.control_signals[0].efferents[0]
        assert M.parameter_states[pnl.THRESHOLD].mod_afferents[0]==C.control_signals[0].efferents[1]
        assert len(G.gating_signals)==1
        assert len(G.gating_signals[0].efferents)==2
        assert M.output_states[pnl.DECISION_VARIABLE].mod_afferents[0]==G.gating_signals[0].efferents[0]
        assert M.output_states[pnl.RESPONSE_TIME].mod_afferents[0]==G.gating_signals[0].efferents[1]

    def test_mapping_projection_with_mech_and_state_name_specs(self):
         R1 = pnl.TransferMechanism(output_states=['OUTPUT_1', 'OUTPUT_2'])
         R2 = pnl.TransferMechanism(default_variable=[[0],[0]],
                                    input_states=['INPUT_1', 'INPUT_2'])
         T = pnl.TransferMechanism(input_states=[{pnl.MECHANISM: R1,
                                                  pnl.OUTPUT_STATES: ['OUTPUT_1', 'OUTPUT_2']}],
                                   output_states=[{pnl.MECHANISM:R2,
                                                   pnl.INPUT_STATES: ['INPUT_1', 'INPUT_2']}])
         assert len(R1.output_states)==2
         assert len(R2.input_states)==2
         assert len(T.input_states)==1
         for input_state in T.input_states:
             for projection in input_state.path_afferents:
                 assert projection.sender.owner is R1
         assert len(T.output_states)==1
         for output_state in T.output_states:
             for projection in output_state.efferents:
                 assert projection.receiver.owner is R2

    def test_mapping_projection_using_2_item_tuple_with_list_of_state_names(self):

        T1 = pnl.TransferMechanism(name='T1', input_states=[[0,0],[0,0,0]])
        T2 = pnl.TransferMechanism(name='T2',
                                   output_states=[(['InputState-0','InputState-1'], T1)])
        assert len(T2.output_states)==1
        assert T2.output_states[0].efferents[0].receiver.name == 'InputState-0'
        assert T2.output_states[0].efferents[0].matrix.shape == (1,2)
        assert T2.output_states[0].efferents[1].receiver.name == 'InputState-1'
        assert T2.output_states[0].efferents[1].matrix.shape == (1,3)

    def test_mapping_projection_using_2_item_tuple_with_index_specs(self):

        T1 = pnl.TransferMechanism(name='T1', input_states=[[0,0],[0,0,0]])
        T2 = pnl.TransferMechanism(name='T2',
                                   input_states=['a','b','c'],
                                   output_states=[(['InputState-0','InputState-1'], T1),
                                                  ('InputState-0', 2, T1),
                                                  (['InputState-0','InputState-1'], 1, T1)])
        assert len(T2.output_states)==3
        assert T2.output_states[0].efferents[0].receiver.name == 'InputState-0'
        assert T2.output_states[0].efferents[0].matrix.shape == (1,2)
        assert T2.output_states[0].efferents[1].receiver.name == 'InputState-1'
        assert T2.output_states[0].efferents[1].matrix.shape == (1,3)
        assert T2.output_states[1].index == 2
        assert T2.output_states[2].index == 1

    def test_2_item_tuple_from_control_signal_to_parameter_state(self):

        D = pnl.DDM(name='D')

        # Single name
        C = pnl.ControlMechanism(control_signals=[(pnl.DRIFT_RATE, D)])
        assert C.control_signals[0].name == 'D[drift_rate] ControlSignal'
        assert C.control_signals[0].efferents[0].receiver.name == 'drift_rate'

        # List of names
        C = pnl.ControlMechanism(control_signals=[([pnl.DRIFT_RATE, pnl.THRESHOLD], D)])
        assert C.control_signals[0].name == 'D[drift_rate, threshold] ControlSignal'
        assert C.control_signals[0].efferents[0].receiver.name == 'drift_rate'
        assert C.control_signals[0].efferents[1].receiver.name == 'threshold'

    def test_2_item_tuple_from_parameter_state_to_control_signals(self):

        C = pnl.ControlMechanism(control_signals=['a','b'])
        D = pnl.DDM(name='D3',
                     function=pnl.BogaczEtAl(drift_rate=(3,C),
                                             threshold=(2,C.control_signals['b']))
                    )
        assert D.parameter_states[pnl.DRIFT_RATE].mod_afferents[0].sender==C.control_signals[0]
        assert D.parameter_states[pnl.THRESHOLD].mod_afferents[0].sender==C.control_signals[1]

    def test_2_item_tuple_from_gating_signal_to_output_states(self):

        D4 = pnl.DDM(name='D4')

        # Single name
        G = pnl.GatingMechanism(gating_signals=[(pnl.DECISION_VARIABLE, D4)])
        assert G.gating_signals[0].name == 'D4[DECISION_VARIABLE] GatingSignal'
        assert G.gating_signals[0].efferents[0].receiver.name == 'DECISION_VARIABLE'

        # List of names
        G = pnl.GatingMechanism(gating_signals=[([pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME], D4)])
        assert G.gating_signals[0].name == 'D4[DECISION_VARIABLE, RESPONSE_TIME] GatingSignal'
        assert G.gating_signals[0].efferents[0].receiver.name == 'DECISION_VARIABLE'
        assert G.gating_signals[0].efferents[1].receiver.name == 'RESPONSE_TIME'

    def test_2_item_tuple_from_input_and_output_states_to_gating_signals(self):

        G = pnl.GatingMechanism(gating_signals=['a','b'])
        T = pnl.TransferMechanism(name='T',
                     input_states=[(3,G)],
                     output_states=[(2,G.gating_signals['b'])]
                                  )
        assert T.input_states[0].mod_afferents[0].sender==G.gating_signals[0]
        assert T.output_states[0].mod_afferents[0].sender==G.gating_signals[1]

    def test_formats_for_control_specification_for_mechanism_and_function_params(self):

        control_spec_list = [
            pnl.CONTROL,
            pnl.CONTROL_SIGNAL,
            pnl.CONTROL_PROJECTION,
            pnl.ControlSignal,
            pnl.ControlSignal(),
            pnl.ControlProjection,
            "CP_OBJECT",
            pnl.ControlMechanism,
            pnl.ControlMechanism(),
            (0.3, pnl.CONTROL),
            (0.3, pnl.CONTROL_SIGNAL),
            (0.3, pnl.CONTROL_PROJECTION),
            (0.3, pnl.ControlSignal),
            (0.3, pnl.ControlSignal()),
            (0.3, pnl.ControlProjection),
            (0.3, "CP_OBJECT"),
            (0.3, pnl.ControlMechanism),
            (0.3, pnl.ControlMechanism())
        ]
        for i, ctl_tuple in enumerate([j for j in zip(control_spec_list, reversed(control_spec_list))]):
            C1, C2 = ctl_tuple

            # This shenanigans is to avoid assigning the same instantiated ControlProjection more than once
            if C1 is 'CP_OBJECT':
                C1 = pnl.ControlProjection()
            elif isinstance(C1, tuple) and C1[1] is 'CP_OBJECT':
                C1 = (C1[0], pnl.ControlProjection())
            if C2 is 'CP_OBJECT':
                C2 = pnl.ControlProjection()
            elif isinstance(C2, tuple) and C2[1] is 'CP_OBJECT':
                C2 = (C2[0], pnl.ControlProjection())
            
            R = pnl.RecurrentTransferMechanism(noise=C1,
                                               function=pnl.Logistic(gain=C2))
            assert R.parameter_states[pnl.NOISE].mod_afferents[0].name in \
                   'ControlProjection for RecurrentTransferMechanism-{}[noise]'.format(i)
            assert R.parameter_states[pnl.GAIN].mod_afferents[0].name in \
                   'ControlProjection for RecurrentTransferMechanism-{}[gain]'.format(i)


    def test_formats_for_gating_specification_of_input_and_output_states(self):

        gating_spec_list = [
            pnl.GATING,
            pnl.GATING_SIGNAL,
            pnl.GATING_PROJECTION,
            pnl.GatingSignal,
            pnl.GatingSignal(),
            pnl.GatingProjection,
            "GP_OBJECT",
            pnl.GatingMechanism,
            pnl.GatingMechanism(),
            (0.3, pnl.GATING),
            (0.3, pnl.GATING_SIGNAL),
            (0.3, pnl.GATING_PROJECTION),
            (0.3, pnl.GatingSignal),
            (0.3, pnl.GatingSignal()),
            (0.3, pnl.GatingProjection),
            (0.3, "GP_OBJECT"),
            (0.3, pnl.GatingMechanism),
            (0.3, pnl.GatingMechanism())
        ]
        for i, gating_tuple in enumerate([j for j in zip(gating_spec_list, reversed(gating_spec_list))]):
            G1, G2 = gating_tuple

            # This shenanigans is to avoid assigning the same instantiated ControlProjection more than once
            if G1 is 'GP_OBJECT':
                G1 = pnl.GatingProjection()
            elif isinstance(G1, tuple) and G1[1] is 'GP_OBJECT':
                G1 = (G1[0], pnl.GatingProjection())
            if G2 is 'GP_OBJECT':
                G2 = pnl.GatingProjection()
            elif isinstance(G2, tuple) and G2[1] is 'GP_OBJECT':
                G2 = (G2[0], pnl.GatingProjection())
            
            T = pnl.TransferMechanism(name='T-GATING-{}'.format(i),
                                      input_states=[G1],
                                      output_states=[G2])
            assert T.input_states[0].mod_afferents[0].name in \
                   'GatingProjection for T-GATING-{}[InputState-0]'.format(i)

            assert T.output_states[0].mod_afferents[0].name in \
                   'GatingProjection for T-GATING-{}[OutputState-0]'.format(i)
