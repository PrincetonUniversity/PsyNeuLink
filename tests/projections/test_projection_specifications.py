import psyneulink as pnl
import numpy as np
import pytest

import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.statefulfunctions.integratorfunctions
import psyneulink.core.components.functions.transferfunctions

class TestProjectionSpecificationFormats:

    def test_projection_specification_formats(self):
        """Test various matrix and Projection specifications
        Also tests assignment of Projections to pathay of Composition using add_linear_processing_pathway:
        - Projection explicitly specified in sequence (M1_M2_proj)
        - Projection pre-constructed and assigned to Mechanisms, but not specified in pathway(M2_M3_proj)
        - Projection specified in pathway that is duplicate one preconstructed and assigned to Mechanisms (M3_M4_proj)
          (currently it should be ignored; in the future, if/when Projections between the same sender and receiver
           in different Compositions are allowed, then it should be used)
        """
        M1 = pnl.ProcessingMechanism(size=2)
        M2 = pnl.ProcessingMechanism(size=5)
        M3 = pnl.ProcessingMechanism(size=4)
        M4 = pnl.ProcessingMechanism(size=3)

        M1_M2_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        M2_M3_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        M3_M4_matrix_A = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 5)
        M3_M4_matrix_B = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

        M1_M2_proj = pnl.MappingProjection(matrix=M1_M2_matrix)
        M2_M3_proj = pnl.MappingProjection(sender=M2,
                                           receiver=M3,
                                           matrix={pnl.VALUE: M2_M3_matrix,
                                                   pnl.FUNCTION: pnl.AccumulatorIntegrator,
                                                   pnl.FUNCTION_PARAMS: {pnl.DEFAULT_VARIABLE: M2_M3_matrix,
                                                                         pnl.INITIALIZER: M2_M3_matrix}})
        M3_M4_proj_A = pnl.MappingProjection(sender=M3, receiver=M4, matrix=M3_M4_matrix_A)
        c = pnl.Composition()
        c.add_linear_processing_pathway(pathway=[M1,
                                                 M1_M2_proj,
                                                 M2,
                                                 M3,
                                                 M3_M4_matrix_B,
                                                 M4])

        assert np.allclose(M2_M3_proj.matrix.base, M2_M3_matrix)
        assert M2.efferents[0] is M2_M3_proj
        assert np.allclose(M3.efferents[0].matrix.base, M3_M4_matrix_A)
        # This is if different Projections are allowed between the same sender and receiver in different Compositions:
        # assert np.allclose(M3.efferents[1].matrix, M3_M4_matrix_B)
        c.run(inputs={M1:[2, -30]})
        # assert np.allclose(c.results, [[-130.19166667, -152.53333333, -174.875]])
        assert np.allclose(c.results, [[ -78.115,  -91.52 , -104.925]])

    def test_multiple_modulatory_projection_specs(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{pnl.PROJECTIONS: [M.parameter_ports[
                                                                         psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE],
                                                                     M.parameter_ports[
                                                                         psyneulink.core.globals.keywords.THRESHOLD]]}])
        G = pnl.GatingMechanism(gating_signals=[{pnl.PROJECTIONS: [M.output_ports[pnl.DECISION_VARIABLE],
                                                                     M.output_ports[pnl.RESPONSE_TIME]]}])
        assert len(C.control_signals)==1
        assert len(C.control_signals[0].efferents)==2
        assert M.parameter_ports[
                   psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE].mod_afferents[0] == C.control_signals[0].efferents[0]
        assert M.parameter_ports[
                   psyneulink.core.globals.keywords.THRESHOLD].mod_afferents[0] == C.control_signals[0].efferents[1]
        assert len(G.gating_signals)==1
        assert len(G.gating_signals[0].efferents)==2
        assert M.output_ports[pnl.DECISION_VARIABLE].mod_afferents[0]==G.gating_signals[0].efferents[0]
        assert M.output_ports[pnl.RESPONSE_TIME].mod_afferents[0]==G.gating_signals[0].efferents[1]

    def test_multiple_modulatory_projections_with_port_Name(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{'DECISION_CONTROL':[M.parameter_ports[
                                                                           psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE],
                                                                       M.parameter_ports[
                                                                           psyneulink.core.globals.keywords.THRESHOLD]]}])
        G = pnl.GatingMechanism(gating_signals=[{'DDM_OUTPUT_GATE':[M.output_ports[pnl.DECISION_VARIABLE],
                                                                    M.output_ports[pnl.RESPONSE_TIME]]}])
        assert len(C.control_signals)==1
        assert C.control_signals[0].name=='DECISION_CONTROL'
        assert len(C.control_signals[0].efferents)==2
        assert M.parameter_ports[
                   psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE].mod_afferents[0] == C.control_signals[0].efferents[0]
        assert M.parameter_ports[
                   psyneulink.core.globals.keywords.THRESHOLD].mod_afferents[0] == C.control_signals[0].efferents[1]
        assert len(G.gating_signals)==1
        assert G.gating_signals[0].name=='DDM_OUTPUT_GATE'
        assert len(G.gating_signals[0].efferents)==2
        assert M.output_ports[pnl.DECISION_VARIABLE].mod_afferents[0]==G.gating_signals[0].efferents[0]
        assert M.output_ports[pnl.RESPONSE_TIME].mod_afferents[0]==G.gating_signals[0].efferents[1]

    def test_multiple_modulatory_projections_with_mech_and_port_Name_specs(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{pnl.MECHANISM: M,
                                                   pnl.PARAMETER_PORTS: [
                                                       psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE,

                                                       psyneulink.core.globals.keywords.THRESHOLD]}])
        G = pnl.GatingMechanism(gating_signals=[{pnl.MECHANISM: M,
                                                 pnl.OUTPUT_PORTS: [pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME]}])
        assert len(C.control_signals)==1
        assert len(C.control_signals[0].efferents)==2
        assert M.parameter_ports[
                   psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE].mod_afferents[0] == C.control_signals[0].efferents[0]
        assert M.parameter_ports[
                   psyneulink.core.globals.keywords.THRESHOLD].mod_afferents[0] == C.control_signals[0].efferents[1]
        assert len(G.gating_signals)==1
        assert len(G.gating_signals[0].efferents)==2
        assert M.output_ports[pnl.DECISION_VARIABLE].mod_afferents[0]==G.gating_signals[0].efferents[0]
        assert M.output_ports[pnl.RESPONSE_TIME].mod_afferents[0]==G.gating_signals[0].efferents[1]

    def test_mapping_projection_with_mech_and_port_Name_specs(self):
        R1 = pnl.TransferMechanism(output_ports=['OUTPUT_1', 'OUTPUT_2'])
        R2 = pnl.TransferMechanism(default_variable=[[0],[0]],
                                   input_ports=['INPUT_1', 'INPUT_2'])
        T = pnl.TransferMechanism(input_ports=[{pnl.MECHANISM: R1,
                                                pnl.OUTPUT_PORTS: ['OUTPUT_1', 'OUTPUT_2']}],
                                  output_ports=[{pnl.MECHANISM:R2,
                                                 pnl.INPUT_PORTS: ['INPUT_1', 'INPUT_2']}])
        assert len(R1.output_ports)==2
        assert len(R2.input_ports)==2
        assert len(T.input_ports)==1
        for input_port in T.input_ports:
            for projection in input_port.path_afferents:
                assert projection.sender.owner is R1
        assert len(T.output_ports)==1
        for output_port in T.output_ports:
            for projection in output_port.efferents:
                assert projection.receiver.owner is R2

    def test_mapping_projection_using_2_item_tuple_with_list_of_port_Names(self):

        T1 = pnl.TransferMechanism(name='T1', input_ports=[[0,0],[0,0,0]])
        T2 = pnl.TransferMechanism(name='T2',
                                   output_ports=[(['InputPort-0','InputPort-1'], T1)])
        assert len(T2.output_ports)==1
        assert T2.output_ports[0].efferents[0].receiver.name == 'InputPort-0'
        assert T2.output_ports[0].efferents[0].matrix.base.shape == (1,2)
        assert T2.output_ports[0].efferents[1].receiver.name == 'InputPort-1'
        assert T2.output_ports[0].efferents[1].matrix.base.shape == (1,3)

    def test_mapping_projection_using_2_item_tuple_and_3_item_tuples_with_index_specs(self):

        T1 = pnl.TransferMechanism(name='T1', input_ports=[[0,0],[0,0,0]])
        T2 = pnl.TransferMechanism(name='T2',
                                   input_ports=['a','b','c'],
                                   output_ports=[(['InputPort-0','InputPort-1'], T1),
                                                  ('InputPort-0', (pnl.OWNER_VALUE, 2), T1),
                                                  (['InputPort-0','InputPort-1'], 1, T1)])
        assert len(T2.output_ports)==3
        assert T2.output_ports[0].efferents[0].receiver.name == 'InputPort-0'
        assert T2.output_ports[0].efferents[0].matrix.base.shape == (1,2)
        assert T2.output_ports[0].efferents[1].receiver.name == 'InputPort-1'
        assert T2.output_ports[0].efferents[1].matrix.base.shape == (1,3)
        assert T2.output_ports[1].owner_value_index == 2
        assert T2.output_ports[2].owner_value_index == 1

    def test_2_item_tuple_from_control_signal_to_parameter_port(self):

        D = pnl.DDM(name='D')

        # Single name
        C = pnl.ControlMechanism(control_signals=[(
                                                  psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE, D)])
        assert C.control_signals[0].name == 'D[drift_rate] ControlSignal'
        assert C.control_signals[0].efferents[0].receiver.name == 'drift_rate'

        # List of names
        C = pnl.ControlMechanism(control_signals=[([
                                                       psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE,
                                                       psyneulink.core.globals.keywords.THRESHOLD], D)])
        assert C.control_signals[0].name == 'D[drift_rate, threshold] ControlSignal'
        assert C.control_signals[0].efferents[0].receiver.name == 'drift_rate'
        assert C.control_signals[0].efferents[1].receiver.name == 'threshold'

    def test_2_item_tuple_from_parameter_port_to_control_signals(self):

        C = pnl.ControlMechanism(control_signals=['a','b'])
        D = pnl.DDM(name='D3',
                     function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(drift_rate=(3, C),
                                                                                                                  threshold=(2,C.control_signals['b']))
                    )
        assert D.parameter_ports[
                   psyneulink.core.components.functions.distributionfunctions.DRIFT_RATE].mod_afferents[0].sender == C.control_signals[0]
        assert D.parameter_ports[
                   psyneulink.core.globals.keywords.THRESHOLD].mod_afferents[0].sender == C.control_signals[1]

    def test_2_item_tuple_from_gating_signal_to_output_ports(self):

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

    def test_2_item_tuple_from_input_and_output_ports_to_gating_signals(self):

        G = pnl.GatingMechanism(gating_signals=['a','b'])
        T = pnl.TransferMechanism(name='T',
                     input_ports=[(3,G)],
                     output_ports=[(2,G.gating_signals['b'])]
                                  )
        assert T.input_ports[0].mod_afferents[0].sender==G.gating_signals[0]
        assert T.output_ports[0].mod_afferents[0].sender==G.gating_signals[1]

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
        pnl.ControlMechanism,
        (0.3, pnl.CONTROL),
        (0.3, pnl.CONTROL_SIGNAL),
        (0.3, pnl.CONTROL_PROJECTION),
        (0.3, pnl.ControlSignal),
        (0.3, pnl.ControlSignal()),
        (0.3, pnl.ControlProjection),
        (0.3, "CP_OBJECT"),
        (0.3, pnl.ControlMechanism),
        (0.3, pnl.ControlMechanism()),
        (0.3, pnl.ControlMechanism)
    ]

    @pytest.mark.parametrize(
        'noise, gain',
        [(noise, gain) for noise, gain in [j for j in zip(control_spec_list, reversed(control_spec_list))]]
    )
    def test_formats_for_control_specification_for_mechanism_and_function_params(self, noise, gain):
        # This shenanigans is to avoid assigning the same instantiated ControlProjection more than once
        if noise == 'CP_OBJECT':
            noise = pnl.ControlProjection()
        elif isinstance(noise, tuple) and noise[1] == 'CP_OBJECT':
            noise = (noise[0], pnl.ControlProjection())
        if gain == 'CP_OBJECT':
            gain = pnl.ControlProjection()
        elif isinstance(gain, tuple) and gain[1] == 'CP_OBJECT':
            gain = (gain[0], pnl.ControlProjection())

        R = pnl.RecurrentTransferMechanism(
            # NOTE: fixed name prevents failures due to registry naming
            # for parallel test runs
            name='R-CONTROL',
            noise=noise,
            function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=gain)
        )
        assert R.parameter_ports[pnl.NOISE].mod_afferents[0].name in \
                'ControlProjection for R-CONTROL[noise]'
        assert R.parameter_ports[pnl.GAIN].mod_afferents[0].name in \
                'ControlProjection for R-CONTROL[gain]'

    gating_spec_list = [
        pnl.GATING,
        pnl.CONTROL,
        pnl.GATING_SIGNAL,
        pnl.CONTROL_SIGNAL,
        pnl.GATING_PROJECTION,
        pnl.CONTROL_PROJECTION,
        pnl.GatingSignal,
        pnl.ControlSignal,
        pnl.GatingSignal(),
        pnl.ControlSignal(),
        pnl.GatingProjection,
        "GP_OBJECT",
        pnl.GatingMechanism,
        pnl.ControlMechanism,
        pnl.GatingMechanism(),
        pnl.ControlMechanism(),
        (0.3, pnl.GATING),
        (0.3, pnl.CONTROL),
        (0.3, pnl.GATING_SIGNAL),
        (0.3, pnl.CONTROL_SIGNAL),
        (0.3, pnl.GATING_PROJECTION),
        (0.3, pnl.CONTROL_PROJECTION),
        (0.3, pnl.GatingSignal),
        (0.3, pnl.ControlSignal),
        (0.3, pnl.GatingSignal()),
        (0.3, pnl.ControlSignal()),
        (0.3, pnl.GatingProjection),
        (0.3, pnl.ControlProjection),
        (0.3, "GP_OBJECT"),
        (0.3, pnl.GatingMechanism),
        (0.3, pnl.ControlMechanism),
        (0.3, pnl.GatingMechanism()),
        (0.3, pnl.ControlMechanism())
    ]

    @pytest.mark.parametrize(
        'input_port, output_port',
        [(inp, outp) for inp, outp in [j for j in zip(gating_spec_list, reversed(gating_spec_list))]]
    )
    def test_formats_for_gating_specification_of_input_and_output_ports(self, input_port, output_port):
        G_IN, G_OUT = input_port, output_port

        # This shenanigans is to avoid assigning the same instantiated ControlProjection more than once
        if G_IN == 'GP_OBJECT':
            G_IN = pnl.GatingProjection()
        elif isinstance(G_IN, tuple) and G_IN[1] == 'GP_OBJECT':
            G_IN = (G_IN[0], pnl.GatingProjection())
        if G_OUT == 'GP_OBJECT':
            G_OUT = pnl.GatingProjection()
        elif isinstance(G_OUT, tuple) and G_OUT[1] == 'GP_OBJECT':
            G_OUT = (G_OUT[0], pnl.GatingProjection())

        if isinstance(G_IN, tuple):
            IN_NAME = G_IN[1]
        else:
            IN_NAME = G_IN
        IN_CONTROL = pnl.CONTROL in repr(IN_NAME).split(".")[-1].upper()
        if isinstance(G_OUT, tuple):
            OUT_NAME = G_OUT[1]
        else:
            OUT_NAME = G_OUT
        OUT_CONTROL = pnl.CONTROL in repr(OUT_NAME).split(".")[-1].upper()

        T = pnl.TransferMechanism(
            name='T-GATING',
            input_ports=[G_IN],
            output_ports=[G_OUT]
        )

        if IN_CONTROL:
            assert T.input_ports[0].mod_afferents[0].name in \
                    'ControlProjection for T-GATING[InputPort-0]'
        else:
            assert T.input_ports[0].mod_afferents[0].name in \
                    'GatingProjection for T-GATING[InputPort-0]'

        if OUT_CONTROL:
            assert T.output_ports[0].mod_afferents[0].name in \
                    'ControlProjection for T-GATING[OutputPort-0]'
        else:
            assert T.output_ports[0].mod_afferents[0].name in \
                    'GatingProjection for T-GATING[OutputPort-0]'

        # with pytest.raises(pnl.ProjectionError) as error_text:
        #     T1 = pnl.ProcessingMechanism(name='T1', input_ports=[pnl.ControlMechanism()])
        # assert 'Primary OutputPort of ControlMechanism-0 (ControlSignal-0) ' \
        #        'cannot be used as a sender of a Projection to InputPort of T1' in error_text.value.args[0]
        #
        # with pytest.raises(pnl.ProjectionError) as error_text:
        #     T2 = pnl.ProcessingMechanism(name='T2', output_ports=[pnl.ControlMechanism()])
        # assert 'Primary OutputPort of ControlMechanism-1 (ControlSignal-0) ' \
        #        'cannot be used as a sender of a Projection to OutputPort of T2' in error_text.value.args[0]

    def test_no_warning_when_matrix_specified(self):

        with pytest.warns(None) as w:
            c = pnl.Composition()
            m0 = pnl.ProcessingMechanism(
                default_variable=[0, 0, 0, 0]
            )
            p0 = pnl.MappingProjection(
                matrix=[[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]
            )
            m1 = pnl.TransferMechanism(
                default_variable=[0, 0, 0, 0]
            )
            c.add_linear_processing_pathway([m0, p0, m1])
            for warn in w:
                if r'elementwise comparison failed; returning scalar instead' in warn.message.args[0]:
                    raise

    # KDM: this is a good candidate for pytest.parametrize
    def test_masked_mapping_projection(self):

        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        proj = pnl.MaskedMappingProjection(sender=t1,
                                    receiver=t2,
                                    matrix=[[1,2],[3,4]],
                                    mask=[[1,0],[0,1]],
                                    mask_operation=pnl.ADD
                                    )
        c = pnl.Composition(pathways=[[t1, proj, t2]])
        val = c.execute(inputs={t1:[1,2]})
        assert np.allclose(val, [[8, 12]])

        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        proj = pnl.MaskedMappingProjection(sender=t1,
                                    receiver=t2,
                                    matrix=[[1,2],[3,4]],
                                    mask=[[1,0],[0,1]],
                                    mask_operation=pnl.MULTIPLY
                                    )
        c = pnl.Composition(pathways=[[t1, proj, t2]])
        val = c.execute(inputs={t1:[1,2]})
        assert np.allclose(val, [[1, 8]])

        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        proj = pnl.MaskedMappingProjection(sender=t1,
                                    receiver=t2,
                                    mask=[[1,2],[3,4]],
                                    mask_operation=pnl.MULTIPLY
                                    )
        c = pnl.Composition(pathways=[[t1, proj, t2]])
        val = c.execute(inputs={t1:[1,2]})
        assert np.allclose(val, [[1, 8]])

    def test_masked_mapping_projection_mask_conficts_with_matrix(self):

        with pytest.raises(pnl.MaskedMappingProjectionError) as error_text:

            t1 = pnl.TransferMechanism(size=2)
            t2 = pnl.TransferMechanism(size=2)
            pnl.MaskedMappingProjection(sender=t1,
                                        receiver=t2,
                                        mask=[[1,2,3],[4,5,6]],
                                        mask_operation=pnl.MULTIPLY
                                        )
        assert "Shape of the 'mask'" in str(error_text.value)
        assert "((2, 3)) must be the same as its 'matrix' ((2, 2))" in str(error_text.value)

    # FIX 7/22/15 [JDC] - REPLACE WITH MORE ELABORATE TESTS OF DUPLICATE PROJECTIONS:
    #                     SAME FROM OutputPort;  SAME TO InputPort
    #                     TEST ERROR MESSAGES GENERATED BY VARIOUS _check_for_duplicates METHODS
    # def test_duplicate_projection_detection_and_warning(self):
    #
    #     with pytest.warns(UserWarning) as record:
    #         T1 = pnl.TransferMechanism(name='T1')
    #         T2 = pnl.TransferMechanism(name='T2')
    #         T3 = pnl.TransferMechanism(name='T3')
    #         T4 = pnl.TransferMechanism(name='T4')
    #
    #         MP1 = pnl.MappingProjection(sender=T1,receiver=T2,name='MP1')
    #         MP2 = pnl.MappingProjection(sender=T1,receiver=T2,name='MP2')
    #         pnl.proc(T1,MP1,T2,T3)
    #         pnl.proc(T1,MP2,T2,T4)
    #
    #     # hack to find a specific warning (other warnings may be generated by the Process construction)
    #     correct_message_found = False
    #     for warning in record:
    #         if "that already has an identical Projection" in str(warning.message):
    #             correct_message_found = True
    #             break
    #
    #     assert len(T2.afferents)==1
    #     assert correct_message_found

    def test_duplicate_projection_creation_error(self):

        from psyneulink.core.components.projections.projection import DuplicateProjectionError
        with pytest.raises(DuplicateProjectionError) as record:
            T1 = pnl.TransferMechanism(name='T1')
            T2 = pnl.TransferMechanism(name='T2')
            pnl.MappingProjection(sender=T1,receiver=T2,name='MP1')
            pnl.MappingProjection(sender=T1,receiver=T2,name='MP2')
        assert 'Attempt to assign Projection to InputPort-0 of T2 that already has an identical Projection.' \
               in record.value.args[0]
