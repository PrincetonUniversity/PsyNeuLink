import pytest
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions


def validate_learning_mechs(sys):

    def get_learning_mech(name):
        return next(lm for lm in sys.learning_mechanisms if lm.name == name)

    REP_IN_to_REP_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REP_IN to REP_HIDDEN')
    REP_HIDDEN_to_REL_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REP_HIDDEN to REL_HIDDEN')
    REL_IN_to_REL_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_IN to REL_HIDDEN')
    REL_HIDDEN_to_REP_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to REP_OUT')
    REL_HIDDEN_to_PROP_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to PROP_OUT')
    REL_HIDDEN_to_QUAL_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to QUAL_OUT')
    REL_HIDDEN_to_ACT_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to ACT_OUT')

    # Validate error_signal Projections for REP_IN to REP_HIDDEN
    assert len(REP_IN_to_REP_HIDDEN_LM.input_ports) == 3
    assert REP_IN_to_REP_HIDDEN_LM.input_ports[pnl.ERROR_SIGNAL].path_afferents[0].sender.owner == \
           REP_HIDDEN_to_REL_HIDDEN_LM

    # Validate error_signal Projections to LearningMechanisms for REP_HIDDEN_to REL_HIDDEN Projections
    assert all(lm in [input_port.path_afferents[0].sender.owner for input_port in
                      REP_HIDDEN_to_REL_HIDDEN_LM.input_ports]
               for lm in {REL_HIDDEN_to_REP_OUT_LM, REL_HIDDEN_to_PROP_OUT_LM,
                          REL_HIDDEN_to_QUAL_OUT_LM, REL_HIDDEN_to_ACT_OUT_LM})

    # Validate error_signal Projections to LearningMechanisms for REL_IN to REL_HIDDEN Projections
    assert all(lm in [input_port.path_afferents[0].sender.owner for input_port in
                      REL_IN_to_REL_HIDDEN_LM.input_ports]
               for lm in {REL_HIDDEN_to_REP_OUT_LM, REL_HIDDEN_to_PROP_OUT_LM,
                          REL_HIDDEN_to_QUAL_OUT_LM, REL_HIDDEN_to_ACT_OUT_LM})


class TestRumelhartSemanticNetwork:
    r"""
    Tests construction and training of network with both convergent and divergent pathways
    with the following structure:

    # Semantic Network:
    #                         _
    #       REP PROP QUAL ACT  |
    #         \___\__/____/    |
    #             |        _   | Output Processes
    #           HIDDEN      | _|
    #            / \        |
    #       HIDDEN REL_IN   |  Input Processes
    #          /            |
    #       REP_IN         _|
    """

    def test_rumelhart_semantic_network_sequential(self):

        rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
        rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
        rep_hidden = pnl.TransferMechanism(size=4, function=psyneulink.core.components.functions.transferfunctions
                                           .Logistic, name='REP_HIDDEN')
        rel_hidden = pnl.TransferMechanism(size=5, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REL_HIDDEN')
        rep_out = pnl.TransferMechanism(size=10, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_OUT')
        prop_out = pnl.TransferMechanism(size=12, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='PROP_OUT')
        qual_out = pnl.TransferMechanism(size=13, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='QUAL_OUT')
        act_out = pnl.TransferMechanism(size=14, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='ACT_OUT')

        rep_hidden_proc = pnl.Process(pathway=[rep_in, rep_hidden, rel_hidden],
                                      learning=pnl.LEARNING,
                                      name='REP_HIDDEN_PROC')
        rel_hidden_proc = pnl.Process(pathway=[rel_in, rel_hidden],
                                      learning=pnl.LEARNING,
                                      name='REL_HIDDEN_PROC')
        rel_rep_proc = pnl.Process(pathway=[rel_hidden, rep_out],
                                   learning=pnl.LEARNING,
                                   name='REL_REP_PROC')
        rel_prop_proc = pnl.Process(pathway=[rel_hidden, prop_out],
                                    learning=pnl.LEARNING,
                                    name='REL_PROP_PROC')
        rel_qual_proc = pnl.Process(pathway=[rel_hidden, qual_out],
                                    learning=pnl.LEARNING,
                                    name='REL_QUAL_PROC')
        rel_act_proc = pnl.Process(pathway=[rel_hidden, act_out],
                                   learning=pnl.LEARNING,
                                   name='REL_ACT_PROC')

        S = pnl.System(processes=[rep_hidden_proc,
                                  rel_hidden_proc,
                                  rel_rep_proc,
                                  rel_prop_proc,
                                  rel_qual_proc,
                                  rel_act_proc])
        # S.show_graph(show_learning=pnl.ALL, show_dimensions=True)
        validate_learning_mechs(S)

        print(S.origin_mechanisms)
        print(S.terminal_mechanisms)
        S.run(learning=True,
              # num_trials=2,
              inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
              # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
              )
        print(S.results)

    def test_rumelhart_semantic_network_convergent(self):

        rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
        rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
        rep_hidden = pnl.TransferMechanism(size=4, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_HIDDEN')
        rel_hidden = pnl.TransferMechanism(size=5, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REL_HIDDEN')
        rep_out = pnl.TransferMechanism(size=10, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_OUT')
        prop_out = pnl.TransferMechanism(size=12, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='PROP_OUT')
        qual_out = pnl.TransferMechanism(size=13, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='QUAL_OUT')
        act_out = pnl.TransferMechanism(size=14, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='ACT_OUT')

        rep_proc = pnl.Process(pathway=[rep_in, rep_hidden, rel_hidden, rep_out],
                               learning=pnl.LEARNING,
                               name='REP_PROC')
        rel_proc = pnl.Process(pathway=[rel_in, rel_hidden],
                               learning=pnl.LEARNING,
                               name='REL_PROC')
        rel_prop_proc = pnl.Process(pathway=[rel_hidden, prop_out],
                                    learning=pnl.LEARNING,
                                    name='REL_PROP_PROC')
        rel_qual_proc = pnl.Process(pathway=[rel_hidden, qual_out],
                                    learning=pnl.LEARNING,
                                    name='REL_QUAL_PROC')
        rel_act_proc = pnl.Process(pathway=[rel_hidden, act_out],
                                   learning=pnl.LEARNING,
                                   name='REL_ACT_PROC')
        S = pnl.System(processes=[rep_proc,
                                  rel_proc,
                                  rel_prop_proc,
                                  rel_qual_proc,
                                  rel_act_proc])
        # S.show_graph(show_learning=pnl.ALL, show_dimensions=True)
        validate_learning_mechs(S)
        print(S.origin_mechanisms)
        print(S.terminal_mechanisms)
        S.run(inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
              # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
              )

    def test_rumelhart_semantic_network_crossing(self):

        rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
        rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
        rep_hidden = pnl.TransferMechanism(size=4, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_HIDDEN')
        rel_hidden = pnl.TransferMechanism(size=5, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REL_HIDDEN')
        rep_out = pnl.TransferMechanism(size=10, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_OUT')
        prop_out = pnl.TransferMechanism(size=12, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='PROP_OUT')
        qual_out = pnl.TransferMechanism(size=13, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='QUAL_OUT')
        act_out = pnl.TransferMechanism(size=14, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='ACT_OUT')

        rep_proc = pnl.Process(pathway=[rep_in, rep_hidden, rel_hidden, rep_out],
                               learning=pnl.LEARNING,
                               name='REP_PROC')
        rel_proc = pnl.Process(pathway=[rel_in, rel_hidden, prop_out],
                               learning=pnl.LEARNING,
                               name='REL_PROC')
        rel_qual_proc = pnl.Process(pathway=[rel_hidden, qual_out],
                                    learning=pnl.LEARNING,
                                    name='REL_QUAL_PROC')
        rel_act_proc = pnl.Process(pathway=[rel_hidden, act_out],
                                   learning=pnl.LEARNING,
                                   name='REL_ACT_PROC')
        S = pnl.System(processes=[rep_proc,
                                  rel_proc,
                                  rel_qual_proc,
                                  rel_act_proc])

        # S.show_graph(show_learning=pnl.ALL, show_dimensions=True)
        validate_learning_mechs(S)
        print(S.origin_mechanisms)
        print(S.terminal_mechanisms)
        S.run(inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
              # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
              )
