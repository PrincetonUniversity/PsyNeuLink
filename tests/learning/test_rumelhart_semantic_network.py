import psyneulink as pnl

class TestRumelhartSemanticNetwork:
    """
    Tests construction and training of network with both convergent and divergent pathways
    with the following structure:

          E F G H
          \_\/_/
            D
           / \
          B  C
         /
        A
    """

    def test_rumelhart_semantic_network(self):

        rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
        rep_hidden = pnl.TransferMechanism(size=4, function=pnl.Logistic, name='REP_HIDDEN')
        rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
        rel_hidden = pnl.TransferMechanism(size=5, function=pnl.Logistic, name='REL_HIDDEN')
        rep_out = pnl.TransferMechanism(size=10, function=pnl.Logistic, name='REP_OUT')
        prop_out = pnl.TransferMechanism(size=12, function=pnl.Logistic, name='PROP_OUT')
        qual_out = pnl.TransferMechanism(size=13, function=pnl.Logistic, name='QUAL_OUT')
        act_out = pnl.TransferMechanism(size=14, function=pnl.Logistic, name='ACT_OUT')

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

        sys = pnl.System(processes=[rep_hidden_proc,
                                    rel_hidden_proc,
                                    rel_rep_proc,
                                    rel_prop_proc,
                                    rel_qual_proc,
                                    rel_act_proc])

        # Structural validation:
        def get_learning_mech(name):
            return next(lm for lm in sys.learning_mechanisms if lm.name == name)

        REP_IN_to_REP_HIDDEN_LM = get_learning_mech('MappingProjection from REP_IN to REP_HIDDEN LearningMechanism')
        REP_HIDDEN_to_REL_HIDDEN_LM = get_learning_mech('MappingProjection from REP_HIDDEN to REL_HIDDEN LearningMechanism')
        REL_IN_to_REL_HIDDEN_LM = get_learning_mech('MappingProjection from REL_IN to REL_HIDDEN LearningMechanism')
        REL_HIDDEN_to_REP_OUT = get_learning_mech('MappingProjection from REL_HIDDEN to REP_OUT LearningMechanism')
        REL_HIDDEN_to_PROP_OUT = get_learning_mech('MappingProjection from REL_HIDDEN to PROP_OUT LearningMechanism')
        REL_HIDDEN_to_QUAL_OUT = get_learning_mech('MappingProjection from REL_HIDDEN to QUAL_OUT LearningMechanism')
        REL_HIDDEN_to_ACT_OUT = get_learning_mech('MappingProjection from REL_HIDDEN to ACT_OUT LearningMechanism')

        # Validate error_signal Projections for REP_IN to REP_HIDDEN
        # assert len(sys.mechanisms[5].input_states) == 3
        assert len(REP_IN_to_REP_HIDDEN_LM.input_states) == 3
        assert REP_IN_to_REP_HIDDEN_LM.input_states[2].path_afferents[0].sender.owner == REP_HIDDEN_to_REL_HIDDEN_LM

        # Validate error_signal Projections to LearningMechanisms for REP_HIDDEN_to REL_HIDDEN Projections
        assert len(REP_HIDDEN_to_REL_HIDDEN_LM.input_states) == 7
        assert REP_HIDDEN_to_REL_HIDDEN_LM.input_states[3].path_afferents[0].sender.owner == REL_HIDDEN_to_REP_OUT
        assert REP_HIDDEN_to_REL_HIDDEN_LM.input_states[4].path_afferents[0].sender.owner == REL_HIDDEN_to_PROP_OUT
        assert REP_HIDDEN_to_REL_HIDDEN_LM.input_states[5].path_afferents[0].sender.owner == REL_HIDDEN_to_QUAL_OUT
        assert REP_HIDDEN_to_REL_HIDDEN_LM.input_states[6].path_afferents[0].sender.owner == REL_HIDDEN_to_ACT_OUT

        # Validate error_signal Projections to LearningMechanisms for REL_IN to REL_HIDDEN Projections
        assert len(REL_IN_to_REL_HIDDEN_LM.input_states) == 7
        assert REL_IN_to_REL_HIDDEN_LM.input_states[3].path_afferents[0].sender.owner == REL_HIDDEN_to_REP_OUT
        assert REL_IN_to_REL_HIDDEN_LM.input_states[4].path_afferents[0].sender.owner == REL_HIDDEN_to_PROP_OUT
        assert REL_IN_to_REL_HIDDEN_LM.input_states[5].path_afferents[0].sender.owner == REL_HIDDEN_to_QUAL_OUT
        assert REL_IN_to_REL_HIDDEN_LM.input_states[6].path_afferents[0].sender.owner == REL_HIDDEN_to_ACT_OUT

