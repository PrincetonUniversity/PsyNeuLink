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
        # Validate error_signal Projections for REP_IN to REP_HIDDEN
        assert len(sys.mechanisms[5].input_states) == 3
        assert sys.mechanisms[5].input_states[2].path_afferents[0].sender.owner == sys.mechanisms[4]
        # Validate error_signal Projections to LearningMechanisms for REP_HIDDEN_to REL_HIDDEN Projections
        assert len(sys.mechanisms[4].input_states) == 7
        assert sys.mechanisms[4].input_states[3].path_afferents[0].sender.owner == sys.mechanisms[11]
        assert sys.mechanisms[4].input_states[4].path_afferents[0].sender.owner == sys.mechanisms[14]
        assert sys.mechanisms[4].input_states[5].path_afferents[0].sender.owner == sys.mechanisms[17]
        assert sys.mechanisms[4].input_states[6].path_afferents[0].sender.owner == sys.mechanisms[20]
        # Validate error_signal Projections to LearningMechanisms for REL_IN to REL_HIDDEN Projections
        assert len(sys.mechanisms[8].input_states) == 7
        assert sys.mechanisms[8].input_states[3].path_afferents[0].sender.owner == sys.mechanisms[11]
        assert sys.mechanisms[8].input_states[4].path_afferents[0].sender.owner == sys.mechanisms[14]
        assert sys.mechanisms[8].input_states[5].path_afferents[0].sender.owner == sys.mechanisms[17]
        assert sys.mechanisms[8].input_states[6].path_afferents[0].sender.owner == sys.mechanisms[20]

