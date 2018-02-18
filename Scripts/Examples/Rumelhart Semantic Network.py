import psyneulink as pnl
import numpy as np
import typecheck as tc

# This script implements the following network, first described in Rumelhart and Todd
# (Rumelhart, D. E., & Todd, P. M. (1993). Learning and connectionist representations. Attention and performance XIV:
#  Synergies in experimental psychology, artificial intelligence, and cognitive neuroscience, 3-30).

# At present, it implements only the structure of the network, as shown below:

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

# It does not yet implement learning or testing.

#Processing Units:
rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
rep_hidden = pnl.TransferMechanism(size=4, function=pnl.Logistic, name='REP_HIDDEN')
rel_hidden = pnl.TransferMechanism(size=5, function=pnl.Logistic, name='REL_HIDDEN')
rep_out = pnl.TransferMechanism(size=10, function=pnl.Logistic, name='REP_OUT')
prop_out = pnl.TransferMechanism(size=12, function=pnl.Logistic, name='PROP_OUT')
qual_out = pnl.TransferMechanism(size=13, function=pnl.Logistic, name='QUAL_OUT')
act_out = pnl.TransferMechanism(size=14, function=pnl.Logistic, name='ACT_OUT')

#Processes that comprise the System:
# NOTE: this is one of several configuration of processes that can be used to construct the full network
#       (see Test/learning/test_rumelhart_semantic_network.py for other configurations)
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

# The System:
S = pnl.System(processes=[rep_hidden_proc, rel_hidden_proc, rel_rep_proc, rel_prop_proc, rel_qual_proc, rel_act_proc])

# Shows just the processing network:
# S.show_graph(show_dimensions=True)

# Shows all of the learning components:
S.show_graph(show_learning=pnl.ALL)
