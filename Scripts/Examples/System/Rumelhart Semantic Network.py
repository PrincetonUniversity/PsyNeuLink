import psyneulink as pnl
import numpy as np
import typecheck as tc

# This script implements the following network, first described in Rumelhart and Todd
# (Rumelhart, D. E., & Todd, P. M. (1993). Learning and connectionist representations. Attention and performance XIV:
#  Synergies in experimental psychology, artificial intelligence, and cognitive neuroscience, 3-30).

# At present, it implements only the structure of the network, as shown below:

# Semantic Network:
#                               _
#   R_STEP P_STEP Q_STEP A_STEP  | Readout Processes
#        |    |    /    / _______|
#       REP PROP QUAL ACT  |
#         \___\__/____/    |
#             |        _   | Output Processes
#           HIDDEN      | _|
#            / \        |
#       HIDDEN REL_IN   |  Input Processes
#          /            |
#       REP_IN         _|

# It does not yet implement learning or testing.
import psyneulink.core.components.functions.transferfunctions


def step(variable):
    if np.sum(variable)<.5:
        out=0
    else:
        out=1
    return(out)

#Processing Units:
rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
rep_hidden = pnl.TransferMechanism(size=4, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_HIDDEN')
rel_hidden = pnl.TransferMechanism(size=5, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REL_HIDDEN')
rep_out = pnl.TransferMechanism(size=10, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='REP_OUT')
prop_out = pnl.TransferMechanism(size=12, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='PROP_OUT')
qual_out = pnl.TransferMechanism(size=13, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='QUAL_OUT')
act_out = pnl.TransferMechanism(size=14, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='ACT_OUT')
r_step = pnl.ProcessingMechanism(size=10, function=step, name='REP_STEP')
p_step = pnl.ProcessingMechanism(size=12, function=step, name='PROP_STEP')
q_step = pnl.ProcessingMechanism(size=13, function=step, name='QUAL_STEP')
a_step = pnl.ProcessingMechanism(size=14, function=step, name='ACT_STEP')

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

rep_step_proc = pnl.Process(pathway=[rep_out, r_step],
                           name='REP_STEP_PROC')
act_step_proc = pnl.Process(pathway=[act_out, a_step],
                           name='ACT_STEP_PROC')
qual_step_proc = pnl.Process(pathway=[qual_out, q_step],
                           name='QUAL_STEP_PROC')
prop_step_proc = pnl.Process(pathway=[prop_out, p_step],
                           name='PROP_STEP_PROC')


# The System:
S = pnl.System(processes=[rep_hidden_proc,
                          rel_hidden_proc,
                          rel_rep_proc,
                          rel_prop_proc,
                          rel_qual_proc,
                          rel_act_proc,
                          rep_step_proc,
                          act_step_proc,
                          qual_step_proc,
                          prop_step_proc])

# Shows just the processing network:
# S.show_graph(show_dimensions=True)

# Shows all of the learning components:
S.show_graph(show_learning=pnl.ALL)
# S.show_graph(show_mechanism_structure=True)
# S.show_graph(show_processes=True)
# S.show_graph()