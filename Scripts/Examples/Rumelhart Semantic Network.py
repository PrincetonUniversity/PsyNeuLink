import psyneulink as pnl
import numpy as np
import typecheck as tc

# Semantic Network:
#
#       REP PROP QUAL ACT
#         \___\__/____/
#             |
#           HIDDEN
#            / \
#       HIDDEN REL_IN
#          /
#       REP_IN

print ("SCRATCH PAD:  Semantic Network")

rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
rep_hidden = pnl.TransferMechanism(size=4, function=pnl.Logistic, name='REP_HIDDEN')
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

sys = pnl.System(processes=[rep_hidden_proc, rel_hidden_proc, rel_rep_proc, rel_prop_proc, rel_qual_proc, rel_act_proc])
assert True

sys.show_graph(show_learning=pnl.ALL, show_dimensions=True)
