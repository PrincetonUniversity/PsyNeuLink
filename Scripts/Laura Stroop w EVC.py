import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

ci = pnl.TransferMechanism(size=2, name='COLORS INPUT')
wi = pnl.TransferMechanism(size=2, name='WORDS INPUT')
ch = pnl.TransferMechanism(size=2, function=pnl.Logistic, name='COLORS HIDDEN')
wh = pnl.TransferMechanism(size=2, function=pnl.Logistic, name='WORDS HIDDEN')
tl = pnl.TransferMechanism(size=2, function=pnl.Logistic(gain=pnl.CONTROL), name='TASK')
rl = pnl.LCA(size=2, function=pnl.Logistic, name='RESPONSE')
cp = pnl.Process(pathway=[ci, ch, rl])
wp = pnl.Process(pathway=[wi, wh, rl])
tc = pnl.Process(pathway=[tl, ch])
tw = pnl.Process(pathway=[tl,wh])
s = pnl.System(processes=[cp, wp, tc, tw],
               controller=pnl.EVCControlMechanism(name='EVC Mechanimsm'),
               monitor_for_control=[rl],
               name='FEEDFORWARD_STROOP_SYSTEM')
s.show_graph(show_control=pnl.ALL)
