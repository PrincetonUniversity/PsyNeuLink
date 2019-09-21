import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions

ci = pnl.TransferMechanism(size=2, name='COLORS INPUT')
wi = pnl.TransferMechanism(size=2, name='WORDS INPUT')
ch = pnl.TransferMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='COLORS HIDDEN')
wh = pnl.TransferMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='WORDS HIDDEN')
tl = pnl.TransferMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=pnl.CONTROL), name='TASK CONTROL')
rl = pnl.LCAMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Logistic, name='RESPONSE')
cp = pnl.Process(pathway=[ci, ch, rl])
wp = pnl.Process(pathway=[wi, wh, rl])
tc = pnl.Process(pathway=[tl, ch])
tw = pnl.Process(pathway=[tl,wh])
s = pnl.System(processes=[tc, tw, cp, wp],
               controller=pnl.EVCControlMechanism(name='EVC Mechanimsm'),
               monitor_for_control=[rl])
s.show_graph()
