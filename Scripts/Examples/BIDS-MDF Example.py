import psyneulink as pnl

A = pnl.TransferMechanism(function=pnl.Linear(slope=2.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
C = pnl.TransferMechanism(function=pnl.Exponential, name='C')
D = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(rate=0.05), name='D')

comp = pnl.Composition(name='comp', pathways=[[A,B,D], [A,C,D]])

comp.show_graph()