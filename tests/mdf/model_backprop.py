import psyneulink as pnl

A = pnl.TransferMechanism(name='A')
B = pnl.TransferMechanism(name='B')
C = pnl.TransferMechanism(name='C')

p = pnl.Pathway(pathway=[A, B, C])

comp = pnl.Composition(name='comp')
comp.add_backpropagation_learning_pathway(pathway=p)
