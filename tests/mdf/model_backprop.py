import psyneulink as pnl

a = pnl.TransferMechanism()
b = pnl.TransferMechanism()
c = pnl.TransferMechanism()

p = pnl.Pathway(pathway=[a, b, c])

comp = pnl.Composition()
comp.add_backpropagation_learning_pathway(pathway=p)
