import psyneulink as pnl

comp = pnl.Composition(name='comp')
A = pnl.TransferMechanism(name='A', input_shapes=2)
B = pnl.TransferMechanism(name='B', input_shapes=3)
C = pnl.TransferMechanism(name='C', input_shapes=4)
D = pnl.TransferMechanism(name='D', input_shapes=5)

for n in [A, B, C, D]:
    comp.add_node(n)

comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3], [4, 5, 6]]), A, B)
comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3, 4], [5, 6, 7, 8]]), A, C)
comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]), B, D)
comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]), C, D)
