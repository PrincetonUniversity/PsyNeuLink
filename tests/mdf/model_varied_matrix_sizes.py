import psyneulink as pnl

comp = pnl.Composition(name='comp')
A = pnl.TransferMechanism(name='A', input_shapes=2)
B = pnl.TransferMechanism(name='B', input_shapes=3)
C = pnl.TransferMechanism(name='C', input_shapes=4)
D = pnl.TransferMechanism(name='D', input_shapes=5)
E = pnl.TransferMechanism(name='E', input_shapes=(3, 3))
F = pnl.TransferMechanism(name='F', input_shapes=4)

for n in [A, B, C, D, E, F]:
    comp.add_node(n)

comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3], [4, 5, 6]]), A, B)
comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3, 4], [5, 6, 7, 8]]), A, C)
comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]), B, D)
comp.add_projection(pnl.MappingProjection(matrix=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]), C, D)
comp.add_projection(pnl.MappingProjection(matrix=pnl.RANDOM_CONNECTIVITY_MATRIX), A, E)
comp.add_projection(pnl.MappingProjection(matrix=pnl.RANDOM_CONNECTIVITY_MATRIX), B, E.input_ports[1])
comp.add_projection(pnl.MappingProjection(matrix=pnl.RANDOM_CONNECTIVITY_MATRIX), C, E)
comp.add_projection(pnl.MappingProjection(matrix=pnl.RANDOM_CONNECTIVITY_MATRIX), D, E.input_ports[1])
comp.add_projection(pnl.MappingProjection(matrix=pnl.RANDOM_CONNECTIVITY_MATRIX), E, F)
