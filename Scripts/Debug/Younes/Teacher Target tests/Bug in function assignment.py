import psyneulink as pnl

DIM = 3
LEARNING_RATE = 0.01

inputs = pnl.ProcessingMechanism(name='INPUTS', input_shapes=DIM)
outputs = pnl.ProcessingMechanism(name='OUTPUTS', input_shapes=DIM)

# * Pathways * #
pw_inputs_outputs = [
    inputs,
    pnl.MappingProjection(
        inputs, outputs,
        matrix=pnl.IDENTITY_MATRIX, learnable=True, learning_rate=LEARNING_RATE
    ),
    outputs,
]

# ** Composition ** #
comp = pnl.AutodiffComposition(
    pathways=[
        pw_inputs_outputs,
    ],
    learning_rate=LEARNING_RATE,
    loss_spec=pnl.Loss.BINARY_CROSS_ENTROPY)
comp.learn(inputs={inputs:[[0,0,0]]})