import psyneulink as pnl
import numpy as np

state_input_layer = pnl.ProcessingMechanism(
    name='in',
    input_shapes=2,
)

state_out = pnl.ProcessingMechanism(
    name='out',
    input_shapes=2,
)

comp = pnl.Composition()


comp.add_nodes([state_input_layer, state_out])

comp.add_projection(
    pnl.MappingProjection(state_input_layer, state_out)
)
comp.add_projection(
    pnl.MappingProjection(state_input_layer, state_input_layer)
)

comp.scheduler.add_condition(state_out, pnl.BeforeNode(state_input_layer))
# comp.scheduler.add_condition(state_input_layer, pnl.BeforeNode(state_out))

comp.run(inputs={
    state_input_layer: [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
})

print(comp.results)