import psyneulink as pnl
import numpy as np
print(pnl.__version__)


# network params
n_input = 2
n_hidden = 5
n_output = 1
max_entries = 7

# training params
num_epochs = 3
learning_rate = .1
wts_init_scale = .1

# layers

input = pnl.TransferMechanism(
    name='input',
    default_variable=np.zeros(n_input)
)

hidden = pnl.TransferMechanism(
    name='hidden',
    default_variable=np.zeros(n_hidden),
    function=pnl.Logistic()
)

output = pnl.TransferMechanism(
    name='output',
    default_variable=np.zeros(n_output),
    function=pnl.Logistic()
)

# weights
w_ih = pnl.MappingProjection(
    name='input_to_hidden',
    matrix=np.random.randn(n_input, n_hidden)*wts_init_scale,
    sender=input,
    receiver=hidden
)

w_ho = pnl.MappingProjection(
    name='hidden_to_output',
    matrix=np.random.randn(n_hidden, n_output)*wts_init_scale,
    sender=hidden,
    receiver=output
)

# dnd
dnd = pnl.EpisodicMemoryMechanism(
    cue_size=n_hidden, assoc_size=n_hidden,
    name='dnd'
)

w_hdc = pnl.MappingProjection(
    name='hidden_to_cue',
    matrix=np.random.randn(n_hidden, n_hidden)*wts_init_scale,
    sender=hidden,
    receiver=dnd.input_states[pnl.CUE_INPUT]
)

w_hda = pnl.MappingProjection(
    name='hidden_to_assoc',
    matrix=np.random.randn(n_hidden, n_hidden)*wts_init_scale,
    sender=hidden,
    receiver=dnd.input_states[pnl.ASSOC_INPUT]
)


w_dh = pnl.MappingProjection(
    name='em_to_hidden',
    matrix=np.random.randn(n_hidden, n_hidden)*wts_init_scale,
    sender=dnd,
    receiver=hidden
)

comp = pnl.Composition(name='xor')

# add all nodes
all_nodes = [input, hidden, output, dnd]
for node in all_nodes:
    comp.add_node(node)
# input-hidden-output pathway
comp.add_projection(sender=input, projection=w_ih, receiver=hidden)
comp.add_projection(sender=hidden, projection=w_ho, receiver=output)
# conneciton, dnd
comp.add_projection(sender=dnd, projection=w_dh, receiver=hidden)
comp.add_projection(
    sender=hidden,
    projection=w_hdc,
    receiver=dnd.input_states[pnl.CUE_INPUT]
)
comp.add_projection(
    sender=hidden,
    projection=w_hda,
    receiver=dnd.input_states[pnl.ASSOC_INPUT]
)
# show graph
comp.show_graph()

# # comp.show()
# # the required inputs for dnd
# print('dnd input_states: ', dnd.input_states.names)
#
# # currently, dnd receive info from the following node
# print('dnd receive: ')
# for dnd_input in dnd.input_states.names:
#     afferents = dnd.input_states[dnd_input].path_afferents
#     if len(afferents) == 0:
#         print(f'- {dnd_input}: NA')
#     else:
#         sending_node_name = afferents[0].sender.owner.name
#         print(f'- {dnd_input}: {sending_node_name}')
#
# print('dnd cue input: ', dnd.input_states.names)
#
# print('hidden receive: ')
# for hidden_afferent in hidden.input_states[0].path_afferents:
#     print('- ', hidden_afferent.sender.owner.name)
