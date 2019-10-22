import psyneulink as pnl
from psyneulink.core.components.functions.transferfunctions import Logistic
import numpy as np

print(pnl.__version__)


# network params
n_input = 2
n_hidden = 5
n_output = 1
key_size = n_hidden
val_size = n_hidden
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
    function=Logistic()
)

output = pnl.TransferMechanism(
    name='output',
    default_variable=np.zeros(n_output),
    function=Logistic()
)

# weights
w_ih = pnl.MappingProjection(
    name='input_to_hidden',
    matrix=np.random.randn(n_input, n_hidden) * wts_init_scale,
    sender=input,
    receiver=hidden
)

w_ho = pnl.MappingProjection(
    name='hidden_to_output',
    matrix=np.random.randn(n_hidden, n_output) * wts_init_scale,
    sender=hidden,
    receiver=output
)


# ContentAddressableMemory
ContentAddressableMemory = pnl.EpisodicMemoryMechanism(
    key_size, val_size,
    name='episodic memory'
)

w_hd = pnl.MappingProjection(
    name='hidden_to_em',
    matrix=np.random.randn(n_hidden, n_hidden) * wts_init_scale,
    sender=hidden,
    receiver=ContentAddressableMemory
)

w_dh = pnl.MappingProjection(
    name='em_to_hidden',
    matrix=np.random.randn(n_hidden, n_hidden) * wts_init_scale,
    sender=ContentAddressableMemory,
    receiver=hidden
)

comp = pnl.Composition(name='xor')
# add all nodes
all_nodes = [input, hidden, output, ContentAddressableMemory]
for node in all_nodes:
    comp.add_node(node)
# input-hidden-output pathway
comp.add_projection(sender=input, projection=w_ih, receiver=hidden)
comp.add_projection(sender=hidden, projection=w_ho, receiver=output)
# conneciton, ContentAddressableMemory
comp.add_projection(sender=hidden, projection=w_hd, receiver=ContentAddressableMemory)
comp.add_projection(sender=ContentAddressableMemory, projection=w_dh, receiver=hidden)

# comp.show_graph()

print(ContentAddressableMemory.input_ports.names)
print(ContentAddressableMemory.input_ports[pnl.CUE_INPUT].path_afferents[0].sender.owner.name)
print(hidden.input_ports[0].path_afferents[0].sender.owner.name)
