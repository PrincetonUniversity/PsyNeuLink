"""
bistable percepts
"""

import numpy as np
import psyneulink as pnl
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# constants
# this code only works for N_PERCEPTS == 2
ALL_PERCEPTS = ['a', 'b']
N_PERCEPTS = len(ALL_PERCEPTS)
assert N_PERCEPTS == 2

# variables
n_nodes_per_percepts = 8
node_dict = {percept: None for percept in ALL_PERCEPTS}

excit_level = 1
inhib_level = 1


def get_node(percept, node_id):
    # helper func for creating a node
    tm_function = pnl.Linear(slope=1, intercept=0)
    tm_integrator_mode = True
    tm_integration_rate = .5
    node_ = pnl.TransferMechanism(
        name=f'{percept}-{node_id}',
        function=tm_function,
        integrator_mode=tm_integrator_mode,
        integration_rate=tm_integration_rate,
        default_variable=np.zeros((1,)),
    )
    return node_


# init all nodes, save them in list and dict form
for percept in ALL_PERCEPTS:
    node_dict[percept] = [
        get_node(percept, i) for i in range(n_nodes_per_percepts)
    ]


# init composition
bp_comp = pnl.Composition()

print('Forming connetions: ')
# within-percept excitation
for percept in ALL_PERCEPTS:
    for node_i, node_j in product(node_dict[percept], node_dict[percept]):
        if node_i is not node_j:
            print(f'\t{node_i} -> excite -> {node_j}')
            bp_comp.add_linear_processing_pathway(
                pathway=(node_i, [excit_level], node_j))

# inter-percepts inhibition
for node_i, node_j in zip(node_dict[ALL_PERCEPTS[0]],
                          node_dict[ALL_PERCEPTS[1]]):
    print(f'\t{node_i} <- inhibit -> {node_j}')
    bp_comp.add_linear_processing_pathway(
        pathway=(node_i, [-inhib_level], node_j))
    bp_comp.add_linear_processing_pathway(
        pathway=(node_j, [-inhib_level], node_i))

# make sure all nodes are both input and outputs
reportOutputPref = False
for node in bp_comp.nodes:
    # # MODIFIED 4/25/20 OLD:
    # bp_comp.add_required_node_role(node, pnl.NodeRole.INPUT)
    # bp_comp.add_required_node_role(node, pnl.NodeRole.OUTPUT)
    # MODIFIED 4/25/20 NEW:
    bp_comp.require_node_roles(node, [pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT])
    # MODIFIED 4/25/20 END
    # turn off report
    node.reportOutputPref = reportOutputPref

# bp_comp.show_graph()

# init the inputs
n_time_steps = 10
input_dict = {
    node_: np.random.normal(size=(n_time_steps,))
    for node_ in bp_comp.nodes
}

# run the model
bp_comp.run(input_dict, num_trials=10)

acts = np.squeeze(bp_comp.results)
f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(acts[:, :n_nodes_per_percepts], color='red')
ax.plot(acts[:, n_nodes_per_percepts:], color='blue')
ax.set_xlabel('Time')
ax.set_ylabel('Activation')
ax.set_title('temporal dyanmics of the bistable percept model')
