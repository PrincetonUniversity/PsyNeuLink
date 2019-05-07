import numpy as np
from psyneulink import *


num_states = 10
context_size = 2
num_actions = 4
rpe_size = 1

def Concatenate(variable):
    return np.append(variable[0],variable[1])

# def ExepctedValueCalc(variable):
#     variable[]

stim_in = ProcessingMechanism(name='Stimulus',
                              size=num_states)

context_in = ProcessingMechanism(name='Context',
                                 size=context_size)

state = ProcessingMechanism(name='State',
                            function=Concatenate,
                            input_states=[{NAME:'STIM',
                                           SIZE:num_states,
                                           PROJECTIONS:stim_in},
                                          {NAME:'CONTEXT',
                                           SIZE:context_size,
                                           PROJECTIONS:context_in}])

action = ProcessingMechanism(name='Actions',
                             size=num_actions,
                             input_states={NAME: 'Q values',
                                           PROJECTIONS:state})

# q_rep = ProcessingMechanism(name='Q rep',
#                             size=num_actions*num_states,
#                             function=SoftMax(output=PROB, gain=1.0))

em = EpisodicMemoryMechanism(name='Episodic Memory',
                             cue_size=num_states+context_size+rpe_size,
                             assoc_size=1,
                             function=ContentAddressableMemory(function=ExepctedValueCalc))

sr = ProcessingMechanism(name='Successor Rep')


comp = Composition(name='Adaptive Replay Model')
comp.add_nodes([stim_in, context_in, state])
# comp.add_reinforcement_learning_pathway([state, action])

# comp.show_graph(show_node_structure=ALL)

# stimuli = {stim_in:[[1, 1, 1],[2, 2, 2]],
#            context_in: [[10, 10, 10],[20, 20, 20]]}

# stimuli = {stim_in:[1, 1, 1],
#            context_in: [10, 10, 10]}

stimuli = {stim_in:np.array([1]*num_states),
           context_in: np.array([10]*context_size)}

print(comp.execute(inputs=stimuli))