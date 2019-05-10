import numpy as np
from psyneulink import *


num_states = 10
context_size = 2
num_actions = 4
rpe_size = 1


# def ExepctedValueCalc(variable):
#     variable[]

stim_in = ProcessingMechanism(name='Stimulus',
                              size=num_states)

context_in = ProcessingMechanism(name='Context',
                                 size=context_size)

perceptual_state = ProcessingMechanism(name='Perceptual State',
                            function=lambda v: np.append(v[0], v[1]),
                            input_states=[{NAME:'STIM',
                                           SIZE:num_states,
                                           PROJECTIONS:stim_in},
                                          {NAME:'CONTEXT',
                                           SIZE:context_size,
                                           PROJECTIONS:context_in}])

agent_state = ProcessingMechanism(name='Agent State',
                                  size=len(perceptual_state.output_state.value))

agent_action = ProcessingMechanism(name='Agent Action',
                             size=num_actions,
                             input_states={NAME: 'Q values',
                                           PROJECTIONS:agent_state})

action = ProcessingMechanism(name='Model Action',
                             size = num_actions)

# q_rep = ProcessingMechanism(name='Q rep',
#                             size=num_actions*num_states,
#                             function=SoftMax(output=PROB, gain=1.0))


agent = Composition(name='Agent')
agent.add_linear_processing_pathway([agent_state, agent_action])

em = EpisodicMemoryMechanism(name='Episodic Memory',
                             content_size=num_states+context_size+rpe_size,
                             assoc_size=1,
                             # function=ContentAddressableMemory(function=ExepctedValueCalc))
                             function=ContentAddressableMemory)

sr = ProcessingMechanism(name='Successor Rep')


model = Composition(name='Adaptive Replay Model')
model.add_nodes([stim_in, context_in, perceptual_state])
model.add_linear_processing_pathway([perceptual_state, agent, action])

# comp.add_reinforcement_learning_pathway([state, action])

model.show_graph(show_node_structure=ALL)

num_trials = 2

stimuli = {stim_in:np.array([[1]*num_states]*num_trials),
           context_in: np.array([[10]*context_size]*num_trials)}
print(model.run(inputs=stimuli))
print('\n\n', model.results)