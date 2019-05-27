import numpy as np
from psyneulink import *


stim_size = 10
context_size = 2
num_actions = 4
rpe_size = 1

def Concatenate(variable):
    return np.append(variable[0],variable[1])

# def ExepctedValueCalc(variable):
#     variable[]

# *********************************************************************************************
#                          PERCEPTUAL AND ACTION MECHANISMS
# *********************************************************************************************
stim_in = ProcessingMechanism(name='Stimulus',
                              size=stim_size)

context_in = ProcessingMechanism(name='Context',
                                 size=context_size)

reward_in = ProcessingMechanism(name='Reward',
                                size=1)

perceptual_state = ProcessingMechanism(name='State',
                            function=Concatenate,
                            input_states=[{NAME:'STIM',
                                           SIZE:stim_size,
                                           PROJECTIONS:stim_in},
                                          {NAME:'CONTEXT',
                                           SIZE:context_size,
                                           PROJECTIONS:context_in}])

action = ProcessingMechanism(name='Actions',
                             size=num_actions,
                             input_states={NAME: 'Q values',
                                           PROJECTIONS:perceptual_state})

# *********************************************************************************************
#                             RL AGENT NESTED COMPOSITION
# *********************************************************************************************
rl_agent_state = ProcessingMechanism(size=5)
rl_agent_action = ProcessingMechanism(size=5)
rl_agent = Composition(name='Agent')
rl_agent.add_reinforcement_learning_pathway([rl_agent_state, rl_agent_action])

# *********************************************************************************************
#                          MEMORY AND CONTROL MECHANISMS
# *********************************************************************************************
# q_rep = ProcessingMechanism(name='Q rep',
#                             size=num_actions*stim_size,
#                             function=SoftMax(output=PROB, gain=1.0))
#
# em = EpisodicMemoryMechanism(name='Episodic Memory',
#                              content_size=stim_size+context_size,
#                              assoc_size=rpe_size,
#                              # function=ContentAddressableMemory(function=ExepctedValueCalc))
#                              function=ContentAddressableMemory)
#
# sr = ProcessingMechanism(name='Successor Rep')

# *********************************************************************************************
#                                   FULL COMPOSITION
# *********************************************************************************************
model = Composition(name='Adaptive Replay Model')
model.add_nodes([stim_in, context_in, reward_in, perceptual_state])
model.add_linear_processing_pathway([perceptual_state, rl_agent, action])

# *********************************************************************************************
#                                  SHOW AND RUN MODEL
# *********************************************************************************************
model.show_graph(show_controller=True)
# model.show_graph(show_node_structure=ALL)

# stimuli = {stim_in:[[1, 1, 1],[2, 2, 2]],
#            context_in: [[10, 10, 10],[20, 20, 20]]}

# stimuli = {stim_in:[1, 1, 1],
#            context_in: [10, 10, 10]}

stimuli = {stim_in:np.array([1]*stim_size),
           context_in: np.array([10]*context_size)}

# print(model.run(inputs=stimuli))