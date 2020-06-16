import numpy as np
from psyneulink import *


stim_size = 10
context_size = 2
num_actions = 4
rpe_size = 1

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

perceptual_state = ProcessingMechanism(name='Current Port',
                            function=Concatenate,
                            input_ports=[{NAME:'STIM',
                                           SIZE:stim_size,
                                           PROJECTIONS:stim_in},
                                          {NAME:'CONTEXT',
                                           SIZE:context_size,
                                           PROJECTIONS:context_in}])

# action = ProcessingMechanism(name='Action',
#                              size=num_actions,
#                              input_ports={NAME: 'Q values',
#                                            PROJECTIONS:perceptual_state})
action = ProcessingMechanism(name='Action',
                             size=num_actions)

# *********************************************************************************************
#                             RL AGENT NESTED COMPOSITION
# *********************************************************************************************
rl_agent_state = ProcessingMechanism(name='RL Agent Port', size=5)
rl_agent_action = ProcessingMechanism(name='RL Agent Action', size=5)
rl_agent = Composition(name='RL Agent')
rl_learning_components = rl_agent.add_reinforcement_learning_pathway([rl_agent_state, rl_agent_action])
# rl_agent.add_required_node_role(rl_agent_action, NodeRole.OUTPUT)
rl_agent._analyze_graph()

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

# Add Components individually -----------------------------------------------------------------
model.add_nodes([stim_in, context_in, reward_in, perceptual_state, rl_agent, action])
model.add_projection(sender=perceptual_state, receiver=rl_agent_state)
model.add_projection(sender=reward_in, receiver=rl_learning_components[TARGET_MECHANISM])
model.add_projection(sender=rl_agent, receiver=action)
model._analyze_graph()
assert True
# # ALTERNATIVE: Use linear_processing_pathway  ------------------------------------------------
# model.add_nodes([stim_in, context_in, reward_in, perceptual_state])
# model.add_linear_processing_pathway([perceptual_state, rl_agent, action])

# *********************************************************************************************
#                                  SHOW AND RUN MODEL
# *********************************************************************************************
# model.show_graph(show_controller=True,
#                  show_nested=DIRECT,
#                  show_nested_args={'show_node_structure':False,
#                               'show_cim':True},
#                  show_node_structure=ALL,
#                  show_cim=False)
# model.show_graph(show_node_structure=ALL)
model.show_graph(show_nested=DIRECT, show_node_structure=ALL, show_cim=True)


num_trials = 2

stimuli = {stim_in:np.array([1] * stim_size),
           context_in: np.array([10] * context_size),
           reward_in:np.array([1])}

print(model.run(inputs=stimuli))
