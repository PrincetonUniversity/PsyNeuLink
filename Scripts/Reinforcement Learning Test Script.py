import numpy as np

from PsyNeuLink import *
from PsyNeuLink.Components.Functions.Function import SoftMax, Reinforcement

import random
random.seed(0)
np.random.seed(0)

input_layer = TransferMechanism(default_input_value=[0,0,0],
                       name='Input Layer')

action_selection = TransferMechanism(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB,
                                             gain=1.0),
                            name='Action Selection')

p = process(default_input_value=[0, 0, 0],
            pathway=[input_layer,action_selection],
            learning=LearningProjection(learning_function=Reinforcement(learning_rate=.05)),
            target=0)

print ('reward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
print ('targetMechanism weights: \n', action_selection.outputState.sendsToProjections[0].matrix)

actions = ['left', 'middle', 'right']
reward_values = [15, 7, 13]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.outputState.value = [0, 0, 1]
# Get reward value for selected action)
reward = lambda : [reward_values[int(np.nonzero(action_selection.outputState.value)[0])]]

# Run process with RL
# for i in range(10):
#
#     # # Execute process, including weight adjustment based on last reward
#     result = p.execute(input=[1, 1, 1], target=reward)
#
#     print ('result: ', result)
#
#     # Note: this shows weights updated on prior trial, not current one
#     #       (this is a result of parameterState "lazy updating" -- only updated when called)
#     print ('\nreward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_weights():
    print ('\nreward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
    print ('action selected: ', action_selection.outputState.value)

p.run(num_executions=10,
      # inputs=[[[1, 1, 1]]],
      # inputs=[ [ [1, 1, 1] ],[ [.2, 1, .2] ]],
      inputs={input_layer:[[1, 1, 1],[.2, 1, .2]]},
      targets=reward,
      call_before_trial=print_header,
      call_after_trial=show_weights
      )

# s = system(processes=[p])
#
# s.run(num_executions=10,
#       # inputs=[[[1, 1, 1]]],
#       inputs=[ [[1, 1, 1] ],[ [.2, 1, .2] ]],
#       targets=reward,
#       call_before_trial=print_header,
#       call_after_trial=show_weights
#       )
