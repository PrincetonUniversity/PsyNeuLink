from PsyNeuLink import *
from PsyNeuLink.Functions.Utility import SoftMax, Reinforcement
import numpy as np

input_layer = Transfer(default_input_value=[0,0,0],
                       name='Input Layer')

action_selection = Transfer(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB,
                                             gain=1.0),
                            name='Action Selection')

p = process(default_input_value=[0, 0, 0],
            configuration=[input_layer,action_selection],
            learning=LearningSignal(function=Reinforcement(learning_rate=.05)))

print ('reward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
print ('comparator weights: \n', action_selection.outputState.sendsToProjections[0].matrix)

actions = ['left', 'middle', 'right']
reward_values = [15, 7, 13]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.outputState.value = [0, 0, 1]
# Get reward value for selected action)
reward = lambda : [reward_values[int(np.nonzero(action_selection.outputState.value)[0])]]

# Run process with RL
for i in range(100):

    # # Execute process, including weight adjustment based on last reward
    result = p.execute([[1, 1, 1], reward])

    print ('result: ', result)

    # Note: this shows weights updated on prior trial, not current one
    #       (this is a result of parameterState "lazy updating" -- only updated when called)
    print ('\nreward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
