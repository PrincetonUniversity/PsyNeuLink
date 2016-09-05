from PsyNeuLink import *
from PsyNeuLink.Functions.Utility import SoftMax, Reinforcement
import numpy as np

input_layer = Transfer(default_input_value=[0,0,0],
                       name='Input Layer')

action_selection = Transfer(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB,
                                             gain=1.0),
                            name='Action Selection')

# # Explicit delcaration of Mapping projection to be learned:
# reward_prediction = Mapping(sender=input_layer,
#                             receiver=action_selection,
#                             matrix=(kwIdentityMatrix, LearningSignal(function=Reinforcement(learning_rate=.05))))
#
# p = process(default_input_value=[0, 0, 0],
#             configuration=[input_layer, reward_prediction, action_selection])

# Learning specified for process (rather than explicitly declared projection)
p = process(default_input_value=[0, 0, 0],
            configuration=[input_layer, kwIdentityMatrix, action_selection],
            learning=LearningSignal(function=Reinforcement(learning_rate=.05)))




print ('reward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
print ('comparator weights: \n', action_selection.outputState.sendsToProjections[0].matrix)


actions = ['left', 'middle', 'right']
reward_values = [15, 7, 13]
first_reward = 0

# Must initialize for reward lambda function to work (won't be used)
action_selection.outputState.value = [0, 0, 1]
reward = lambda : [reward_values[int(np.nonzero(action_selection.outputState.value)[0])]]

# Run process with RL
for i in range(100):

    # # Execute process, including weight adjustment based on last reward
    # result = p.execute([[1, 1, 1], [reward]])
    # result = p.execute([[1, 1, 1], reward(result)])
    result = p.execute([[1, 1, 1], reward])

    print ('result: ', result)
    # print ('reward: ', reward())

    # DOESN"T SHOW NEW WEIGHTS, SINCE UPDATING OF PARAMETER STATES IS "LAZY" (NOT DONE UNTIL CALLED)
    print ('\nreward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)

    # # Get index for action taken
    # action_taken = int(np.nonzero(result)[0])
    #
    # # Set select reward for action taken
    # reward = reward_values[action_taken]

    # print ('result: ', result)
    # print ('action taken: ', actions[action_taken])
    # print ('reward: ', reward)
    # print ('weights: \n', reward_prediction.matrix)

