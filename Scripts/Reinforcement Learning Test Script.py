from PsyNeuLink import *
from PsyNeuLink.Functions.Utility import SoftMax, Reinforcement
import numpy as np

input_layer = Transfer(default_input_value=[0,0,0],
                       name='Input Layer')

action_selection = Transfer(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB,
                                             gain=.5),
                            name='Action Selection')

reward_prediction = Mapping(sender=input_layer,
                            receiver=action_selection,
                            matrix=(kwIdentityMatrix, LearningSignal(function=Reinforcement(learning_rate=.1))))

p = process(default_input_value=[0, 0, 0],
            configuration=[input_layer, reward_prediction, action_selection])


print ('weights: \n', reward_prediction.matrix)


actions = ['left', 'middle', 'right']
reward_values = [15, 7, 13]
first_reward = 0
reward = first_reward

for i in range(100):

    # Execute process, including weight adjustment based on last reward
    result = p.execute([[1, 1, 1], [reward]])

    # Get index for action taken
    action_taken = int(np.nonzero(result)[0])

    # Set select reward for action taken
    reward = reward_values[action_taken]

    print ('result: ', result)
    print ('action taken: ', actions[action_taken])
    print ('reward: ', reward)
    print ('weights: \n', reward_prediction.matrix)
