from PsyNeuLink import *
from PsyNeuLink.Functions.Utility import SoftMax, Reinforcement

input_layer = Transfer(default_input_value=[0,0,0],
                       name='Input Layer')
action_selection = Transfer(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB),
                            name='Action Selection')
reward_prediction = Mapping(sender=input_layer,
                            receiver=action_selection,
                            matrix=(kwIdentityMatrix, LearningSignal(function=Reinforcement)))
                            # matrix=(kwIdentityMatrix, LearningSignal(function=BackPropagation(activation_function=Linear))))
                            # matrix=(kwIdentityMatrix, LearningSignal))
# reward_prediction = Mapping(matrix=(kwIdentityMatrix, LearningSignal))

# reward_monitor = Comparator(default_sample_and_target=[[0],[0]],
#                             name='Comparator')
# action_feedback = Mapping(sender=action_selection,
#                           receiver=reward_monitor)

# learning_signal = LearningSignal(sender=reward_monitor,
#                                  receiver=reward_prediction)


p = process(default_input_value=[0, 0, 0],
            # configuration=[input_layer, action_selection])
            configuration=[input_layer, reward_prediction, action_selection])

p.execute([[1, 1, 1],[0]])
