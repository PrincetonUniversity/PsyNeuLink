from PsyNeuLink import *

input_layer = Transfer(default_input_value=[0,0,0],
                       name='Input Layer')
action_selection = Transfer(default_input_value=[0,0,0],
                            function=SoftMax(output=SoftMax.PROB),
                            name='Action Selection')
reward_prediction = Mapping(sender=input_layer,
                            receiver=action_selection,
                            matrix=kwIdentityMatrix)
reward_monitor = Comparator(name='Comparator')
learning_signal = LearningSignal(sender=reward_monitor,
                                 receiver=reward_prediction.parameterStates['matrix'])
action_feedback = Mapping(sender=action_selection,
                          receiver=reward_monitor)

p = process(default_input_value=[0, 0, 0],
            configuration=[input_layer, action_selection])

p.execute([1, 1, 1])
