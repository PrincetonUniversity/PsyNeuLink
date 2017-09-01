
# coding: utf-8

# In[ ]:

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system

#The following code starts to build a 3 layer neural network

input_layer = TransferMechanism(name='Input Layer',
                       function=Logistic,
                       default_variable = np.zeros((2,)))

hidden_layer = TransferMechanism(name='Hidden Layer', 
                                 function = Linear, 
                                 default_variable =[0])

output_layer = TransferMechanism(name='Output Layer',
                        function=Linear,
                        default_variable =[0])



input_hidden_weights = MappingProjection(name='Input-Hidden Weights',
                                         matrix = np.random.rand(2,1)*1 - .5)

hidden_output_weights = MappingProjection(name='Hidden-Output Weights',
                                         matrix = np.random.rand(1,1)*1 - .5)

input_output_weights = MappingProjection(name = 'Input-Output Weights',
                                         matrix = np.random.rand(2,1)*1 - .5)



input_hidden_process = process(default_variable=[0, 0],
                               pathway=[input_layer, input_hidden_weights, hidden_layer],
                               learning=LEARNING,
                               learning_rate=1.0,
                               target=[1],
                               # name='INPUT TO HIDDEN PROCESS',
                               prefs={VERBOSE_PREF: False,
                                      REPORT_OUTPUT_PREF: False})

hidden_output_process = process(default_variable=[0],
                               pathway=[hidden_layer, hidden_output_weights, output_layer],
                               learning=LEARNING,
                               learning_rate=1.0,
                               target=[1],
                               # name='HIDDEN TO OUTPUT PROCESS',
                               prefs={VERBOSE_PREF: False,
                                      REPORT_OUTPUT_PREF: False})

input_output_process = process(default_variable=[0, 0],
                               pathway=[input_layer, input_output_weights, output_layer],
                               learning=LEARNING,
                               learning_rate=1.0,
                               target=[1],
                               # name='INPUT TO OUTPUT PROCESS',
                               prefs={VERBOSE_PREF: False,
                                      REPORT_OUTPUT_PREF: False})


three_layer_net = system(processes=[input_hidden_process, hidden_output_process, input_output_process,],
                         targets=[0],
                         # targets=[[0],[0]],
                         learning_rate=1,
                         prefs={VERBOSE_PREF: False,
                                REPORT_OUTPUT_PREF: True})

three_layer_net.show_graph()

input_list = {input_layer:[[0, 0], [0, 1], [1, 0], [1, 1]]}
target_list = {output_layer:[[0], [1], [1], [0]]}
# target_list = {output_layer:[[[0],[0]], [[1],[1]], [[1],[1]], [[0],[0]]]}


def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial+1)
    
def show_target():
    i = three_layer_net.input
    t = three_layer_net.target_input_states[0].value
    print('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))
    print('INPUT-OUTPUT WEIGHTS:')
    print(input_hidden_weights.matrix)
    print('HIDDEN-OUTPUT WEIGHTS:')
    print(hidden_output_weights.matrix)
    

three_layer_net.run(num_trials=100,
                  inputs=input_list,
                  targets=target_list,
                  call_before_trial=print_header,
                  call_after_trial=show_target)

