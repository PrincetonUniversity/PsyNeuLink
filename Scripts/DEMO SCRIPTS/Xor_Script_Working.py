
# coding: utf-8

# In[ ]:

import numpy as np
from PsyNeuLink.Components.Functions.Function import Logistic, Linear
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system


#The following code starts to build a 3 layer neural network 

input_layer = TransferMechanism(name='Input Layer',
                       function=Linear,
                       default_input_value = np.zeros((3,)))

hidden_layer = TransferMechanism(name='Hidden Layer', 
                                 function = Logistic,
                                 default_input_value =[0, 0, 0])

output_layer = TransferMechanism(name='Output Layer',
                        function=Logistic,
                        default_input_value =[0])

input_hidden_weights = MappingProjection(name='Input-Hidden Weights',
                                         matrix = np.random.uniform(-3, 3,(3,3)))

hidden_output_weights = MappingProjection(name='Hidden-Output Weights',
                                         matrix = np.random.uniform(-3, 3,(3,1)))

LearningRate = 0.3

xor_process = process(default_input_value=[0, 0, 1],
                                   pathway=[input_layer,
                                            input_hidden_weights,
                                            hidden_layer,
                                            hidden_output_weights,
                                            output_layer],
                                   learning=LEARNING,
                                   learning_rate=LearningRate,
                                   target=[1],
                                   # name='INPUT TO HIDDEN PROCESS',
                                   prefs={VERBOSE_PREF: False,
                                          REPORT_OUTPUT_PREF: False})


xor_system = system(processes=[xor_process],
                         targets=[0],
                         # targets=[[0],[0]],
                         learning_rate=LearningRate,
                         prefs={VERBOSE_PREF: False,
                                REPORT_OUTPUT_PREF: True})

# xor_system.show_graph()

input_list = {input_layer:[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]}
target_list = {output_layer:[[0], [1], [1], [0]]}

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial+1)
    
def show_target():
    i = xor_system.input
    t = xor_system.targetInputStates[0].value
    print('\nSTIMULI:\n\n- Input: {}\n- Target: {}'.format(i, t))
    print('- Output: {}'.format(output_layer.value))
    # # print('INPUT-OUTPUT WEIGHTS:')
    # # print(input_hidden_weights.matrix)
    # print('HIDDEN-OUTPUT WEIGHTS:')
    # print(hidden_output_weights.matrix)
    

xor_system.run(num_executions=3000,
                  inputs=input_list,
                  targets=target_list,
                  # call_before_trial=print_header,
                  # call_after_trial=show_target
               )

