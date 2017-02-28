from PsyNeuLink.Components.Functions.Function import Linear, Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.System import *
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
import numpy as np

process_prefs = {REPORT_OPUTPUT_PREF: True,
                 VERBOSE_PREF: False}

system_prefs = {REPORT_OPUTPUT_PREF: True,
                VERBOSE_PREF: False}

colors = TransferMechanism(default_input_value=[0,0],
                        function=Linear,
                        name="Colors")

words = TransferMechanism(default_input_value=[0,0],
                        function=Linear,
                        name="Words")

hidden = TransferMechanism(default_input_value=[0,0],
                           function=Logistic,
                           name="Hidden")

response = TransferMechanism(default_input_value=[0,0],
                           function=Logistic(),
                           name="Response")

output = TransferMechanism(default_input_value=[0,0],
                           function=Logistic,
                           name="Output")

CH_Weights_matrix = np.arange(4).reshape((2,2))
WH_Weights_matrix = np.arange(4).reshape((2,2))
HO_Weights_matrix = np.arange(4).reshape((2,2))

CH_Weights = MappingProjection(name='Color-Hidden Weights',
                        matrix=CH_Weights_matrix
                        )
WH_Weights = MappingProjection(name='Word-Hidden Weights',
                        matrix=WH_Weights_matrix
                        )
HO_Weights = MappingProjection(name='Hidden-Output Weights',
                        matrix=HO_Weights_matrix
                        )

color_naming_process = process(
    default_input_value=[1, 2.5],
    pathway=[colors, CH_Weights, hidden, HO_Weights, response],
    learning=LEARNING,
    target=[2,2],
    name='Color Naming',
    prefs=process_prefs)

word_reading_process = process(
    default_input_value=[.5, 3],
    pathway=[words, WH_Weights, hidden],
    name='Word Reading',
    learning=LEARNING,
    target=[3,3],
    prefs=process_prefs)

mySystem = system(processes=[color_naming_process, word_reading_process],
                  targets=[0,0],
                  name='Stroop Model',
                  prefs=system_prefs)

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_target():
    print ('\nColor Naming\n\tInput: {}\n\tTarget: {}'.
           # format(color_naming_process.input, color_naming_process.target))
           format(colors.inputValue, mySystem.targets))
    print ('Wording Reading:\n\tInput: {}\n\tTarget: {}\n'.
           # format(word_reading_process.input, word_reading_process.target))
           format(words.inputValue, mySystem.targets))
    print ('Response: \n', response.outputValue[0])
    print ('Hidden-Output:')
    print (HO_Weights.matrix)
    print ('Color-Hidden:')
    print (CH_Weights.matrix)
    print ('Word-Hidden:')
    print (WH_Weights.matrix)


stim_list_dict = {colors:[[1, 1]],
                  words:[[-2, -2]]}

target_list_dict = {response:[[1, 1]]}

mySystem.run(num_executions=2,
            inputs=stim_list_dict,
            targets=target_list_dict,
            call_before_trial=print_header,
            call_after_trial=show_target)

# print()
# for m in mySystem.learningExecutionList:
#     print (m.name)
#
# PsyNeuLink response & weights after 1st trial:
#
# Response:
#  [ 0.50899214  0.54318254]
# Hidden-Output:
# [[ 0.01462766  1.01351195]
#  [ 2.00220713  3.00203878]]
# Color-Hidden:
# [[ 0.01190129  1.0103412 ]
#  [ 2.01190129  3.0103412 ]]
# Word-Hidden:
# [[-0.02380258  0.9793176 ]
#  [ 1.97619742  2.9793176 ]]

# Correct response & weights after 1st trial:
#
# response
#     0.5090    0.5432
# hidden-output weights
#     0.0146    1.0135
#     2.0022    3.0020
# color-hidden weights
#     0.0119    1.0103
#     2.0119    3.0103
# words-hidden weights
#    -0.0238    0.9793
#     1.9762    2.9793