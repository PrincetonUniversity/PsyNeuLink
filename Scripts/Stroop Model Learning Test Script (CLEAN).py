from PsyNeuLink.Components.Functions.Function import Linear, Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.System import *
from PsyNeuLink.Globals.Keywords import *

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

decision = TransferMechanism(default_input_value=[0,0],
                           function=Logistic,
                           name="Decision")

response = TransferMechanism(default_input_value=[0,0],
                           function=Logistic,
                           name="Response")

color_naming_process = process(
    default_input_value=[1, 2.5],
    pathway=[colors, FULL_CONNECTIVITY_MATRIX, decision, FULL_CONNECTIVITY_MATRIX, response],
    learning=LEARNING,
    target=[2,2],
    name='Color Naming',
    prefs=process_prefs)

word_reading_process = process(
    default_input_value=[.5, 3],
    pathway=[words, FULL_CONNECTIVITY_MATRIX, decision],
    name='Word Reading',
    learning=LEARNING,
    target=[3,3],
    prefs=process_prefs)

mySystem = system(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model',
                  prefs=system_prefs)

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_target():
    print ('\nColor Naming\n\tInput: {}\n\tTarget: {}'.
           format(color_naming_process.input, color_naming_process.target))
    print ('Wording Reading:\n\tInput: {}\n\tTarget: {}\n'.
           format(word_reading_process.input, word_reading_process.target))
    print ('Response: \n', response.outputValue[0])

stim_list_dict = {colors:[1, 1],
                  words:[2, 2]}

target_list_dict = {response:[1, 1]}

mySystem.run(num_executions=10,
            inputs=stim_list_dict,
            targets=target_list_dict,
            call_before_trial=print_header,
            call_after_trial=show_target)
#
