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

response2 = TransferMechanism(default_input_value=[0,0],
                           function=Logistic,
                           name="Response2")


color_naming_process = process(
    default_input_value=[1, 2.5],
    # pathway=[(colors, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
    # pathway=[colors, FULL_CONNECTIVITY_MATRIX, response],
    pathway=[colors, FULL_CONNECTIVITY_MATRIX, decision, FULL_CONNECTIVITY_MATRIX, response],
    learning=LEARNING,
    target=[2,2],
    name='Color Naming',
    prefs=process_prefs
)

word_reading_process = process(
    default_input_value=[.5, 3],
    # pathway=[words, FULL_CONNECTIVITY_MATRIX, response],
    pathway=[words, FULL_CONNECTIVITY_MATRIX, decision, FULL_CONNECTIVITY_MATRIX, response2],
    name='Word Reading',
    learning=LEARNING,
    target=[3,3],
    prefs=process_prefs
)

mySystem = system(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model',
                  prefs=system_prefs
                  )

# TEST REPORT_OUTPUT_PREFs:
# colors.reportOutputPref = True
# words.reportOutputPref = True
# response.reportOutputPref = True
# color_naming_process.reportOutputPref = False
# word_reading_process.reportOutputPref =  False
# process_prefs.reportOutputPref = PreferenceEntry(True, PreferenceLevel.CATEGORY)

# mySystem.reportOutputPref = True

# # Execute processes:
# for i in range(10):
#     color_naming_process.execute(input=[1, 1],target=[0,1])
#     print(response.inputState.receivesFromProjections[0].matrix)
#     print(response.inputState.receivesFromProjections[1].matrix)
#
#     word_reading_process.execute(input=[1, 1], target=[1,0])
#     print(response.inputState.receivesFromProjections[0].matrix)
#     print(response.inputState.receivesFromProjections[1].matrix)
#
# # Execute system:
# mySystem.execute(inputs=[[1,1],[1,1]])

# SHOWIONS:
# mySystem.show()
# mySystem.controller.show()

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_target():
    print ('\nColor Naming\n\tInput: {}\n\tTarget: {}'.
           format(color_naming_process.input, color_naming_process.target))
    print ('Wording Reading:\n\tInput: {}\n\tTarget: {}\n'.
           format(word_reading_process.input, word_reading_process.target))
    # print ('\nInput Weights: \n', Input_Weights.matrix)
    # print ('Middle Weights: \n', Middle_Weights.matrix)
    # print ('Output Weights: \n', Output_Weights.matrix)
    print ('MSE: \n', response.outputValue)


mySystem.run(num_executions=10,
      # inputs=stim_list,
      inputs=[[1, 1],[1, 1]],
      targets=[[1, 1]],
      call_before_trial=print_header,
      call_after_trial=show_target)

