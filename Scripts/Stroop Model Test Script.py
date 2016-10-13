from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

process_prefs = {REPORT_OPUTPUT_PREF: True,
                 VERBOSE_PREF: False}

system_prefs = {REPORT_OPUTPUT_PREF: True,
                VERBOSE_PREF: False}

colors = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Colors")

words = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Words")

response = Transfer(default_input_value=[0,0],
                           function=Logistic,
                           name="Response")

color_naming_process = process(
    default_input_value=[1, 2.5],
    # configuration=[(colors, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
    configuration=[colors, FULL_CONNECTIVITY_MATRIX, response],
    learning=LEARNING_SIGNAL,
    target=[2,2],
    name='Color Naming',
    prefs=process_prefs
)

word_reading_process = process(
    default_input_value=[.5, 3],
    configuration=[words, FULL_CONNECTIVITY_MATRIX, response],
    name='Word Reading',
    learning=LEARNING_SIGNAL,
    target=[3,3],
    prefs=process_prefs
)

mySystem = system(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model',
                  prefs=system_prefs,
                  )

# TEST REPORT_OUTPUT_PREFs:
# colors.reportOutputPref = True
# words.reportOutputPref = True
# response.reportOutputPref = True
# color_naming_process.reportOutputPref = False
# word_reading_process.reportOutputPref =  False
# process_prefs.reportOutputPref = PreferenceEntry(True, PreferenceLevel.CATEGORY)

# mySystem.reportOutputPref = True

# Execute processes:
for i in range(10):
    color_naming_process.execute(input=[1, 1],target=[0,1])
    print(response.inputState.receivesFromProjections[0].matrix)
    print(response.inputState.receivesFromProjections[1].matrix)

    word_reading_process.execute(input=[1, 1], target=[1,0])
    print(response.inputState.receivesFromProjections[0].matrix)
    print(response.inputState.receivesFromProjections[1].matrix)

# Execute system:
mySystem.execute(inputs=[[1,1],[1,1]])

# SHOWIONS:
# mySystem.show()
# mySystem.controller.show()
