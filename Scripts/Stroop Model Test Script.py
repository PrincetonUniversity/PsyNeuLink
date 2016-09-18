from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

process_prefs = {REPORT_OPUTPUT_PREF: True,
                 VERBOSE_PREF: False}

system_prefs = {REPORT_OPUTPUT_PREF: True,
                VERBOSE_PREF: True}

colors = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Colors")

words = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Words")

response = Transfer(default_input_value=[0,0],
                           function=Logistic,
                           name="Response")

color_naming_process = process(default_input_value=[1, 2.5],
                               configuration=[(colors, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
                               name='Color Naming',
                               prefs=process_prefs
                               )

word_reading_process = process(default_input_value=[.5, 3],
                               configuration=[(words, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
                               name='Word Reading',
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

# TEST VERBOSE_PREFs:
# colors.verbosePref = True
# words.verbosePref = True
color_naming_process.verbosePref = True
word_reading_process.verbosePref = True
mySystem.verbosePref = True

# Execute processes:
# color_naming_process.execute([2, 2])
# word_reading_process.execute([3, 3])

# Execute system:
mySystem.execute(inputs=[[1,1],[1,1]])

# INSPECTIONS:
# mySystem.inspect()
# mySystem.controller.inspect()
