import random as rnd

from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True, PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True, PreferenceLevel.INSTANCE))

colors = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Colors")

words = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Words")

verbal_response = Transfer(default_input_value=[0,0],
                           function=Logistic,
                           name="Verbal Response")

color_naming_process = process(default_input_value=[1, 2.5],
                               configuration=[(colors, 0), FULL_CONNECTIVITY_MATRIX, (verbal_response,0)],
                               # configuration=[(colors), FULL_CONNECTIVITY_MATRIX, (verbal_response)],
                               name='Color Naming',
                               # prefs=process_prefs
                               )

# color_naming_process.prefs.reportOutputPref = PreferenceEntry(True, PreferenceLevel.INSTANCE)
# color_naming_pref = color_naming_process.prefs
# color_naming_pref = color_naming_process.prefs.reportOutputPref
# color_naming_pref = color_naming_process.prefs._report_output_pref.setting
# color_naming_pref = color_naming_process.reportOutputPref


word_reading_process = process(default_input_value=[.5, 3],
                               configuration=[(words, 0), FULL_CONNECTIVITY_MATRIX, (verbal_response,0)],
                               # configuration=[(words), FULL_CONNECTIVITY_MATRIX, (verbal_response)],
                               name='Word Reading',
                               # prefs=process_prefs
                               )

# word_reading_process.prefs.reportOutputPref = PreferenceEntry(True, PreferenceLevel.INSTANCE)
# word_reading_pref = word_reading_process.prefs
# word_reading_pref = word_reading_process.prefs.reportOutputPref
# word_reading_pref = word_reading_process.prefs._report_output_pref.setting
# word_reading_pref = word_reading_process.reportOutputPref
#

mySystem = system(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model')

# colors.reportOutputPref = False
# color_pref = colors.prefs
# color_pref = colors.prefs.reportOutputPref
# color_pref = colors.prefs._report_output_pref.setting
# color_pref = colors.reportOutputPref
#
#
# mySystem.prefs.reportOutputPref = False
# mySystem_pref = mySystem.prefs
# mySystem_pref = mySystem.prefs.reportOutputPref
# mySystem_pref = mySystem.prefs._report_output_pref.setting
# mySystem_pref = mySystem.reportOutputPref

colors.reportOutputPref = False
words.reportOutputPref = False
verbal_response.reportOutputPref = False
# color_naming_process.reportOutputPref = PreferenceEntry(True, PreferenceLevel.INSTANCE)
# word_reading_process.reportOutputPref =  PreferenceEntry(True, PreferenceLevel.INSTANCE)
color_naming_process.reportOutputPref = True
word_reading_process.reportOutputPref =  True
# mySystem.reportOutputPref = PreferenceEntry(True, PreferenceLevel.INSTANCE)
mySystem.reportOutputPref = True
# colors.verbosePref = False
# words.verbosePref = False
# color_naming_process.verbosePref = True
# word_reading_process.verbosePref = False
# mySystem.verbosePref = False

color_naming_process.execute([2, 2])
word_reading_process.execute([3, 3])

# mySystem.execute(inputs=[[1,1],[1,1]])


# #region Inspect
# mySystem.inspect()
# mySystem.controller.inspect()
# #endregion

# #region Run
# numTrials = 10
# for i in range(0, numTrials):
#     stimulus = rnd.random()*3 - 2
#
#     # Present stimulus:
#     CentralClock.time_step = 0
#     mySystem.execute([[stimulus, stimulus],[0, 0]])
#     print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateLabels,
#                                mySystem.terminalMechanisms.outputStateValues))
#
#     # Present feedback:
#     CentralClock.time_step = 1
#     mySystem.execute([[0,0],[1,1]])
#     print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateLabels,
#                                mySystem.terminalMechanisms.outputStateValues))
#

#endregion