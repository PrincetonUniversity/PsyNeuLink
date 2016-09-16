import random as rnd

from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

# process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True, PreferenceLevel.INSTANCE),
#                                       verbose_pref=PreferenceEntry(True, PreferenceLevel.INSTANCE))

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
                               configuration=[(colors, 0), FULL_CONNECTIVITY_MATRIX, verbal_response],
                               name='Color Naming',
                               # prefs=process_prefs
                               )

word_reading_process = process(default_input_value=[.5, 3],
                               configuration=[(words, 0), FULL_CONNECTIVITY_MATRIX, verbal_response],
                               name='Word Reading',
                               # prefs=process_prefs
                               )

mySystem = system(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model')

colors.reportOutputPref = False
color_naming_process.reportOutputPref = True


# colors.reportOutputPref = False
words.reportOutputPref = False
# colors.verbosePref = False
words.verbosePref = False
# color_naming_process.reportOutputPref = True
word_reading_process.reportOutputPref = False
# color_naming_process.verbosePref = True
word_reading_process.verbosePref = False
mySystem.reportOutputPref = True
mySystem.verbosePref = False


mySystem.execute(inputs=[[1,1],[1,1]])


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