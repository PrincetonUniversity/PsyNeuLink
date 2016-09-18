import random as rnd

from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

process_prefs = {
    REPORT_OPUTPUT_PREF: True,
    VERBOSE_PREF: False
}

system_prefs = {
    REPORT_OPUTPUT_PREF: True,
    VERBOSE_PREF: True
}


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
                               # configuration=[(colors), FULL_CONNECTIVITY_MATRIX, (response)],
                               name='Color Naming',
                               prefs=process_prefs
                               )

word_reading_process = process(default_input_value=[.5, 3],
                               configuration=[(words, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
                               # configuration=[(words), FULL_CONNECTIVITY_MATRIX, (response)],
                               name='Word Reading',
                               prefs=process_prefs
                               )

mySystem = system(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model',
                  prefs=system_prefs
                  )


# colors.reportOutputPref = True
# words.reportOutputPref = True
# response.reportOutputPref = True
# color_naming_process.reportOutputPref = False
# word_reading_process.reportOutputPref =  False
# process_prefs.reportOutputPref = PreferenceEntry(True, PreferenceLevel.CATEGORY)


# mySystem.reportOutputPref = True

# colors.verbosePref = True
# words.verbosePref = True
color_naming_process.verbosePref = True
word_reading_process.verbosePref = True
mySystem.verbosePref = True

# color_naming_process.execute([2, 2])
# word_reading_process.execute([3, 3])

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