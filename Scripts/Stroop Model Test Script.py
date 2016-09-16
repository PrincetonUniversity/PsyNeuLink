import random as rnd

from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

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
                               name='Color Naming')

word_reading_process = process(default_input_value=[.5, 3],
                               configuration=[(words, 0), FULL_CONNECTIVITY_MATRIX, verbal_response],
                               name='Word Reading')

mySystem = system(processes=[color_naming_process, word_reading_process],
                  enable_controller=False,
                  name='Stroop Model')

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