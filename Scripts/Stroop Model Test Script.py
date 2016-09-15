import random as rnd

from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utilities.Utility import Linear, Logistic
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.System import *
from PsyNeuLink.Globals.Keywords import *

color_naming = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Color Naming"
                        )

word_reading = Transfer(default_input_value=[0,0],
                        function=Linear,
                        name="Word Reading")

verbal_response = Transfer(default_input_value=[0,0],
                           function=Logistic)

color_naming_process = process(default_input_value=[1, 2.5],
                               configuration=[(color_naming, 0), FULL_CONNECTIVITY_MATRIX, verbal_response],
                               name='Color Naming')

word_reading_process = process(default_input_value=[.5, 3],
                               configuration=[(word_reading, 0), FULL_CONNECTIVITY_MATRIX, verbal_response],
                               name='Word Naming')

mySystem = System_Base(params={kwProcesses:[color_naming_process, word_reading_process]},
                       name='Stroop Model')

mySystem.execute([1,1],[1,1])


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