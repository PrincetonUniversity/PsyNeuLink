import random as rnd

from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.EVCMechanism import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Functions.System import System_Base
from PsyNeuLink.Globals.Keywords import *


#region Preferences
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))
#endregion

#region Mechanisms
Input = LinearMechanism(name='Input')
Reward = LinearMechanism(name='Reward')
Decision = DDM(params={FUNCTION_PARAMS:{DRIFT_RATE:(1.0, CONTROL_SIGNAL),
                                              #   THRESHOLD:(10.0, CONTROL_SIGNAL)
                                              },
                          kwDDM_AnalyticSolution:kwBogaczEtAl},
                  prefs = DDM_prefs,
                  name='Decision'
                  )
#endregion

#region Processes
TaskExecutionProcess = Process_Base(default_input_value=[0],
                                    params={CONFIGURATION:[(Input, 0),
                                                             IDENTITY_MATRIX,
                                                             (Decision, 0)]},
                                    prefs = process_prefs,
                                    name = 'TaskExecutionProcess')

RewardProcess = Process_Base(default_input_value=[0],
                             params={CONFIGURATION:[(Reward, 1)]},
                             prefs = process_prefs,
                             name = 'RewardProcess')
#endregion

#region System
mySystem = System_Base(params={kwProcesses:[TaskExecutionProcess, RewardProcess]},
                       name='EVC Test System')
#endregion

#region Show
mySystem.show()
mySystem.controller.show()
#endregion

outputStateLabels = mySystem.terminalMechanisms.outputStateLabels
#region Run
numTrials = 10
for i in range(0, numTrials):
    stimulus = rnd.random()*3 - 2

    # Present stimulus:
    CentralClock.time_step = 0
    mySystem.execute([[stimulus],[0]])
    print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateLabels,
                               mySystem.terminalMechanisms.outputStateValues))

    # Present feedback:
    CentralClock.time_step = 1
    mySystem.execute([[0],[1]])
    print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateLabels,
                               mySystem.terminalMechanisms.outputStateValues))

# # Run EVC:
# CentralClock.time_step = 2
# mySystem.execute([[0],[0],[0]])

#endregion