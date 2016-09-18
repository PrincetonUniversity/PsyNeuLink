from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.AdaptiveIntegrator import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
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
                                              THRESHOLD:(1.0),
                                              NOISE:(0.5),
                                              STARTING_POINT:(0),
                                              NON_DECISION_TIME:(0.45)
                                                 # THRESHOLD:(10.0, CONTROL_SIGNAL)
                                              },
                       kwDDM_AnalyticSolution:BOGACZ_ET_AL},
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
mySystem = System_Base(params={kwProcesses:[TaskExecutionProcess, RewardProcess],
                               MONITORED_OUTPUT_STATES:[Reward, kwDDM_Probability_upperBound,(kwDDM_RT_Mean, -1, 1)]},
                       name='EVC Test System')
#endregion

#region Inspect
mySystem.inspect()
mySystem.controller.inspect()
#endregion

#region Run

inputList = [0.5, 0.123]
rewardList = [20, 20]

for i in range(0,2):

    print("############################ TRIAL {} ############################".format(i));

    stimulusInput = inputList[i]
    rewardInput = rewardList[i]

    # Present stimulus:
    CentralClock.time_step = 0
    mySystem.execute([[stimulusInput],[0]])
    print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateNames,
                               mySystem.terminalMechanisms.outputStateValues))

    # Present feedback:
    CentralClock.time_step = 1
    mySystem.execute([[0],[rewardInput]])
    print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateNames,
                               mySystem.terminalMechanisms.outputStateValues))

#endregion

# output states in EVCMechanism DDM_Error_Rate and DDM_RT_Mean are flipped
# first control intensity in allocation list is 0 but appears to be 1 when multiplied times drift
# how to specify stimulus learning rate? currently there appears to be no learning
# no learning rate