from Functions.Mechanisms.AdaptiveIntegrator import *
from Functions.Mechanisms.LinearMechanism import *

from Functions.Mechanisms.ProcessingMechanisms.DDM import *
from Functions.Process import Process_Base
from Functions.System import System_Base
from Globals.Keywords import *

#region Preferences
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))
#endregion

#region Mechanisms
TargetInput = LinearMechanism(params={kwExecuteMethodParams:{kwLinearMechanism_Slope:(1.0, kwControlSignal)}},
                              name='TargetInput')

DistractorInput = LinearMechanism(params={kwExecuteMethodParams:{kwLinearMechanism_Slope:(1.0, kwControlSignal)}},
                                  name = 'DistractorInput')


Reward = LinearMechanism(name='Reward')
Decision = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:(1.0),
                                              kwDDM_Threshold:(1.0),
                                              kwDDM_Noise:(0.5),
                                              kwDDM_T0:(0.45)
                                                 # kwDDM_Threshold:(10.0, kwControlSignal)
                                              },
                       kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                  prefs = DDM_prefs,
                  name='Decision'
                  )
#endregion

#region Processes
TargetProcess = Process_Base(default_input_value=[0],
                                    params={kwConfiguration:[(TargetInput, 0),
                                                             kwIdentityMatrix,
                                                             (Decision, 0)]},
                                    prefs = process_prefs,
                                    name = 'TaskExecutionProcess')

DistractorProcess = Process_Base(default_input_value=[0],
                             params={kwConfiguration: [(DistractorInput, 0),
                                                       kwIdentityMatrix,
                                                       (Decision, 0)]},
                             prefs=process_prefs,
                             name='TaskExecutionProcess')

RewardProcess = Process_Base(default_input_value=[0],
                             params={kwConfiguration:[(Reward, 1)]},
                             prefs = process_prefs,
                             name = 'RewardProcess')
#endregion

#region System
mySystem = System_Base(params={kwProcesses:[TargetProcess, DistractorProcess, RewardProcess],
                               kwMonitoredOutputStates:[Reward, kwDDM_Error_Rate,(kwDDM_RT_Mean, -1, 1)]},
                       name='EVC Test System')
#endregion

#region Inspect
mySystem.inspect()
mySystem.controller.inspect()
#endregion

#region Run
nTrials = 2;

inputList = [0.123, 0.345]
rewardList = [10, 10];

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

# how to set default bias parameter in DDM?
# output states in EVCMechanism DDM_Error_Rate and DDM_RT_Mean are flipped
# no cost function assigned
# first control intensity in allocation list is 0 but appears to be 1 when multiplied times drift
# how to specify stimulus learning rate? currently there appears to be no learning
# no learning rate