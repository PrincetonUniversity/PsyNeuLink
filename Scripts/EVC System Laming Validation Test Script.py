# from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Projections.ControlSignal import ControlSignal
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
Input = Transfer(name='Input')
Reward = Transfer(name='Reward')
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0, ControlSignal(function=Linear)),
                                   threshold=(1.0),
                                   noise=(0.5),
                                   starting_point=(0),
                                   T0=0.45),
               prefs = DDM_prefs,
               name='Decision')
#endregion

#region Processes
TaskExecutionProcess = process(default_input_value=[0],
                               configuration=[(Input, 0), IDENTITY_MATRIX, (Decision, 0)],
                               prefs = process_prefs,
                               name = 'TaskExecutionProcess')

RewardProcess = process(default_input_value=[0],
                        configuration=[(Reward, 1)],
                        prefs = process_prefs,
                        name = 'RewardProcess')
#endregion

#region System
mySystem = System_Base(processes=[TaskExecutionProcess, RewardProcess],
                       monitored_output_states=[Reward, kwDDM_Probability_upperBound,(kwDDM_RT_Mean, -1, 1)],
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

    print("\n############################ TRIAL {} ############################".format(i));

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