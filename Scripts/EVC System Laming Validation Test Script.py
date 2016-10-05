# from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Projections.ControlSignal import ControlSignal
from PsyNeuLink.Functions.System import system
from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
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
TaskExecutionProcess = process(
    default_input_value=[0],
    configuration=[(Input, 0), IDENTITY_MATRIX, (Decision, 0)],
    prefs = process_prefs,
    name = 'TaskExecutionProcess')

RewardProcess = process(
    default_input_value=[0],
    configuration=[(Reward, 1)],
    prefs = process_prefs,
    name = 'RewardProcess')
#endregion

#region System
mySystem = system(processes=[TaskExecutionProcess, RewardProcess],
                  controller=EVCMechanism,
                  enable_controller=True,
                  monitored_output_states=[Reward, PROBABILITY_UPPER_BOUND,(RT_MEAN, -1, 1)],
                  # monitored_output_states=[Reward, DECISION_VARIABLE,(RT_MEAN, -1, 1)],
                  name='EVC Test System')
#endregion

#region Inspect
mySystem.inspect()
mySystem.controller.inspect()
#endregion

#region Run

inputList = [0.5, 0.123]
rewardList = [20, 20]

def show_trial_header():
    print("\n############################ TRIAL {} ############################".format(CentralClock.trial))

def show_results():
    results = sorted(zip(mySystem.terminalMechanisms.outputStateNames, mySystem.terminalMechanisms.outputStateValues))
    print('\nRESULTS:')
    print ('\tControl signal (from EVC): {}'.format(Decision.parameterStates[DRIFT_RATE].value))
    for result in results:
        print("\t{}: {}".format(result[0], result[1]))

mySystem.run(num_trials=2,
             call_before=show_trial_header,
             inputs=[[[[0.5],[0]],[[0],[20]]],[[[0.123],[0]],[[0],[20]]]],
             call_after=show_results)


# for i in range(0,2):
#
#     print("\n############################ TRIAL {} ############################".format(i));
#
#     stimulusInput = inputList[i]
#     rewardInput = rewardList[i]
#
#     print("\n Time Step {} ----------------------------------------------------".format(0));
#
#     # Present stimulus:
#     CentralClock.time_step = 0
#     mySystem.execute([[stimulusInput],[0]])
#     results = sorted(zip(mySystem.terminalMechanisms.outputStateNames, mySystem.terminalMechanisms.outputStateValues))
#     print('\nRESULTS:')
#     print ('\tControl signal (from EVC): {}'.format(Decision.parameterStates[DRIFT_RATE].value))
#     for result in results:
#         print("\t{}: {}".format(result[0], result[1]))
#
#     print("\n Time Step {} ----------------------------------------------------".format(1));
#
#     # Present feedback:
#     CentralClock.time_step = 1
#     mySystem.execute([[0],[rewardInput]])
#     # results = sorted(zip(mySystem.terminalMechanisms.outputStateNames, mySystem.terminalMechanisms.outputStateValues))
#     # print('\nRESULTS:')
#     # for result in results:
#     #     print("\t{}: {}".format(result[0], result[1]))

#endregion
