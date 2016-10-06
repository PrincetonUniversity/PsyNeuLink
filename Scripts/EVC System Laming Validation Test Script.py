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

# Two ways to specify stimuli:

# - as a dictionary of stimulus lists; for each entry:
#     key is name of an origin mechanism in the system
#     value is a list of its stimuli (one for each trial)
inputList = [0.5, 0.123]
rewardList = [20, 20]
stim_lists = {Input:[0.5, 0.123],
          Reward:[20, 20]}
stimListInput = mySystem.construct_input(stim_lists)

# - as a list of trials;
#     each item in the list is a sublist of stimuli,
#     one for each origin mechanism in the system
trial_list = [[0.5, 20], [0.123, 20]]
trialListInput = mySystem.construct_input(trial_list)


#region Set up print out

def show_trial_header():
    print("\n############################ TRIAL {} ############################".format(CentralClock.trial))

def show_results():
    results = sorted(zip(mySystem.terminalMechanisms.outputStateNames, mySystem.terminalMechanisms.outputStateValues))
    print('\nRESULTS (time step {}): '.format(CentralClock.time_step))
    print ('\tControl signal (from EVC): {}'.format(Decision.parameterStates[DRIFT_RATE].value))
    for result in results:
        print("\t{}: {}".format(result[0], result[1]))
#endregion Set up print out


#region Run
mySystem.run(num_trials=2,
             call_before_trial=show_trial_header,
             # inputs=[[[[0.5],[0]],[[0],[20]]],[[[0.123],[0]],[[0],[20]]]],
             inputs=trialListInput,
             call_after_time_step=show_results
             )
#endregion