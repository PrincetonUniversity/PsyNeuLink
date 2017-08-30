# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *

from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms import EVCMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import *

random.seed(0)
np.random.seed(0)

# Preferences:
DDM_prefs = ComponentPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = ComponentPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

# Mechanisms:
Input = TransferMechanism(name='Input',
                 # params={MONITOR_FOR_CONTROL:[MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]}
                 )
Reward = TransferMechanism(name='Reward',
                 # params={MONITOR_FOR_CONTROL:[PROBABILITY_UPPER_THRESHOLD,(RESPONSE_TIME, -1, 1)]}
                  )
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0, ControlProjection(function=Linear,
                                                                      control_signal={
                                                                          ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)}
                                                                      )),
                                   threshold=(1.0, ControlProjection(function=Linear,
                                                                     control_signal={
                                                                         ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)}
                                                                     )),
                                   noise=(0.5),
                                   starting_point=(0),
                                   t0=0.45),
               prefs = DDM_prefs,
               name='Decision')

# Processes:
TaskExecutionProcess = process(
    default_variable=[0],
    pathway=[(Input, 0), IDENTITY_MATRIX, (Decision, 0)],
    prefs = process_prefs,
    name = 'TaskExecutionProcess')

RewardProcess = process(
    default_variable=[0],
    pathway=[(Reward, 1)],
    prefs = process_prefs,
    name = 'RewardProcess')

# System:
mySystem = system(processes=[TaskExecutionProcess, RewardProcess],
                  controller=EVCMechanism,
                  # controller=EVCMechanism(monitor_for_control=[Reward,
                  #                                              DDM_PROBABILITY_UPPER_THRESHOLD,
                  #                                              DDM_RESPONSE_TIME],
                  #                         outcome_function=LinearCombination(exponents=[1, 1, -1])),
                  enable_controller=True,
                  monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
                  # monitor_for_control=[Input, PROBABILITY_UPPER_THRESHOLD,(RESPONSE_TIME, -1, 1)],
                  # monitor_for_control=[MonitoredOutputStatesOption.ALL_OUTPUT_STATES],
                  name='EVC Test System')

# Show characteristics of system:
# mySystem.show()
# mySystem.controller.show()

# Specify stimuli for run:
# #   two ways to do so:
#
# #   - as a dictionary of stimulus lists; for each entry:
# #     key is name of an origin mechanism in the system
# #     value is a list of its sequence of stimuli (one for each trial)
# inputList = [0.5, 0.123]
# rewardList = [20, 20]
# # stim_list_dict = {Input:[0.5, 0.123],
# #               Reward:[20, 20]}

# stim_list_dict = {Input:[0.5, 0.123],
#                   Reward:[20, 20]}

stim_list_dict = {Input:[0.5],
                  Reward:[20]}


# #   - as a list of trials;
# #     each item in the list contains the stimuli for a given trial,
# #     one for each origin mechanism in the system
# trial_list = [[0.5, 20], [0.123, 20]]
# reversed_trial_list = [[Reward, Input], [20, 0.5], [20, 0.123]]

# Create printouts function (to call in run):
def show_trial_header():
    print("\n############################ TRIAL {} ############################".format(CentralClock.trial))

def show_results():
    import re
    results = sorted(zip(mySystem.terminal_mechanisms.outputStateNames, mySystem.terminal_mechanisms.outputStateValues))
    # print('\nRESULTS (time step {}): [RANDOM: {}]'.format(CentralClock.time_step, np.random.random()))
    print('\nRESULTS (time step {}): [RANDOM: {}]'.format(CentralClock.time_step, np.random.random()))
    print ('\tDrift rate control signal (from EVC):'
           '\n\t\tDecision.parameterState: {}'
           '\n\t\tControlSignal: {}'
           '\n\t\tControlProjection: {}'.
           # format(re.sub('[\[,\],\n]','',str(float(Decision.parameterStates[DRIFT_RATE].value)))))
           format(re.sub('[\[,\],\n]','',str("{:0.3}".format(float(Decision._parameter_states[DRIFT_RATE].value)))),
                  mySystem.controller.output_states['drift_rate_ControlSignal'].value,
                  Decision._parameter_states[DRIFT_RATE].path_afferents[0].value
                  ))
    print ('\tThreshold control signal (from EVC):'
           '\n\t\tDecision.parameterState: {}'
           '\n\t\tControlSignal: {}'
           '\n\t\tControlProjection: {}'.
           format(re.sub('[\[,\],\n]','',str(float(Decision._parameter_states[THRESHOLD].value))),
                  mySystem.controller.output_states['threshold_ControlSignal'].value,
                  Decision._parameter_states[THRESHOLD].path_afferents[0].value
                  ))
    for result in results:
        print("\t{}: {}".format(result[0],
                                re.sub('[\[,\],\n]','',str("{:0.3}".format(float(result[1]))))))

# Run system:

mySystem.controller.reportOutputPref = False

# mySystem.show_graph(direction='LR')

# mySystem.run(inputs=trial_list,
# # mySystem.run(inputs=reversed_trial_list,
mySystem.run(inputs=stim_list_dict,
             call_before_trial=show_trial_header,
             # call_after_time_step=show_results
             call_after_trial=show_results
             )
