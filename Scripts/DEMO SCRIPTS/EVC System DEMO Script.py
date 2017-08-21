# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *

import random
random.seed(0)
np.random.seed(0)

#region
# Preferences:
DDM_prefs = ComponentPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})
#endregion

process_prefs = ComponentPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

# Mechanisms:
Input = TransferMechanism(name='Input',
                 # params={MONITOR_FOR_CONTROL:[MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]}

                 )
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0, ControlProjection(function=Linear,
                                                                      control_signal={
                                                                          ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.1)}
                                                                      )),
                                   threshold=(1.0, ControlProjection(function=Linear,
                                                                     control_signal={
                                                                         ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.1)}
                                                                     )),
                                   noise=(0.5),
                                   starting_point=(0),
                                   t0=0.45),
               prefs = DDM_prefs,
               name='Decision')

Reward = TransferMechanism(name='Reward')


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
                  enable_controller=True,
                  monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
                  name='EVC Test System')

#region
#  Show characteristics of system:
# mySystem.show()
# mySystem.controller.show()
mySystem.controller.reportOutputPref = True
#endregion

stim_list_dict = {Input:[0.5, 0.123],
                  Reward:[20, 20]}

#region
# Create printouts function (to call in run):
def show_trial_header():
    print("\n############################ TRIAL {} ############################".format(CentralClock.trial))

def show_results():
    import re
    results = sorted(zip(mySystem.terminal_mechanisms.outputStateNames, mySystem.terminal_mechanisms.outputStateValues))
    print('\nRESULTS (time step {}): '.format(CentralClock.time_step))
    print ('\tDrift rate control signal (from EVC): {}'.
           # format(re.sub('[\[,\],\n]','',str(float(Decision.parameterStates[DRIFT_RATE].value)))))
           format(re.sub('[\[,\],\n]','',str("{:0.3}".format(float(Decision._parameter_states[DRIFT_RATE].value))))))
    print ('\tThreshold control signal (from EVC): {}'.
           format(re.sub('[\[,\],\n]','',str(float(Decision._parameter_states[THRESHOLD].value))),
                  mySystem.controller.output_states['threshold_ControlSignal'].value,
                  Decision._parameter_states[THRESHOLD].path_afferents[0].value
                  ))
    for result in results:
        print("\t{}: {}".format(result[0],
                                re.sub('[\[,\],\n]','',str("{:0.3}".format(float(result[1]))))))
#endregion

# Run system:

mySystem.show_graph()
mySystem.show_graph_with_control()

mySystem.run(inputs=stim_list_dict,
             call_before_trial=show_trial_header,
             # call_after_time_step=show_results
             call_after_trial=show_results
             )
