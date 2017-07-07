# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
import numpy as np
import random as rand

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *


def test_search_function(controller=None, **kwargs):
    result = np.array(controller.allocationPolicy).reshape(len(controller.allocationPolicy), -1)
    return result

def test_outcome_function(**kwargs):
    result = np.array([0])
    return result


# Preferences:
mechanism_prefs = ComponentPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = ComponentPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

# Control Parameters
signalSearchRange = np.arange(0, 5.1, 0.1)

# Stimulus Mechanisms
Target_Stim = TransferMechanism(name='Target Stimulus', function=Linear(slope = 1))
Flanker1_Stim = TransferMechanism(name='Flanker 1 Stimulus', function=Linear(slope = 1))
Flanker2_Stim = TransferMechanism(name='Flanker 2 Stimulus', function=Linear(slope = 1))

# Processing Mechanisms (Control)
Target_Rep = TransferMechanism(name='Target Representation',
                               function=Linear(slope=(1.0,
                                                     ControlProjection(function=Linear,
                                                                       control_signal={ALLOCATION_SAMPLES:signalSearchRange}
                                                                       ))),
                               prefs=mechanism_prefs)

Flanker1_Rep = TransferMechanism(name='Flanker 1 Representation',
                               function=Linear(slope=(1.0,
                                                     ControlProjection(function=Linear,
                                                                       control_signal={ALLOCATION_SAMPLES:signalSearchRange}
                                                                       ))),
                               prefs=mechanism_prefs)
Flanker2_Rep = TransferMechanism(name='Flanker 2 Representation',
                               function=Linear(slope=(1.0,
                                                     ControlProjection(function=Linear,
                                                                       control_signal={ALLOCATION_SAMPLES:signalSearchRange}
                                                                       ))),
                               prefs=mechanism_prefs)

# Processing Mechanism (Automatic)
Automatic_Component = TransferMechanism(name='Automatic Component',
                               function=Linear(slope=(1.0)),
                               prefs=mechanism_prefs)

# Decision Mechanisms
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0),
                                   threshold=(0.1654),
                                   noise=(0.5),
                                   starting_point=(0),
                                   t0=0.25),
               prefs = mechanism_prefs,
               name='Decision')

# Outcome Mechanisms:
Reward = TransferMechanism(name='Reward')


# Processes:
TargetControlProcess = process(
    default_input_value=[0],
    pathway=[Target_Stim, Target_Rep, Decision],
    prefs = process_prefs,
    name = 'Target Control Process')

Flanker1ControlProcess = process(
    default_input_value=[0],
    pathway=[Flanker1_Stim, Flanker1_Rep, Decision],
    prefs = process_prefs,
    name = 'Flanker 2 Control Process')

Flanker2ControlProcess = process(
    default_input_value=[0],
    pathway=[Flanker2_Stim, Flanker2_Rep, Decision],
    prefs = process_prefs,
    name = 'Flanker 1 Control Process')

TargetAutomaticProcess = process(
    default_input_value=[0],
    pathway=[Target_Stim, Automatic_Component, Decision],
    prefs = process_prefs,
    name = 'Target Automatic Process')

Flanker1AutomaticProcess = process(
    default_input_value=[0],
    pathway=[Flanker1_Stim, Automatic_Component, Decision],
    prefs = process_prefs,
    name = 'Flanker1 Automatic Process')

Flanker2AutomaticProcess = process(
    default_input_value=[0],
    pathway=[Flanker2_Stim, Automatic_Component, Decision],
    prefs = process_prefs,
    name = 'Flanker2 Automatic Process')


RewardProcess = process(
    default_input_value=[0],
    pathway=[(Reward, 1)],
    prefs = process_prefs,
    name = 'RewardProcess')

# System:
mySystem = system(processes=[TargetControlProcess, Flanker1ControlProcess, Flanker2ControlProcess,
                             TargetAutomaticProcess, Flanker1AutomaticProcess, Flanker2AutomaticProcess,
                             RewardProcess],
                  controller=EVCMechanism,
                  enable_controller=True,
                  monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD],
                  name='EVC Gratton System')


# Show characteristics of system:
mySystem.show()
mySystem.controller.show()


# generate stimulus environment

nTrials = 1
targetFeatures = [1]
flankerFeatures = [-1] # for full simulation: flankerFeatures = [-1,1]
reward = 100

targetInputList = np.random.choice(targetFeatures, nTrials).tolist()
flanker1InputList = np.random.choice(flankerFeatures, nTrials).tolist()
flanker2InputList = flanker1InputList
rewardList = np.random.choice(reward, nTrials).tolist()

stim_list_dict = {Target_Stim:targetInputList,
                  Flanker1_Stim:flanker1InputList,
                  Flanker2_Stim:flanker2InputList,
                  Reward:rewardList}


# Create printouts function (to call in run):
def show_trial_header():
    print("\n############################ TRIAL {} ############################".format(CentralClock.trial))

def show_results():
    import re
    results = sorted(zip(mySystem.terminalMechanisms.outputStateNames, mySystem.terminalMechanisms.outputStateValues))
    print('\nRESULTS (time step {}): '.format(CentralClock.time_step))
    print ('\tDrift rate control signal (from EVC): {}'.
           # format(re.sub('[\[,\],\n]','',str(float(Decision.parameterStates[DRIFT_RATE].value)))))
           format(re.sub('[\[,\],\n]','',str("{:0.3}".format(float(Decision._parameter_states[DRIFT_RATE].value))))))
    # print ('\tThreshold control signal (from EVC): {}'.
    #        format(re.sub('[\[,\],\n]','',str(float(Decision.parameterStates[THRESHOLD].value)))))
    for result in results:
        print("\t{}: {}".format(result[0],
                                re.sub('[\[,\],\n]','',str("{:0.3}".format(float(result[1]))))))


# Plot system:
# mySystem.show_graph()

# Run system:

mySystem.controller.reportOutputPref = False
mySystem.run(num_trials=nTrials,
             inputs=stim_list_dict,
             call_before_trial=show_trial_header,
             call_after_time_step=show_results
             )


# Bug
# In the very first trial of system execution, the input to 'Target Stimulus_PredictionMechanism' is 2.0 instead of 1.0
# (the correct input value for Target_Stim is 1.0 as listed in targetInputList)
