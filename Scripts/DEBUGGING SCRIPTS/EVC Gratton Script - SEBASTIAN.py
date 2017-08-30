# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *

from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms import EVCMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import *


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
signalSearchRange = np.arange(1.0, 2.0, 0.2)

# Stimulus Mechanisms
Target_Stim = TransferMechanism(name='Target Stimulus', function=Linear(slope = 0.3324))
Flanker_Stim = TransferMechanism(name='Flanker Stimulus', function=Linear(slope = 0.3545221843))

# Processing Mechanisms (Control)
Target_Rep = TransferMechanism(name='Target Representation',
                               function=Linear(slope=(1.0,
                                                     ControlProjection(function=Linear,
                                                                       control_signal_params={
                                                                           ALLOCATION_SAMPLES:signalSearchRange}
                                                                       ))),
                               prefs=mechanism_prefs)
Flanker_Rep = TransferMechanism(name='Flanker Representation',
                               function=Linear(slope=(1.0,
                                                     ControlProjection(function=Linear,
                                                                       control_signal_params={
                                                                           ALLOCATION_SAMPLES:signalSearchRange}
                                                                       ))),
                               prefs=mechanism_prefs)

# Processing Mechanism (Automatic)
Automatic_Component = TransferMechanism(name='Automatic Component',
                               function=Linear(slope=(1.0)),
                               prefs=mechanism_prefs)

# Decision Mechanisms
Decision = DDM(function=BogaczEtAl(drift_rate=1.0,
                                   threshold=0.2645,
                                   noise=0.5,
                                   starting_point=0,
                                   t0=0.15),
               output_states=[DECISION_VARIABLE,
                              RESPONSE_TIME,
                              PROBABILITY_UPPER_THRESHOLD],
               prefs = mechanism_prefs,
               name='Decision')

# Outcome Mechanisms:
Reward = TransferMechanism(name='Reward')


# Processes:
TargetControlProcess = process(
    default_variable=[0],
    pathway=[Target_Stim, Target_Rep, Decision],
    prefs = process_prefs,
    name = 'Target Control Process')

FlankerControlProcess = process(
    default_variable=[0],
    pathway=[Flanker_Stim, Flanker_Rep, Decision],
    prefs = process_prefs,
    name = 'Flanker Control Process')

TargetAutomaticProcess = process(
    default_variable=[0],
    pathway=[Target_Stim, Automatic_Component, Decision],
    prefs = process_prefs,
    name = 'Target Automatic Process')

FlankerAutomaticProcess = process(
    default_variable=[0],
    pathway=[Flanker_Stim, Automatic_Component, Decision],
    prefs = process_prefs,
    name = 'Flanker1 Automatic Process')


RewardProcess = process(
    default_variable=[0],
    pathway=[Reward],
    prefs = process_prefs,
    name = 'RewardProcess')

# System:
mySystem = system(processes=[TargetControlProcess, FlankerControlProcess,
                             TargetAutomaticProcess, FlankerAutomaticProcess,
                             RewardProcess],
                  controller=EVCMechanism,
                  enable_controller=True,
                  monitor_for_control=[Reward, Decision.PROBABILITY_UPPER_THRESHOLD],
                  # monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
                  name='EVC Gratton System')


# Show characteristics of system:
mySystem.show()
mySystem.controller.show()

# configure EVC components
mySystem.controller.control_signals[0].intensity_cost_function = Exponential(rate =  0.8046).function
mySystem.controller.control_signals[1].intensity_cost_function = Exponential(rate =  0.8046).function


# Loop over the KEYS in this dict
# for mech in mySystem.controller.prediction_mechanisms.keys():
for mech in mySystem.controller.prediction_mechanisms.mechanisms:

    # mySystem.controller.prediction_mechanisms is dictionary organized into key-value pairs where the key is a
    # (transfer) mechanism, and the value is the corresponding prediction (integrator) mechanism

    # For example: the key which is a transfer mechanism with the name 'Flanker Stimulus'
    # acceses an integrator mechanism with the name 'Flanker Stimulus_PredictionMechanism'

    if mech.name == 'Flanker Stimulus Prediction Mechanism' or mech.name == 'Target Stimulus Prediction Mechanism':

        # when you find a key mechanism (transfer mechanism) with the correct name, print its name
        print(mech.name)

        # then use that key to access its *value* in the dictionary, which will be an integrator mechanism
        # that integrator mechanism is the one whose rate we want to change ( I think!)
        # mySystem.controller.prediction_mechanisms[mech].function_object.rate = 0.3481
        # mySystem.controller.prediction_mechanisms[mech].parameterStates['rate'].base_value = 0.3481
        # mech.parameterStates['rate'].base_value = 0.3481
        # mySystem.controller.prediction_mechanisms[mech].function_object.rate = 0.3481
        mech.function_object.rate = 1.0
        x = mech.function_object.rate

    if 'Reward' in mech.name :
        print(mech.name)
        mech.function_object.rate = 1.0
        # mySystem.controller.prediction_mechanisms[mech].parameterStates['rate'].base_value = 1.0




print('new rate of integration mechanisms before System execution:')
# for mech in mySystem.controller.prediction_mechanisms.keys():
for mech in mySystem.controller.prediction_mechanisms.mechanisms:
    print(mech.name)
    print(mech.function_object.rate)
    print('----')

# generate stimulus environment

# nTrials = 20
# targetFeatures = [1]
# flankerFeatures = [-1] # for full simulation: flankerFeatures = [-1,1]
# reward = 100

nTrials = 3
targetFeatures = [1 , 1, 1]
flankerFeatures = [1, -1, 1] # for full simulation: flankerFeatures = [-1,1]
reward = 100


# targetInputList = np.random.choice(targetFeatures, nTrials).tolist()
# flankerInputList = np.random.choice(flankerFeatures, nTrials).tolist()
# rewardList = (np.ones(nTrials) * reward).tolist() #np.random.choice(reward, nTrials).tolist()

targetInputList = [1, 1, 1]
flankerInputList = [1, -1, 1]
rewardList = [100, 100, 100]

stim_list_dict = {Target_Stim:targetInputList,
                  Flanker_Stim:flankerInputList,
                  Reward:rewardList}


mySystem.controller.reportOutputPref = True

mySystem.run(num_trials=nTrials,
             inputs=stim_list_dict,
             )

