import numpy as np
import psyneulink as pnl


def test_search_function(controller=None, **kwargs):
    result = np.array(controller.allocationPolicy).reshape(len(controller.allocationPolicy), -1)
    return result


def test_outcome_function(**kwargs):
    result = np.array([0])
    return result


# Preferences:
mechanism_prefs = pnl.ComponentPreferenceSet(
    prefs={
        pnl.VERBOSE_PREF: pnl.PreferenceEntry(False, pnl.PreferenceLevel.INSTANCE),
        pnl.REPORT_OUTPUT_PREF: pnl.PreferenceEntry(True, pnl.PreferenceLevel.INSTANCE)
    }
)

process_prefs = pnl.ComponentPreferenceSet(
    reportOutput_pref=pnl.PreferenceEntry(False, pnl.PreferenceLevel.INSTANCE),
    verbose_pref=pnl.PreferenceEntry(True, pnl.PreferenceLevel.INSTANCE)
)

# Control Parameters
signalSearchRange = np.arange(0.8, 2.0, 0.2)

# Stimulus Mechanisms
Target_Stim = pnl.TransferMechanism(name='Target Stimulus', function=pnl.Linear(slope=0.3324))
Flanker_Stim = pnl.TransferMechanism(name='Flanker Stimulus', function=pnl.Linear(slope=0.3545221843))

# Processing Mechanisms (Control)
Target_Rep = pnl.TransferMechanism(
    name='Target Representation',
    function=pnl.Linear(
        slope=(
            1.0,
            pnl.ControlProjection(
                function=pnl.Linear,
                control_signal_params={pnl.ALLOCATION_SAMPLES: signalSearchRange}
            )
        )
    ),
    prefs=mechanism_prefs
)
Flanker_Rep = pnl.TransferMechanism(
    name='Flanker Representation',
    function=pnl.Linear(
        slope=(
            1.0,
            pnl.ControlProjection(
                function=pnl.Linear,
                control_signal_params={pnl.ALLOCATION_SAMPLES: signalSearchRange}
            )
        )
    ),
    prefs=mechanism_prefs
)

# Processing Mechanism (Automatic)
Automatic_Component = pnl.TransferMechanism(
    name='Automatic Component',
    function=pnl.Linear(slope=(1.0)),
    prefs=mechanism_prefs
)

# Decision Mechanisms
Decision = pnl.DDM(
    function=pnl.BogaczEtAl(
        drift_rate=(1.0),
        threshold=(0.2645),
        noise=(0.5),
        starting_point=(0),
        t0=0.15
    ),
    prefs=mechanism_prefs,
    name='Decision',
    output_states=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD,
        {
            pnl.NAME: 'OFFSET RT',
            pnl.INDEX: 2,
            pnl.CALCULATE: pnl.Linear(0, slope=0.3, intercept=1).function
        }
    ],
)

# Outcome Mechanisms:
Reward = pnl.TransferMechanism(name='Reward')

# Processes:
TargetControlProcess = pnl.Process(
    default_variable=[0],
    pathway=[Target_Stim, Target_Rep, Decision],
    prefs=process_prefs,
    name='Target Control Process'
)

FlankerControlProcess = pnl.Process(
    default_variable=[0],
    pathway=[Flanker_Stim, Flanker_Rep, Decision],
    prefs=process_prefs,
    name='Flanker Control Process'
)

TargetAutomaticProcess = pnl.Process(
    default_variable=[0],
    pathway=[Target_Stim, Automatic_Component, Decision],
    prefs=process_prefs,
    name='Target Automatic Process'
)

FlankerAutomaticProcess = pnl.Process(
    default_variable=[0],
    pathway=[Flanker_Stim, Automatic_Component, Decision],
    prefs=process_prefs,
    name='Flanker1 Automatic Process'
)

RewardProcess = pnl.Process(
    default_variable=[0],
    pathway=[Reward],
    prefs=process_prefs,
    name='RewardProcess'
)

# System:
mySystem = pnl.System(
    processes=[
        TargetControlProcess,
        FlankerControlProcess,
        TargetAutomaticProcess,
        FlankerAutomaticProcess,
        RewardProcess
    ],
    controller=pnl.EVCControlMechanism,
    enable_controller=True,
    monitor_for_control=[
        Reward,
        Decision.PROBABILITY_UPPER_THRESHOLD,
        ('OFFSET RT', 1, -1),
    ],
    # monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
    name='EVC Gratton System'
)

# Show characteristics of system:
mySystem.show()
mySystem.controller.show()
mySystem.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)

# configure EVC components
mySystem.controller.control_signals[0].intensity_cost_function = pnl.Exponential(rate=0.8046).function
mySystem.controller.control_signals[1].intensity_cost_function = pnl.Exponential(rate=0.8046).function

for mech in mySystem.controller.prediction_mechanisms.mechanisms:
    if mech.name == 'Flanker Stimulus Prediction Mechanism' or mech.name == 'Target Stimulus Prediction Mechanism':
        # when you find a key mechanism (transfer mechanism) with the correct name, print its name
        print(mech.name)
        mech.function_object.rate = 1.0

    if 'Reward' in mech.name:
        print(mech.name)
        mech.function_object.rate = 0.8
        # mySystem.controller.prediction_mechanisms[mech].parameterStates['rate'].base_value = 1.0

print('new rate of integration mechanisms before System execution:')
# for mech in mySystem.controller.prediction_mechanisms.keys():
for mech in mySystem.controller.prediction_mechanisms.mechanisms:
    print(mech.name)
    print(mech.function_object.rate)
    print('----')

# generate stimulus environment

nTrials = 3
targetFeatures = [1, 1, 1]
flankerFeatures = [1, -1, 1]  # for full simulation: flankerFeatures = [-1,1]
reward = [100, 100, 100]

targetInputList = targetFeatures
flankerInputList = flankerFeatures
rewardList = reward

# targetInputList = np.random.choice(targetFeatures, nTrials).tolist()
# flankerInputList = np.random.choice(flankerFeatures, nTrials).tolist()
# rewardList = (np.ones(nTrials) * reward).tolist() #np.random.choice(reward, nTrials).tolist()

stim_list_dict = {
    Target_Stim: targetInputList,
    Flanker_Stim: flankerInputList,
    Reward: rewardList
}

mySystem.controller.reportOutputPref = True


mySystem.run(
    num_trials=nTrials,
    inputs=stim_list_dict,
)
