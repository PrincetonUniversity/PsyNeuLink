import numpy as np
import psyneulink as pnl


# Preferences:
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.transferfunctions

mechanism_prefs = pnl.BasePreferenceSet(
    prefs={
        pnl.VERBOSE_PREF: pnl.PreferenceEntry(False, pnl.PreferenceLevel.INSTANCE),
        # pnl.REPORT_OUTPUT_PREF: pnl.PreferenceEntry(True, pnl.PreferenceLevel.INSTANCE)
    }
)

process_prefs = pnl.BasePreferenceSet(
    reportOutput_pref=pnl.PreferenceEntry(False, pnl.PreferenceLevel.INSTANCE),
    verbose_pref=pnl.PreferenceEntry(True, pnl.PreferenceLevel.INSTANCE)
)

# Control Parameters
signalSearchRange = np.arange(0.8, 2.0, 0.2)

# Stimulus Mechanisms
Target_Stim = pnl.TransferMechanism(name='Target Stimulus', function=psyneulink.core.components.functions
                                    .transferfunctions.Linear(slope=0.3324))
Flanker_Stim = pnl.TransferMechanism(name='Flanker Stimulus', function=psyneulink.core.components.functions.transferfunctions.Linear(slope=0.3545221843))

# Processing Mechanisms (Control)
Target_Rep = pnl.TransferMechanism(
    name='Target Representation',
    function=psyneulink.core.components.functions.transferfunctions.Linear(
        slope=(
            1.0,
            pnl.ControlProjection(
                function=psyneulink.core.components.functions.transferfunctions.Linear,
                control_signal_params={pnl.ALLOCATION_SAMPLES: signalSearchRange}
            )
        )
    ),
    prefs=mechanism_prefs
)
Flanker_Rep = pnl.TransferMechanism(
    name='Flanker Representation',
    function=psyneulink.core.components.functions.transferfunctions.Linear(
        slope=(
            1.0,
            pnl.ControlProjection(
                function=psyneulink.core.components.functions.transferfunctions.Linear,
                control_signal_params={pnl.ALLOCATION_SAMPLES: signalSearchRange}
            )
        )
    ),
    prefs=mechanism_prefs
)

# Processing Mechanism (Automatic)
Automatic_Component = pnl.TransferMechanism(
    name='Automatic Component',
    function=psyneulink.core.components.functions.transferfunctions.Linear(slope=(1.0)),
    prefs=mechanism_prefs
)

# Decision Mechanisms
Decision = pnl.DDM(
    function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(
        drift_rate=(1.0),
        threshold=(0.2645),
        noise=(0.5),
        starting_point=(0),
        t0=0.15
    ),
    prefs=mechanism_prefs,
    name='Decision',
    output_ports=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD,
        {
            pnl.NAME: 'OFFSET RT',
            pnl.VARIABLE: (pnl.OWNER_VALUE, 2),
            pnl.FUNCTION: psyneulink.core.components.functions.transferfunctions.Linear(0, slope=0.3, intercept=1)
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
    controller=pnl.EVCControlMechanism(name='Task Controller', ),
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

# Show graph of system (with control components)
# mySystem.show_graph()
# mySystem.show_graph(show_dimensions=pnl.ALL, show_projection_labels=True)
# mySystem.show_graph(show_control=True, show_projection_labels=False)
# mySystem.show_graph(show_control=True, show_processes=True, show_headers=False)
# mySystem.show_graph(show_control=True, show_mechanism_structure=True, show_headers=False)
# mySystem.show_graph(show_control=True, show_mechanism_structure=pnl.VALUES)
mySystem.show_graph(show_control=True, show_mechanism_structure=pnl.ALL, show_headers=False)

# configure EVC components
mySystem.controller.control_signals[0].intensity_cost_function = psyneulink.core.components.functions.transferfunctions.Exponential(rate=0.8046).function
mySystem.controller.control_signals[1].intensity_cost_function = psyneulink.core.components.functions.transferfunctions.Exponential(rate=0.8046).function

for mech in mySystem.controller.prediction_mechanisms.mechanisms:
    if mech.name == 'Flanker Stimulus Prediction Mechanism' or mech.name == 'Target Stimulus Prediction Mechanism':
        # when you find a key mechanism (transfer mechanism) with the correct name, print its name
        print(mech.name)
        mech.function.rate = 1.0

    if 'Reward' in mech.name:
        print(mech.name)
        mech.function.rate = 0.8
        # mySystem.controller.prediction_mechanisms[mech].parameterPorts['rate'].base_value = 1.0

print('new rate of integration mechanisms before System execution:')
# for mech in mySystem.controller.prediction_mechanisms.keys():
for mech in mySystem.controller.prediction_mechanisms.mechanisms:
    print(mech.name)
    print(mech.function.rate)
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

Flanker_Rep.set_log_conditions(('mod_slope', pnl.ContextFlags.CONTROL))

mySystem.run(
        num_trials=nTrials,
        inputs=stim_list_dict,
        animate={'show_control':pnl.ALL, pnl.UNIT: pnl.EXECUTION_SET}
)

Flanker_Rep.log.print_entries()

