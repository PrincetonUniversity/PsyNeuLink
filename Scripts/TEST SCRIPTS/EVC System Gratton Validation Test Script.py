
from psyneulink.components.functions.function import Exponential
from psyneulink.components.mechanisms.processing.transfermechanism import *
from psyneulink.components.process import process
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.components.system import MonitoredOutputStatesOption, system
from psyneulink.globals.keywords import *
from psyneulink.globals.preferences.componentpreferenceset import *
from psyneulink.library.mechanisms.processing.integrator.ddm import *
from psyneulink.library.subsystems.evc.evccontrolmechanism import EVCControlMechanism

random.seed(0)
np.random.seed(0)

# Preferences:
mechanism_prefs = ComponentPreferenceSet(
    prefs={
        kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)
    }
)

process_prefs = ComponentPreferenceSet(
    reportOutput_pref=PreferenceEntry(False, PreferenceLevel.INSTANCE),
    verbose_pref=PreferenceEntry(True, PreferenceLevel.INSTANCE)
)

# Control Parameters
signalSearchRange = np.arange(1.0, 2.0, 0.2)

# Stimulus Mechanisms
Target_Stim = TransferMechanism(name='Target Stimulus', function=Linear(slope=0.3324))
Flanker_Stim = TransferMechanism(name='Flanker Stimulus', function=Linear(slope=0.3545221843))

# Processing Mechanisms (Control)
Target_Rep = TransferMechanism(
    name='Target Representation',
    function=Linear(
        slope=(
            1.0,
            ControlProjection(
                function=Linear,
                control_signal_params={ALLOCATION_SAMPLES: signalSearchRange}
            )
        )
    ),
    prefs=mechanism_prefs
)
Flanker_Rep = TransferMechanism(
    name='Flanker Representation',
    function=Linear(
        slope=(
            1.0,
            ControlProjection(
                function=Linear,
                control_signal_params={ALLOCATION_SAMPLES: signalSearchRange}
            )
        )
    ),
    prefs=mechanism_prefs
)

# Processing Mechanism (Automatic)
Automatic_Component = TransferMechanism(
    name='Automatic Component',
    function=Linear(slope=(1.0)),
    prefs=mechanism_prefs
)

# Decision Mechanisms
Decision = DDM(
    function=BogaczEtAl(
        drift_rate=(1.0),
        threshold=(0.2645),
        noise=(0.5),
        starting_point=(0),
        t0=0.15
    ),
    prefs=mechanism_prefs,
    name='Decision',
    output_states=[
        DECISION_VARIABLE,
        RESPONSE_TIME,
        PROBABILITY_UPPER_THRESHOLD
    ],
)

# Outcome Mechanisms:
Reward = TransferMechanism(name='Reward')

# Processes:
TargetControlProcess = process(
    default_variable=[0],
    pathway=[Target_Stim, Target_Rep, Decision],
    prefs=process_prefs,
    name='Target Control Process'
)

FlankerControlProcess = process(
    default_variable=[0],
    pathway=[Flanker_Stim, Flanker_Rep, Decision],
    prefs=process_prefs,
    name='Flanker Control Process'
)

TargetAutomaticProcess = process(
    default_variable=[0],
    pathway=[Target_Stim, Automatic_Component, Decision],
    prefs=process_prefs,
    name='Target Automatic Process'
)

FlankerAutomaticProcess = process(
    default_variable=[0],
    pathway=[Flanker_Stim, Automatic_Component, Decision],
    prefs=process_prefs,
    name='Flanker1 Automatic Process'
)

RewardProcess = process(
    default_variable=[0],
    pathway=[Reward],
    prefs=process_prefs,
    name='RewardProcess'
)

# System:
mySystem = system(
    processes=[
        TargetControlProcess,
        FlankerControlProcess,
        TargetAutomaticProcess,
        FlankerAutomaticProcess,
        RewardProcess
    ],
    controller=EVCControlMechanism,
    enable_controller=True,
    monitor_for_control=[
        Reward,
        (Decision.PROBABILITY_UPPER_THRESHOLD, 1, -1)
    ],
    # monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
    name='EVC Gratton System'
)

# Show characteristics of system:
mySystem.show()
mySystem.controller.show()
# mySystem.show_graph(show_control=True)

# configure EVC components
mySystem.controller.control_signals[0].intensity_cost_function = Exponential(rate=0.8046).function
mySystem.controller.control_signals[1].intensity_cost_function = Exponential(rate=0.8046).function

for mech in mySystem.controller.prediction_mechanisms.mechanisms:
    if mech.name == 'Flanker Stimulus Prediction Mechanism' or mech.name == 'Target Stimulus Prediction Mechanism':
        # when you find a key mechanism (transfer mechanism) with the correct name, print its name
        print(mech.name)
        mech.function_object.rate = 1.0

    if 'Reward' in mech.name:
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

stim_list_dict = {Target_Stim: targetInputList,
                  Flanker_Stim: flankerInputList,
                  Reward: rewardList}

mySystem.controller.reportOutputPref = True

expected_results_array = [
    0.2645,  0.32257753,  0.94819408, 100.,
    0.2645,  0.31663196,  0.95508757, 100.,
    0.2645,  0.31093566,  0.96110142, 100.,
    0.2645,  0.30548947,  0.96633839, 100.,
    0.2645,  0.30029103,  0.97089165, 100.,
    0.2645,  0.3169957,  0.95468427, 100.,
    0.2645,  0.31128378,  0.9607499, 100.,
    0.2645,  0.30582202,  0.96603252, 100.,
    0.2645,  0.30060824,  0.9706259, 100.,
    0.2645,  0.29563774,  0.97461444, 100.,
    0.2645,  0.31163288,  0.96039533, 100.,
    0.2645,  0.30615555,  0.96572397, 100.,
    0.2645,  0.30092641,  0.97035779, 100.,
    0.2645,  0.2959409,  0.97438178, 100.,
    0.2645,  0.29119255,  0.97787196, 100.,
    0.2645,  0.30649004,  0.96541272, 100.,
    0.2645,  0.30124552,  0.97008732, 100.,
    0.2645,  0.29624499,  0.97414704, 100.,
    0.2645,  0.29148205,  0.97766847, 100.,
    0.2645,  0.28694892,  0.98071974, 100.,
    0.2645,  0.30156558,  0.96981445, 100.,
    0.2645,  0.29654999,  0.97391021, 100.,
    0.2645,  0.29177245,  0.97746315, 100.,
    0.2645,  0.28722523,  0.98054192, 100.,
    0.2645,  0.28289958,  0.98320731, 100.,
    0.2645,  0.28289958,  0.98320731, 100.,
    0.2645,  0.42963678,  0.47661181, 100.,
    0.2645,  0.42846471,  0.43938586, 100.,
    -0.2645,  0.42628176,  0.40282965, 100.,
    0.2645,  0.42314468,  0.36732207, 100.,
    -0.2645,  0.41913221,  0.333198, 100.,
    0.2645,  0.42978939,  0.51176048, 100.,
    0.2645,  0.42959394,  0.47427693, 100.,
    -0.2645,  0.4283576,  0.43708106, 100.,
    0.2645,  0.4261132,  0.40057958, 100.,
    -0.2645,  0.422919,  0.36514906, 100.,
    0.2645,  0.42902209,  0.54679323, 100.,
    0.2645,  0.42980788,  0.50942101, 100.,
    -0.2645,  0.42954704,  0.47194318, 100.,
    -0.2645,  0.42824656,  0.43477897, 100.,
    0.2645,  0.42594094,  0.3983337, 100.,
    -0.2645,  0.42735293,  0.58136855, 100.,
    -0.2645,  0.42910149,  0.54447221, 100.,
    0.2645,  0.42982229,  0.50708112, 100.,
    -0.2645,  0.42949608,  0.46961065, 100.,
    -0.2645,  0.42813159,  0.43247968, 100.,
    -0.2645,  0.42482049,  0.61516258, 100.,
    0.2645,  0.42749136,  0.57908829, 100.,
    0.2645,  0.42917687,  0.54214925, 100.,
    -0.2645,  0.42983261,  0.50474093, 100.,
    -0.2645,  0.42944107,  0.46727945, 100.,
    -0.2645,  0.42944107,  0.46727945, 100.,
    0.2645,  0.32257753,  0.94819408, 100.,
    0.2645,  0.31663196,  0.95508757, 100.,
    0.2645,  0.31093566,  0.96110142, 100.,
    0.2645,  0.30548947,  0.96633839, 100.,
    0.2645,  0.30029103,  0.97089165, 100.,
    0.2645,  0.3169957,  0.95468427, 100.,
    0.2645,  0.31128378,  0.9607499, 100.,
    0.2645,  0.30582202,  0.96603252, 100.,
    0.2645,  0.30060824,  0.9706259, 100.,
    0.2645,  0.29563774,  0.97461444, 100.,
    0.2645,  0.31163288,  0.96039533, 100.,
    0.2645,  0.30615555,  0.96572397, 100.,
    0.2645,  0.30092641,  0.97035779, 100.,
    0.2645,  0.2959409,  0.97438178, 100.,
    0.2645,  0.29119255,  0.97787196, 100.,
    0.2645,  0.30649004,  0.96541272, 100.,
    0.2645,  0.30124552,  0.97008732, 100.,
    0.2645,  0.29624499,  0.97414704, 100.,
    0.2645,  0.29148205,  0.97766847, 100.,
    0.2645,  0.28694892,  0.98071974, 100.,
    0.2645,  0.30156558,  0.96981445, 100.,
    0.2645,  0.29654999,  0.97391021, 100.,
    0.2645,  0.29177245,  0.97746315, 100.,
    0.2645,  0.28722523,  0.98054192, 100.,
    0.2645,  0.28289958,  0.98320731, 100.,
    0.2645,  0.28289958,  0.98320731, 100.,
]

mySystem.run(
    num_trials=nTrials,
    inputs=stim_list_dict,
)
