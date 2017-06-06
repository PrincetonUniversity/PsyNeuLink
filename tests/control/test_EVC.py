import numpy as np

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM, DECISION_VARIABLE, PROBABILITY_UPPER_THRESHOLD, RESPONSE_TIME
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Functions.Function import Linear, BogaczEtAl, Exponential
from PsyNeuLink.Globals.Keywords import ALLOCATION_SAMPLES, IDENTITY_MATRIX, MEAN, RESULT, VARIANCE
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import ComponentPreferenceSet, kpVerbosePref, kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel


def test_EVC():
    # Mechanisms
    Input = TransferMechanism(
        name='Input'
    )
    Reward = TransferMechanism(
        output_states=[RESULT, MEAN, VARIANCE],
        name='Reward'
    )
    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        output_states=[
            DECISION_VARIABLE,
            RESPONSE_TIME,
            PROBABILITY_UPPER_THRESHOLD
        ],
        name='Decision',
    )

    # Processes:
    TaskExecutionProcess = process(
        default_input_value=[0],
        pathway=[(Input), IDENTITY_MATRIX, (Decision)],
        name='TaskExecutionProcess',
    )

    RewardProcess = process(
        default_input_value=[0],
        pathway=[(Reward)],
        name='RewardProcess',
    )

    # System:
    mySystem = system(
        processes=[TaskExecutionProcess, RewardProcess],
        controller=EVCMechanism,
        enable_controller=True,
        monitor_for_control=[
            Reward,
            Decision.PROBABILITY_UPPER_THRESHOLD,
            (Decision.RESPONSE_TIME, -1, 1)
        ],
        name='EVC Test System',
    )

    # Stimuli
    stim_list_dict = {
        Input: [0.5, 0.123],
        Reward: [20, 20]
    }

    mySystem.run(
        inputs=stim_list_dict,
    )

    RewardPrediction = mySystem.executionList[3]
    InputPrediction = mySystem.executionList[4]

    # rearranging mySystem.results into a format that we can compare with pytest
    results_array = []
    for elem in mySystem.results:
        elem_array = []
        for inner_elem in elem:
            elem_array.append(float(inner_elem))
        results_array.append(elem_array)

    # mySystem.results expected output properly formatted
    expected_results_array = [
        [10., 10.0, 0.0, -0.1, 0.48999867, 0.50499983],
        [10., 10.0, 0.0, -0.4, 1.08965888, 0.51998934],
        [10., 10.0, 0.0, 0.7, 2.40680493, 0.53494295],
        [10., 10.0, 0.0, -1., 4.43671978, 0.549834],
        [10., 10.0, 0.0, 0.1, 0.48997868, 0.51998934],
        [10., 10.0, 0.0, -0.4, 1.08459402, 0.57932425],
        [10., 10.0, 0.0, 0.7, 2.36033556, 0.63645254],
        [10., 10.0, 0.0, 1., 4.24948962, 0.68997448],
        [10., 10.0, 0.0, 0.1, 0.48993479, 0.53494295],
        [10., 10.0, 0.0, 0.4, 1.07378304, 0.63645254],
        [10., 10.0, 0.0, 0.7, 2.26686573, 0.72710822],
        [10., 10.0, 0.0, 1., 3.90353015, 0.80218389],
        [10., 10.0, 0.0, 0.1, 0.4898672, 0.549834],
        [10., 10.0, 0.0, -0.4, 1.05791834, 0.68997448],
        [10., 10.0, 0.0, 0.7, 2.14222978, 0.80218389],
        [10., 10.0, 0.0, 1., 3.49637662, 0.88079708],
        [10., 10.0, 0.0, 1., 3.49637662, 0.88079708],
        [15., 15.0, 0.0, 0.1, 0.48999926, 0.50372993],
        [15., 15.0, 0.0, -0.4, 1.08981011, 0.51491557],
        [15., 15.0, 0.0, 0.7, 2.40822035, 0.52608629],
        [15., 15.0, 0.0, 1., 4.44259627, 0.53723096],
        [15., 15.0, 0.0, 0.1, 0.48998813, 0.51491557],
        [15., 15.0, 0.0, 0.4, 1.0869779, 0.55939819],
        [15., 15.0, 0.0, -0.7, 2.38198336, 0.60294711],
        [15., 15.0, 0.0, 1., 4.33535807, 0.64492386],
        [15., 15.0, 0.0, 0.1, 0.48996368, 0.52608629],
        [15., 15.0, 0.0, 0.4, 1.08085171, 0.60294711],
        [15., 15.0, 0.0, 0.7, 2.32712843, 0.67504223],
        [15., 15.0, 0.0, 1., 4.1221271, 0.7396981],
        [15., 15.0, 0.0, 0.1, 0.48992596, 0.53723096],
        [15., 15.0, 0.0, -0.4, 1.07165729, 0.64492386],
        [15., 15.0, 0.0, 0.7, 2.24934228, 0.7396981],
        [15., 15.0, 0.0, 1., 3.84279648, 0.81637827],
        [15., 15.0, 0.0, 1., 3.84279648, 0.81637827]
    ]

    expected_output = [
        # Decision Output | Second Trial
        (Decision.output_states[0].value, np.array(1.0)),

        # Input Prediction Output | Second Trial
        (InputPrediction.output_states[0].value, np.array(0.1865)),

        # RewardPrediction Output | Second Trial
        (RewardPrediction.output_states[0].value, np.array(15.0)),

        # --- Decision Mechanism ---

        #   ControlSignal Values
        #       drift rate
        (mySystem.controller.control_signals[0].value, np.array(1.0)),
        #       threshold
        (mySystem.controller.control_signals[1].value, np.array(1.0)),

        #    Output State Values
        #       decision variable
        (Decision.output_states[DECISION_VARIABLE].value, np.array([1.0])),
        #       response time
        (Decision.output_states[RESPONSE_TIME].value, np.array([3.84279648])),
        #       upper bound
        (Decision.output_states[PROBABILITY_UPPER_THRESHOLD].value, np.array([0.81637827])),
        #       lower bound
        # (round(float(Decision.output_states['DDM_probability_lowerBound'].value),3), 0.184),

        # --- Reward Mechanism ---
        #    Output State Values
        #       transfer mean
        (Reward.output_states[RESULT].value, np.array([15.])),
        #       transfer_result
        (Reward.output_states[MEAN].value, np.array(15.0)),
        #       transfer variance
        (Reward.output_states[VARIANCE].value, np.array(0.0)),

        # System Results Array
        #   (all intermediate output values of system)
        (results_array, expected_results_array)
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))


def test_EVC_gratton():

    def test_search_function(controller=None, **kwargs):
        result = np.array(controller.allocationPolicy).reshape(len(controller.allocationPolicy), -1)
        return result

    def test_outcome_function(**kwargs):
        result = np.array([0])
        return result

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
                    control_signal={ALLOCATION_SAMPLES: signalSearchRange}
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
                    control_signal={ALLOCATION_SAMPLES: signalSearchRange}
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
        default_input_value=[0],
        pathway=[Target_Stim, Target_Rep, Decision],
        prefs=process_prefs,
        name='Target Control Process'
    )

    FlankerControlProcess = process(
        default_input_value=[0],
        pathway=[Flanker_Stim, Flanker_Rep, Decision],
        prefs=process_prefs,
        name='Flanker Control Process'
    )

    TargetAutomaticProcess = process(
        default_input_value=[0],
        pathway=[Target_Stim, Automatic_Component, Decision],
        prefs=process_prefs,
        name='Target Automatic Process'
    )

    FlankerAutomaticProcess = process(
        default_input_value=[0],
        pathway=[Flanker_Stim, Automatic_Component, Decision],
        prefs=process_prefs,
        name='Flanker1 Automatic Process'
    )

    RewardProcess = process(
        default_input_value=[0],
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
        controller=EVCMechanism,
        enable_controller=True,
        monitor_for_control=[
            Reward,
            Decision.PROBABILITY_UPPER_THRESHOLD
        ],
        # monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
        name='EVC Gratton System'
    )

    # Show characteristics of system:
    mySystem.show()
    mySystem.controller.show()

    # configure EVC components
    mySystem.controller.control_signals[0].intensity_cost_function = Exponential(rate=0.8046).function
    mySystem.controller.control_signals[1].intensity_cost_function = Exponential(rate=0.8046).function

    # Loop over the KEYS in this dict
    # for mech in mySystem.controller.prediction_mechanisms.keys():
    #
    #     # mySystem.controller.prediction_mechanisms is dictionary organized into key-value pairs where the key is a
    #     # (transfer) mechanism, and the value is the corresponding prediction (integrator) mechanism
    #
    #     # For example: the key which is a transfer mechanism with the name 'Flanker Stimulus'
    #     # acceses an integrator mechanism with the name 'Flanker Stimulus_PredictionMechanism'
    #
    #     if mech.name is 'Flanker Stimulus' or mech.name is 'Target Stimulus':
    #
    #         # when you find a key mechanism (transfer mechanism) with the correct name, print its name
    #         print(mech.name)
    #
    #         # then use that key to access its *value* in the dictionary, which will be an integrator mechanism
    #         # that integrator mechanism is the one whose rate we want to change ( I think!)
    #         # mySystem.controller.prediction_mechanisms[mech].function_object.rate = 0.3481
    #         # mySystem.controller.prediction_mechanisms[mech].parameter_states['rate'].base_value = 0.3481
    #         # mech.parameter_states['rate'].base_value = 0.3481
    #         # mySystem.controller.prediction_mechanisms[mech].function_object.rate = 1.0 # 0.3481
    #         mySystem.controller.prediction_mechanisms[mech].parameter_states['rate'].base_value = 1 # 0.3481
    #
    #     if mech.name is 'Reward':
    #         print(mech.name)
    #         # mySystem.controller.prediction_mechanisms[mech].function_object.rate = 1.0
    #         mySystem.controller.prediction_mechanisms[mech].parameter_states['rate'].base_value = 1.0
    #

    for mech in mySystem.controller.predictionMechanisms.mechanisms:
        if 'Reward' in mech.name:
            mech._parameter_states['rate'].base_value = 1.0
        if 'Flanker' in mech.name or 'Target' in mech.name:
            mech._parameter_states['rate'].base_value = 1.0

    # print('new rate of integration mechanisms before system execution:')
    # for mech in mySystem.controller.prediction_mechanisms.keys():
    #     print( mySystem.controller.prediction_mechanisms[mech].name)
    #     print( mySystem.controller.prediction_mechanisms[mech].function_object.rate)
    #     print('----')

    # generate stimulus environment

    nTrials = 3
    targetFeatures = [1]
    flankerFeatures = [1]  # for full simulation: flankerFeatures = [-1,1]
    reward = 100

    targetInputList = np.random.choice(targetFeatures, nTrials).tolist()
    flankerInputList = np.random.choice(flankerFeatures, nTrials).tolist()
    rewardList = (np.ones(nTrials) * reward).tolist()  # np.random.choice(reward, nTrials).tolist()

    stim_list_dict = {Target_Stim: targetInputList,
                      Flanker_Stim: flankerInputList,
                      Reward: rewardList}

    mySystem.controller.reportOutputPref = True

    # mySystem.show_graph()

    mySystem.run(
        num_executions=nTrials,
        inputs=stim_list_dict,
    )
