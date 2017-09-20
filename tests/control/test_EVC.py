import numpy as np
import pytest

from PsyNeuLink.Components.Functions.Function import Linear, BogaczEtAl, Exponential, DRIFT_RATE, THRESHOLD
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import ALLOCATION_SAMPLES, IDENTITY_MATRIX, MEAN, RESULT, VARIANCE
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import ComponentPreferenceSet, kpVerbosePref, \
    kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVC.EVCControlMechanism import EVCControlMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms.DDM import DDM, DECISION_VARIABLE, \
    PROBABILITY_UPPER_THRESHOLD, RESPONSE_TIME


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
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
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

    # Input.prefs.paramValidationPref = False
    # Reward.prefs.paramValidationPref = False
    # Decision.prefs.paramValidationPref = False
    # Decision.input_state.prefs.paramValidationPref = False
    # for mech in [Input, Reward, Decision]:
    #     mech.function_object.prefs.paramValidationPref = False
    #     for os in mech.output_states:
    #         os.prefs.paramValidationPref = False
    #     for instate in mech.input_states:
    #         instate.prefs.paramValidationPref = False
    #     for pstate in mech._parameter_states:
    #         pstate.prefs.paramValidationPref = False

    # Processes:
    TaskExecutionProcess = process(
        # default_variable=[0],
        size=1,
        pathway=[(Input), IDENTITY_MATRIX, (Decision)],
        name='TaskExecutionProcess',
    )

    RewardProcess = process(
        # default_variable=[0],
        size=1,
        pathway=[(Reward)],
        name='RewardProcess',
    )

    # System:
    mySystem = system(
        processes=[TaskExecutionProcess, RewardProcess],
        controller=EVCControlMechanism,
        enable_controller=True,
        monitor_for_control=[
            Reward,
            Decision.PROBABILITY_UPPER_THRESHOLD,
            (Decision.RESPONSE_TIME, -1, 1)
        ],
        name='EVC Test System',
    )

    # TaskExecutionProcess.prefs.paramValidationPref = False
    # RewardProcess.prefs.paramValidationPref = False
    # mySystem.prefs.paramValidationPref = False

    # Stimuli
    stim_list_dict = {
        Input: [0.5, 0.123],
        Reward: [20, 20]
    }

    mySystem.run(
        inputs=stim_list_dict,
    )

    RewardPrediction = mySystem.execution_list[3]
    InputPrediction = mySystem.execution_list[4]

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

    np.testing.assert_allclose(
        pytest.helpers.expand_np_ndarray(mySystem.results),
        expected_results_array,
        atol=1e-08,
        verbose=True,
    )


def test_laming_validation_specify_control_signals():
    # Mechanisms:
    Input = TransferMechanism(
        name='Input'
    )
    Reward = TransferMechanism(
        name='Reward',
        output_states=[RESULT, MEAN, VARIANCE]
    )
    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=1.0,
            threshold=1.0,
            noise=0.5,
            starting_point=0,
            t0=0.45
        ),
        output_states=[
            DECISION_VARIABLE,
            RESPONSE_TIME,
            PROBABILITY_UPPER_THRESHOLD
        ],
        name='Decision'
    )

    # Processes:
    TaskExecutionProcess = process(
        default_variable=[0],
        pathway=[Input, IDENTITY_MATRIX, Decision],
        name='TaskExecutionProcess'
    )

    RewardProcess = process(
        default_variable=[0],
        pathway=[Reward],
        name='RewardProcess'
    )

    # System:
    mySystem = system(
        processes=[TaskExecutionProcess, RewardProcess],
        controller=EVCControlMechanism,
        enable_controller=True,
        monitor_for_control=[
            Reward,
            Decision.PROBABILITY_UPPER_THRESHOLD,
            (Decision.RESPONSE_TIME, -1, 1)
        ],
        control_signals=[
            (DRIFT_RATE, Decision),
            (THRESHOLD, Decision)
        ],
        name='EVC Test System'
    )
    # Stimulus
    stim_list_dict = {
        Input: [0.5, 0.123],
        Reward: [20, 20]
    }

    # Run system:
    mySystem.run(
        inputs=stim_list_dict
    )

    RewardPrediction = mySystem.execution_list[3]
    InputPrediction = mySystem.execution_list[4]

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
        # ALT: float(Decision._parameter_states[DRIFT_RATE].value
        # (mySystem.controller.control_signals[0].value, np.array(1.0)),
        # #       threshold
        #
        # # ALT: float(Decision._parameter_states[THRESHOLD].value
        # (mySystem.controller.control_signals[1].value, np.array(1.0)),

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

    np.testing.assert_almost_equal(
        Decision._parameter_states[DRIFT_RATE].value,
        Decision._parameter_states[DRIFT_RATE].mod_afferents[0].value * Decision._parameter_states[DRIFT_RATE].function_object.value
    )
    np.testing.assert_almost_equal(
        Decision._parameter_states[THRESHOLD].value,
        Decision._parameter_states[THRESHOLD].mod_afferents[0].value * Decision._parameter_states[THRESHOLD].function_object.value
    )
