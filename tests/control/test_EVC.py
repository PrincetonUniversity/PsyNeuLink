import numpy as np

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM, DECISION_VARIABLE, PROBABILITY_UPPER_THRESHOLD, RESPONSE_TIME
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Functions.Function import Linear, BogaczEtAl
from PsyNeuLink.Globals.Keywords import ALLOCATION_SAMPLES, IDENTITY_MATRIX, MEAN, RESULT, VARIANCE


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
