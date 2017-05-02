from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Globals.TimeScale import TimeScale
from PsyNeuLink.scheduling.condition import AfterNCalls
import numpy as np
import random
random.seed(0)
np.random.seed(0)

def test_EVC():
    # Mechanisms
    Input = TransferMechanism(
        name='Input'
    )
    Reward = TransferMechanism(
        name='Reward'
    )
    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        name='Decision',
    )

    # Processes:
    TaskExecutionProcess = process(
        default_input_value=[0],
        pathway=[(Input), IDENTITY_MATRIX, (Decision)],
        name = 'TaskExecutionProcess',
    )

    RewardProcess = process(
        default_input_value=[0],
        pathway=[(Reward)],
        name = 'RewardProcess',
    )

    # System:
    mySystem = system(
        processes=[TaskExecutionProcess, RewardProcess],
        controller=EVCMechanism,
        enable_controller=True,
        monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
        name='EVC Test System',
    )

    # Stimuli
    stim_list_dict = {Input:[0.5, 0.123],
                      Reward:[20, 20]}

    mySystem.run(
        inputs=stim_list_dict,
        # call_before_trial=show_trial_header,
        # # call_after_time_step=show_results
        # call_after_trial=show_results,
        termination_processing={TimeScale.TRIAL: AfterNCalls(Decision, 1)},
    )

    RewardPrediction = mySystem.executionList[3][0]
    InputPrediction = mySystem.executionList[4][0]

    # # Decision
    # print(Decision.inputState.value, 0.123)
    # print(Decision.outputState.value, 1.0)
    #
    # # Input Prediction
    # print(InputPrediction.inputState.value, 0.123)
    # print(InputPrediction.outputState.value, 0.186)
    #
    # # RewardPrediction
    # print(RewardPrediction.inputState.value, 20.0)
    # print(RewardPrediction.outputState.value, 15.0)
    #
    # # Decision
    # # drift rate
    # print(mySystem.controller.controlSignals[0].value, 1.0)
    # # threshold
    # print(mySystem.controller.controlSignals[1].value, 1.0000000000000002)
    #
    # print(Decision.outputStates['DDM_decision_variable'].value, 1.0)
    # print(Decision.outputStates['DDM_response_time'].value, 0.184)
    # print(Decision.outputStates['DDM_probability_upperBound'].value, 0.816)
    # print(Decision.outputStates['DDM_probability_lowerBound'].value, 3.84)
    #
    # # Reward
    # print(Reward.outputStates['transfer_mean '].value, 15.0)
    # print(Reward.outputStates['transfer_result'].value, 15.0)
    # print(Reward.outputStates['transfer_variance'].value, 0.0)
    #
    #

    expected_output = [
        (Decision.inputState.value, 0.123),
        (Decision.outputState.value, 1.0),

        # Input Prediction
        (InputPrediction.inputState.value, 0.123),
        (InputPrediction.outputState.value, 0.186),

        # RewardPrediction
        (RewardPrediction.inputState.value, 20.0),
        (RewardPrediction.outputState.value, 15.0),

        # Decision
        # drift rate
        (mySystem.controller.controlSignals[0].value, 1.0),
        # threshold
        (mySystem.controller.controlSignals[1].value, 1.0000000000000002),

        (Decision.outputStates['DDM_decision_variable'].value, 1.0),
        (Decision.outputStates['DDM_response_time'].value, 0.184),
        (Decision.outputStates['DDM_probability_upperBound'].value, 0.816),
        (Decision.outputStates['DDM_probability_lowerBound'].value, 3.84),

        # Reward
        (Reward.outputStates['transfer_mean '].value, 15.0),
        (Reward.outputStates['transfer_result'].value, 15.0),
        (Reward.outputStates['transfer_variance'].value, 0.0),
    ]

    # objective_response = s.mechanisms[3]
    # objective_hidden = s.mechanisms[7]
    # expected_output = [
    #     (colors.outputState.value, np.array([1., 1.])),
    #     (words.outputState.value, np.array([-2., -2.])),
    #     (hidden.outputState.value, np.array([0.13227553, 0.01990677])),
    #     (response.outputState.value, np.array([0.51044657, 0.5483048])),
    #     (objective_response.outputState.value, np.array([0.48955343, 0.4516952])),
    #     (objective_response.outputStates[TARGET_MSE].value, np.array(0.22184555903789838)),
    #     (objective_hidden.outputState.value, np.array([0., 0.])),
    #     (CH_Weights.matrix, np.array([
    #         [0.01190129, 1.0103412],
    #         [2.01190129, 3.0103412]
    #     ])),
    #     (WH_Weights.matrix, np.array([
    #         [-0.02380258, 0.9793176],
    #         [1.97619742, 2.9793176]
    #     ])),
    #     (HO_Weights.matrix, np.array([
    #         [0.01462766, 1.01351195],
    #         [2.00220713, 3.00203878]
    #     ])),
    # ]

    # for i in range(len(expected_output)):
    #     val, expected = expected_output[i]
    #     # setting absolute tolerance to be in accordance with reference_output precision
    #     # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
    #     # which WILL FAIL unless you gather higher precision values to use as reference
    #     np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
