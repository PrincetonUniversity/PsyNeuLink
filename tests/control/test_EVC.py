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

    print(Input.outputState.value)
    print(Decision.outputState.value)
    print(Reward.outputState.value)
    print(RewardPrediction.outputState.value)
    print(InputPrediction.outputState.value)

    from pprint import pprint
    pprint(mySystem.__dict__)

    expected_output = [
        # (Input.outputState.value, ),
        (Decision.outputState.value, 1.0),
        # (Reward.outputState.value, ),
        (RewardPrediction.outputState.value, 10.0),
        (InputPrediction.outputState.value, 0.25)
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
