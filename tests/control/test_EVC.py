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
    )

    RewardPrediction = mySystem.executionList[3][0]
    InputPrediction = mySystem.executionList[4][0]

    expected_output = [
        # (Decision.inputState.value, 0.123),
        (Decision.outputState.value, 1.0),

        # Input Prediction
        # (InputPrediction.inputState.value, 0.123),
        (InputPrediction.outputState.value, 0.1865),

        # RewardPrediction
        # (RewardPrediction.inputState.value, 20.0),
        (RewardPrediction.outputState.value, 15.0),

        # Decision
        # drift rate
        (mySystem.controller.controlSignals[0].value, 1.0),
        # threshold
        (mySystem.controller.controlSignals[1].value, 1.0000000000000002),

        (Decision.outputStates['DDM_decision_variable'].value, 1.0),
        (round(float(Decision.outputStates['DDM_response_time'].value),2), 3.84),
        (round(float(Decision.outputStates['DDM_probability_upperBound'].value),3), 0.816),
        (round(float(Decision.outputStates['DDM_probability_lowerBound'].value),3), 0.184),

        # Reward
        (Reward.outputStates['transfer_mean '].value, 15.0),
        (Reward.outputStates['transfer_result'].value, 15.0),
        (Reward.outputStates['transfer_variance'].value, 0.0),
    ]



    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
