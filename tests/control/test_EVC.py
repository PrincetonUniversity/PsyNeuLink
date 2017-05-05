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

    # rearranging mySystem.results into a format that we can compare with pytest
    results_array = []
    for elem in mySystem.results:
        elem_array = []
        for inner_elem in elem:
            elem_array.append(float(inner_elem))
        results_array.append(elem_array)

    # mySystem.results expected output properly formatted
    expected_results_array = [[10.0, 10.0, 0.0, -0.1, 0.48999866671999787, 0.5049998333399996, 0.4950001666600003],
                            [10.0, 10.0, 0.0, -0.4, 1.0896588849786173, 0.5199893401555817, 0.48001065984441826],
                            [10.0, 10.0, 0.0, 0.7000000000000001, 2.4068049288600117, 0.5349429451582145,
                                0.4650570548417855],
                            [10.0, 10.0, 0.0, -1.0000000000000002, 4.4367197849982345, 0.5498339973124778,
                                0.45016600268752216],
                            [10.0, 10.0, 0.0, 0.1, 0.4899786803111636, 0.5199893401555817, 0.48001065984441826],
                            [10.0, 10.0, 0.0, -0.4, 1.0845940171899957, 0.5793242521487494, 0.42067574785125056],
                            [10.0, 10.0, 0.0, 0.7000000000000001, 2.3603355639419292, 0.6364525402815664,
                                0.36354745971843366],
                            [10.0, 10.0, 0.0, 1.0000000000000002, 4.249489622552251, 0.6899744811276125,
                                0.3100255188723875],
                            [10.0, 10.0, 0.0, 0.1, 0.4899347944665309, 0.5349429451582145, 0.4650570548417855],
                            [10.0, 10.0, 0.0, 0.4, 1.0737830412871605, 0.6364525402815664, 0.36354745971843366],
                            [10.0, 10.0, 0.0, 0.7000000000000001, 2.2668657307290365, 0.7271082163411295,
                                0.2728917836588704],
                            [10.0, 10.0, 0.0, 1.0000000000000002, 3.903530154955222, 0.8021838885585819,
                                0.19781611144141814],
                            [10.0, 10.0, 0.0, 0.1, 0.48986719784998234, 0.5498339973124778, 0.45016600268752216],
                            [10.0, 10.0, 0.0, -0.4, 1.0579183396083598, 0.6899744811276125, 0.31002551887238755],
                            [10.0, 10.0, 0.0, 0.7000000000000001, 2.142229775928058, 0.8021838885585819,
                                0.19781611144141814],
                            [10.0, 10.0, 0.0, 1.0000000000000002, 3.4963766238230605, 0.8807970779778825,
                                0.11920292202211748],
                            [10.0, 10.0, 0.0, 1.0000000000000002, 3.4963766238230605, 0.8807970779778825,
                                0.11920292202211748],
                            [15.0, 15.0, 0.0, 0.1, 0.4899992579951842, 0.503729930808051, 0.49627006919194905],
                            [15.0, 15.0, 0.0, -0.4, 1.0898101101714857, 0.5149155731933728, 0.4850844268066272],
                            [15.0, 15.0, 0.0, 0.7000000000000001, 2.4082203479469535, 0.5260862924922933,
                                0.4739137075077066],
                            [15.0, 15.0, 0.0, 1.0000000000000002, 4.442596267412592, 0.5372309601936224,
                                0.4627690398063776],
                            [15.0, 15.0, 0.0, 0.1, 0.4899881318857179, 0.5149155731933728, 0.4850844268066272],
                            [15.0, 15.0, 0.0, 0.4, 1.0869779015554775, 0.5593981893200483, 0.44060181067995174],
                            [15.0, 15.0, 0.0, -0.7000000000000001, 2.3819833629908955, 0.6029471134850862,
                                0.39705288651491377],
                            [15.0, 15.0, 0.0, 1.0000000000000002, 4.335358066652099, 0.6449238558861232,
                                0.3550761441138768],
                            [15.0, 15.0, 0.0, 0.1, 0.489963680570346, 0.5260862924922933, 0.4739137075077066],
                            [15.0, 15.0, 0.0, 0.4, 1.0808517103643738, 0.6029471134850862, 0.39705288651491377],
                            [15.0, 15.0, 0.0, 0.7000000000000001, 2.327128433145911, 0.6750422263908562,
                                0.32495777360914385],
                            [15.0, 15.0, 0.0, 1.0000000000000002, 4.122127099075572, 0.7396980963921578,
                                0.26030190360784217],
                            [15.0, 15.0, 0.0, 0.1, 0.4899259626741259, 0.5372309601936224, 0.4627690398063776],
                            [15.0, 15.0, 0.0, -0.4, 1.0716572906643353, 0.6449238558861232, 0.3550761441138768],
                            [15.0, 15.0, 0.0, 0.7000000000000001, 2.2493422785470294, 0.7396980963921578,
                                0.26030190360784217],
                            [15.0, 15.0, 0.0, 1.0000000000000002, 3.842796481342382, 0.8163782718851771,
                                0.18362172811482289],
                            [15.0, 15.0, 0.0, 1.0000000000000002, 3.842796481342382, 0.8163782718851771,
                             0.18362172811482289]]
    expected_output = [
        # Decision Output | Second Trial
        (Decision.outputState.value, 1.0),

        # Input Prediction Output | Second Trial
        (InputPrediction.outputState.value, 0.1865),

        # RewardPrediction Output | Second Trial
        (RewardPrediction.outputState.value, 15.0),

        # --- Decision Mechanism ---

        #   ControlSignal Values
        #       drift rate
        (mySystem.controller.controlSignals[0].value, 1.0),
        #       threshold
        (mySystem.controller.controlSignals[1].value, 1.0000000000000002),

        #    Output State Values
        #       decision variable
        (Decision.outputStates['DDM_decision_variable'].value, 1.0),
        #       response time
        (round(float(Decision.outputStates['DDM_response_time'].value),2), 3.84),
        #       upper bound
        (round(float(Decision.outputStates['DDM_probability_upperBound'].value),3), 0.816),
        #       lower bound
        (round(float(Decision.outputStates['DDM_probability_lowerBound'].value),3), 0.184),

        # --- Reward Mechanism ---
        #    Output State Values
        #       transfer mean
        (Reward.outputStates['transfer_mean '].value, 15.0),
        #       transfer_result
        (Reward.outputStates['transfer_result'].value, 15.0),
        #       transfer variance
        (Reward.outputStates['transfer_variance'].value, 0.0),

        # System Results Array
        #   (all intermediate output values of system)
        (results_array, expected_results_array)
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
