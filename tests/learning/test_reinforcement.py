import random
import numpy as np

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Functions.Function import PROB
from PsyNeuLink.Components.Functions.Function import SoftMax, Reinforcement
from PsyNeuLink.Components.System import System_Base, system
from PsyNeuLink.Globals.TimeScale import CentralClock

def test_reinforcement():
    input_layer = TransferMechanism(
        default_input_value=[0,0,0
            ],
        name='Input Layer',
    )

    action_selection = TransferMechanism(
        default_input_value=[0,0,0
            ],
        function=SoftMax(
            output=PROB,
            gain=1.0,
        ),
        name='Action Selection',
    )

    p = process(
        default_input_value=[0, 0, 0
            ],
        pathway=[input_layer,action_selection
            ],
        learning=LearningProjection(learning_function=Reinforcement(learning_rate=0.05)),
        target=0,
    )

    print ('reward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
    print ('targetMechanism weights: \n', action_selection.outputState.sendsToProjections[0].matrix)

    reward_values = [10, 10, 10]

    # Must initialize reward (won't be used, but needed for declaration of lambda function)
    action_selection.outputState.value = [0, 0, 1]
    # Get reward value for selected action)
    reward = lambda : [reward_values[int(np.nonzero(action_selection.outputState.value)[0])]]

    def print_header():
        print("\n\n**** TRIAL: ", CentralClock.trial)

    def show_weights():
        print ('Reward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
        print ('\nAction selected:  {}; predicted reward: {}'.format(
            np.nonzero(action_selection.outputState.value)[0][0
            ],
            action_selection.outputState.value[np.nonzero(action_selection.outputState.value)][0
            ],
            )
        )

    input_list = {input_layer:[[1, 1, 1]]}

    s = system(
        processes=[p
            ],
        # learning_rate=0.05,
        targets=[0
            ],
    )

    results = s.run(
        num_executions=10,
        inputs=input_list,
        targets=reward,
        call_before_trial=print_header,
        call_after_trial=show_weights,
    )

    results_list = []
    for elem in s.results:
        for nested_elem in elem:
            results_list.append(nested_elem.tolist())

    expected_results_list = [[0.0, 0.0, 2.283625],
                             0.7612083333333333,
                             1.1588762534722221,
                             [0.0, 3.7149643351562496, 0.0],
                             1.2383214450520832,
                             3.066880002551759,
                             [0.0, 4.029216118398437, 0.0],
                             1.3430720394661455,
                             3.607685006391504,
                             [1.0, 0.0, 0.0],
                             0.3333333333333333,
                             0.22222222222222224,
                             [0.0, 4.327755312478515, 0.0],
                             1.442585104159505,
                             4.162103565485778,
                             [0.0, 4.611367546854589, 0.0],
                             1.5371225156181962,
                             4.725491256040825,
                             [0.0, 4.880799169511859, 0.0],
                             1.626933056503953,
                             5.29382234069059,
                             [0.0, 0.0, 2.6694437499999997],
                             0.8898145833333332,
                             1.583539985425347,
                             [0.0, 5.136759211036266, 0.0],
                             1.7122530703454222,
                             5.863621153814649,
                             [0.0, 5.379921250484453, 0.0],
                             1.7933070834948177,
                             6.431900591425378]

    mech_objective_action = s.mechanisms[2]
    mech_learning_input_to_action = s.mechanisms[3]

    reward_prediction_weights = action_selection.inputState.receivesFromProjections[0]

    expected_output = [
        (input_layer.outputState.value, np.array([1., 1., 1.])),
        (action_selection.outputState.value, np.array([ 0., 3.38417298, 0.])),
        (mech_objective_action.outputState.value, np.array([6.61582702])),
        (mech_learning_input_to_action.outputState.value, np.array([
            [0.        , 0.        , 0.
            ],
            [0.        , 0.33079135, 0.
            ],
            [0.        , 0.        , 0.        ]])
        ),
        (reward_prediction_weights.matrix, np.array([
            [1.        , 0.        , 0.
            ],
            [0.        , 3.38417298, 0.
            ],
            [0.        , 0.        , 2.283625  ]])
        ),
        # (results_list, expected_results_list)
        # (results, [
        #     [
        #         np.array([ 0., 0., 2.283625]), np.array(0.7612083333333333), np.array(1.1588762534722221)
        #     ],
        #     [
        #         np.array([ 0., 3.71496434, 0.]), np.array(1.2383214450520832), np.array(3.066880002551759)
        #     ],
        #     [
        #         np.array([ 0., 4.02921612, 0.]), np.array(1.3430720394661455), np.array(3.607685006391504)
        #     ],
        #     [
        #         np.array([ 1., 0., 0.]), np.array(0.3333333333333333), np.array(0.22222222222222224)
        #     ],
        #     [
        #         np.array([ 0., 4.32775531, 0.]), np.array(1.442585104159505), np.array(4.162103565485778)
        #     ],
        #     [
        #         np.array([ 0., 4.61136755, 0.]), np.array(1.5371225156181962), np.array(4.725491256040825)
        #     ],
        #     [
        #         np.array([ 0., 4.88079917, 0.]), np.array(1.626933056503953), np.array(5.29382234069059)
        #     ],
        #     [
        #         np.array([ 0., 0., 2.66944375]), np.array(0.8898145833333332), np.array(1.583539985425347)
        #     ],
        #     [
        #         np.array([ 0., 5.13675921, 0.]), np.array(1.7122530703454222), np.array(5.863621153814649)
        #     ],
        #     [
        #         np.array([ 0., 5.37992125, 0.]), np.array(1.7933070834948177), np.array(6.431900591425378)
        #     ],
        # ])
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
