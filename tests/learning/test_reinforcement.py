import random
import numpy as np

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Functions.Function import PROB
from PsyNeuLink.Components.Functions.Function import SoftMax, Reinforcement
from PsyNeuLink.Components.System import System_Base, system
from PsyNeuLink.Globals.TimeScale import CentralClock

random.seed(0)
np.random.seed(0)

def test_reinforcement():
    input_layer = TransferMechanism(
        default_input_value=[0,0,0],
        name='Input Layer',
    )

    action_selection = TransferMechanism(
        default_input_value=[0,0,0],
        function=SoftMax(
            output=PROB,
            gain=1.0,
        ),
        name='Action Selection',
    )

    p = process(
        default_input_value=[0, 0, 0],
        pathway=[input_layer,action_selection],
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
            np.nonzero(action_selection.outputState.value)[0][0],
            action_selection.outputState.value[np.nonzero(action_selection.outputState.value)][0],
            )
        )

    input_list = {input_layer:[[1, 1, 1]]}

    s = system(
        processes=[p],
        # learning_rate=0.05,
        targets=[0],
    )

    s.run(
        num_executions=10,
        inputs=input_list,
        targets=reward,
        call_before_trial=print_header,
        call_after_trial=show_weights,
    )

    mech_objective_action = s.mechanisms[2]
    mech_learning_input_to_action = s.mechanisms[3]

    reward_prediction_weights = action_selection.inputState.receivesFromProjections[0]

    expected_output = [
        (input_layer.outputState.value, np.array([1., 1., 1.])),
        (action_selection.outputState.value, np.array([ 0., 3.38417298, 0.])),
        (mech_objective_action.outputState.value, np.array([6.61582702])),
        (mech_learning_input_to_action.outputState.value, np.array([
            [0.        , 0.        , 0.        ],
            [0.        , 0.33079135, 0.        ],
            [0.        , 0.        , 0.        ]])
        ),
        (reward_prediction_weights.matrix, np.array([
            [1.        , 0.        , 0.        ],
            [0.        , 3.38417298, 0.        ],
            [0.        , 0.        , 2.283625  ]])
        ),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
