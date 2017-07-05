import numpy as np
import pytest

from PsyNeuLink.Components.Functions.Function import PROB
from PsyNeuLink.Components.Functions.Function import Reinforcement, SoftMax
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Scheduling.TimeScale import CentralClock


def test_reinforcement():
    input_layer = TransferMechanism(
        default_input_value=[0, 0, 0],
        name='Input Layer',
    )

    action_selection = TransferMechanism(
        default_input_value=[0, 0, 0],
        function=SoftMax(
            output=PROB,
            gain=1.0,
        ),
        name='Action Selection',
    )

    p = process(
        default_input_value=[0, 0, 0],
        pathway=[input_layer, action_selection],
        learning=LearningProjection(learning_function=Reinforcement(learning_rate=0.05)),
        target=0,
    )

    # print ('reward prediction weights: \n', action_selection.input_states[0].path_afferents[0].matrix)
    # print ('targetMechanism weights: \n', action_selection.output_states.sendsToProjections[0].matrix)

    reward_values = [10, 10, 10]

    # Must initialize reward (won't be used, but needed for declaration of lambda function)
    action_selection.output_states.value = [0, 0, 1]
    # Get reward value for selected action)
    reward = lambda: [reward_values[int(np.nonzero(action_selection.output_states.value)[0])]]

    def print_header():
        print("\n\n**** TRIAL: ", CentralClock.trial)

    def show_weights():
        print('Reward prediction weights: \n', action_selection.input_states[0].path_afferents[0].matrix)
        print('\nAction selected:  {}; predicted reward: {}'.format(
            np.nonzero(action_selection.output_states.value)[0][0],
            action_selection.output_states.value[np.nonzero(action_selection.output_states.value)[0][0]],
        ))

    input_list = {input_layer: [[1, 1, 1]]}

    s = system(
        processes=[p],
        # learning_rate=0.05,
        targets=[0],
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
            nested_elem = nested_elem.tolist()
            try:
                iter(nested_elem)
            except TypeError:
                nested_elem = [nested_elem]
            results_list.extend(nested_elem)

    mech_objective_action = s.mechanisms[2]
    mech_learning_input_to_action = s.mechanisms[3]

    reward_prediction_weights = action_selection.input_states[0].path_afferents[0]

    expected_output = [
        (input_layer.output_states.values, [np.array([1., 1., 1.])]),
        (action_selection.output_states.values, [np.array([0., 3.38417298, 0.])]),
        (pytest.helpers.expand_np_ndarray(mech_objective_action.output_states.values), pytest.helpers.expand_np_ndarray([np.array([6.61582702]), np.array(43.7691671006736)])),
        (pytest.helpers.expand_np_ndarray(mech_learning_input_to_action.output_states.values), pytest.helpers.expand_np_ndarray([np.array(
                [0.        , 0.33079135, 0.        ]
            ),
            np.array([
                [0.        , 0.        , 0.        ],
                [0.        , 0.33079135, 0.        ],
                [0.        , 0.        , 0.        ],
            ])
        ])),
        (reward_prediction_weights.matrix, np.array([
            [ 1.,          0.,          0.,        ],
            [ 0.,          3.71496434,  0.,        ],
            [ 0.,          0.,          2.283625,  ],
        ])),
        (results, [
            [np.array([0., 1., 0.])],
            [np.array([0., 1.45, 0.])],
            [np.array([0., 1.8775, 0.])],
            [np.array([0., 2.283625, 0.])],
            [np.array([0., 0., 1.])],
            [np.array([0., 0., 1.45])],
            [np.array([0., 2.66944375, 0.])],
            [np.array([0., 0., 1.8775])],
            [np.array([0., 3.03597156, 0.])],
            [np.array([0., 3.38417298, 0.])]
        ]),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
