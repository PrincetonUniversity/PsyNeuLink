import functools
import numpy as np
import pytest
import psyneulink as pnl


from psyneulink.core.components.functions.learningfunctions import Reinforcement
from psyneulink.core.components.functions.transferfunctions import SoftMax
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.system import System
from psyneulink.core.globals.keywords import PROB


def test_reinforcement():
    input_layer = TransferMechanism(
        default_variable=[0, 0, 0],
        name='Input Layer',
    )

    action_selection = TransferMechanism(
        default_variable=[0, 0, 0],
        function=SoftMax(
            output=PROB,
            gain=1.0,
        ),
        name='Action Selection',
    )

    p = Process(
        default_variable=[0, 0, 0],
        size=3,
        pathway=[input_layer, action_selection],
        learning=LearningProjection(learning_function=Reinforcement(learning_rate=0.05)),
        target=0,
    )

    # print ('reward prediction weights: \n', action_selection.input_ports[0].path_afferents[0].matrix)
    # print ('targetMechanism weights: \n', action_selection.output_ports.sendsToProjections[0].matrix)

    reward_values = [10, 10, 10]

    # Must initialize reward (won't be used, but needed for declaration of lambda function)
    action_selection.output_port.value = [0, 0, 1]
    # Get reward value for selected action)
    reward = lambda: [reward_values[int(np.nonzero(action_selection.output_port.value)[0])]]

    def print_header(system):
        print("\n\n**** TRIAL: ", system.scheduler.clock.simple_time)

    def show_weights():
        print('Reward prediction weights: \n', action_selection.input_ports[0].path_afferents[0].get_mod_matrix(s))
        print('\nAction selected:  {}; predicted reward: {}'.format(
            np.nonzero(action_selection.output_port.value)[0][0],
            action_selection.output_port.value[np.nonzero(action_selection.output_port.value)[0][0]],
        ))

    input_list = {input_layer: [[1, 1, 1]]}

    s = System(
        processes=[p],
        # learning_rate=0.05,
        targets=[0],
    )

    results = s.run(
        num_trials=10,
        inputs=input_list,
        targets=reward,
        call_before_trial=functools.partial(print_header, s),
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

    reward_prediction_weights = action_selection.input_ports[0].path_afferents[0]

    expected_output = [
        (input_layer.get_output_values(s), [np.array([1., 1., 1.])]),
        (action_selection.get_output_values(s), [np.array([0.      , 3.71496434      , 0.])]),
        (pytest.helpers.expand_np_ndarray(mech_objective_action.get_output_values(s)), pytest.helpers.expand_np_ndarray([np.array([6.28503566484375]), np.array(39.50167330835792)])),
        (pytest.helpers.expand_np_ndarray(mech_learning_input_to_action.get_output_values(s)), pytest.helpers.expand_np_ndarray([
            [np.array([0.        , 0.31425178324218755 , 0.]), np.array([0.        , 0.31425178324218755 , 0. ])]
        ])),
        (reward_prediction_weights.get_mod_matrix(s), np.array([
            [1.,         0.,         0.        ],
            [0.,         4.02921612, 0.        ],
            [0.,         0.,         1.8775    ],
        ])),
        (results, [
            [np.array([0., 1., 0.])],
            [np.array([0.  , 1.45, 0.  ])],
            [np.array([0., 0., 1.])],
            [np.array([0.    , 1.8775, 0.    ])],
            [np.array([0.      , 2.283625, 0.      ])],
            [np.array([0.        , 2.66944375, 0.        ])],
            [np.array([0.  , 0.  , 1.45])],
            [np.array([0.        , 3.03597156, 0.        ])],
            [np.array([0.      , 3.38417298, 0.])],
            [np.array([0.      , 3.71496434, 0.])],
        ]),
    ]

    for i, exp in enumerate(expected_output):
        val, expected = exp
        np.testing.assert_allclose(val, expected, err_msg='Failed on expected_output[{0}]'.format(i))

def test_reinforcement_fixed_targets():
    input_layer = TransferMechanism(size=2,
                                    name='Input Layer',
    )

    action_selection = pnl.DDM(input_format=pnl.ARRAY,
                               function=pnl.DriftDiffusionAnalytical(),
                               output_ports=[pnl.SELECTED_INPUT_ARRAY],
                               name='DDM')

    p = Process(pathway=[input_layer, action_selection],
                learning=LearningProjection(learning_function=Reinforcement(learning_rate=0.05)))

    input_list = {input_layer: [[1, 1], [1, 1]]}
    s = System(
        processes=[p],
        # learning_rate=0.05,
    )
    targets = [[10.], [10.]]

    # logged_mechanisms = [input_layer, action_selection]
    # for mech in s.learning_mechanisms:
    #     logged_mechanisms.append(mech)
    #
    # for mech in logged_mechanisms:
    #     mech.log.set_log_conditions(items=[pnl.VALUE])

    results = s.run(
        inputs=input_list,
        targets=targets
    )

    assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                [2.08614798], [1.85006765], [2.30401336], [2.08614798], [1.85006765]])
