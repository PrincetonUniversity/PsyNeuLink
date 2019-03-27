import psyneulink as pnl
import numpy as np
import pytest

class TestHebbian:

    def test_simple_hebbian(self):
        Hebb_C = pnl.Composition()
        size = 9

        Hebb2 = pnl.RecurrentTransferMechanism(
            size=size,
            function=pnl.Linear,
            enable_learning=True,
            hetero=0.,
            auto=0.,
            name='Hebb2',
        )

        Hebb_C.add_node(Hebb2)

        src = [1, 0, 0, 1, 0, 0, 1, 0, 0]

        inputs_dict = {Hebb2: np.array(src)}

        Hebb_C.run(num_trials=5,
                   inputs=inputs_dict)
        activity = Hebb2.value

        assert np.allclose(activity, [[1.86643089, 0., 0., 1.86643089, 0., 0., 1.86643089, 0., 0.]])

class TestReinforcement:

    def test_rl(self):
            input_layer = pnl.TransferMechanism(
                default_variable=[0, 0, 0],
                name='Input Layer',
            )

            action_selection = pnl.TransferMechanism(
                default_variable=[0, 0, 0],
                function=pnl.SoftMax(
                    output=pnl.PROB,
                    gain=1.0,
                ),
                name='Action Selection',
            )

            reward_values = [10, 10, 10]
            action_selection.output_state.value = [0, 0, 1]
            reward = lambda: [reward_values[int(np.nonzero(action_selection.output_state.value)[0])]]
            comp = pnl.Composition(name='comp')
            # comp.add_linear_processing_pathway([input_layer, action_selection])
            comp.add_reinforcement_learning_pathway(pathway=[input_layer, action_selection],
                                                    learning_rate=0.05,
                                                    # target=reward
                                                    )


            def print_header():
                print("\n\n**** TIME: ", comp.scheduler_processing.clock.simple_time)

            # def show_weights():
            #     print('Reward prediction weights: \n',
            #           action_selection.input_states[0].path_afferents[0].get_mod_matrix(s))
            #     print('\nAction selected:  {}; predicted reward: {}'.format(
            #         np.nonzero(action_selection.output_state.value)[0][0],
            #         action_selection.output_state.value[np.nonzero(action_selection.output_state.value)[0][0]],
            #     ))
            #
            inputs_dict = {input_layer: [[1, 1, 1]],
                          comp.target_mechanism: [[0., 0., 10.]]}
            #
            comp._analyze_graph()
            comp.show_graph()
            print("\n\n\n\n", )

            print("\n\n\n\n\nRUN:\n")
            results = comp.run(
                num_trials=3,
                inputs=inputs_dict,
                call_before_time_step=print_header
            )

            print("\n\nResults:\n")
            print(results)
            print("\n\nNodes:\n")
            for node in comp.nodes:
                print(node.name)
            print("\n\nProjections:\n")
            for proj in comp.projections:
                print(proj.name)
            #
            # results_list = []
            # for elem in comp.results:
            #     for nested_elem in elem:
            #         nested_elem = nested_elem.tolist()
            #         try:
            #             iter(nested_elem)
            #         except TypeError:
            #             nested_elem = [nested_elem]
            #         results_list.extend(nested_elem)
            #
            # mech_objective_action = comp.nodes[2]
            # mech_learning_input_to_action = comp.nodes[3]
            #
            # reward_prediction_weights = action_selection.input_states[0].path_afferents[0]
            #
            # expected_output = [
            #     (input_layer.get_output_values(comp), [np.array([1., 1., 1.])]),
            #     (action_selection.get_output_values(comp), [np.array([0., 0., 2.283625])]),
            #     (pytest.helpers.expand_np_ndarray(mech_objective_action.get_output_values(comp)),
            #      pytest.helpers.expand_np_ndarray([np.array([7.716375]), np.array(59.542443140625004)])),
            #     (pytest.helpers.expand_np_ndarray(mech_learning_input_to_action.get_output_values(comp)),
            #      pytest.helpers.expand_np_ndarray([
            #          [np.array([0., 0., 0.38581875]), np.array([0., 0., 0.38581875])]
            #      ])),
            #     (reward_prediction_weights.get_mod_matrix(comp), np.array([
            #         [1., 0., 0.],
            #         [0., 3.38417298, 0.],
            #         [0., 0., 2.66944375],
            #     ])),
            #     (results, [
            #         [np.array([0., 1., 0.])],
            #         [np.array([0., 1.45, 0.])],
            #         [np.array([0., 1.8775, 0.])],
            #         [np.array([0., 0., 1.])],
            #         [np.array([0., 0., 1.45])],
            #         [np.array([0., 2.283625, 0.])],
            #         [np.array([0., 0., 1.8775])],
            #         [np.array([0., 2.66944375, 0.])],
            #         [np.array([0., 3.03597156, 0.])],
            #         [np.array([0., 0., 2.283625])]
            #     ]),
            # ]
            #
            # for i in range(len(expected_output)):
            #     val, expected = expected_output[i]
            #     # setting absolute tolerance to be in accordance with reference_output precision
            #     # if you do not specify, assert_allclose will use a relative tolerance of 1e-07,
            #     # which WILL FAIL unless you gather higher precision values to use as reference
            #     np.testing.assert_allclose(val, expected, atol=1e-08,
            #                                err_msg='Failed on expected_output[{0}]'.format(i))
