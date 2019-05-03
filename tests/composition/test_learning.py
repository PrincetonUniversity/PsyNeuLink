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
            input_layer = pnl.TransferMechanism(size=2,
                                                name='Input Layer')
            input_layer.log.set_log_conditions(items=pnl.VALUE)
            action_selection =  pnl.DDM(input_format=pnl.ARRAY,
                                        function=pnl.DriftDiffusionAnalytical(),
                                        output_states=[pnl.SELECTED_INPUT_ARRAY],
                                        name='DDM')
            action_selection.log.set_log_conditions(items=pnl.SELECTED_INPUT_ARRAY)

            comp = pnl.Composition(name='comp')
            learning_components = comp.add_reinforcement_learning_pathway(pathway=[input_layer, action_selection],
                                                                          learning_rate=0.05)
            learned_projection = learning_components[pnl.LEARNED_PROJECTION]
            learning_mechanism = learning_components[pnl.LEARNING_MECHANISM]
            target_mechanism = learning_components[pnl.TARGET_MECHANISM]
            comparator_mechanism = learning_components[pnl.COMPARATOR_MECHANISM]

            learned_projection.log.set_log_conditions(items=["matrix", "mod_matrix"])

            inputs_dict = {input_layer: [[1., 1.], [1., 1.]],
                           target_mechanism: [[10.], [10.]]
                           }
            learning_mechanism.log.set_log_conditions(items=[pnl.VALUE])
            comparator_mechanism.log.set_log_conditions(items=[pnl.VALUE])

            target_mechanism.log.set_log_conditions(items=pnl.VALUE)
            comp.run(inputs=inputs_dict)
            # comp.show_graph()

            # input_layer.log.print_entries()
            # action_selection.log.print_entries()
            # comparator_mechanism.log.print_entries()
            # learning_mechanism.log.print_entries()
            # learned_projection.log.print_entries()

            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336], \
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

# class TestBackprop:
#
#     def test_backprop(self):
#         # create processing components
#         input_layer = pnl.TransferMechanism(
#             name='input_layer',
#             function=pnl.Logistic,
#             size=2,
#         )
#
#         hidden_layer = pnl.TransferMechanism(
#             name='hidden_layer',
#             function=pnl.Logistic,
#             size=5
#         )
#
#         output_layer = pnl.TransferMechanism(
#             name='output_layer',
#             function=pnl.Logistic,
#             size=3
#         )
#
#         # assemble composition & create learning components
#         comp = pnl.Composition(name='back-prop-comp')
#         comp.add_linear_processing_pathway([input_layer, hidden_layer])
#         learning_components = comp.add_back_propagation_pathway([hidden_layer, output_layer])
#
#         # unpack learning components
#         learned_projection = learning_components[pnl.LEARNED_PROJECTION]
#         learning_mechanism = learning_components[pnl.LEARNING_MECHANISM]
#         target_mechanism = learning_components[pnl.TARGET_MECHANISM]
#         comparator_mechanism = learning_components[pnl.COMPARATOR_MECHANISM]
#
#         inputs_dict = {input_layer: [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
#                        target_mechanism: [[10.], [10.], [10.], [10.], [10.], [10.]]
#                        }
#         learning_mechanism.log.set_log_conditions(items=[pnl.VALUE])
#         comparator_mechanism.log.set_log_conditions(items=[pnl.VALUE])
#
#         target_mechanism.log.set_log_conditions(items=pnl.VALUE)
#         comp.run(inputs=inputs_dict)
#
#         print(comp.results)

