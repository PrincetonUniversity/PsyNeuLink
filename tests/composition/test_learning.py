import psyneulink as pnl
import numpy as np
import pytest
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336], \
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

    def test_td_montague_et_al_figure_c(self):

        # create processing mechanisms
        sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                       name=pnl.SAMPLE)

        action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                                 function=pnl.Linear(slope=1.0, intercept=0.01),
                                                 name='Action Selection')

        sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                           receiver=action_selection,
                                                           matrix=np.zeros((60, 60)))

        comp = pnl.Composition(name='TD_Learning')
        pathway = [sample_mechanism, sample_to_action_selection, action_selection]
        learning_related_components = comp.add_td_learning_pathway(pathway, learning_rate=0.3)

        comparator_mechanism = learning_related_components[pnl.COMPARATOR_MECHANISM]
        comparator_mechanism.log.set_log_conditions(pnl.VALUE)
        target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]
        # learned_projection = learning_related_components[pnl.LEARNED_PROJECTION]
        comp.show_graph()
        stimulus_onset = 41
        reward_delivery = 54

        # build input dictionary
        samples = []
        targets = []
        for trial in range(120):
            target = [0.]*60
            target[reward_delivery] = 1.
            if trial in {14, 29, 44, 59, 74, 89}:
                target[reward_delivery] = 0.
            targets.append(target)

            sample = [0.]*60
            for i in range(stimulus_onset, 60):
                sample[i] =1.
            samples.append(sample)

        inputs = {sample_mechanism: samples,
                  target_mechanism: targets}


        comp.run(inputs=inputs)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        with plt.style.context('seaborn'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_vals, y_vals = np.meshgrid(np.arange(120), np.arange(40, 60, step=1))
            d_vals = [d.flatten() for d in delta_vals[40:60]]

            ax.plot(x_vals, y_vals, d_vals)
            ax.invert_yaxis()
            ax.set_xlabel("Trial")
            ax.set_ylabel("Timestep")
            ax.set_zlabel("∂")
            ax.set_title("Montague et. al. (1996) -- Figure 5B")
            plt.show()


    def test_td_montague_et_al_figure_b(self):

        # create processing mechanisms
        sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                                 name=pnl.SAMPLE)

        action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                                 function=pnl.Linear(slope=1.0, intercept=1.0),
                                                 name='Action Selection')

        sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                           receiver=action_selection,
                                                           matrix=np.zeros((60, 60)))

        comp = pnl.Composition(name='TD_Learning')
        pathway = [sample_mechanism, sample_to_action_selection, action_selection]
        learning_related_components = comp.add_td_learning_pathway(pathway, learning_rate=0.3)

        comparator_mechanism = learning_related_components[pnl.COMPARATOR_MECHANISM]
        comparator_mechanism.log.set_log_conditions(pnl.VALUE)
        target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]
        # learned_projection = learning_related_components[pnl.LEARNED_PROJECTION]
        # comp.show_graph()

        # build input dictionary
        stimulus_onset = 41
        reward_delivery = 54

        samples = []
        targets = []
        for trial in range(120):
            target = [0.]*60
            if trial not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 29, 44, 59, 74, 89, 104, 119}:
                target[reward_delivery] = 1.
            targets.append(target)

            sample = [0.]*60
            for i in range(stimulus_onset, 60):
                sample[i] = 1.
            samples.append(sample)

        inputs = {sample_mechanism: samples,
                  target_mechanism: targets}


        comp.run(inputs=inputs)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        with plt.style.context('seaborn'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_vals, y_vals = np.meshgrid(np.arange(120), np.arange(40, 60, step=1))
            d_vals = np.array(delta_vals[40:60])

            print("\n\n ------------------------------ X ------------------------------")
            print(x_vals)
            print("\n\n ------------------------------ Y ------------------------------")
            print(y_vals)
            print("\n\n ------------------------------ D ------------------------------")
            for d in d_vals:
                print(d)
            ax.plot(x_vals, y_vals,
                   d_vals.transpose())
            ax.invert_yaxis()
            ax.set_xlabel("Trial")
            ax.set_ylabel("Timestep")
            ax.set_zlabel("∂")
            ax.set_title("Montague et. al. (1996) -- Figure 5B")
            plt.show()


    def test_td_montague_et_al_figure_a(self):

        # create processing mechanisms
        sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                       name=pnl.SAMPLE)

        action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                                 function=pnl.Linear(slope=1.0, intercept=0.01),
                                                 name='Action Selection')

        sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                           receiver=action_selection,
                                                           matrix=np.zeros((60, 60)))

        comp = pnl.Composition(name='TD_Learning')
        pathway = [sample_mechanism, sample_to_action_selection, action_selection]
        learning_related_components = comp.add_td_learning_pathway(pathway, learning_rate=0.3)

        comparator_mechanism = learning_related_components[pnl.COMPARATOR_MECHANISM]
        comparator_mechanism.log.set_log_conditions(pnl.VALUE)
        target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]
        # learned_projection = learning_related_components[pnl.LEARNED_PROJECTION]
        comp.show_graph()
        stimulus_onset = 41
        reward_delivery = 54

        # build input dictionary
        samples = []
        targets = []
        for trial in range(120):
            target = [0.]*60
            target[reward_delivery] = 1.
            if trial in {14, 29, 44, 59, 74, 89}:
                target[reward_delivery] = 0.
            targets.append(target)

            sample = [0.]*60
            for i in range(stimulus_onset, 60):
                sample[i] =1.
            samples.append(sample)

        inputs = {sample_mechanism: samples,
                  target_mechanism: targets}


        comp.run(inputs=inputs)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        with plt.style.context('seaborn'):
            plt.plot(delta_vals[0][0], "-o", label="Trial 1")
            plt.plot(delta_vals[29][0], "-s", label="Trial 30")
            plt.plot(delta_vals[49][0], "-o", label="Trial 50")
            plt.title("Montague et. al. (1996) -- Figure 5A")
            plt.xlabel("Timestep")
            plt.ylabel("∂")
            plt.legend()
            plt.xlim(xmin=35)
            plt.xticks()
            plt.show()

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

