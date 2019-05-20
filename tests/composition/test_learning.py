import psyneulink as pnl
import numpy as np

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

        # comp.show_graph()

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

        trial_1_expected = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.003,  0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., -0.003,  0.]

        trial_30_expected = [0.]*40
        trial_30_expected +=[.0682143186, .0640966042, .0994344173, .133236921, .152270799, .145592903, .113949692,
                             .0734420009, .0450652924, .0357386468, .0330810871, .0238007805, .0102892090, -.998098988,
                             -.0000773996815, -.0000277845011, -.00000720338916, -.00000120056486, -.0000000965971727, 0.]
        trial_50_expected = [0.]*40
        trial_50_expected += [.717416347, .0816522429, .0595516548, .0379308899, .0193587853, .00686581694,
                              .00351883747, .00902310583, .0149133617, .000263272179, -.0407611997, -.0360124387,
                              .0539085146,  .0723714910, -.000000550934336, -.000000111783778, -.0000000166486478,
                              -.00000000161861854, -.0000000000770770722, 0.]

        assert np.allclose(trial_1_expected, delta_vals[0][0])
        assert np.allclose(trial_30_expected, delta_vals[29][0])
        assert np.allclose(trial_50_expected, delta_vals[49][0])

