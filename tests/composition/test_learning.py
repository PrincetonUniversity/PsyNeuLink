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
        output = Hebb_C.learn(num_trials=5,
                   inputs=inputs_dict)

        activity = Hebb2.value

        assert np.allclose(activity, [[1.86643089, 0., 0., 1.86643089, 0., 0., 1.86643089, 0., 0.]])

class TestReinforcement:

    def test_rl(self):
            input_layer = pnl.TransferMechanism(size=2,
                                                name='Input Layer')
            input_layer.log.set_log_conditions(items=pnl.VALUE)
            action_selection = pnl.DDM(input_format=pnl.ARRAY,
                                       function=pnl.DriftDiffusionAnalytical(),
                                       output_ports=[pnl.SELECTED_INPUT_ARRAY],
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
            comp.learn(inputs=inputs_dict)


            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
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
        for trial in range(50):
            target = [0.] * 60
            target[reward_delivery] = 1.
            # {14, 29, 44, 59, 74, 89}
            if trial in {14, 29, 44}:
                target[reward_delivery] = 0.
            targets.append(target)

            sample = [0.] * 60
            for i in range(stimulus_onset, 60):
                sample[i] =1.
            samples.append(sample)

        inputs = {sample_mechanism: samples,
                  target_mechanism: targets}


        comp.learn(inputs=inputs)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        trial_1_expected = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.003,  0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., -0.003,  0.]

        trial_30_expected = [0.] * 40
        trial_30_expected +=[.0682143186, .0640966042, .0994344173, .133236921, .152270799, .145592903, .113949692,
                             .0734420009, .0450652924, .0357386468, .0330810871, .0238007805, .0102892090, -.998098988,
                             -.0000773996815, -.0000277845011, -.00000720338916, -.00000120056486, -.0000000965971727, 0.]
        trial_50_expected = [0.] * 40
        trial_50_expected += [.717416347, .0816522429, .0595516548, .0379308899, .0193587853, .00686581694,
                              .00351883747, .00902310583, .0149133617, .000263272179, -.0407611997, -.0360124387,
                              .0539085146,  .0723714910, -.000000550934336, -.000000111783778, -.0000000166486478,
                              -.00000000161861854, -.0000000000770770722, 0.]

        assert np.allclose(trial_1_expected, delta_vals[0][0])
        assert np.allclose(trial_30_expected, delta_vals[29][0])
        assert np.allclose(trial_50_expected, delta_vals[49][0])

    def test_rl_enable_learning_false(self):
            input_layer = pnl.TransferMechanism(size=2,
                                                name='Input Layer')
            input_layer.log.set_log_conditions(items=pnl.VALUE)
            action_selection = pnl.DDM(input_format=pnl.ARRAY,
                                       function=pnl.DriftDiffusionAnalytical(),
                                       output_ports=[pnl.SELECTED_INPUT_ARRAY],
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
            comp.learn(inputs=inputs_dict)


            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

            # Pause learning -- values are the same as the previous trial (because we pass in the same inputs)
            inputs_dict = {input_layer: [[1., 1.], [1., 1.]]}
            comp.run(inputs=inputs_dict)
            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

            # Resume learning
            inputs_dict = {input_layer: [[1., 1.], [1., 1.]],
                           target_mechanism: [[10.], [10.]]}
            comp.learn(inputs=inputs_dict)
            assert np.allclose(learning_mechanism.value, [np.array([0.38581875, 0.]), np.array([0.38581875, 0.])])
            assert np.allclose(action_selection.value, [[1.], [0.978989672], [0.99996], [0.0000346908466], [0.978989672],
                                                        [0.118109771], [1.32123733], [0.978989672], [0.118109771],
                                                        [1.32123733]])

    def test_td_enabled_learning_false(self):

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
        for trial in range(50):
            target = [0.] * 60
            target[reward_delivery] = 1.
            # {14, 29, 44, 59, 74, 89}
            if trial in {14, 29, 44}:
                target[reward_delivery] = 0.
            targets.append(target)

            sample = [0.] * 60
            for i in range(stimulus_onset, 60):
                sample[i] =1.
            samples.append(sample)

        inputs1 = {sample_mechanism: samples[0:30],
                  target_mechanism: targets[0:30]}

        inputs2 = {sample_mechanism: samples[30:50],
                   target_mechanism: targets[30:50]}

        comp.learn(inputs=inputs1)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        trial_1_expected = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.003,  0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., -0.003,  0.]

        trial_30_expected = [0.] * 40
        trial_30_expected +=[.0682143186, .0640966042, .0994344173, .133236921, .152270799, .145592903, .113949692,
                             .0734420009, .0450652924, .0357386468, .0330810871, .0238007805, .0102892090, -.998098988,
                             -.0000773996815, -.0000277845011, -.00000720338916, -.00000120056486, -.0000000965971727, 0.]


        assert np.allclose(trial_1_expected, delta_vals[0][0])
        assert np.allclose(trial_30_expected, delta_vals[29][0])

        # Pause Learning
        comp.run(inputs={sample_mechanism: samples[0:3]})

        # Resume Learning
        comp.learn(inputs=inputs2)
        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        trial_50_expected = [0.] * 40
        trial_50_expected += [.717416347, .0816522429, .0595516548, .0379308899, .0193587853, .00686581694,
                              .00351883747, .00902310583, .0149133617, .000263272179, -.0407611997, -.0360124387,
                              .0539085146, .0723714910, -.000000550934336, -.000000111783778, -.0000000166486478,
                              -.00000000161861854, -.0000000000770770722, 0.]

        assert np.allclose(trial_50_expected, delta_vals[49][0])


class TestNestedLearning:

    def test_nested_learning(self):
        stim_size = 10
        context_size = 2
        num_actions = 4

        def Concatenate(variable):
            return np.append(variable[0], variable[1])

        stim_in = pnl.ProcessingMechanism(name='Stimulus',
                                          size=stim_size)
        context_in = pnl.ProcessingMechanism(name='Context',
                                             size=context_size)
        reward_in = pnl.ProcessingMechanism(name='Reward',
                                            size=1)

        perceptual_state = pnl.ProcessingMechanism(name='Current Port',
                                                   function=Concatenate,
                                                   input_ports=[{pnl.NAME: 'STIM',
                                                                  pnl.SIZE: stim_size,
                                                                  pnl.PROJECTIONS: stim_in},
                                                                 {pnl.NAME: 'CONTEXT',
                                                                  pnl.SIZE: context_size,
                                                                  pnl.PROJECTIONS: context_in}])

        action = pnl.ProcessingMechanism(name='Action',
                                         size=num_actions)

        # Nested Composition
        rl_agent_state = pnl.ProcessingMechanism(name='RL Agent Port',
                                                 size=5)
        rl_agent_action = pnl.ProcessingMechanism(name='RL Agent Action',
                                                  size=5)
        rl_agent = pnl.Composition(name='RL Agent')
        rl_learning_components = rl_agent.add_reinforcement_learning_pathway([rl_agent_state,
                                                                              rl_agent_action])
        rl_agent._analyze_graph()

        model = pnl.Composition(name='Adaptive Replay Model')
        model.add_nodes([stim_in, context_in, reward_in, perceptual_state, rl_agent, action])
        model.add_projection(sender=perceptual_state, receiver=rl_agent_state)
        model.add_projection(sender=reward_in, receiver=rl_learning_components[pnl.TARGET_MECHANISM])
        model.add_projection(sender=rl_agent_action, receiver=action)
        model.add_projection(sender=rl_agent, receiver=action)

        # model.show_graph(show_controller=True, show_nested=True, show_node_structure=True)

        stimuli = {stim_in: np.array([1] * stim_size),
                   context_in: np.array([10] * context_size)}
        #
        # print(model.run(inputs=stimuli))

    def test_nested_learn_then_run(self):
        iSs = np.array(
            [np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996])
             ],
        )

        cSs = np.array(
            [np.array(
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])]
        )

        oSs = np.array(
            [np.array([0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([1., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 1., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., -0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])]
        )

        nf = 3
        nd = 5
        nh = 200

        D_i = nf * nd
        D_c = nd ** 2
        D_h = nh
        D_o = nf * nd

        wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
        wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
        wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
        who = np.random.rand(D_h, D_o) * 0.02 - 0.01

        il = pnl.TransferMechanism(size=D_i, name='input')
        cl = pnl.TransferMechanism(size=D_c, name='control')
        hl = pnl.TransferMechanism(size=D_h, name='hidden',
                                   function=pnl.Logistic(bias=-2))
        ol = pnl.TransferMechanism(size=D_o, name='output',
                                   function=pnl.Logistic(bias=-2))
        pih = pnl.MappingProjection(matrix=wih)
        pch = pnl.MappingProjection(matrix=wch)
        pco = pnl.MappingProjection(matrix=wco)
        pho = pnl.MappingProjection(matrix=who)

        mnet = pnl.Composition()

        target_mech = mnet.add_backpropagation_learning_pathway(
            [il, pih, hl, pho, ol],
            learning_rate=100
        )[pnl.TARGET_MECHANISM]

        mnet.add_backpropagation_learning_pathway(
            [cl, pch, hl, pho, ol],
            learning_rate=100
        )

        mnet.add_backpropagation_learning_pathway(
            [cl, pco, ol],
            learning_rate=100
        )

        mnet._analyze_graph()

        inputs = {
            il: iSs,
            cl: cSs,
            target_mech: oSs
        }

        outer = Composition("outer-composition")
        outer.add_node(mnet)
        mnet.learn(inputs=inputs)

        del inputs[target_mech]
        # This run should not error, as we are no longer in learning mode (and hence, we shouldn't need the target mech inputs)
        outer.run(inputs={mnet: inputs})

class TestBackProp:

    @pytest.mark.pytorch
    def test_back_prop(self):

        input_layer = pnl.TransferMechanism(name="input",
                                            size=2,
                                            function=pnl.Logistic())

        hidden_layer = pnl.TransferMechanism(name="hidden",
                                             size=2,
                                             function=pnl.Logistic())

        output_layer = pnl.TransferMechanism(name="output",
                                             size=2,
                                             function=pnl.Logistic())

        comp = pnl.Composition(name="backprop-composition")
        learning_components = comp.add_backpropagation_learning_pathway(pathway=[input_layer, hidden_layer, output_layer],
                                                                learning_rate=0.5)
        # learned_projection = learning_components[pnl.LEARNED_PROJECTION]
        # learned_projection.log.set_log_conditions(pnl.MATRIX)
        learning_mechanism = learning_components[pnl.LEARNING_MECHANISM]
        target_mechanism = learning_components[pnl.TARGET_MECHANISM]
        # comparator_mechanism = learning_components[pnl.COMPARATOR_MECHANISM]
        for node in comp.nodes:
            node.log.set_log_conditions(pnl.VALUE)
        # comp.show_graph(show_node_structure=True)
        eid="eid"

        comp.learn(inputs={input_layer: [[1.0, 1.0]],
                         target_mechanism: [[1.0, 1.0]]},
                 num_trials=5,
                 context=eid)

        # for node in comp.nodes:
        #     try:
        #         log = node.log.nparray_dictionary()
        #     except ValueError:
        #         continue
        #     if eid in log:
        #         print(node.name, " values:")
        #         values = log[eid][pnl.VALUE]
        #         for i, val in enumerate(values):
        #             print("     Trial ", i, ":  ", val)
        #         print("\n - - - - - - - - - - - - - - - - - - \n")
        #     else:
        #         print(node.name, " EMPTY LOG!")

    def test_multilayer(self):

        input_layer = pnl.TransferMechanism(name='input_layer',
                                            function=pnl.Logistic,
                                            size=2)

        hidden_layer_1 = pnl.TransferMechanism(name='hidden_layer_1',
                                               function=pnl.Logistic,
                                               size=5)

        hidden_layer_2 = pnl.TransferMechanism(name='hidden_layer_2',
                                               function=pnl.Logistic,
                                               size=4)

        output_layer = pnl.TransferMechanism(name='output_layer',
                                             function=pnl.Logistic,
                                             size=3)

        input_weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        middle_weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        output_weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

        # This projection will be used by the process below by referencing it in the process' pathway;
        #    note: sender and receiver args don't need to be specified
        input_weights = pnl.MappingProjection(
            name='Input Weights',
            matrix=input_weights_matrix,
        )

        # This projection will be used by the process below by assigning its sender and receiver args
        #    to mechanismss in the pathway
        middle_weights = pnl.MappingProjection(
            name='Middle Weights',
            sender=hidden_layer_1,
            receiver=hidden_layer_2,
            matrix=middle_weights_matrix,
        )

        # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
        output_weights = pnl.MappingProjection(
            name='Output Weights',
            sender=hidden_layer_2,
            receiver=output_layer,
            matrix=output_weights_matrix,
        )

        comp = pnl.Composition(name='multilayer')

        p = [input_layer, input_weights, hidden_layer_1, middle_weights, hidden_layer_2, output_weights, output_layer]
        learning_components = comp.add_backpropagation_learning_pathway(
            pathway=p,
            loss_function='sse',
            learning_rate=1.
        )

        target_node = learning_components[pnl.TARGET_MECHANISM]

        input_dictionary = {target_node: [[0., 0., 1.]],
                            input_layer: [[-1., 30.]]}

        # comp.show_graph()

        comp.learn(inputs=input_dictionary,
                 num_trials=10)
    
        objective_output_layer = comp.nodes[5]

        expected_output = [
            (output_layer.get_output_values(comp), [np.array([0.22686074, 0.25270212, 0.91542149])]),
            # error here? why still MSE
            (objective_output_layer.output_ports[pnl.MSE].parameters.value.get(comp), np.array(0.04082589331852094)),
            (input_weights.get_mod_matrix(comp), np.array([
                [ 0.09900247, 0.19839653, 0.29785764, 0.39739191, 0.49700232],
                [ 0.59629092, 0.69403786, 0.79203411, 0.89030237, 0.98885379],
            ])),
            (middle_weights.get_mod_matrix(comp), np.array([
                [ 0.09490249, 0.10488719, 0.12074013, 0.1428774 ],
                [ 0.29677354, 0.30507726, 0.31949676, 0.3404652 ],
                [ 0.49857336, 0.50526254, 0.51830509, 0.53815062],
                [ 0.70029406, 0.70544225, 0.71717037, 0.73594383],
                [ 0.90192903, 0.90561554, 0.91609668, 0.93385292],
            ])),
            (output_weights.get_mod_matrix(comp), np.array([
                [-0.74447522, -0.71016859, 0.31575293],
                [-0.50885177, -0.47444784, 0.56676582],
                [-0.27333719, -0.23912033, 0.8178167 ],
                [-0.03767547, -0.00389039, 1.06888608],
            ])),
            (comp.parameters.results.get(comp), [
                [np.array([0.8344837 , 0.87072018, 0.89997433])],
                [np.array([0.77970193, 0.83263138, 0.90159627])],
                [np.array([0.70218502, 0.7773823 , 0.90307765])],
                [np.array([0.60279149, 0.69958079, 0.90453143])],
                [np.array([0.4967927 , 0.60030321, 0.90610082])],
                [np.array([0.4056202 , 0.49472391, 0.90786617])],
                [np.array([0.33763025, 0.40397637, 0.90977675])],
                [np.array([0.28892812, 0.33633532, 0.9117193 ])],
                [np.array([0.25348771, 0.28791896, 0.9136125 ])],
                [np.array([0.22686074, 0.25270212, 0.91542149])]
            ]),
        ]

        # Test nparray output of log for Middle_Weights

        for i in range(len(expected_output)):
            val, expected = expected_output[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    @pytest.mark.parametrize('models', [
        # [pnl.SYSTEM,pnl.COMPOSITION],
        # [pnl.SYSTEM,'AUTODIFF'],
        [pnl.COMPOSITION,'AUTODIFF']
    ])
    @pytest.mark.pytorch
    def test_xor_training_identicalness_standard_composition_vs_autodiff(self, models):
        """Test equality of results for running 3-layered xor network using System, Composition and Autodiff"""

        num_epochs=2

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])
    
        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])
    
        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)
    
        # SET UP MODELS --------------------------------------------------------------------------------
    
        # System
        if pnl.SYSTEM in models:
    
            input_sys = pnl.TransferMechanism(name='input_sys',
                                           default_variable=np.zeros(2))
    
            hidden_sys = pnl.TransferMechanism(name='hidden_sys',
                                            default_variable=np.zeros(10),
                                            function=pnl.Logistic())
    
            output_sys = pnl.TransferMechanism(name='output_sys',
                                            default_variable=np.zeros(1),
                                            function=pnl.Logistic())
    
            in_to_hidden_sys = pnl.MappingProjection(name='in_to_hidden_sys',
                                        matrix=in_to_hidden_matrix.copy(),
                                        sender=input_sys,
                                        receiver=hidden_sys)
    
            hidden_to_out_sys = pnl.MappingProjection(name='hidden_to_out_sys',
                                        matrix=hidden_to_out_matrix.copy(),
                                        sender=hidden_sys,
                                        receiver=output_sys)
    
            xor_process = pnl.Process(pathway=[input_sys,
                                           in_to_hidden_sys,
                                           hidden_sys,
                                           hidden_to_out_sys,
                                           output_sys],
                                  learning=pnl.LEARNING)

            xor_sys = pnl.System(processes=[xor_process],
                             learning_rate=10)
    
        # STANDARD Composition
        if pnl.COMPOSITION in models:
    
            input_comp = pnl.TransferMechanism(name='input_comp',
                                       default_variable=np.zeros(2))
    
            hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                        default_variable=np.zeros(10),
                                        function=pnl.Logistic())
    
            output_comp = pnl.TransferMechanism(name='output_comp',
                                        default_variable=np.zeros(1),
                                        function=pnl.Logistic())
    
            in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                        matrix=in_to_hidden_matrix.copy(),
                                        sender=input_comp,
                                        receiver=hidden_comp)
    
            hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                        matrix=hidden_to_out_matrix.copy(),
                                        sender=hidden_comp,
                                        receiver=output_comp)
    
            xor_comp = pnl.Composition()
    
            learning_components = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                         in_to_hidden_comp,
                                                                         hidden_comp,
                                                                         hidden_to_out_comp,
                                                                         output_comp],
                                                                        learning_rate=10)
            target_mech = learning_components[pnl.TARGET_MECHANISM]

        # AutodiffComposition
        if 'AUTODIFF' in models:
    
            input_autodiff = pnl.TransferMechanism(name='input',
                                       default_variable=np.zeros(2))
    
            hidden_autodiff = pnl.TransferMechanism(name='hidden',
                                        default_variable=np.zeros(10),
                                        function=pnl.Logistic())
    
            output_autodiff = pnl.TransferMechanism(name='output',
                                        default_variable=np.zeros(1),
                                        function=pnl.Logistic())
    
            in_to_hidden_autodiff = pnl.MappingProjection(name='in_to_hidden',
                                        matrix=in_to_hidden_matrix.copy(),
                                        sender=input_autodiff,
                                        receiver=hidden_autodiff)
    
            hidden_to_out_autodiff = pnl.MappingProjection(name='hidden_to_out',
                                        matrix=hidden_to_out_matrix.copy(),
                                        sender=hidden_autodiff,
                                        receiver=output_autodiff)
    
            xor_autodiff = pnl.AutodiffComposition(param_init_from_pnl=True,
                                      learning_rate=10,
                                      optimizer_type='sgd')
    
            xor_autodiff.add_node(input_autodiff)
            xor_autodiff.add_node(hidden_autodiff)
            xor_autodiff.add_node(output_autodiff)
    
            xor_autodiff.add_projection(sender=input_autodiff, projection=in_to_hidden_autodiff, receiver=hidden_autodiff)
            xor_autodiff.add_projection(sender=hidden_autodiff, projection=hidden_to_out_autodiff, receiver=output_autodiff)
            xor_autodiff.infer_backpropagation_learning_pathways()
    
            inputs_dict = {"inputs": {input_autodiff:xor_inputs},
                           "targets": {output_autodiff:xor_targets},
                           "epochs": num_epochs}
        # RUN MODELS -----------------------------------------------------------------------------------
    
        if pnl.SYSTEM in models:
            results_sys = xor_sys.run(inputs={input_sys:xor_inputs},
                                      targets={output_sys:xor_targets},
                                      num_trials=(num_epochs * xor_inputs.shape[0]),
                                      )
        if pnl.COMPOSITION in models:
            result = xor_comp.learn(inputs={input_comp:xor_inputs,
                                          target_mech:xor_targets},
                                  num_trials=(num_epochs * xor_inputs.shape[0]),
                                  )
        if 'AUTODIFF' in models:
            result = xor_autodiff.learn(inputs=inputs_dict)
            autodiff_weights = xor_autodiff.get_parameters()
    
        # COMPARE WEIGHTS FOR PAIRS OF MODELS ----------------------------------------------------------
    
        if all(m in models for m in {pnl.SYSTEM, 'AUTODIFF'}):
            assert np.allclose(autodiff_weights[in_to_hidden_autodiff], in_to_hidden_sys.get_mod_matrix(xor_sys))
            assert np.allclose(autodiff_weights[hidden_to_out_autodiff], hidden_to_out_sys.get_mod_matrix(xor_sys))
    
        if all(m in models for m in {pnl.SYSTEM, pnl.COMPOSITION}):
            assert np.allclose(in_to_hidden_comp.get_mod_matrix(xor_comp), in_to_hidden_sys.get_mod_matrix(xor_sys))
            assert np.allclose(hidden_to_out_comp.get_mod_matrix(xor_comp), hidden_to_out_sys.get_mod_matrix(xor_sys))
    
        if all(m in models for m in {pnl.COMPOSITION, 'AUTODIFF'}):
            assert np.allclose(autodiff_weights[in_to_hidden_autodiff], in_to_hidden_comp.get_mod_matrix(xor_comp))
            assert np.allclose(autodiff_weights[hidden_to_out_autodiff], hidden_to_out_comp.get_mod_matrix(xor_comp))

    @pytest.mark.parametrize('configuration', [
        'Y UP',
        'BRANCH UP',
        'EXTEND UP',
        'EXTEND DOWN BRANCH UP',
        'CROSS',
        'Y UP AND DOWN',
        'BRANCH DOWN',
        'EXTEND DOWN',
        'BOW',
        'COMPLEX'
        'JOIN BY TERMINAL'
    ])
    def test_backprop_with_various_intersecting_pathway_configurations(self, configuration, show_graph=False):
        '''Test add_backpropgation using various configuration of intersecting pathways

        References in description are to attachment point of added pathway (always A)
        Branches created/added left to right

        '''

        if 'Y UP' == configuration:
            # 1) First mech is already an origin (Y UP)
            #
            #    E            C
            #     \         /
            #      D       B
            #       \     /
            #        A + A
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[A,D,E])
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'BRANCH UP' == configuration:
            # 2) First mech is intermediate (BRANCH UP)
            #
            #            C
            #             \
            #         E   B
            #       /      \
            #      B   +    A
            #     /
            #    D
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,B,E])
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'EXTEND UP' == configuration:
            # 3) First mech is already a terminal (EXTEND UP)
            #
            #                  C
            #                /
            #               B
            #              /
            #         A + A
            #       /
            #      E
            #     /
            #    D
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,E,A])
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'EXTEND DOWN BRANCH UP' == configuration:
            # 4) Intermediate mech is already an origin (EXTEND DOWN BRANCH UP)
            #
            #    D       C
            #     \     /
            #      A + A
            #         /
            #        B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[A,D])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'CROSS' == configuration:
            # 5) Intermediate mech is already an intermediate (CROSS)
            #
            #    E       C
            #     \     /
            #      A + A
            #     /     \
            #    D       B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,A,E])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'Y UP AND DOWN' == configuration:
            # 6) Intermediate mech is already a terminal (Y UP AND DOWN)
            #
            #          C
            #          \
            #      A + A
            #     /     \
            #    D      B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,A])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'BRANCH DOWN' == configuration:
            # 7) Last mech is already an intermediate (BRANCH DOWN)
            #
            #    D
            #     \
            #      A + A
            #     /     \
            #    C       B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[C,A,D])
            comp.add_backpropagation_learning_pathway(pathway=[B,A])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'EXTEND DOWN' == configuration:
            # 8) Last mech is already a terminal (EXTEND DOWN)
            #
            #        A + A
            #       /     \
            #      E       B
            #     /         \
            #    D           C
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,E,A])
            comp.add_backpropagation_learning_pathway(pathway=[C,B,A])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'BOW' == configuration:
            # 9) Bow
            #
            #            F
            #           /
            #      C + C
            #     /     \
            #    B       D
            #     \     /
            #      A + A
            #     /
            #    E
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            F = pnl.ProcessingMechanism(name='F')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[E,A,B,C])
            comp.add_backpropagation_learning_pathway(pathway=[A,D,C,F])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'COMPLEX' == configuration:
            # 10) Complex
            #
            #          C        I
            #          \         \
            #      A + A      F   G
            #     /     \    /     \
            #    D      B + B   +  D
            #              /        \
            #             E         H
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            F = pnl.ProcessingMechanism(name='F')
            G = pnl.ProcessingMechanism(name='G')
            H = pnl.ProcessingMechanism(name='H')
            I = pnl.ProcessingMechanism(name='I')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,A])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            comp.add_backpropagation_learning_pathway(pathway=[E,B,F])
            comp.add_backpropagation_learning_pathway(pathway=[H,D,G,I])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')

        if 'JOIN BY TERMINAL' == configuration:
            # 8) Last mech is already a terminal (EXTEND DOWN)
            #
            #        A     F   A
            #       /     /     \
            #      E  +  B   +   B
            #     /       \
            #    D         C
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            F = pnl.ProcessingMechanism(name='F')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,E,A])
            comp.add_backpropagation_learning_pathway(pathway=[C,B,F])
            comp.add_backpropagation_learning_pathway(pathway=[B,A])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            print(f'Completed configuration: {configuration}')


    @pytest.mark.parametrize('order', [
        'color_full',
        'word_partial',
        'word_full',
        'full_overlap'
    ])
    def test_stroop_model_learning(self, order):
        """Test backpropagation learning for simple convergent/overlapping pathways"""

        # CONSTRUCT MODEL ---------------------------------------------------------------------------

        num_trials = 2

        color_to_hidden_wts = np.arange(4).reshape((2, 2))
        word_to_hidden_wts = np.arange(4).reshape((2, 2))
        hidden_to_response_wts = np.arange(4).reshape((2, 2))

        color_comp = pnl.TransferMechanism(size=2, name='Color')
        word_comp = pnl.TransferMechanism(size=2, name='Word')
        hidden_comp = pnl.TransferMechanism(size=2, function=pnl.Logistic(), name='Hidden')
        response_comp = pnl.TransferMechanism(size=2, function=pnl.Logistic(), name='Response')

        if order == 'color_full':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp,
                             hidden_to_response_wts.copy(),
                             response_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp]
        elif order == 'word_full':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp,
                            hidden_to_response_wts.copy(),
                            response_comp]
        elif order == 'word_partial':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp,
                             hidden_to_response_wts.copy(),
                             response_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp,
                            # FIX: CROSSED_PATHWAYS 7/28/19 [JDC]: THE FOLLOWING LINES CRASHES:
                            # response_comp
                            ]
        elif order == 'full_overlap':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp,
                             hidden_to_response_wts.copy(),
                             response_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp,
                            hidden_to_response_wts.copy(),
                            response_comp
                            ]
        else:
            assert False, 'Bad order specified for test_stroop_model_learning'

        comp = pnl.Composition(name='Stroop Model - Composition')
        comp.add_backpropagation_learning_pathway(pathway=color_pathway,
                                          learning_rate=1)
        comp.add_backpropagation_learning_pathway(pathway=word_pathway,
                                          learning_rate=1)
        # comp.show_graph(show_learning=True)

        # RUN MODEL ---------------------------------------------------------------------------

        # print('\nEXECUTING COMPOSITION-----------------------\n')
        target = comp.get_nodes_by_role(pnl.NodeRole.TARGET)[0]
        results_comp = comp.learn(inputs={color_comp: [[1, 1]],
                                          word_comp: [[-2, -2]],
                                          target: [[1, 1]]},
                                  num_trials=num_trials)
        # print('\nCOMPOSITION RESULTS')
        # print(f'Results: {comp.results}')
        # print(f'color_to_hidden_comp: {comp.projections[0].get_mod_matrix(comp)}')
        # print(f'word_to_hidden_comp: {comp.projections[15].get_mod_matrix(comp)}')

        # VALIDATE RESULTS ---------------------------------------------------------------------------
        # Note:  numbers based on test of System in tests/learning/test_stroop

        composition_and_expected_outputs = [
            (color_comp.output_ports[0].parameters.value.get(comp), np.array([1., 1.])),
            (word_comp.output_ports[0].parameters.value.get(comp), np.array([-2., -2.])),
            (hidden_comp.output_ports[0].parameters.value.get(comp), np.array([0.13227553, 0.01990677])),
            (response_comp.output_ports[0].parameters.value.get(comp), np.array([0.51044657, 0.5483048])),
            (comp.nodes['Comparator'].output_ports[0].parameters.value.get(comp), np.array([0.48955343, 0.4516952])),
            (comp.nodes['Comparator'].output_ports[pnl.MSE].parameters.value.get(comp), np.array(
                    0.22184555903789838)),
            (comp.projections['MappingProjection from Color[RESULT] to Hidden[InputPort-0]'].get_mod_matrix(comp),
             np.array([
                 [ 0.02512045, 1.02167245],
                 [ 2.02512045, 3.02167245],
             ])),
            (comp.projections['MappingProjection from Word[RESULT] to Hidden[InputPort-0]'].get_mod_matrix(comp),
             np.array([
                 [-0.05024091, 0.9566551 ],
                 [ 1.94975909, 2.9566551 ],
             ])),
            (comp.projections['MappingProjection from Hidden[RESULT] to Response[InputPort-0]'].get_mod_matrix(comp),
             np.array([
                 [ 0.03080958, 1.02830959],
                 [ 2.00464242, 3.00426575],
             ])),
            ([results_comp[-1][0]], [np.array([0.51044657, 0.5483048])]),
        ]

        for i in range(len(composition_and_expected_outputs)):
            val, expected = composition_and_expected_outputs[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    def test_pytorch_equivalence_with_learning_enabled_composition(self):
        iSs = np.array(
            [np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996])
             ],
        )

        cSs = np.array(
            [np.array(
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])]
        )

        oSs = np.array(
            [np.array([0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([1., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 1., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., -0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])]
        )

        nf = 3
        nd = 5
        nh = 200

        D_i = nf * nd
        D_c = nd ** 2
        D_h = nh
        D_o = nf * nd

        wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
        wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
        wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
        who = np.random.rand(D_h, D_o) * 0.02 - 0.01

        il = pnl.TransferMechanism(size=D_i, name='input')
        cl = pnl.TransferMechanism(size=D_c, name='control')
        hl = pnl.TransferMechanism(size=D_h, name='hidden',
                                   function=pnl.Logistic(bias=-2))
        ol = pnl.TransferMechanism(size=D_o, name='output',
                                   function=pnl.Logistic(bias=-2))
        pih = pnl.MappingProjection(matrix=wih)
        pch = pnl.MappingProjection(matrix=wch)
        pco = pnl.MappingProjection(matrix=wco)
        pho = pnl.MappingProjection(matrix=who)

        mnet = pnl.Composition()

        target_mech = mnet.add_backpropagation_learning_pathway(
            [il, pih, hl, pho, ol],
            learning_rate=100
        )[pnl.TARGET_MECHANISM]

        mnet.add_backpropagation_learning_pathway(
            [cl, pch, hl, pho, ol],
            learning_rate=100
        )

        mnet.add_backpropagation_learning_pathway(
            [cl, pco, ol],
            learning_rate=100
        )

        mnet._analyze_graph()

        inputs = {
            il: iSs,
            cl: cSs,
            target_mech: oSs
        }

        mnet.learn(inputs=inputs)
        mnet.run(inputs=inputs)
        
        comparator = np.array([0.02288846, 0.11646781, 0.03473711, 0.0348004, 0.01679579,
                             0.04851733, 0.05857743, 0.04819957, 0.03004438, 0.05113508,
                             0.06849843, 0.0442623, 0.00967315, 0.06998125, 0.03482444,
                             0.05856816, 0.00724313, 0.03676571, 0.03668758, 0.01761947,
                             0.0516829, 0.06260267, 0.05160782, 0.03140498, 0.05462971,
                             0.07360401, 0.04687923, 0.00993319, 0.07662302, 0.03687142,
                             0.0056837, 0.03411045, 0.03615285, 0.03606166, 0.01774354,
                             0.04700402, 0.09696857, 0.06843472, 0.06108671, 0.0485631,
                             0.07194324, 0.04485926, 0.00526768, 0.07442083, 0.0364541,
                             0.02819926, 0.03804169, 0.04091214, 0.04091113, 0.04246229,
                             0.05583883, 0.06643675, 0.05630667, 0.01540373, 0.05948422,
                             0.07721549, 0.05081813, 0.01205326, 0.07998289, 0.04084186,
                             0.02859247, 0.03794089, 0.04111452, 0.04139213, 0.01222424,
                             0.05677404, 0.06736114, 0.05614553, 0.03573626, 0.05983103,
                             0.07867571, 0.09971621, 0.01203033, 0.08107789, 0.04110497,
                             0.02694072, 0.03592752, 0.03878366, 0.03895513, 0.01852774,
                             0.05097689, 0.05753834, 0.05090328, 0.03405996, 0.05293719,
                             0.07037981, 0.03474316, 0.02861534, 0.12504038, 0.0387827,
                             0.02467716, 0.03373265, 0.03676382, 0.03677551, 0.00758558,
                             0.089832, 0.06330426, 0.0514472, 0.03120581, 0.05535174,
                             0.07494839, 0.04169744, 0.00698747, 0.0771042, 0.03659954,
                             0.03008443, 0.0393799, 0.0423592, 0.04237004, 0.00965198,
                             0.09863199, 0.06813933, 0.05675321, 0.03668943, 0.0606036,
                             0.07898065, 0.04662618, 0.00954765, 0.08093391, 0.04218842,
                             0.02701085, 0.03660227, 0.04058368, 0.04012464, 0.02030738,
                             0.047633, 0.06693405, 0.055821, 0.03456592, 0.10166267,
                             0.07870758, 0.04935871, 0.01065449, 0.08012213, 0.04036544,
                             0.02576563, 0.03553382, 0.03920509, 0.03914452, 0.01907667,
                             0.05106766, 0.06555857, 0.05434728, 0.03335726, 0.05074808,
                             0.07715102, 0.04839309, 0.02494798, 0.08001304, 0.03921895,
                             0.00686952, 0.03941704, 0.04128484, 0.04117602, 0.02217508,
                             0.05152296, 0.10361618, 0.07488737, 0.0707186, 0.05289282,
                             0.07557573, 0.04978292, 0.00705783, 0.07787788, 0.04164007,
                             0.00574239, 0.03437231, 0.03641445, 0.03631848, 0.01795791,
                             0.04723996, 0.09732232, 0.06876138, 0.06156679, 0.04878423,
                             0.07214104, 0.04511085, 0.00535038, 0.07459818, 0.0367153,
                             0.02415251, 0.03298647, 0.03586635, 0.0360273, 0.01624523,
                             0.04829838, 0.05523439, 0.04821285, 0.03115052, 0.05034625,
                             0.06836408, 0.03264844, 0.0241706, 0.12190507, 0.03585727,
                             0.02897192, 0.03925683, 0.04250414, 0.04253885, 0.02175426,
                             0.05683923, 0.06547528, 0.05705267, 0.03742978, 0.05951711,
                             0.12675475, 0.05216411, 0.00181494, 0.08218002, 0.04234364,
                             0.02789848, 0.036924, 0.03976586, 0.03993866, 0.01932489,
                             0.05186586, 0.05829845, 0.05179337, 0.03504668, 0.05379566,
                             0.07103772, 0.03544133, 0.03019486, 0.12605846, 0.03976812])

        assert np.allclose(comparator, np.array(mnet.parameters.results.get(mnet)[-15:]).reshape(225))

import pytest
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions
from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.compositions.composition import Composition, NodeRole

def validate_learning_mechs(comp):

    def get_learning_mech(name):
        return next(lm for lm in comp.get_nodes_by_role(NodeRole.LEARNING) if lm.name == name)

    REP_IN_to_REP_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REP_IN to REP_HIDDEN')
    REP_HIDDEN_to_REL_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REP_HIDDEN to REL_HIDDEN')
    REL_IN_to_REL_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_IN to REL_HIDDEN')
    REL_HIDDEN_to_REP_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to REP_OUT')
    REL_HIDDEN_to_PROP_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to PROP_OUT')
    REL_HIDDEN_to_QUAL_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to QUAL_OUT')
    REL_HIDDEN_to_ACT_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to ACT_OUT')

    # Validate error_signal Projections for REP_IN to REP_HIDDEN
    assert len(REP_IN_to_REP_HIDDEN_LM.input_ports) == 3
    assert REP_IN_to_REP_HIDDEN_LM.input_ports[pnl.ERROR_SIGNAL].path_afferents[0].sender.owner == \
           REP_HIDDEN_to_REL_HIDDEN_LM

    # Validate error_signal Projections to LearningMechanisms for REP_HIDDEN_to REL_HIDDEN Projections
    assert all(lm in [input_port.path_afferents[0].sender.owner for input_port in
                      REP_HIDDEN_to_REL_HIDDEN_LM.input_ports]
               for lm in {REL_HIDDEN_to_REP_OUT_LM, REL_HIDDEN_to_PROP_OUT_LM,
                          REL_HIDDEN_to_QUAL_OUT_LM, REL_HIDDEN_to_ACT_OUT_LM})

    # Validate error_signal Projections to LearningMechanisms for REL_IN to REL_HIDDEN Projections
    assert all(lm in [input_port.path_afferents[0].sender.owner for input_port in
                      REL_IN_to_REL_HIDDEN_LM.input_ports]
               for lm in {REL_HIDDEN_to_REP_OUT_LM, REL_HIDDEN_to_PROP_OUT_LM,
                          REL_HIDDEN_to_QUAL_OUT_LM, REL_HIDDEN_to_ACT_OUT_LM})


class TestRumelhartSemanticNetwork:
    r"""
    Tests construction and training of network with both convergent and divergent pathways
    with the following structure:

    # Semantic Network:
    #                        __
    #    REP PROP QUAL ACT     |
    #      \   \  /   /   __   | Output Processes
    #       REL_HIDDEN      |__|
    #          /   \        |
    #  REP_HIDDEN  REL_IN   |  Input Processes
    #       /               |
    #   REP_IN           ___|
    """

    def test_rumelhart_semantic_network_sequential(self):

        rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
        rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
        rep_hidden = pnl.TransferMechanism(size=4,
                                           function=psyneulink.core.components.functions.transferfunctions.Logistic,
                                           name='REP_HIDDEN')
        rel_hidden = pnl.TransferMechanism(size=5, function=Logistic, name='REL_HIDDEN')
        rep_out = pnl.TransferMechanism(size=10, function=Logistic, name='REP_OUT')
        prop_out = pnl.TransferMechanism(size=12, function=Logistic, name='PROP_OUT')
        qual_out = pnl.TransferMechanism(size=13, function=Logistic, name='QUAL_OUT')
        act_out = pnl.TransferMechanism(size=14, function=Logistic, name='ACT_OUT')

        comp = Composition()

        # comp.add_backpropagation_learning_pathway(pathway=[rep_in, rep_hidden, rel_hidden])
        comp.add_backpropagation_learning_pathway(pathway=[rel_in, rel_hidden])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, rep_out])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, prop_out])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, qual_out])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, act_out])
        comp.add_backpropagation_learning_pathway(pathway=[rep_in, rep_hidden, rel_hidden])

        # comp.show_graph(show_learning=True)
        # validate_learning_mechs(comp)

        comp.learn(
              num_trials=2,
              inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
              # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
              )
        print(comp.results)

    # def test_rumelhart_semantic_network_convergent(self):
    #
    #     rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
    #     rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
    #     rep_hidden = pnl.TransferMechanism(size=4, function=Logistic, name='REP_HIDDEN')
    #     rel_hidden = pnl.TransferMechanism(size=5, function=Logistic, name='REL_HIDDEN')
    #     rep_out = pnl.TransferMechanism(size=10, function=Logistic, name='REP_OUT')
    #     prop_out = pnl.TransferMechanism(size=12, function=Logistic, name='PROP_OUT')
    #     qual_out = pnl.TransferMechanism(size=13, function=Logistic, name='QUAL_OUT')
    #     act_out = pnl.TransferMechanism(size=14, function=Logistic, name='ACT_OUT')
    #
    #     rep_proc = pnl.Process(pathway=[rep_in, rep_hidden, rel_hidden, rep_out],
    #                            learning=pnl.LEARNING,
    #                            name='REP_PROC')
    #     rel_proc = pnl.Process(pathway=[rel_in, rel_hidden],
    #                            learning=pnl.LEARNING,
    #                            name='REL_PROC')
    #     rel_prop_proc = pnl.Process(pathway=[rel_hidden, prop_out],
    #                                 learning=pnl.LEARNING,
    #                                 name='REL_PROP_PROC')
    #     rel_qual_proc = pnl.Process(pathway=[rel_hidden, qual_out],
    #                                 learning=pnl.LEARNING,
    #                                 name='REL_QUAL_PROC')
    #     rel_act_proc = pnl.Process(pathway=[rel_hidden, act_out],
    #                                learning=pnl.LEARNING,
    #                                name='REL_ACT_PROC')
    #     S = pnl.System(processes=[rep_proc,
    #                               rel_proc,
    #                               rel_prop_proc,
    #                               rel_qual_proc,
    #                               rel_act_proc])
    #     # S.show_graph(show_learning=pnl.ALL, show_dimensions=True)
    #     validate_learning_mechs(S)
    #     print(S.origin_mechanisms)
    #     print(S.terminal_mechanisms)
    #     S.run(inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                   rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    #           # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    #           #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    #           #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    #           #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
    #           )
    #
    # def test_rumelhart_semantic_network_crossing(self):
    #
    #     rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
    #     rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
    #     rep_hidden = pnl.TransferMechanism(size=4, function=Logistic, name='REP_HIDDEN')
    #     rel_hidden = pnl.TransferMechanism(size=5, function=Logistic, name='REL_HIDDEN')
    #     rep_out = pnl.TransferMechanism(size=10, function=Logistic, name='REP_OUT')
    #     prop_out = pnl.TransferMechanism(size=12, function=Logistic, name='PROP_OUT')
    #     qual_out = pnl.TransferMechanism(size=13, function=Logistic, name='QUAL_OUT')
    #     act_out = pnl.TransferMechanism(size=14, function=Logistic, name='ACT_OUT')
    #
    #     rep_proc = pnl.Process(pathway=[rep_in, rep_hidden, rel_hidden, rep_out],
    #                            learning=pnl.LEARNING,
    #                            name='REP_PROC')
    #     rel_proc = pnl.Process(pathway=[rel_in, rel_hidden, prop_out],
    #                            learning=pnl.LEARNING,
    #                            name='REL_PROC')
    #     rel_qual_proc = pnl.Process(pathway=[rel_hidden, qual_out],
    #                                 learning=pnl.LEARNING,
    #                                 name='REL_QUAL_PROC')
    #     rel_act_proc = pnl.Process(pathway=[rel_hidden, act_out],
    #                                learning=pnl.LEARNING,
    #                                name='REL_ACT_PROC')
    #     S = pnl.System(processes=[rep_proc,
    #                               rel_proc,
    #                               rel_qual_proc,
    #                               rel_act_proc])
    #
    #     # S.show_graph(show_learning=pnl.ALL, show_dimensions=True)
    #     validate_learning_mechs(S)
    #     print(S.origin_mechanisms)
    #     print(S.terminal_mechanisms)
    #     S.run(inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                   rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    #           # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    #           #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    #           #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    #           #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
    #           )

class TestLearningPathwayMethods:
    def test_multiple_of_same_learning_pathway(self):
        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)

        input_comp = pnl.TransferMechanism(name='input_comp',
                                       default_variable=np.zeros(2))
    
        hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                    default_variable=np.zeros(10),
                                    function=pnl.Logistic())

        output_comp = pnl.TransferMechanism(name='output_comp',
                                    default_variable=np.zeros(1),
                                    function=pnl.Logistic())

        in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                    matrix=in_to_hidden_matrix.copy(),
                                    sender=input_comp,
                                    receiver=hidden_comp)

        hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                    matrix=hidden_to_out_matrix.copy(),
                                    sender=hidden_comp,
                                    receiver=output_comp)

        xor_comp = pnl.Composition()

        learning_components = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                        in_to_hidden_comp,
                                                                        hidden_comp,
                                                                        hidden_to_out_comp,
                                                                        output_comp],
                                                                    learning_rate=10)
        # Try readd the same learning pathway (shouldn't error)
        learning_components = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                        in_to_hidden_comp,
                                                                        hidden_comp,
                                                                        hidden_to_out_comp,
                                                                        output_comp],
                                                                    learning_rate=10)
    def test_run_no_targets(self):
        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)

        input_comp = pnl.TransferMechanism(name='input_comp',
                                       default_variable=np.zeros(2))
    
        hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                    default_variable=np.zeros(10),
                                    function=pnl.Logistic())

        output_comp = pnl.TransferMechanism(name='output_comp',
                                    default_variable=np.zeros(1),
                                    function=pnl.Logistic())

        in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                    matrix=in_to_hidden_matrix.copy(),
                                    sender=input_comp,
                                    receiver=hidden_comp)

        hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                    matrix=hidden_to_out_matrix.copy(),
                                    sender=hidden_comp,
                                    receiver=output_comp)

        xor_comp = pnl.Composition()

        learning_components = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                        in_to_hidden_comp,
                                                                        hidden_comp,
                                                                        hidden_to_out_comp,
                                                                        output_comp],
                                                                    learning_rate=10)
        # Try to run without any targets (non-learning
        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
        xor_comp.run(inputs={input_comp:xor_inputs})
    
