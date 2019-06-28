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
        for trial in range(50):
            target = [0.]*60
            target[reward_delivery] = 1.
            # {14, 29, 44, 59, 74, 89}
            if trial in {14, 29, 44}:
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

    def test_rl_learning_enabled_false(self):
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

            # Pause learning -- values are the same as the previous trial (because we pass in the same inputs)
            comp.learning_enabled = False
            inputs_dict = {input_layer: [[1., 1.], [1., 1.]]}
            comp.run(inputs=inputs_dict)
            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336], \
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

            # Resume learning
            comp.learning_enabled = True
            inputs_dict = {input_layer: [[1., 1.], [1., 1.]],
                           target_mechanism: [[10.], [10.]]}
            comp.run(inputs=inputs_dict)
            assert np.allclose(learning_mechanism.value, [np.array([0.38581875, 0.]), np.array([0.38581875, 0.])])
            assert np.allclose(action_selection.value, [[1.], [0.978989672], [0.99996], [0.0000346908466], [0.978989672], \
                                                        [0.118109771], [1.32123733], [0.978989672], [0.118109771],
                                                        [1.32123733]])

    def test_td_learning_enabled_false(self):

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
            target = [0.]*60
            target[reward_delivery] = 1.
            # {14, 29, 44, 59, 74, 89}
            if trial in {14, 29, 44}:
                target[reward_delivery] = 0.
            targets.append(target)

            sample = [0.]*60
            for i in range(stimulus_onset, 60):
                sample[i] =1.
            samples.append(sample)

        inputs1 = {sample_mechanism: samples[0:30],
                  target_mechanism: targets[0:30]}

        inputs2 = {sample_mechanism: samples[30:50],
                   target_mechanism: targets[30:50]}

        comp.run(inputs=inputs1)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        trial_1_expected = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.003,  0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., -0.003,  0.]

        trial_30_expected = [0.]*40
        trial_30_expected +=[.0682143186, .0640966042, .0994344173, .133236921, .152270799, .145592903, .113949692,
                             .0734420009, .0450652924, .0357386468, .0330810871, .0238007805, .0102892090, -.998098988,
                             -.0000773996815, -.0000277845011, -.00000720338916, -.00000120056486, -.0000000965971727, 0.]


        assert np.allclose(trial_1_expected, delta_vals[0][0])
        assert np.allclose(trial_30_expected, delta_vals[29][0])

        # Pause Learning
        comp.learning_enabled = False
        comp.run(inputs={sample_mechanism: samples[0:3]})

        # Resume Learning
        comp.learning_enabled = True
        comp.run(inputs=inputs2)
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

        perceptual_state = pnl.ProcessingMechanism(name='Current State',
                                                   function=Concatenate,
                                                   input_states=[{pnl.NAME: 'STIM',
                                                                  pnl.SIZE: stim_size,
                                                                  pnl.PROJECTIONS: stim_in},
                                                                 {pnl.NAME: 'CONTEXT',
                                                                  pnl.SIZE: context_size,
                                                                  pnl.PROJECTIONS: context_in}])

        action = pnl.ProcessingMechanism(name='Action',
                                         size=num_actions)

        # Nested Composition
        rl_agent_state = pnl.ProcessingMechanism(name='RL Agent State',
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

        model.show_graph(show_controller=True, show_nested=True, show_node_structure=True)

        stimuli = {stim_in: np.array([1] * stim_size),
                   context_in: np.array([10] * context_size)}
        #
        # print(model.run(inputs=stimuli))


class TestBackProp:

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
        learning_components = comp.add_back_propagation_pathway(pathway=[input_layer, hidden_layer, output_layer],
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

        comp.run(inputs={input_layer: [[1.0, 1.0]],
                         target_mechanism: [[1.0, 1.0]]},
                 num_trials=5,
                 execution_id=eid)

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

    def test_stroop_with_backprop(self):
        # Mechanisms
        color = pnl.TransferMechanism(size=2,
                                      function=pnl.Linear(),
                                      name='color')
        word = pnl.TransferMechanism(size=2,
                                     function=pnl.Linear(),
                                     name='word')
        hidden = pnl.TransferMechanism(size=2,
                                       function=pnl.Logistic(),
                                       name='hidden')
        response = pnl.TransferMechanism(size=2,
                                         function=pnl.Logistic(),
                                         name='response')
        # Weights
        ch_weights = np.arange(4).reshape((2, 2))
        wh_weights = np.arange(4).reshape((2, 2))
        ho_weights = np.arange(4).reshape((2, 2))

        # Assemble Composition
        comp = pnl.Composition(name='stroop-with-learning')
        comp.add_back_propagation_pathway(pathway=[color, ch_weights, hidden, ho_weights, response])
        learning_components = comp.add_back_propagation_pathway(pathway=[word, wh_weights, hidden, ho_weights, response
                                                   ])
        target = learning_components[pnl.TARGET_MECHANISM]
        # comp.add_back_propagation_pathway(pathway=[hidden, ho_weights, response])
        comp.show_graph()
        comp.run(inputs={color: [[1, 1]],
                         word: [[-2, -2]],
                         target: [[1, 1]]},
                 num_trials=2)
        print(response.value)

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

        input_weights = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        middle_weights = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        output_weights = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

        comp = pnl.Composition(name='multilayer')

        p = [input_layer, input_weights, hidden_layer_1, middle_weights, hidden_layer_2, output_weights, output_layer]
        learning_components = comp.add_back_propagation_pathway(pathway=p,
                                                                learning_rate=1.)

        target_node = learning_components[pnl.TARGET]

        input_dictionary = {target_node: [[0., 0., 1.]],
                            input_layer: [[-1., 30.]]}

        comp.show_graph()

        comp.run(inputs=input_dictionary,
                 num_trials=10)


