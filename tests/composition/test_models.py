import psyneulink as pnl
import numpy as np

class TestStroopModels:

    # This implements the model by Cohen, J. D., & Huston, T. A. (1994). Progress in the use of interactive
    # models for understanding attention and performance. In C. Umilta & M. Moscovitch(Eds.),
    # AttentionandperformanceXV(pp.453-456). Cam- bridge, MA: MIT Press.
    # The model aims to capute top-down effects of selective attention and the bottom-up effects of attentional capture.

    def test_cohen_huston_1994(self):

        # Define Variables -------------------------------------------------------------------------------------
        rate = 0.1  # modified from the original code from 0.01 to 0.1
        inhibition = -2.0  # lateral inhibition
        bias = 4.0  # bias is positive since Logistic equation has - sing already implemented
        threshold = 0.55  # modified from original code from 0.6 to 0.55 because incongruent condition won't reach 0.6
        settle_trials = 50  # cycles until model settles

        # Create mechanisms ------------------------------------------------------------------------------------

        #   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')

        colors_input_layer = pnl.TransferMechanism(
            size=3,
            function=pnl.Linear,
            name='COLORS_INPUT'
        )

        words_input_layer = pnl.TransferMechanism(
            size=3,
            function=pnl.Linear,
            name='WORDS_INPUT'
        )

        task_input_layer = pnl.TransferMechanism(
            size=2,
            function=pnl.Linear,
            name='TASK_INPUT'
        )

        #   Task layer, tasks: ('name the color', 'read the word')
        task_layer = pnl.RecurrentTransferMechanism(
            size=2,
            function=pnl.Logistic(),
            hetero=-2,
            integrator_mode=True,
            integration_rate=0.1,
            name='TASK'
        )

        #   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
        colors_hidden_layer = pnl.RecurrentTransferMechanism(
            size=3,
            function=pnl.Logistic(bias=4.0),
            integrator_mode=True,
            hetero=-2.0,
            # noise=pnl.NormalDist(mean=0.0, standard_dev=.0).function,
            integration_rate=0.1,  # cohen-huston text says 0.01
            name='COLORS HIDDEN'
        )

        words_hidden_layer = pnl.RecurrentTransferMechanism(
            size=3,
            function=pnl.Logistic(bias=4.0),
            hetero=-2,
            integrator_mode=True,
            # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
            integration_rate=0.1,
            name='WORDS HIDDEN'
        )
        #   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
        response_layer = pnl.RecurrentTransferMechanism(
            size=2,
            function=pnl.Logistic(),
            hetero=-2.0,
            integrator_mode=True,
            integration_rate=0.1,
            name='RESPONSE'
        )
        # Connect mechanisms --------------------------------------------------------------------------------------------------
        # (note that response layer projections are set to all zero first for initialization

        color_input_weights = pnl.MappingProjection(
            matrix=np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        )

        word_input_weights = pnl.MappingProjection(
            matrix=np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        )

        task_input_weights = pnl.MappingProjection(
            matrix=np.array([
                [1.0, 0.0],
                [0.0, 1.0]
            ])
        )

        color_task_weights = pnl.MappingProjection(
            matrix=np.array([
                [4.0, 0.0],
                [4.0, 0.0],
                [4.0, 0.0]
            ])
        )

        task_color_weights = pnl.MappingProjection(
            matrix=np.array([
                [4.0, 4.0, 4.0],
                [0.0, 0.0, 0.0]
            ])
        )

        word_task_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 4.0],
                [0.0, 4.0],
                [0.0, 4.0]
            ])
        )

        task_word_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0]
            ])
        )

        response_color_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        )

        response_word_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        )

        color_response_weights = pnl.MappingProjection(
            matrix=np.array([
                [1.5, 0.0],
                [0.0, 1.5],
                [0.0, 0.0]
            ])
        )
        word_response_weights = pnl.MappingProjection(
            matrix=np.array([
                [2.5, 0.0],
                [0.0, 2.5],
                [0.0, 0.0]
            ])
        )
        bidirectional_stroop = pnl.Composition(name="bidirectional_stroop")

        color_response_pathway = [colors_input_layer,
                                  color_input_weights,
                                  colors_hidden_layer,
                                  color_response_weights,
                                  response_layer]
        bidirectional_stroop.add_linear_processing_pathway(pathway=color_response_pathway)

        color_response_pathway_2 = [response_layer,
                                    response_color_weights,
                                    colors_hidden_layer]
        bidirectional_stroop.add_linear_processing_pathway(pathway=color_response_pathway_2)


        word_response_pathway = [words_input_layer,
                                 word_input_weights,
                                 words_hidden_layer,
                                 word_response_weights,
                                 response_layer]
        bidirectional_stroop.add_linear_processing_pathway(pathway=word_response_pathway)

        word_response_pathway_2 = [response_layer,
                                   response_word_weights,
                                   words_hidden_layer]
        bidirectional_stroop.add_linear_processing_pathway(pathway=word_response_pathway_2)


        task_color_response_pathway = [task_input_layer,
                                       task_input_weights,
                                       task_layer,
                                       task_color_weights,
                                       colors_hidden_layer]

        bidirectional_stroop.add_linear_processing_pathway(pathway=task_color_response_pathway)

        task_color_response_pathway_2 = [colors_hidden_layer,
                                         color_task_weights,
                                         task_layer]

        bidirectional_stroop.add_linear_processing_pathway(pathway=task_color_response_pathway_2)

        task_word_response_pathway = [task_input_layer,
                                      task_layer,
                                      task_word_weights,
                                      words_hidden_layer]
        bidirectional_stroop.add_linear_processing_pathway(pathway=task_word_response_pathway)

        task_word_response_pathway_2 = [words_hidden_layer,
                                        word_task_weights,
                                        task_layer]

        bidirectional_stroop.add_linear_processing_pathway(pathway=task_word_response_pathway_2)
        bidirectional_stroop.add_required_c_node_role(response_layer, pnl.CNodeRole.TERMINAL)
        bidirectional_stroop._analyze_graph()

        input_dict = {colors_input_layer: [0, 0, 0],
                      words_input_layer: [0, 0, 0],
                      task_input_layer: [0, 1]}

        bidirectional_stroop.run(inputs=input_dict)
        print(bidirectional_stroop.get_c_nodes_by_role(pnl.CNodeRole.TERMINAL))
        print(bidirectional_stroop.output_values)
        for node in bidirectional_stroop.c_nodes:
            print(node.name, " Value: ", node.output_values)

    def test_DDM(self):
        myMechanism = pnl.DDM(
            function=pnl.BogaczEtAl(
                drift_rate=(1.0),
                threshold=(10.0),
                starting_point=0.0,
            ),
            name='My_DDM',
        )

        myMechanism_2 = pnl.DDM(
            function=pnl.BogaczEtAl(
                drift_rate=2.0,
                threshold=20.0),
            name='My_DDM_2'
        )

        myMechanism_3 = pnl.DDM(
            function=pnl.BogaczEtAl(
                drift_rate=3.0,
                threshold=30.0
            ),
            name='My_DDM_3',
        )

        z = pnl.Composition()
        z.add_linear_processing_pathway([myMechanism,
                                        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX),
                                        myMechanism_2,
                                         pnl.MappingProjection(matrix=pnl.FULL_CONNECTIVITY_MATRIX),
                                        myMechanism_3])

        result = z.run(inputs={myMechanism: [[40]]})[0][0]

        expected_output = [
            (myMechanism.input_states[0].value, np.array([40.])),
            (myMechanism.output_states[0].value, np.array([10.])),
            (myMechanism_2.input_states[0].value, np.array([10.])),
            (myMechanism_2.output_states[0].value, np.array([20.])),
            (myMechanism_3.input_states[0].value, np.array([20.])),
            (myMechanism_3.output_states[0].value, np.array([30.])),
            (result, np.array([30.])),
        ]

        for i in range(len(expected_output)):
            val, expected = expected_output[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    def test_lauras_cohen_1990_model(self):
        #  INPUT UNITS

        #  colors: ('red', 'green'), words: ('RED','GREEN')
        colors_input_layer = pnl.TransferMechanism(size=2,
                                                   function=pnl.Linear,
                                                   name='COLORS_INPUT')

        words_input_layer = pnl.TransferMechanism(size=2,
                                                  function=pnl.Linear,
                                                  name='WORDS_INPUT')

        #   Task layer, tasks: ('name the color', 'read the word')
        task_layer = pnl.TransferMechanism(size=2,
                                           function=pnl.Linear,
                                           name='TASK')

        #   HIDDEN LAYER UNITS

        #   colors_hidden: ('red','green')
        #   Logistic activation function, Gain = 1.0, Bias = -4.0 (in PNL bias is subtracted so enter +4.0 to get negative bias)
        #   randomly distributed noise to the net input
        #   time averaging = integration_rate = 0.1
        unit_noise = 0.005
        colors_hidden_layer = pnl.TransferMechanism(size=2,
                                                    function=pnl.Logistic(gain=1.0, bias=4.0),
                                                    # should be able to get same result with offset = -4.0
                                                    integrator_mode=True,
                                                    noise=pnl.NormalDist(mean=0, standard_dev=unit_noise).function,
                                                    integration_rate=0.1,
                                                    name='COLORS HIDDEN')
        #    words_hidden: ('RED','GREEN')
        words_hidden_layer = pnl.TransferMechanism(size=2,
                                                   function=pnl.Logistic(gain=1.0, bias=4.0),
                                                   integrator_mode=True,
                                                   noise=pnl.NormalDist(mean=0, standard_dev=unit_noise).function,
                                                   integration_rate=0.1,
                                                   name='WORDS HIDDEN')

        #    OUTPUT UNITS

        #   Response layer, provide input to accumulator, responses: ('red', 'green')
        #   time averaging = tau = 0.1
        #   randomly distributed noise to the net input
        response_layer = pnl.TransferMechanism(size=2,
                                               function=pnl.Logistic,
                                               name='RESPONSE',
                                               integrator_mode=True,
                                               noise=pnl.NormalDist(mean=0, standard_dev=unit_noise).function,
                                               integration_rate=0.1)
        #   Respond red accumulator
        #   alpha = rate of evidence accumlation = 0.1
        #   sigma = noise = 0.1
        #   noise will be: squareroot(time_step_size * noise) * a random sample from a normal distribution
        accumulator_noise = 0.1
        respond_red_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist(mean=0,
                                                                                                             standard_dev=accumulator_noise).function,
                                                                                        rate=0.1),
                                                          name='respond_red_accumulator')
        #   Respond green accumulator
        respond_green_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist(mean=0,
                                                                                                               standard_dev=accumulator_noise).function,
                                                                                          rate=0.1),
                                                            name='respond_green_accumulator')

        #   LOGGING
        colors_hidden_layer.set_log_conditions('value')
        words_hidden_layer.set_log_conditions('value')
        response_layer.set_log_conditions('value')
        respond_red_accumulator.set_log_conditions('value')
        respond_green_accumulator.set_log_conditions('value')

        #   SET UP CONNECTIONS

        #   rows correspond to sender
        #   columns correspond to: weighting of the contribution that a given sender makes to the receiver

        #   INPUT TO HIDDEN
        # row 0: input_'red' to hidden_'red', hidden_'green'
        # row 1: input_'green' to hidden_'red', hidden_'green'
        color_weights = pnl.MappingProjection(matrix=np.matrix([[2.2, -2.2],
                                                                [-2.2, 2.2]]),
                                              name='COLOR_WEIGHTS')
        # row 0: input_'RED' to hidden_'RED', hidden_'GREEN'
        # row 1: input_'GREEN' to hidden_'RED', hidden_'GREEN'
        word_weights = pnl.MappingProjection(matrix=np.matrix([[2.6, -2.6],
                                                               [-2.6, 2.6]]),
                                             name='WORD_WEIGHTS')

        #   HIDDEN TO RESPONSE
        # row 0: hidden_'red' to response_'red', response_'green'
        # row 1: hidden_'green' to response_'red', response_'green'
        color_response_weights = pnl.MappingProjection(matrix=np.matrix([[1.3, -1.3],
                                                                         [-1.3, 1.3]]),
                                                       name='COLOR_RESPONSE_WEIGHTS')
        # row 0: hidden_'RED' to response_'red', response_'green'
        # row 1: hidden_'GREEN' to response_'red', response_'green'
        word_response_weights = pnl.MappingProjection(matrix=np.matrix([[2.5, -2.5],
                                                                        [-2.5, 2.5]]),
                                                      name='WORD_RESPONSE_WEIGHTS')

        #   TASK TO HIDDEN LAYER
        #   row 0: task_CN to hidden_'red', hidden_'green'
        #   row 1: task_WR to hidden_'red', hidden_'green'
        task_CN_weights = pnl.MappingProjection(matrix=np.matrix([[4.0, 4.0],
                                                                  [0, 0]]),
                                                name='TASK_CN_WEIGHTS')

        #   row 0: task_CN to hidden_'RED', hidden_'GREEN'
        #   row 1: task_WR to hidden_'RED', hidden_'GREEN'
        task_WR_weights = pnl.MappingProjection(matrix=np.matrix([[0, 0],
                                                                  [4.0, 4.0]]),
                                                name='TASK_WR_WEIGHTS')

        #   RESPONSE UNITS TO ACCUMULATORS
        #   row 0: response_'red' to respond_red_accumulator
        #   row 1: response_'green' to respond_red_accumulator
        respond_red_differencing_weights = pnl.MappingProjection(matrix=np.matrix([[1.0], [-1.0]]),
                                                                 name='RESPOND_RED_WEIGHTS')

        #   row 0: response_'red' to respond_green_accumulator
        #   row 1: response_'green' to respond_green_accumulator
        respond_green_differencing_weights = pnl.MappingProjection(matrix=np.matrix([[-1.0], [1.0]]),
                                                                   name='RESPOND_GREEN_WEIGHTS')

        #   CREATE PATHWAYS
        #   Words pathway
        words_process = pnl.Process(pathway=[words_input_layer,
                                             word_weights,
                                             words_hidden_layer,
                                             word_response_weights,
                                             response_layer], name='WORDS_PROCESS')

        #   Colors pathway
        colors_process = pnl.Process(pathway=[colors_input_layer,
                                              color_weights,
                                              colors_hidden_layer,
                                              color_response_weights,
                                              response_layer], name='COLORS_PROCESS')

        #   Task representation pathway
        task_CN_process = pnl.Process(pathway=[task_layer,
                                               task_CN_weights,
                                               colors_hidden_layer],
                                      name='TASK_CN_PROCESS')
        task_WR_process = pnl.Process(pathway=[task_layer,
                                               task_WR_weights,
                                               words_hidden_layer],
                                      name='TASK_WR_PROCESS')

        #   Evidence accumulation pathway
        respond_red_process = pnl.Process(pathway=[response_layer,
                                                   respond_red_differencing_weights,
                                                   respond_red_accumulator],
                                          name='RESPOND_RED_PROCESS')
        respond_green_process = pnl.Process(pathway=[response_layer,
                                                     respond_green_differencing_weights,
                                                     respond_green_accumulator],
                                            name='RESPOND_GREEN_PROCESS')

        #   CREATE SYSTEM
        my_Stroop = pnl.System(processes=[colors_process,
                                          words_process,
                                          task_CN_process,
                                          task_WR_process,
                                          respond_red_process,
                                          respond_green_process],
                               name='FEEDFORWARD_STROOP_SYSTEM')

        # my_Stroop.show()
        # my_Stroop.show_graph(show_dimensions=pnl.ALL)

        # Function to create test trials
        # a RED word input is [1,0] to words_input_layer and GREEN word is [0,1]
        # a red color input is [1,0] to colors_input_layer and green color is [0,1]
        # a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]

        def trial_dict(red_color, green_color, red_word, green_word, CN, WR):

            trialdict = {
                colors_input_layer: [red_color, green_color],
                words_input_layer: [red_word, green_word],
                task_layer: [CN, WR]
            }
            return trialdict

        #   CREATE THRESHOLD FUNCTION
        # first value of DDM's value is DECISION_VARIABLE
        def pass_threshold(mech1, mech2, thresh):
            results1 = mech1.output_states[0].value
            results2 = mech2.output_states[0].value
            for val in results1:
                if val >= thresh:
                    return True
            for val in results2:
                if val >= thresh:
                    return True
            return False

        accumulator_threshold = 1.0

        mechanisms_to_update = [colors_hidden_layer, words_hidden_layer, response_layer]

        def switch_integrator_mode(mechanisms, mode):
            for mechanism in mechanisms:
                mechanism.integrator_mode = mode

        def switch_noise(mechanisms, noise):
            for mechanism in mechanisms:
                mechanism.noise = noise

        def switch_to_initialization_trial(mechanisms):
            # Turn off accumulation
            switch_integrator_mode(mechanisms, False)
            # Turn off noise
            switch_noise(mechanisms, 0)
            # Execute once per trial
            my_Stroop.termination_processing = {pnl.TimeScale.TRIAL: pnl.AllHaveRun()}

        def switch_to_processing_trial(mechanisms):
            # Turn on accumulation
            switch_integrator_mode(mechanisms, True)
            # Turn on noise
            switch_noise(mechanisms, pnl.NormalDist(mean=0, standard_dev=unit_noise).function)
            # Execute until one of the accumulators crosses the threshold
            my_Stroop.termination_processing = {pnl.TimeScale.TRIAL: pnl.While(pass_threshold,
                                                                               respond_red_accumulator,
                                                                               respond_green_accumulator,
                                                                               accumulator_threshold)}

        def switch_trial_type():
            # Next trial will be a processing trial
            if isinstance(my_Stroop.termination_processing[pnl.TimeScale.TRIAL], pnl.AllHaveRun):
                switch_to_processing_trial(mechanisms_to_update)
            # Next trial will be an initialization trial
            else:
                switch_to_initialization_trial(mechanisms_to_update)

        CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 1, 0)

        WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 1)

        # Start with an initialization trial
        switch_to_initialization_trial(mechanisms_to_update)

        my_Stroop.run(inputs=trial_dict(0, 1, 1, 0, 1, 0),
                      # termination_processing=change_termination_processing,
                      num_trials=4,
                      call_after_trial=switch_trial_type)

        # {colors_input_layer: [[0, 0], [1, 0]],
        #                       words_input_layer: [[0, 0], [1, 0]],
        #                       task_layer: [[0, 1], [0, 1]]}


