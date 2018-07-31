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

        my_Stroop = pnl.Composition()

        #   CREATE PATHWAYS
        #   Words pathway
        words_pathway = [words_input_layer,
                         word_weights,
                         words_hidden_layer,
                         word_response_weights,
                         response_layer]
        my_Stroop.add_linear_processing_pathway(words_pathway)

        #   Colors pathway
        colors_pathway = [colors_input_layer,
                          color_weights,
                          colors_hidden_layer,
                          color_response_weights,
                          response_layer]
        my_Stroop.add_linear_processing_pathway(colors_pathway)

        #   Task representation pathway
        task_CN_pathway = [task_layer,
                           task_CN_weights,
                           colors_hidden_layer]
        my_Stroop.add_linear_processing_pathway(task_CN_pathway)

        #   Task representation pathway
        task_WR_pathway = [task_layer,
                           task_WR_weights,
                           words_hidden_layer]
        my_Stroop.add_linear_processing_pathway(task_WR_pathway)

        #   Evidence accumulation pathway
        respond_red_pathway = [response_layer,
                               respond_red_differencing_weights,
                               respond_red_accumulator]
        my_Stroop.add_linear_processing_pathway(respond_red_pathway)

        respond_green_pathway = [response_layer,
                                 respond_green_differencing_weights,
                                 respond_green_accumulator]
        my_Stroop.add_linear_processing_pathway(respond_green_pathway)

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
                      call_after_trial=switch_trial_type
                      )

        # {colors_input_layer: [[0, 0], [1, 0]],
        #                       words_input_layer: [[0, 0], [1, 0]],
        #                       task_layer: [[0, 1], [0, 1]]}

    def test_kalanthroff(self):
        # Implements the Kalanthroff, Davelaar, Henik, Goldfarb & Usher model: Task Conflict and Proactive Control:
        # A Computational Theory of the Stroop Task. Psychol Rev. 2018 Jan;125(1):59-82. doi: 10.1037/rev0000083.
        # Epub 2017 Oct 16.
        # #https://www.ncbi.nlm.nih.gov/pubmed/29035077

        # Define Variables ------------------------------------------------------------------------------------------
        Lambda = 0.03  # PsyNeuLink has Euler integration constant reversed (1-0.97)
        pc_high = 0.15  # High proactive control from Figure 6 in Paper
        pc_low = 0.025  # Low proactive control from Figure 6 in Paper
        pc = pc_low  # Select proactive control
        inhibition = -1.3  # Inhibition between units within a layer
        inhibition_task = -1.9  # Inhibition between units within task layer
        bias = -0.3  # bias input to color feature layer and word feature layer
        threshold = 0.70
        settle = 200  # Number of trials until system settles

        # Create mechanisms -----------------------------------------------------------------------------------------
        # 4 Input layers for color, word, task & bias
        colors_input_layer = pnl.TransferMechanism(
            size=2,
            function=pnl.Linear,
            name='COLORS_INPUT'
        )

        words_input_layer = pnl.TransferMechanism(
            size=2,
            function=pnl.Linear,
            name='WORDS_INPUT'
        )

        task_input_layer = pnl.TransferMechanism(
            size=2,
            function=pnl.Linear,
            name='PROACTIVE_CONTROL'
        )

        bias_input = pnl.TransferMechanism(
            size=2,
            function=pnl.Linear,
            name='BIAS'
        )

        # Built python function to ensure that the logistic function outputs 0 when input is <= 0


        def my_special_Logistic(variable):
            maxi = variable - 0.0180
            output = np.fmax([0], maxi)
            return output

        # Built python function that takes output of special logistic function and computes conflict by multiplying
        # output both task units with each over times 500


        def my_conflict_function(variable):
            maxi = variable - 0.0180
            new = np.fmax([0], maxi)
            out = [new[0] * new[1] * 500]
            return out

        # Create color feature layer, word feature layer, task demand layer and response layer
        color_feature_layer = pnl.RecurrentTransferMechanism(
            size=2,  # Define unit size
            function=pnl.Logistic(  # Define Logistic fucntion & set gain
                gain=4, bias=1
            ),  # to 4 & bias to 1
            integrator_mode=True,  # Set Integrator mode to True
            integration_rate=Lambda,  # smoothing factor ==  integration rate
            hetero=inhibition,  # Inhibition among units within a layer
            output_states=[{  # Create new output state by applying
                pnl.NAME: 'SPECIAL_LOGISTIC',  # the "my_special_Logistic" function
                pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                pnl.FUNCTION: my_special_Logistic
            }],
            name='COLOR_LAYER')

        # The word_feature_layer is set up as the color_feature_layer
        word_feature_layer = pnl.RecurrentTransferMechanism(
            size=2,  # Define unit size
            function=pnl.Logistic(  # Define Logistic fucntion & set gain
                gain=4, bias=1
            ),  # to 4 & bias to 1
            integrator_mode=True,  # Set Integrator mode to True
            integration_rate=Lambda,  # smoothing factor ==  integration rate
            hetero=inhibition,  # Inhibition among units within a layer
            output_states=[{  # Create new output state by applying
                pnl.NAME: 'SPECIAL_LOGISTIC',  # the "my_special_Logistic" function
                pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                pnl.FUNCTION: my_special_Logistic
            }],
            name='WORD_LAYER')

        # The response_layer is set up as the color_feature_layer & the word_feature_layer
        response_layer = pnl.RecurrentTransferMechanism(
            size=2,  # Define unit size
            function=pnl.Logistic(  # Define Logistic fucntion & set gain
                gain=4, bias=1
            ),  # to 4 & bias to 1
            integrator_mode=True,  # Set Integrator mode to True
            integration_rate=Lambda,  # smoothing factor ==  integration rate
            hetero=inhibition,  # Inhibition among units within a layer
            output_states=[{  # Create new output state by applying
                pnl.NAME: 'SPECIAL_LOGISTIC',  # the "my_special_Logistic" function
                pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                pnl.FUNCTION: my_special_Logistic
            }],
            name='RESPONSE_LAYER'
        )

        # The task_demand_layer is set up as the color_feature_layer but with a different python function on it's output state
        # and a differnet inhibition weight on the hetero
        task_demand_layer = pnl.RecurrentTransferMechanism(
            size=2,  # Define unit size
            function=pnl.Logistic(  # Define Logistic fucntion & set gain
                gain=4, bias=1
            ),  # to 4 & bias to 1
            integrator_mode=True,  # Set Integrator mode to True
            integration_rate=Lambda,  # smoothing factor ==  integration rate
            hetero=inhibition_task,  # Inhibition among units within a layer
            output_states=[  # Create new output state by applying
                {
                    pnl.NAME: 'SPECIAL_LOGISTIC',  # the "my_conflict_function" function
                    pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                    pnl.FUNCTION: my_special_Logistic
                },
                {
                    pnl.NAME: 'CONFLICT',
                    pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                    pnl.FUNCTION: my_conflict_function
                }
            ],
            name='TASK_LAYER'
        )

        # Log mechanisms ------------------------------------------------------------------------------------------------------
        color_feature_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output of my_special_Logistic function
        word_feature_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output of my_special_Logistic function
        response_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output of my_special_Logistic function
        task_demand_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output of my_special_Logistic function

        task_demand_layer.set_log_conditions('CONFLICT')  # Log outout of my_conflict_function function

        # Connect mechanisms --------------------------------------------------------------------------------------------------
        color_input_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 0.0],  # response layer projections are set to all
                [0.0, 0.0]  # zero for initialization period first
            ])
        )
        word_input_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 0.0],
                [0.0, 0.0]
            ])
        )
        color_task_weights = pnl.MappingProjection(
            matrix=np.array([
                [2.0, 0.0],  # color to task projection
                [2.0, 0.0]
            ])
        )
        word_task_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 2.0],  # word to task projection
                [0.0, 2.0]
            ])
        )
        task_color_weights = pnl.MappingProjection(
            matrix=np.array([
                [1.0, 1.0],  # task to color projection
                [0.0, 0.0]
            ])
        )
        task_word_weights = pnl.MappingProjection(
            matrix=np.array([
                [0.0, 0.0],  # task to word projection
                [1.0, 1.0]
            ])
        )
        color_response_weights = pnl.MappingProjection(
            matrix=np.array([
                [2.0, 0.0],  # color to response projection
                [0.0, 2.0]
            ])
        )
        word_response_weights = pnl.MappingProjection(
            matrix=np.array([
                [2.5, 0.0],  # word to response projection
                [0.0, 2.5]
            ])
        )
        task_input_weights = pnl.MappingProjection(
            matrix=np.array([
                [1.0, 0.0],  # proactive control to task
                [0.0, 1.0]
            ])
        )

        # to send a control signal from the task demand layer to the response layer,
        # set matrix to -1 to reduce response layer activation
        # specify the sender of the projection which is the second output state the task demand layer
        # specify the receiver of the projection
        task_conflict_to_response_weights = pnl.MappingProjection(
            matrix=np.array([[-1.0, -1.0]]),
            sender=task_demand_layer.output_states[1],
            receiver=response_layer
        )

        # Create pathways -----------------------------------------------------------------------------------------------------
        color_response_pathway = [colors_input_layer,
                                  color_input_weights,
                                  color_feature_layer,
                                  color_response_weights,
                                  response_layer]

        word_response_pathway = [words_input_layer,
                                 word_input_weights,
                                 word_feature_layer,
                                 word_response_weights,
                                 response_layer]

        task_color_pathway_1 = [task_input_layer,
                                task_input_weights,
                                task_demand_layer,
                                task_color_weights,
                                color_feature_layer]

        task_color_pathway_2 = [color_feature_layer,
                                color_task_weights,
                                task_demand_layer]

        task_word_pathway_1 = [task_input_layer,
                               task_demand_layer,
                               task_word_weights,
                               word_feature_layer]

        task_word_pathway_2 = [word_feature_layer,
                               word_task_weights,
                               task_demand_layer]

        bias_color_pathway = [bias_input, color_feature_layer]

        bias_word_pathway = [bias_input, word_feature_layer]

        conflict_pathway = [task_demand_layer,
                            task_conflict_to_response_weights,
                            response_layer]

        PCTC = pnl.Composition(name="PCTC")

        composition_pathways = [color_response_pathway,
                                word_response_pathway,
                                task_color_pathway_1,
                                task_word_pathway_1,
                                task_color_pathway_2,
                                task_word_pathway_2,
                                bias_color_pathway,
                                bias_word_pathway,
                                conflict_pathway]

        for pathway in composition_pathways:
            PCTC.add_linear_processing_pathway(pathway)

        # reinitialize_mechanisms_when=pnl.Never(),

        def pass_threshold(response_layer, thresh):
            results1 = response_layer.output_states.values[0][0]  # red response
            results2 = response_layer.output_states.values[0][1]  # green response
            if results1 >= thresh or results2 >= thresh:
                return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)
        }

        # Create test trials function -----------------------------------------------------------------------
        # a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
        # a blue color input is [1,0] to colors_input_layer and green color is [0,1]
        # a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]


        def trial_dict(blue_color, green_color, blue_word, green_word, PC_CN, PC_WR, bias):

            trialdict = {
                colors_input_layer: [blue_color, green_color],
                words_input_layer: [blue_word, green_word],
                task_input_layer: [PC_CN, PC_WR],
                bias_input: [bias, bias]
            }
            return trialdict

        initialize_input = trial_dict(1.0, 0.0, 1.0, 0.0, pc, 0.0, bias)

        # Run congruent trial --------------------------------------------------------------------------------
        congruent_input = trial_dict(1.0, 0.0, 1.0, 0.0, pc, 0.0, bias)  # specify congruent trial input
        PCTC.run(inputs=initialize_input,
                 num_trials=settle)  # run system to settle for 200 trials with congruent stimuli input

        color_input_weights.matrix = np.array([
            [1.0, 0.0],  # set color input projections to 1 on the diagonals to e.g.
            [0.0, 1.0]
        ])  # send a green color input to the green unit of the color layer
        word_input_weights.matrix = np.array([
            [1.0, 0.0],  # the same for word input projections
            [0.0, 1.0]
        ])

        results = PCTC.run(inputs=congruent_input,
                 termination_processing=terminate_trial)  # run system with congruent stimulus input until
        # threshold in of of the response layer units is reached
        print(results)
        # Store values from run -----------------------------------------------------------------------------------------------
        # t = task_demand_layer.log.nparray_dictionary(
        #     'SPECIAL_LOGISTIC')  # Log task output from special logistic function
        # tt = t['SPECIAL_LOGISTIC']
        # n_con = tt.shape[0]
        # ttt_cong = tt.reshape(n_con, 2)
        # conflict_con = ttt_cong[200:, 0] * ttt_cong[200:, 1] * 100  # Compute conflict for plotting (as in MATLAB code)
        #
        # c = color_feature_layer.log.nparray_dictionary(
        #     'SPECIAL_LOGISTIC')  # Log color output from special logistic function
        # cc = c['SPECIAL_LOGISTIC']
        # ccc_cong = cc.reshape(n_con, 2)
        # w = word_feature_layer.log.nparray_dictionary(
        #     'SPECIAL_LOGISTIC')  # Log word output from special logistic function
        # ww = w['SPECIAL_LOGISTIC']
        # www_cong = ww.reshape(n_con, 2)
        # r = response_layer.log.nparray_dictionary(
        #     'SPECIAL_LOGISTIC')  # Log response output from special logistic function
        # rr = r['SPECIAL_LOGISTIC']
        # rrr_cong = rr.reshape(n_con, 2)
        #
        # # Clear log & reinitialize --------------------------------------------------------------------------------------------
        # response_layer.log.clear_entries(delete_entry=False)
        # color_feature_layer.log.clear_entries(delete_entry=False)
        # word_feature_layer.log.clear_entries(delete_entry=False)
        # task_demand_layer.log.clear_entries(delete_entry=False)
        #
        # color_feature_layer.reinitialize([[0, 0]])
        # word_feature_layer.reinitialize([[0, 0]])
        # response_layer.reinitialize([[0, 0]])
        # task_demand_layer.reinitialize([[0, 0]])
        #
        # # Run neutral trials --------------------------------------------------------------------------------------------------
        # # Set input projections back to 0 for settling period
        # color_input_weights.matrix = np.array([
        #     [0.0, 0.0],
        #     [0.0, 0.0]
        # ])
        # word_input_weights.matrix = np.array([
        #     [0.0, 0.0],
        #     [0.0, 0.0]
        # ])
        #
        # neutral_input = trial_dict(1.0, 0.0, 0.0, 0.0, pc, 0.0, bias)  # create neutral stimuli input
        # PCTC.run(inputs=initialize_input,
        #          num_trials=settle)  # run system to settle for 200 trials with neutral stimuli input
        #
        # color_input_weights.matrix = np.array([
        #     [1.0, 0.0],  # Set input projections to 1 for stimulus presentation period
        #     [0.0, 1.0]
        # ])
        # word_input_weights.matrix = np.array([
        #     [1.0, 0.0],
        #     [0.0, 1.0]
        # ])
        #
        # PCTC.run(inputs=neutral_input,
        #          termination_processing=terminate_trial)  # run system with neutral stimulus input until
        # # threshold in of of the response layer units is reached
        #
        # # Store values from neutral run ---------------------------------------------------------------------------------------
        # t = task_demand_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # tt = t['SPECIAL_LOGISTIC']
        # n_neutral = tt.shape[0]
        # ttt_neutral = tt.reshape(n_neutral, 2)
        # conflict_neutral = ttt_neutral[200:, 0] * ttt_neutral[200:, 1] * 100
        #
        # c = color_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # cc = c['SPECIAL_LOGISTIC']
        # ccc_neutral = cc.reshape(n_neutral, 2)
        # w = word_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # ww = w['SPECIAL_LOGISTIC']
        # www_neutral = ww.reshape(n_neutral, 2)
        # r = response_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # rr = r['SPECIAL_LOGISTIC']
        # rrr_neutral = rr.reshape(n_neutral, 2)
        # # Clear log & reinitialize --------------------------------------------------------------------------------------------
        #
        # response_layer.log.clear_entries(delete_entry=False)
        # color_feature_layer.log.clear_entries(delete_entry=False)
        # word_feature_layer.log.clear_entries(delete_entry=False)
        # task_demand_layer.log.clear_entries(delete_entry=False)
        #
        # color_feature_layer.reinitialize([[0, 0]])
        # word_feature_layer.reinitialize([[0, 0]])
        # response_layer.reinitialize([[0, 0]])
        # task_demand_layer.reinitialize([[0, 0]])
        #
        # # Run incongruent trials ----------------------------------------------------------------------------------------------
        # # Set input projections back to 0 for settling period
        # color_input_weights.matrix = np.array([
        #     [0.0, 0.0],
        #     [0.0, 0.0]
        # ])
        # word_input_weights.matrix = np.array([
        #     [0.0, 0.0],
        #     [0.0, 0.0]
        # ])
        #
        # incongruent_input = trial_dict(1.0, 0.0, 0.0, 1.0, pc, 0.0, bias)
        # PCTC.run(inputs=initialize_input, num_trials=settle)
        #
        # color_input_weights.matrix = np.array([
        #     [1.0, 0.0],
        #     [0.0, 1.0]
        # ])
        # word_input_weights.matrix = np.array([
        #     [1.0, 0.0],
        #     [0.0, 1.0]
        # ])
        #
        # PCTC.run(inputs=incongruent_input, termination_processing=terminate_trial)
        #
        # # Store values from neutral run ---------------------------------------------------------------------------------------
        #
        # t = task_demand_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # tt = t['SPECIAL_LOGISTIC']
        # n_incon = tt.shape[0]
        # ttt_incong = tt.reshape(n_incon, 2)
        # conflict_incon = ttt_incong[200:, 0] * ttt_incong[200:, 1] * 100
        #
        # c = color_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # cc = c['SPECIAL_LOGISTIC']
        # ccc_incong = cc.reshape(n_incon, 2)
        # w = word_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # ww = w['SPECIAL_LOGISTIC']
        # www_incong = ww.reshape(n_incon, 2)
        # r = response_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
        # rr = r['SPECIAL_LOGISTIC']
        # rrr_incong = rr.reshape(n_incon, 2)
        #
        # # Plotting ------------------------------------------------------------------------------------------------------------
        # # Set up plot structure
        # fig, axes = plt.subplots(nrows=3, ncols=4, sharey=True, sharex=True)
        # axes[0, 0].set_ylabel('Congruent')
        # axes[1, 0].set_ylabel('Neutral')
        # axes[2, 0].set_ylabel('Incongruent')
        #
        # axes[0, 0].set_title('Task demand units', fontsize=9)
        # axes[0, 1].set_title('Response units', fontsize=9)
        # axes[0, 2].set_title('Color feature map', fontsize=9)
        # axes[0, 3].set_title('Word feature map', fontsize=9)
        # plt.setp(
        #     axes,
        #     xticks=[0, 400, 780],
        #     yticks=[0, 0.4, 0.79],
        #     yticklabels=['0', '0.4', '0.8'],
        #     xticklabels=['0', '400', '800']
        # )
        #
        # # Plot congruent output --------------------------
        # axes[0, 0].plot(ttt_cong[settle:, 0], 'c')
        # axes[0, 0].plot(ttt_cong[settle:, 1], 'k')
        # axes[0, 0].plot(conflict_con, 'r')
        #
        # axes[0, 1].plot(rrr_cong[settle:, 0], 'b')
        # axes[0, 1].plot(rrr_cong[settle:, 1], 'g')
        # axes[0, 1].plot([0, n_con - settle], [threshold, threshold], 'k')
        # axes[0, 2].plot(ccc_cong[settle:, 0], 'b')
        # axes[0, 2].plot(ccc_cong[settle:, 1], 'g')
        #
        # axes[0, 3].plot(www_cong[settle:, 0], 'b')
        # axes[0, 3].plot(www_cong[settle:, 1], 'g')
        #
        # # Plot neutral output --------------------------
        # axes[1, 0].plot(ttt_neutral[settle:, 0], 'c')
        # axes[1, 0].plot(ttt_neutral[settle:, 1], 'k')
        # axes[1, 0].plot(conflict_neutral, 'r')
        #
        # axes[1, 1].plot(rrr_neutral[settle:, 0], 'b')
        # axes[1, 1].plot(rrr_neutral[settle:, 1], 'g')
        # axes[1, 1].plot([0, n_neutral - settle], [threshold, threshold], 'k')
        # axes[1, 2].plot(ccc_neutral[settle:, 0], 'b')
        # axes[1, 2].plot(ccc_neutral[settle:, 1], 'g')
        #
        # axes[1, 3].plot(www_neutral[settle:, 0], 'b')
        # axes[1, 3].plot(www_neutral[settle:, 1], 'g')
        #
        # # Plot incongruent output --------------------------
        # axes[2, 0].plot(ttt_incong[settle:, 0], 'c')
        # axes[2, 0].plot(ttt_incong[settle:, 1], 'k')
        # axes[2, 0].plot(conflict_incon, 'r')
        #
        # axes[2, 1].plot(rrr_incong[settle:, 0], 'b')
        # axes[2, 1].plot(rrr_incong[settle:, 1], 'g')
        # axes[2, 1].plot([0, n_incon - settle], [threshold, threshold], 'k')
        # axes[2, 2].plot(ccc_incong[settle:, 0], 'b')
        # axes[2, 2].plot(ccc_incong[settle:, 1], 'g')
        #
        # axes[2, 3].plot(www_incong[settle:, 0], 'b')
        # axes[2, 3].plot(www_incong[settle:, 1], 'g')
        #
        # plt.show()

