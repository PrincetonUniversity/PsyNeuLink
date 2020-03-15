import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.statefulfunctions.integratorfunctions
import psyneulink.core.components.functions.transferfunctions


class TestStroop:
    def test_lauras_cohen_1990_model(self):
        #  INPUT UNITS

        #  colors: ('red', 'green'), words: ('RED','GREEN')
        colors_input_layer = pnl.TransferMechanism(size=2,
                                                   function=psyneulink.core.components.functions.transferfunctions.Linear,
                                                   name='COLORS_INPUT')

        words_input_layer = pnl.TransferMechanism(size=2,
                                                  function=psyneulink.core.components.functions.transferfunctions.Linear,
                                                  name='WORDS_INPUT')

        #   Task layer, tasks: ('name the color', 'read the word')
        task_layer = pnl.TransferMechanism(size=2,
                                           function=psyneulink.core.components.functions.transferfunctions.Linear,
                                           name='TASK')

        #   HIDDEN LAYER UNITS

        #   colors_hidden: ('red','green')
        #   Logistic activation function, Gain = 1.0, Bias = -4.0 (in PNL bias is subtracted so enter +4.0 to get negative bias)
        #   randomly distributed noise to the net input
        #   time averaging = integration_rate = 0.1
        unit_noise = 0.005
        colors_hidden_layer = pnl.TransferMechanism(size=2,
                                                    function=psyneulink.core.components.functions.transferfunctions
                                                    .Logistic(gain=1.0, x_0=4.0),
                                                    # should be able to get same result with offset = -4.0
                                                    integrator_mode=True,
                                                    noise=psyneulink.core.components.functions.distributionfunctions
                                                    .NormalDist(mean=0, standard_deviation=unit_noise).function,
                                                    integration_rate=0.1,
                                                    name='COLORS HIDDEN')
        #    words_hidden: ('RED','GREEN')
        words_hidden_layer = pnl.TransferMechanism(size=2,
                                                   function=pnl.Logistic(gain=1.0, x_0=4.0),
                                                   integrator_mode=True,
                                                   noise=pnl.NormalDist(mean=0,
                                                                        standard_deviation=unit_noise).function,
                                                   integration_rate=0.1,
                                                   name='WORDS HIDDEN')

        #    OUTPUT UNITS

        #   Response layer, provide input to accumulator, responses: ('red', 'green')
        #   time averaging = tau = 0.1
        #   randomly distributed noise to the net input
        response_layer = pnl.TransferMechanism(size=2,
                                               function=psyneulink.core.components.functions.transferfunctions.Logistic,
                                               name='RESPONSE',
                                               integrator_mode=True,
                                               noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                                               integration_rate=0.1)
        #   Respond red accumulator
        #   alpha = rate of evidence accumlation = 0.1
        #   sigma = noise = 0.1
        #   noise will be: squareroot(time_step_size * noise) * a random sample from a normal distribution
        accumulator_noise = 0.1
        respond_red_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist(mean=0,
                                                                                                             standard_deviation=accumulator_noise).function,
                                                                                        rate=0.1),
                                                          name='respond_red_accumulator')
        #   Respond green accumulator
        respond_green_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist(mean=0,
                                                                                                               standard_deviation=accumulator_noise).function,
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
        color_weights = pnl.MappingProjection(matrix=np.atleast_2d([[2.2, -2.2],
                                                                [-2.2, 2.2]]),
                                              name='COLOR_WEIGHTS')
        # row 0: input_'RED' to hidden_'RED', hidden_'GREEN'
        # row 1: input_'GREEN' to hidden_'RED', hidden_'GREEN'
        word_weights = pnl.MappingProjection(matrix=np.atleast_2d([[2.6, -2.6],
                                                               [-2.6, 2.6]]),
                                             name='WORD_WEIGHTS')

        #   HIDDEN TO RESPONSE
        # row 0: hidden_'red' to response_'red', response_'green'
        # row 1: hidden_'green' to response_'red', response_'green'
        color_response_weights = pnl.MappingProjection(matrix=np.atleast_2d([[1.3, -1.3],
                                                                         [-1.3, 1.3]]),
                                                       name='COLOR_RESPONSE_WEIGHTS')
        # row 0: hidden_'RED' to response_'red', response_'green'
        # row 1: hidden_'GREEN' to response_'red', response_'green'
        word_response_weights = pnl.MappingProjection(matrix=np.atleast_2d([[2.5, -2.5],
                                                                        [-2.5, 2.5]]),
                                                      name='WORD_RESPONSE_WEIGHTS')

        #   TASK TO HIDDEN LAYER
        #   row 0: task_CN to hidden_'red', hidden_'green'
        #   row 1: task_WR to hidden_'red', hidden_'green'
        task_CN_weights = pnl.MappingProjection(matrix=np.atleast_2d([[4.0, 4.0],
                                                                  [0, 0]]),
                                                name='TASK_CN_WEIGHTS')

        #   row 0: task_CN to hidden_'RED', hidden_'GREEN'
        #   row 1: task_WR to hidden_'RED', hidden_'GREEN'
        task_WR_weights = pnl.MappingProjection(matrix=np.atleast_2d([[0, 0],
                                                                  [4.0, 4.0]]),
                                                name='TASK_WR_WEIGHTS')

        #   RESPONSE UNITS TO ACCUMULATORS
        #   row 0: response_'red' to respond_red_accumulator
        #   row 1: response_'green' to respond_red_accumulator
        respond_red_differencing_weights = pnl.MappingProjection(matrix=np.atleast_2d([[1.0], [-1.0]]),
                                                                 name='RESPOND_RED_WEIGHTS')

        #   row 0: response_'red' to respond_green_accumulator
        #   row 1: response_'green' to respond_green_accumulator
        respond_green_differencing_weights = pnl.MappingProjection(matrix=np.atleast_2d([[-1.0], [1.0]]),
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
        # context is always passed to Condition functions and is the context
        # in which the function gets called - below, during system execution
        def pass_threshold(mech1, mech2, thresh, context=None):
            results1 = mech1.output_ports[0].parameters.value.get(context)
            results2 = mech2.output_ports[0].parameters.value.get(context)
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
            switch_noise(mechanisms, psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function)
            # Execute until one of the accumulators crosses the threshold
            my_Stroop.termination_processing = {
                pnl.TimeScale.TRIAL: pnl.While(
                    pass_threshold,
                    respond_red_accumulator,
                    respond_green_accumulator,
                    accumulator_threshold
                )
            }

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
