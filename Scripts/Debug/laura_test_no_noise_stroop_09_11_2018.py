# import psyneulink as pnl
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas
#
#
# def test_lauras_cohen_1990_model(red_color, green_color, red_word, green_word, CN, WR, n_trials):
#     #  INPUT UNITS
#
#     #  colors: ('red', 'green'), words: ('RED','GREEN')
#     colors_input_layer = pnl.TransferMechanism(size=2,
#                                                function=pnl.Linear,
#                                                name='COLORS_INPUT')
#
#     words_input_layer = pnl.TransferMechanism(size=2,
#                                               function=pnl.Linear,
#                                               name='WORDS_INPUT')
#
#     #   Task layer, tasks: ('name the color', 'read the word')
#     task_layer = pnl.TransferMechanism(size=2,
#                                        function=pnl.Linear,
#                                        name='TASK')
#
#     #   HIDDEN LAYER UNITS
#
#     #   colors_hidden: ('red','green')
#     #   Logistic activation function, Gain = 1.0, Bias = -4.0 (in PNL bias is subtracted so enter +4.0 to get negative bias)
#     #   randomly distributed noise to the net input
#     #   time averaging = integration_rate = 0.1
#     unit_noise = 0.005
#     # colors_hidden_layer = pnl.TransferMechanism(size=2,
#     #                                             function=pnl.Logistic(gain=1.0, bias=4.0),
#     #                                             # should be able to get same result with offset = -4.0
#     #                                             integrator_mode=True,
#     #                                             noise=pnl.NormalDist(mean=0, standard_deviation=unit_noise).function,
#     #                                             integration_rate=0.1,
#     #                                             name='COLORS HIDDEN')
#
#     colors_hidden_layer = pnl.TransferMechanism(size=2,
#                                                 function=pnl.Logistic(gain=1.0, x_0=4.0),
#                                                 # should be able to get same result with offset = -4.0
#                                                 integrator_mode=True,
#                                                 noise=0.0,
#                                                 integration_rate=0.1,
#                                                 name='COLORS HIDDEN')
#     #    words_hidden: ('RED','GREEN')
#     # words_hidden_layer = pnl.TransferMechanism(size=2,
#     #                                            function=pnl.Logistic(gain=1.0, bias=4.0),
#     #                                            integrator_mode=True,
#     #                                            noise=pnl.NormalDist(mean=0, standard_deviation=unit_noise).function,
#     #                                            integration_rate=0.1,
#     #                                            name='WORDS HIDDEN')
#     words_hidden_layer = pnl.TransferMechanism(size=2,
#                                                function=pnl.Logistic(gain=1.0, x_0=4.0),
#                                                integrator_mode=True,
#                                                noise=0.0,
#                                                integration_rate=0.1,
#                                                name='WORDS HIDDEN')
#
#     #    OUTPUT UNITS
#
#     #   Response layer, provide input to accumulator, responses: ('red', 'green')
#     #   time averaging = tau = 0.1
#     #   randomly distributed noise to the net input
#     # response_layer = pnl.TransferMechanism(size=2,
#     #                                        function=pnl.Logistic,
#     #                                        name='RESPONSE',
#     #                                        integrator_mode=True,
#     #                                        noise=pnl.NormalDist(mean=0, standard_deviation=unit_noise).function,
#     #                                        integration_rate=0.1)
#     response_layer = pnl.TransferMechanism(size=2,
#                                            function=pnl.Logistic,
#                                            name='RESPONSE',
#                                            integrator_mode=True,
#                                            noise=0.0,
#                                            integration_rate=0.1)
#     #   Respond red accumulator
#     #   alpha = rate of evidence accumlation = 0.1
#     #   sigma = noise = 0.1
#     #   noise will be: squareroot(time_step_size * noise) * a random sample from a normal distribution
#     accumulator_noise = 0.1
#     # respond_red_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist(mean=0,
#     #                                                                                                      standard_deviation= accumulator_noise).function,
#     #                                                                                 rate=0.1),
#     #                                                   name='respond_red_accumulator')
#     respond_red_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=0.0,
#                                                                                     rate=0.1),
#                                                       name='respond_red_accumulator')
#     #   Respond green accumulator
#     # respond_green_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist(mean=0,
#     #                                                                                                        standard_deviation=accumulator_noise).function,
#     #                                                                                   rate=0.1),
#     #                                                     name='respond_green_accumulator')
#     respond_green_accumulator = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=0.0,
#                                                                                       rate=0.1),
#                                                         name='respond_green_accumulator')
#
#     #   LOGGING
#     colors_hidden_layer.set_log_conditions('value')
#     words_hidden_layer.set_log_conditions('value')
#     response_layer.set_log_conditions('value')
#     respond_red_accumulator.set_log_conditions('value')
#     respond_green_accumulator.set_log_conditions('value')
#
#     logged_mechanisms = [colors_hidden_layer, words_hidden_layer, response_layer, respond_red_accumulator, respond_green_accumulator]
#
#     #   SET UP CONNECTIONS
#
#     #   rows correspond to sender
#     #   columns correspond to: weighting of the contribution that a given sender makes to the receiver
#
#     #   INPUT TO HIDDEN
#     # row 0: input_'red' to hidden_'red', hidden_'green'
#     # row 1: input_'green' to hidden_'red', hidden_'green'
#     color_weights = pnl.MappingProjection(matrix=np.matrix([[2.2, -2.2],
#                                                             [-2.2, 2.2]]),
#                                           name='COLOR_WEIGHTS')
#     # row 0: input_'RED' to hidden_'RED', hidden_'GREEN'
#     # row 1: input_'GREEN' to hidden_'RED', hidden_'GREEN'
#     word_weights = pnl.MappingProjection(matrix=np.matrix([[2.6, -2.6],
#                                                            [-2.6, 2.6]]),
#                                          name='WORD_WEIGHTS')
#
#     #   HIDDEN TO RESPONSE
#     # row 0: hidden_'red' to response_'red', response_'green'
#     # row 1: hidden_'green' to response_'red', response_'green'
#     color_response_weights = pnl.MappingProjection(matrix=np.matrix([[1.3, -1.3],
#                                                                      [-1.3, 1.3]]),
#                                                    name='COLOR_RESPONSE_WEIGHTS')
#     # row 0: hidden_'RED' to response_'red', response_'green'
#     # row 1: hidden_'GREEN' to response_'red', response_'green'
#     word_response_weights = pnl.MappingProjection(matrix=np.matrix([[2.5, -2.5],
#                                                                     [-2.5, 2.5]]),
#                                                   name='WORD_RESPONSE_WEIGHTS')
#
#     #   TASK TO HIDDEN LAYER
#     #   row 0: task_CN to hidden_'red', hidden_'green'
#     #   row 1: task_WR to hidden_'red', hidden_'green'
#     task_CN_weights = pnl.MappingProjection(matrix=np.matrix([[4.0, 4.0],
#                                                               [0, 0]]),
#                                             name='TASK_CN_WEIGHTS')
#
#     #   row 0: task_CN to hidden_'RED', hidden_'GREEN'
#     #   row 1: task_WR to hidden_'RED', hidden_'GREEN'
#     task_WR_weights = pnl.MappingProjection(matrix=np.matrix([[0, 0],
#                                                               [4.0, 4.0]]),
#                                             name='TASK_WR_WEIGHTS')
#
#     #   RESPONSE UNITS TO ACCUMULATORS
#     #   row 0: response_'red' to respond_red_accumulator
#     #   row 1: response_'green' to respond_red_accumulator
#     respond_red_differencing_weights = pnl.MappingProjection(matrix=np.matrix([[1.0], [-1.0]]),
#                                                              name='RESPOND_RED_WEIGHTS')
#
#     #   row 0: response_'red' to respond_green_accumulator
#     #   row 1: response_'green' to respond_green_accumulator
#     respond_green_differencing_weights = pnl.MappingProjection(matrix=np.matrix([[-1.0], [1.0]]),
#                                                                name='RESPOND_GREEN_WEIGHTS')
#
#     #   CREATE PATHWAYS
#     #   Words pathway
#     words_process = pnl.Process(pathway=[words_input_layer,
#                                          word_weights,
#                                          words_hidden_layer,
#                                          word_response_weights,
#                                          response_layer], name='WORDS_PROCESS')
#
#     #   Colors pathway
#     colors_process = pnl.Process(pathway=[colors_input_layer,
#                                           color_weights,
#                                           colors_hidden_layer,
#                                           color_response_weights,
#                                           response_layer], name='COLORS_PROCESS')
#
#     #   Task representation pathway
#     task_CN_process = pnl.Process(pathway=[task_layer,
#                                            task_CN_weights,
#                                            colors_hidden_layer],
#                                   name='TASK_CN_PROCESS')
#     task_WR_process = pnl.Process(pathway=[task_layer,
#                                            task_WR_weights,
#                                            words_hidden_layer],
#                                   name='TASK_WR_PROCESS')
#
#     #   Evidence accumulation pathway
#     respond_red_process = pnl.Process(pathway=[response_layer,
#                                                respond_red_differencing_weights,
#                                                respond_red_accumulator],
#                                       name='RESPOND_RED_PROCESS')
#     respond_green_process = pnl.Process(pathway=[response_layer,
#                                                  respond_green_differencing_weights,
#                                                  respond_green_accumulator],
#                                         name='RESPOND_GREEN_PROCESS')
#
#     #   CREATE SYSTEM
#     my_Stroop = pnl.System(processes=[colors_process,
#                                       words_process,
#                                       task_CN_process,
#                                       task_WR_process,
#                                       respond_red_process,
#                                       respond_green_process],
#                            name='FEEDFORWARD_STROOP_SYSTEM')
#
#     # my_Stroop.show()
#     # my_Stroop.show_graph(show_dimensions=pnl.ALL)
#
#     # Function to create test trials
#     # a RED word input is [1,0] to words_input_layer and GREEN word is [0,1]
#     # a red color input is [1,0] to colors_input_layer and green color is [0,1]
#     # a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]
#
#     def trial_dict(red_color, green_color, red_word, green_word, CN, WR):
#
#         trialdict = {
#             colors_input_layer: [[0, 0], [red_color, green_color]],
#             words_input_layer: [[0, 0], [red_word, green_word]],
#             task_layer: [[CN, WR], [CN, WR]]
#         }
#         return trialdict
#
#     #   CREATE THRESHOLD FUNCTION
#     # first value of DDM's value is DECISION_VARIABLE
#     def pass_threshold(mech1, mech2, thresh):
#         results1 = mech1.output_ports[0].value
#         results2 = mech2.output_ports[0].value
#         for val in results1:
#             if val >= thresh:
#                 return True
#         for val in results2:
#             if val >= thresh:
#                 return True
#         return False
#
#     accumulator_threshold = 1.0
#
#     mechanisms_to_update = [colors_hidden_layer, words_hidden_layer, response_layer]
#
#     def switch_integrator_mode(mechanisms, mode):
#         for mechanism in mechanisms:
#             mechanism.integrator_mode = mode
#
#     def switch_noise(mechanisms, noise):
#         for mechanism in mechanisms:
#             mechanism.noise = noise
#
#     def switch_to_initialization_trial(mechanisms):
#         # Turn off accumulation
#         switch_integrator_mode(mechanisms, False)
#         # Turn off noise
#         switch_noise(mechanisms, 0)
#         # Execute once per trial
#         my_Stroop.termination_processing = {pnl.TimeScale.TRIAL: pnl.AllHaveRun()}
#
#     def switch_to_processing_trial(mechanisms):
#         # Turn on accumulation
#         switch_integrator_mode(mechanisms, True)
#         # Turn on noise
#         # switch_noise(mechanisms, pnl.NormalDist(mean=0, standard_deviation=unit_noise).function)
#         # Execute until one of the accumulators crosses the threshold
#         my_Stroop.termination_processing = {pnl.TimeScale.TRIAL: pnl.While(pass_threshold,
#                                                                            respond_red_accumulator,
#                                                                            respond_green_accumulator,
#                                                                            accumulator_threshold)}
#
#     def switch_trial_type():
#         # Next trial will be a processing trial
#         if isinstance(my_Stroop.termination_processing[pnl.TimeScale.TRIAL], pnl.AllHaveRun):
#             switch_to_processing_trial(mechanisms_to_update)
#         # Next trial will be an initialization trial
#         else:
#             switch_to_initialization_trial(mechanisms_to_update)
#
#     def _extract_rt_cycles(mechanism):
#         # Grab the log dictionary from the output layer
#         log_dict = mechanism.log.nparray_dictionary()
#
#         # Extract out the relevant keys from the log to a single numpy array
#         relevant_key_arrays = [np.array([x[0] for x in log_dict[key]]) for key in ('Run', 'Trial', 'Pass')]
#         table = np.stack(relevant_key_arrays, axis=1)
#
#         # Filter out only the last run
#         last_run = np.max(table[:, 0])
#         table = table[table[:, 0] == last_run]
#
#         # Filter out only the last pass of each trial
#         trial_ends = (table[1:, 1] - table[:-1, 1]) != 0
#         trial_ends = np.append(trial_ends, True)
#         last_passes = table[trial_ends, :]
#
#         # Filter out only odd trials
#         last_passes = last_passes[last_passes[:, 1] % 2 == 1, :]
#         return last_passes[:, 2]
#
#     def last_run_to_dataframe(mechanism_list):
#         dataframes = []
#         first = True
#         for log_layer in mechanism_list:
#             layer_size = log_layer.size[0]
#             log_dict = log_layer.log.nparray_dictionary()
#
#             # Extract out all keys, treating value specially since it's already an np array
#             arrays = [np.array([x[0] for x in log_dict[key]]) for key in ('Run', 'Trial', 'Pass', 'Time_step')]
#             arrays.extend([np.squeeze(log_dict['value'][:, :, i]) for i in range(layer_size)])
#             table = np.stack(arrays, axis=1)
#
#             # Filter out only the last run
#             last_run = np.max(table[:, 0])
#             table = table[table[:, 0] == last_run]
#
#             # Create as dataframe and add to the list of dataframes
#             if first:
#                 df = pandas.DataFrame(table, columns=['Run', 'Trial', 'Pass', 'Time_step'] +
#                                                      [f'{log_layer.name}_{i}' for i in range(layer_size)])
#                 first = False
#
#             else:
#                 df = pandas.DataFrame(table[:, -1 * layer_size:], columns=[f'{log_layer.name}_{i}'
#                                                                            for i in range(layer_size)])
#
#             dataframes.append(df)
#
#         return pandas.concat(dataframes, axis=1, join='inner')
#
#
#     # Start with an initialization trial
#     switch_to_initialization_trial(mechanisms_to_update)
#
#     my_Stroop.run(inputs=trial_dict(red_color, green_color, red_word, green_word, CN, WR),
#                   # termination_processing=change_termination_processing,
#                   num_trials=n_trials,
#                   call_after_trial=switch_trial_type)
#
#     # respond_red_accumulator.log.print_entries()
#     # respond_green_accumulator.log.print_entries()
#     # response_layer.log.print_entries()
#     my_Stroop_rt_cycles = _extract_rt_cycles(respond_green_accumulator)
#     my_Stroop_DataFrame = last_run_to_dataframe(logged_mechanisms)
#     # print(my_Stroop_DataFrame)
#     respond_red_accumulator.log.print_entries
#
#     return my_Stroop_rt_cycles
#
# incong_results = test_lauras_cohen_1990_model(0, 1, 1, 0, 1, 0, 10)
#
# print(incong_results)
#
# plt.hist(incong_results)
# plt.title('Reaction times for incongruent CN trial where green is correct response')
# plt.xlabel("number of cycles")
# plt.ylabel("number of trials out of 100")
# plt.show()
#
# #test WR control trial
# results_WR_control_trial = test_lauras_cohen_1990_model(0, 0, 1, 0, 0, 1, 50)
#
# #test WR congruent trial (should have the least cycles)
# results_WR_congruent_trial = test_lauras_cohen_1990_model(1, 0, 1, 0, 0, 1, 50)
#
# #test WR incongruent trial, should see that color doesn't affect word (same number of cycles as WR control)
# results_WR_incongruent_trial = test_lauras_cohen_1990_model(0, 1, 1, 0, 0, 1, 50)
#
# #test CN control trial
# results_CN_control_trial = test_lauras_cohen_1990_model(1, 0, 0, 0, 1, 0, 50)
#
# #test CN congruent trial (should have more cycles than WR congruent)
# results_CN_congruent_trial = test_lauras_cohen_1990_model(1, 0, 1, 0, 1, 0, 50)
#
# #test CN incongruent trial, should see that word interferes with color (should have most cycles + more than CN control)
# results_CN_incongruent_trial = test_lauras_cohen_1990_model(1, 0, 0, 1, 1, 0, 50)
#
#
# print(results_CN_incongruent_trial)
# cycles_mean = [np.mean(results_WR_control_trial[0]),
#      np.mean(results_WR_incongruent_trial[0]),
#      np.mean(results_WR_congruent_trial[0]),
#      np.mean(results_CN_control_trial[0]),
#      np.mean(results_CN_incongruent_trial[0]),
#      np.mean(results_CN_congruent_trial[0])]
# cycles_std = [np.std(results_WR_control_trial),
#          np.std(results_WR_incongruent_trial),
#          np.std(results_WR_congruent_trial),
#          np.std(results_CN_control_trial),
#          np.std(results_CN_incongruent_trial),
#          np.std(results_CN_congruent_trial)]
# cycles_x = np.array([0, 1, 2, 0, 1, 2])
# labs = ['control',
#         'conflict',
#         'congruent']
# legend = ['WR trial',
#           'CN trial']
# colors = ['b', 'c']
#
# print(np.mean(results_WR_congruent_trial[0]))
# print('of 0',np.mean(results_WR_congruent_trial))
#
# plt.plot(cycles_x[0:3], cycles_mean[0:3], color=colors[0])
# plt.errorbar(cycles_x[0:3], cycles_mean[0:3], xerr=0, yerr=cycles_std[0:3], ecolor=colors[0], fmt='none')
# plt.scatter(cycles_x[0], cycles_mean[0], marker='x', color=colors[0])
# plt.scatter(cycles_x[1], cycles_mean[1], marker='x', color=colors[0])
# plt.scatter(cycles_x[2], cycles_mean[2], marker='x', color=colors[0])
# plt.plot(cycles_x[3:6], cycles_mean[3:6], color=colors[1])
# plt.errorbar(cycles_x[3:6], cycles_mean[3:6], xerr=0, yerr=cycles_std[3:6], ecolor=colors[1], fmt='none')
# plt.scatter(cycles_x[3], cycles_mean[3], marker='o', color=colors[1])
# plt.scatter(cycles_x[4], cycles_mean[4], marker='o', color=colors[1])
# plt.scatter(cycles_x[5], cycles_mean[5], marker='o', color=colors[1])
#
# plt.xticks(cycles_x, labs, rotation=15)
# plt.tick_params(axis='x', labelsize=9)
# plt.title('Mean Number of Cycles by trial type')
# plt.legend(legend)
# plt.show()