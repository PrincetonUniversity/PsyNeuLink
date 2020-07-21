import numpy as np
import psyneulink as pnl

#  INPUT UNITS
#  colors: ('red', 'green'), words: ('RED','GREEN')
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.statefulfunctions.integratorfunctions
import psyneulink.core.components.functions.transferfunctions

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
unit_noise = 0.001
colors_hidden_layer = pnl.TransferMechanism(size=2,
                                            function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),  #should be able to get same result with offset = -4.0
                                            integrator_mode=True,
                                            noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                                            integration_rate=0.1,
                                            name='COLORS HIDDEN')
#    words_hidden: ('RED','GREEN')
words_hidden_layer = pnl.TransferMechanism(size=2,
                                           function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),
                                           integrator_mode=True,
                                           noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                                           integration_rate=0.1,
                                           name='WORDS HIDDEN')

#    OUTPUT UNITS

#   Response layer, provide input to accumulator, responses: ('red', 'green')
response_layer = pnl.TransferMechanism(size=2,
                                       function=psyneulink.core.components.functions.transferfunctions.Logistic,
                                       integrator_mode=True,
                                       noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                                       integration_rate=0.1,
                                       name='RESPONSE')
#   Respond red accumulator
#   alpha = rate of evidence accumlation = 0.1
#   sigma = noise = 0.1
#   noise will be: squareroot(time_step_size * noise) * a random sample from a normal distribution
accumulator_noise = 0.01 #0.03
respond_red_accumulator = pnl.IntegratorMechanism(
                                   function=psyneulink.core.components.functions.statefulfunctions.integratorfunctions.SimpleIntegrator(
                                                   noise=psyneulink.core.components.functions.distributionfunctions
                                                       .NormalDist(mean=0, standard_deviation=accumulator_noise).function,
                                                   rate=0.1),
                                   name='respond_red_accumulator')
#   Respond green accumulator
respond_green_accumulator = pnl.IntegratorMechanism(
                                   function=psyneulink.core.components.functions.statefulfunctions.integratorfunctions.SimpleIntegrator(
                                                   noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=accumulator_noise).function,
                                                   rate=0.1),
                                   name='respond_green_accumulator')

#   LOGGING
# Here we set up logs to keep track of what the model is doing.
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
response_layer.set_log_conditions('value')
respond_red_accumulator.set_log_conditions('value')
respond_green_accumulator.set_log_conditions('value')

#   Create the connections between the mechanisms
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

#   Create pathways as processes

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

# Define initialization trials separately
# input just task and run once so system asymptotes
WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 1)

CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 1, 0)

m = my_Stroop.run(inputs=CN_trial_initialize_input)

#   CREATE THRESHOLD FUNCTION
# first value of DDM's value is DECISION_VARIABLE
def pass_threshold(mech1, mech2, thresh):
    results1 = mech1.output_ports[0].value
    results2 = mech2.output_ports[0].value
    for val in results1:
        if val >= thresh:
            return True
    for val in results2:
        if val >= thresh:
            return True
    return False
accumulator_threshold = 1.0

terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, respond_red_accumulator, respond_green_accumulator, accumulator_threshold)
}


# function to test a particular trial type
def testtrialtype(test_trial_input, initialize_trial_input, ntrials, condition):
    # create variable to store results
    results = np.empty((10, 0))
    # clear log
    respond_red_accumulator.log.clear_entries(delete_entry=False)
    respond_red_accumulator.reset(0)
    respond_green_accumulator.reset(0)
    print('\n\n{}'.format(condition))
    for trial in range(ntrials):
        print('.', end='', flush=True)
        if not ((trial + 1) % 10):
            print('  ', end='', flush=True)
        # run system once (with integrator mode off and no noise for hidden units) with only task so asymptotes
        colors_hidden_layer.integrator_mode = False
        words_hidden_layer.integrator_mode = False
        response_layer.integrator_mode = False
        colors_hidden_layer.noise = 0
        words_hidden_layer.noise = 0
        response_layer.noise = 0

        my_Stroop.run(inputs=initialize_trial_input)
        # but didn't want to run accumulators so set those back to zero
        respond_green_accumulator.reset(0)
        respond_red_accumulator.reset(0)

        # now put back in integrator mode and noise
        colors_hidden_layer.integrator_mode = True
        words_hidden_layer.integrator_mode = True
        response_layer.integrator_mode = True
        colors_hidden_layer.noise = psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function
        words_hidden_layer.noise = psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function
        response_layer.noise = psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function

        # run system with test pattern
        my_Stroop.run(inputs=test_trial_input, termination_processing=terminate_trial)

        # store results
        my_red_accumulator_results = respond_red_accumulator.log.nparray_dictionary()
        # print('respond_red_accumulator.log.nparray_dictionary(): ',respond_red_accumulator.log.nparray_dictionary())
        # how many cycles to run? count the length of the log
        num_timesteps = np.asarray(np.size(my_red_accumulator_results['value'])).reshape(1, 1)
        # print('num_timesteps; ', num_timesteps)
        # value of parts of the system
        red_activity = np.asarray(respond_red_accumulator.value).reshape(1, 1)
        green_activity = np.asarray(respond_green_accumulator.value).reshape(1, 1)
        colors_hidden_layer_value = np.asarray(colors_hidden_layer.value).reshape(2, 1)
        # print('colors_hidden_layer_value: ', colors_hidden_layer_value)

        words_hidden_layer_value = np.asarray(words_hidden_layer.value).reshape(2, 1)
        response_layer_value = np.asarray(response_layer.value).reshape(2, 1)
        # which response hit threshold first?
        if red_activity > green_activity:
            respond_red = np.array([1]).reshape(1, 1)
        else:
            respond_red = np.array([0]).reshape(1, 1)

        # print('num_timesteps: ', num_timesteps)
        # print('respond_red: ', respond_red)
        # print('red_activity: ', red_activity)
        # print('green_activity: ', green_activity)
        # print('colors_hidden_layer_value: ', colors_hidden_layer_value)
        # print('words_hidden_layer_value: ', words_hidden_layer_value)
        # print('response_layer_value: ', response_layer_value)
        tmp_results = np.concatenate((num_timesteps,
                                      respond_red,
                                      red_activity,
                                      green_activity,
                                      colors_hidden_layer_value,
                                      words_hidden_layer_value,
                                      response_layer_value), axis=0)
        results = np.append(results, tmp_results, axis=1)

        # print('tmp_results: ', tmp_results)

        # after a run we want to reset the activations of the integrating units
        # so we can test many trials and examine the distrubtion of responses
        words_hidden_layer.reset([0, 0])
        colors_hidden_layer.reset([0, 0])
        response_layer.reset([0, 0])
        # clear log to get num_timesteps for next run
        respond_red_accumulator.log.clear_entries(delete_entry=False)

    return results

ntrials = 10
print('\nRunning {} trials per condition (one dot for each trial, space every 10)'.format(ntrials))

WR_control_trial_input = trial_dict(0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
results_WR_control_trial = testtrialtype(WR_control_trial_input,
                                         WR_trial_initialize_input,
                                         ntrials,
                                         'Word Reading Control Condition')

#test WR congruent trial (should have the least cycles)
WR_congruent_trial_input = trial_dict(1, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
results_WR_congruent_trial = testtrialtype(WR_congruent_trial_input,
                                           WR_trial_initialize_input,
                                           ntrials,
                                           'Word Reading Congruent Condition')

#test WR incongruent trial, should see that color doesn't affect word (same number of cycles as WR control)
WR_incongruent_trial_input = trial_dict(1, 0, 0, 1, 0, 1) #red_color, green color, red_word, green word, CN, WR
results_WR_incongruent_trial = testtrialtype(WR_incongruent_trial_input,
                                             WR_trial_initialize_input,
                                             ntrials,
                                             'Word Reading Incongruent Condition')

CN_control_trial_input = trial_dict(1, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_control_trial = testtrialtype(CN_control_trial_input,
                                         CN_trial_initialize_input,
                                         ntrials,
                                         'Color Naming Control Condition')

CN_congruent_trial_input = trial_dict(1, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_congruent_trial = testtrialtype(CN_congruent_trial_input,
                                           CN_trial_initialize_input,
                                           ntrials,
                                           'Color Naming Congruent Condition')

# # #test CN incongruent trial, should see that word interferes with color (should have most cycles + more than CN control)
CN_incongruent_trial_input = trial_dict(1, 0, 0, 1, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_incongruent_trial = testtrialtype(CN_incongruent_trial_input,
                                             CN_trial_initialize_input,
                                             ntrials,
                                             'Color Naming Incongruent Condition')

# print("\nPlotting results...")
#
# plt.hist(12 * results_CN_control_trial[0][np.where(results_CN_control_trial[1]==1)]+ 206)
# plt.show()
# plt.hist(12 * results_CN_congruent_trial[0][np.where(results_CN_congruent_trial[1]==1)]+ 206)
# plt.show()
# plt.hist(12 * results_CN_incongruent_trial[0][np.where(results_CN_incongruent_trial[1]==1)]+ 206)
# plt.show()
#
#
# W_control = 12 * results_WR_control_trial[0]+ 206
# W_congruent= 12 * results_WR_congruent_trial[0]+ 206
# W_incongruent = 12 * results_WR_incongruent_trial[0]+ 206
#
#
# C_control = 12 * results_CN_control_trial[0]+ 206
# C_congruent= 12 * results_CN_congruent_trial[0]+ 206
# C_incongruent = 12 * results_CN_incongruent_trial[0]+ 206
#
#
# cycles_mean = [np.mean(W_control),
#      np.mean(W_incongruent),
#      np.mean(W_congruent),
#      np.mean(C_control),
#      np.mean(C_incongruent),
#      np.mean(C_congruent)]
# cycles_std = [np.std(W_control),
#          np.std(W_incongruent),
#          np.std(W_congruent),
#          np.std(C_control),
#          np.std(C_incongruent),
#          np.std(C_congruent)]
# cycles_x = np.array([0, 1, 2, 0, 1, 2])
# labs = ['control',
#         'conflict',
#         'congruent']
# legend = ['WR trial',
#           'CN trial']
# colors = ['b', 'c']
#
# print("Still plotting....")
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
# plt.legend(legend)
# plt.show()
