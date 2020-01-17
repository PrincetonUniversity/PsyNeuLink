import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# SET UP MECHANISMS
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
import psyneulink.core.components.functions.objectivefunctions
import psyneulink.core.components.functions.transferfunctions

colors_input_layer = pnl.TransferMechanism(size=2,
                                           function=psyneulink.core.components.functions.transferfunctions.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=2,
                                          function=psyneulink.core.components.functions.transferfunctions.Linear,
                                          name='WORDS_INPUT')

# Specify signalSearchRange for control_signal_params (not sure if needed)
#signalSearchRange = np.array([1.0])#np.arange(1.0,2.1,0.5) # why 0.8 to 2.0 in increments of 0.2 )#

#   Task layer, tasks: ('name the color', 'read the word')
task_layer = pnl.TransferMechanism(size=2,
                                   function=psyneulink.core.components.functions.transferfunctions.Logistic(
                                       gain=(1.0, pnl.ControlProjection(  # receiver= response_layer.output_ports[1],
                                           # 'DECISION_ENERGY'
                                           # modulation=pnl.OVERRIDE,#what to implement here
                                       ))),
                                   name='TASK')
task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')

task_layer.loggable_items

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
#   Logistic activation function, Gain = 1.0, Bias = -4.0
#should be randomly distributed noise to the net input of each unit (except input unit)
#should have tau = integration_rate = 0.1
colors_hidden_layer = pnl.TransferMechanism(size=2,
                                            function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),
                                            integrator_mode=True,
                                            #  noise=pnl.NormalDist(mean=0.0, standard_deviation=.005).function,
                                            integration_rate=0.1,
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.TransferMechanism(size=2,
                                           function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),
                                           integrator_mode=True,
                                           #    noise=pnl.NormalDist(mean=0.0, standard_deviation=.005).function,
                                           integration_rate=0.1,
                                           name='WORDS HIDDEN')

#log hidden layer activation
# colors_input_layer.set_log_conditions('value')
# colors_hidden_layer.set_log_conditions('value')
# colors_hidden_layer.set_log_conditions('RESULT')
#
# words_input_layer.set_log_conditions('value')
# words_hidden_layer.set_log_conditions('value')
# words_hidden_layer.set_log_conditions('RESULT')




#   Response layer, responses: ('red', 'green')
#tau = 0.1 (here, smoothing factor)
#should be randomly distributed noise to the net input of each unit (except input unit)

# Now a RecurrentTransferMechanism compared to Lauda's Stroop model!
response_layer = pnl.RecurrentTransferMechanism(size=2,  #Recurrent
                                                function=psyneulink.core.components.functions.transferfunctions.Logistic,  #pnl.Stability(matrix=np.matrix([[0.0, -1.0], [-1.0, 0.0]])),
                                                name='RESPONSE',
                                                output_ports = [pnl.RESULT,
                                          {pnl.NAME: 'DECISION_ENERGY',
                                          pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                           pnl.FUNCTION: psyneulink.core.components.functions.objectivefunctions
                                                .Stability(default_variable=np.array([0.0, -1.0]),
                                                                                                                           metric=pnl.ENERGY,
                                                                                                                           matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]))}],
                                                integrator_mode=True,  #)
                                                # noise=pnl.NormalDist(mean=0.0, standard_deviation=.01).function)
                                                integration_rate=0.1)

#response_layer.set_log_conditions('value')
#response_layer.set_log_conditions('gain')


#   SET UP CONNECTIONS
#   rows correspond to sender
#   columns correspond to: weighting of the contribution that a given sender makes to the receiver
#   in linear algebra terms can write out the matrix
#   Input to hidden
# column 0: input_'red' to hidden_'red', hidden_'green'
# column 1: input_'green' to hidden_'red', hidden_'green'
color_weights = pnl.MappingProjection(matrix=np.matrix([[2.2, -2.2],
                                                        [-2.2, 2.2]]),
                                      name='COLOR_WEIGHTS')
# column 0: input_'RED' to hidden_'RED', hidden_'GREEN'
# column 1: input_'GREEN' to hidden_'RED', hidden_'GREEN'
word_weights = pnl.MappingProjection(matrix=np.matrix([[2.6, -2.6],
                                                       [-2.6, 2.6]]),
                                     name='WORD_WEIGHTS')

#   Hidden to response
# column 0: hidden_'red' to response_'red', response_'green'
# column 1: hidden_'green' to response_'red', response_'green'
color_response_weights = pnl.MappingProjection(matrix=np.matrix([[1.3, -1.3],
                                                                 [-1.3, 1.3]]),
                                               name='COLOR_RESPONSE_WEIGHTS')
# column 0: hidden_'RED' to response_'red', response_'green'
# column 1: hidden_'GREEN' to response_'red', response_'green'
word_response_weights = pnl.MappingProjection(matrix=np.matrix([[2.5, -2.5],
                                                                [-2.5, 2.5]]),
                                              name='WORD_RESPONSE_WEIGHTS')

#   Task to hidden layer
# column 0: task_CN to hidden_'red', hidden_'green'
# column 1: task_WR to hidden_'red', hidden_'green'
# task_CN_weights = pnl.MappingProjection(matrix=np.matrix([[4.0, 0],
#                                                           [4.0, 0]]),
task_CN_weights = pnl.MappingProjection(matrix=np.matrix([[4.0, 4.0],
                                                          [0.0, 0.0]]),
                                        name='TASK_CN_WEIGHTS')

# column 0: task_CN to hidden_'RED', hidden_'GREEN'
# column 1: task_WR to hidden_'RED', hidden_'GREEN'
task_WR_weights = pnl.MappingProjection(matrix=np.matrix([[0, 0.0],
                                                          [4.0, 4.0]]),
                                        name='TASK_WR_WEIGHTS')



# In[ ]:


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

#   CREATE SYSTEM
my_Stroop = pnl.System(processes=[colors_process,
                                  words_process,
                                  task_CN_process,
                                  task_WR_process],
                      controller=pnl.ControlMechanism,
                       monitor_for_control=[response_layer],
                       enable_controller=True,
                         # objective_mechanism =pnl.ObjectiveMechanism(default_variable=[0.0, 0.0],
                         #                                             monitored_output_ports=[response_layer.output_ports[0]],
                         #                                             function=pnl.Linear(default_variable= [0.0, 0.0]),
                          #                                            name="Objective Mechanism"),
                       # monitor_for_control=
                                      # function=pnl.LinearCombination(operation=pnl.ENERGY))
                                  # respond_red_process,
                                  # respond_green_process],
                       name='FEEDFORWARD_STROOP_SYSTEM')

# my_Stroop.controller.set_log_conditions('TASK[gain] ControlSignal')
# my_Stroop.controller.loggable_items

#   CREATE THRESHOLD FUNCTION
# first value of DDM's value is DECISION_VARIABLE
def pass_threshold(mech1, thresh):
    results1 = mech1.output_ports.values[0][0] #red response
    results2 = mech1.output_ports.values[0][1] #red response
    print(results1)
    print(results2)
    if results1  >= thresh or results2 >= thresh:
        return True
    # for val in results1:
    #     if val >= thresh:
    #         return True
    return False
accumulator_threshold = 0.8

#terminate_trial = {
 #   pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, accumulator_threshold)
#}

# my_Stroop.show_graph(show_mechanism_structure=pnl.VALUES)
# my_Stroop.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)



# # Function to create test trials
# # a RED word input is [1,0] to words_input_layer and GREEN word is [0,1]
# # a red color input is [1,0] to colors_input_layer and green color is [0,1]
# # a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]
# def trial_dict(red_color, green_color, red_word, green_word, CN, WR):
#
#     trialdict = {
#     colors_input_layer: [red_color, green_color],
#     words_input_layer: [red_word, green_word],
#     task_layer: [CN, WR]
#     }
#     return trialdict
#
# # Set initialization trials to turn on control for task layer
# # run once so system asymptotes - give input to task layer only
# WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 1)
# CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 1, 0)

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

# function to test a particular trial type
# def testtrialtype(test_trial_input, initialize_trial_input, ntrials, plot_title, trial_test_counter):
#     # create variable to store results
#     results = np.empty((10, 0))
#
#     for trial in range(ntrials):
#         # run system once (with integrator mode off and no noise for hidden units) with only task so asymptotes
#         colors_hidden_layer.integrator_mode = False
#         words_hidden_layer.integrator_mode = False
#         response_layer.integrator_mode = False
#         #colors_hidden_layer.noise = 0
#         #words_hidden_layer.noise = 0
#         #response_layer.noise = 0
#         my_Stroop.run(inputs=initialize_trial_input)
#         # now put back in integrator mode and noise
#         colors_hidden_layer.integrator_mode = True
#         words_hidden_layer.integrator_mode = True
#         response_layer.integrator_mode = True
#         #colors_hidden_layer.noise = pnl.NormalDist(mean=0, standard_deviation=unit_noise).function
#         #words_hidden_layer.noise = pnl.NormalDist(mean=0, standard_deviation=unit_noise).function
#         #response_layer.noise = pnl.NormalDist(mean=0, standard_deviation=unit_noise).function
#         # run system with test pattern
#         my_Stroop.run(inputs=test_trial_input) #termination_processing=terminate_trial)
#
#     return results

#function to test a particular trial type
def testtrialtype(test_trial_input, initialize_trial_input, ntrials):#, plot_title, trial_test_counter):
    # create variable to store results
    results = np.empty((10, 0))
    # clear log
    # respond_red_accumulator.log.clear_entries(delete_entry=False)
    # respond_red_accumulator.reinitialize(0)
    # respond_green_accumulator.reinitialize(0)
    for trial in range(ntrials):
        # run system once (with integrator mode off and no noise for hidden units) with only task so asymptotes
        colors_hidden_layer.integrator_mode = False
        words_hidden_layer.integrator_mode = False
        response_layer.integrator_mode = False
        # colors_hidden_layer.noise = 0
        # words_hidden_layer.noise = 0
        # response_layer.noise = 0

        my_Stroop.run(inputs=initialize_trial_input)
        # but didn't want to run accumulators so set those back to zero
        # respond_green_accumulator.reinitialize(0)
        # respond_red_accumulator.reinitialize(0)

        # now put back in integrator mode and noise
        colors_hidden_layer.integrator_mode = True
        words_hidden_layer.integrator_mode = True
        response_layer.integrator_mode = True
        #colors_hidden_layer.noise = pnl.NormalDist(mean=0, standard_deviation=unit_noise).function
        #words_hidden_layer.noise = pnl.NormalDist(mean=0, standard_deviation=unit_noise).function
        #response_layer.noise = pnl.NormalDist(mean=0, standard_deviation=unit_noise).function

        # run system with test pattern
        my_Stroop.run(inputs=test_trial_input)

        # store results
        # my_red_accumulator_results = respond_red_accumulator.log.nparray_dictionary()
        # how many cycles to run? count the length of the log
        # num_timesteps = np.asarray(np.size(my_red_accumulator_results['value'])).reshape(1, 1)
        # value of parts of the system
        # red_activity = np.asarray(respond_red_accumulator.value).reshape(1, 1)
        # green_activity = np.asarray(respond_green_accumulator.value).reshape(1, 1)
        # colors_hidden_layer_value = np.asarray(colors_hidden_layer.value).reshape(2, 1)
        # words_hidden_layer_value = np.asarray(words_hidden_layer.value).reshape(2, 1)
        # response_layer_value = np.asarray(response_layer.value).reshape(2, 1)

    return results



# trial_test_counter = 1
# #test WR control trial
# ntrials = 50
# WR_control_trial_title = 'RED word (control) WR trial where Red correct'
# WR_control_trial_input = trial_dict(0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# results_WR_control_trial = testtrialtype(WR_control_trial_input,
#                                          WR_trial_initialize_input,
#                                          ntrials,
#                                          WR_control_trial_title,
#                                          trial_test_counter)

# ntrials = 50
# WR_congruent_trial_title = 'congruent WR trial where Red correct'
# WR_congruent_trial_input = trial_dict(1, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# results_WR_congruent_trial = testtrialtype(WR_congruent_trial_input,
#                                            WR_trial_initialize_input,
#                                            ntrials,
#                                            WR_congruent_trial_title,
#                                            trial_test_counter)

# ntrials = 150
# WR_incongruent_trial_title = 'incongruent WR trial where Red correct'
# WR_incongruent_trial_input = trial_dict(1, 0, 0, 1, 0, 1) #red_color, green color, red_word, green word, CN, WR
# results_WR_incongruent_trial = testtrialtype(WR_incongruent_trial_input,
#                                              WR_trial_initialize_input,
#                                              ntrials,
#                                              WR_incongruent_trial_title,
#                                              trial_test_counter)
#
# print(response_layer.value)

# ntrials = 50
# CN_control_trial_title = 'red color (control) CN trial where Red correct'
# CN_control_trial_input = trial_dict(1, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# results_CN_control_trial = testtrialtype(CN_control_trial_input,
#                                          CN_trial_initialize_input,
#                                          ntrials,
#                                          CN_control_trial_title,
#                                          trial_test_counter)


# ntrials = 50
# CN_congruent_trial_title = 'congruent CN trial where Red correct'
# CN_congruent_trial_input = trial_dict(1, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# results_CN_congruent_trial = testtrialtype(CN_congruent_trial_input,
#                                            CN_trial_initialize_input,
#                                            ntrials,
#                                            CN_congruent_trial_title,
#                                            trial_test_counter)

ntrials = 10
CN_incongruent_trial_input = trial_dict(1, 0, 0, 1, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_incongruent_trial = testtrialtype(CN_incongruent_trial_input,
                                             CN_trial_initialize_input,
                                             ntrials)

#my_Stroop.run(inputs=CN_trial_initialize_input,
 #           )
print(response_layer.value)

# my_Stroop.run(inputs=test_trial_input)
# #run system once with only task so asymptotes
# nTrials = 5
# #my_Stroop.run(inputs=CN_incongruent_trial_input, num_trials=nTrials)
# #but didn't want to run accumulators so set those back to zero
# #respond_green_accumulator.reinitialize(0)
# #respond_red_accumulator.reinitialize(0)
# # now run test trial
# #my_Stroop.show_graph(show_mechanism_structure=pnl.VALUES)
# my_Stroop.run(inputs=CN_incongruent_trial_input, num_trials=1)
#
# response_layer.log.print_entries()
# my_Stroop.log.print_entries()

