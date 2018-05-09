import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# Define Variables ----------------------------------------------------------------------------------------------------

rate = 0.01
inhibition = -2.0
bias = 4.0
threshold = 0.6
settle_trials = 500
# Create mechanisms ---------------------------------------------------------------------------------------------------
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(size=2,
                                           function=pnl.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=2,
                                          function=pnl.Linear,
                                          name='WORDS_INPUT')

task_input_layer = pnl.TransferMechanism(size=2,
                                          function=pnl.Linear,
                                          name='TASK_INPUT')

#   Task layer, tasks: ('name the color', 'read the word')
task_layer = pnl.RecurrentTransferMechanism(size=2,
                                            function=pnl.Logistic(),
                                            hetero=-2,
                                            integrator_mode=True,
                                            smoothing_factor=0.01,
                                            name='TASK')

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=2,
                                            function=pnl.Logistic(bias=4.0),
                                            integrator_mode=True,
                                                     hetero=-2,
                                           # noise=pnl.NormalDist(mean=0.0, standard_dev=.0).function,
                                            smoothing_factor=0.01, # cohen-huston text says 0.01
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[1, 1, 1]]),
                                                    size=2,
                                           function=pnl.Logistic(bias=4.0),
                                                    hetero=-2,
                                           integrator_mode=True,
                                          # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
                                           smoothing_factor=0.01,
                                           name='WORDS HIDDEN')

# Log mechanisms ------------------------------------------------------------------------------------------------------
#task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
task_layer.set_log_conditions('InputState-0')

colors_hidden_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('InputState-0')

words_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('InputState-0')






#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(size=2,  #Recurrentdefault_variable=np.array([[3.1, 3.1]]),
                                                function=pnl.Logistic(),
                                                hetero=-2.0,
                                                integrator_mode=True,
                                                smoothing_factor=0.01,
                                                name='RESPONSE')


response_layer.set_log_conditions('value')
response_layer.set_log_conditions('InputState-0')
# Connect mechanisms --------------------------------------------------------------------------------------------------
# (note that response layer projections are set to all zero first for initialization

color_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0],
                                                             [0.0, 1.0]]))

word_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))

task_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))

color_task_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 0.0],
                                                             [4.0, 0.0]]))

task_color_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 4.0],
                                                             [0.0, 0.0]]))

word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                           [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
                                                           [4.0, 4.0]]))

# color_response_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
#                                    [0.0, 0.0],
#                                    [0.0, 0.0]]))
#
# word_response_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
#                                    [0.0, 0.0],
#                                    [0.0, 0.0]]))
#
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
                                   [0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
                                   [0.0, 0.0]]))
#
# response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
#                                    [0.0, 1.5, 0.0]]))
#
# response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
#                                    [0.0, 2.5, 0.0]]))

color_response_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
                                   [0.0, 1.5]]))
word_response_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
                                   [0.0, 2.5]]))
#
# Create pathways -----------------------------------------------------------------------------------------------------
color_response_process = pnl.Process(pathway=[colors_input_layer,
                                              color_input_weights,
                                              colors_hidden_layer,
                                              color_response_weights,
                                              response_layer,
                                              response_color_weights,
                                              colors_hidden_layer],
                                     name='COLORS_RESPONSE_PROCESS')

word_response_process = pnl.Process(pathway=[words_input_layer,
                                             word_input_weights,
                                             words_hidden_layer,
                                             word_response_weights,
                                             response_layer,
                                             response_word_weights,
                                             words_hidden_layer],
                                     name='WORDS_RESPONSE_PROCESS')

task_color_response_process = pnl.Process(pathway=[task_input_layer,
                                                   task_input_weights,
                                                   task_layer,
                                                   task_color_weights,
                                                   colors_hidden_layer,
                                                   color_task_weights,
                                                   task_layer])

task_word_response_process = pnl.Process(pathway=[task_input_layer,
                                                  task_layer,
                                                  task_word_weights,
                                                  words_hidden_layer,
                                                  word_task_weights,
                                                  task_layer])


# Create system -------------------------------------------------------------------------------------------------------
Bidirectional_Stroop = pnl.System(processes=[color_response_process,
                                                   word_response_process,
                                                   task_color_response_process,
                                                   task_word_response_process],
                                        name='FEEDFORWARD_STROOP_SYSTEM')


# LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')


Bidirectional_Stroop.show()
# Bidirectional_Stroop.show_graph(show_dimensions=pnl.ALL)#,show_mechanism_structure=pnl.VALUES) # Uncomment to show graph of the system

# Create threshold function -------------------------------------------------------------------------------------------
def pass_threshold(response_layer, thresh):
    results1 = response_layer.output_states.values[0][0] #red response
    results2 = response_layer.output_states.values[0][1] #green response
    # print(results1)
    # print(results2)
    if results1  >= thresh or results2 >= thresh:
        return True
    return False

terminate_trial = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)
}

# Create test trials function -----------------------------------------------------------------------------------------
# a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a blue color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]
def trial_dict(red_color, green_color, red_word, green_word, CN, WR):

    trialdict = {
    colors_input_layer: [red_color, green_color],
    words_input_layer: [red_word, green_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
WR_initialize_input = trial_dict(0, 0, 0, 0, 0, 1)
CN_initialize_input = trial_dict(0, 0, 0, 0, 1, 0)

CN_incongruent_trial_input = trial_dict(1, 0, 0, 1, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR

WR_congruent_trial_input = trial_dict(1, 0, 1, 0,  0, 1) #red_color, green color, red_word, green word, CN, WR
WR_incongruent_trial_input = trial_dict(1, 0, 0, 1, 0, 1) #red_color, green color, red_word, green word, CN, WR
WR_control_trial_input = trial_dict(1, 0, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR

# Run congruent trial -------------------------------------------------------------------------------------------------


Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
                                   [0.0, 1.5]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
                                   [0.0, 2.5]]))

# colors_hidden_layer.reinitialize([[0,0,0]])
# words_hidden_layer.reinitialize([[0,0,0]])
# response_layer.reinitialize([[0,0]])
# task_layer.reinitialize([[0,0]])

Bidirectional_Stroop.run(inputs=CN_congruent_trial_input,num_trials=600)# termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
ccc_cong = cc.reshape(n_con,2)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
www_cong = ww.reshape(n_con,2)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
rrr_cong = rr.reshape(n_con,2)

# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

# colors_hidden_layer.reinitialize([[0,0]])
# words_hidden_layer.reinitialize([[0,0]])
# response_layer.reinitialize([[0,0]])
# task_layer.reinitialize([[0,0]])
#
# #   CREATE THRESHOLD FUNCTION
# #first value of DDM's value is DECISION_VARIABLE
# # def pass_threshold(response_layer, thresh):
# #     results1 = response_layer.output_states.values[0][0] #red response
# #     results2 = response_layer.output_states.values[0][1] #green response
# #     # print(results1)
# #     # print(results2)
# #     if results1  >= thresh or results2 >= thresh:
# #         return True
# #     # for val in results1:
# #     #     if val >= thresh:
# #     #         return True
# #     return False
# # accumulator_threshold = 0.6
# #
# # terminate_trial = {
# #    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, accumulator_threshold)
# # }
#
# def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
#
#     trialdict = {
#     colors_input_layer: [red_color, green_color, neutral_color],
#     words_input_layer: [red_word, green_word, neutral_word],
#     task_input_layer: [CN, WR]
#     }
#     return trialdict
#
# # Define initialization trials separately
# WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 1)
# CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)
#
# # Initialize System:
# # colors_hidden_layer.integrator_mode = False
# # words_hidden_layer.integrator_mode = False
# # task_layer.integrator_mode = False
# # response_layer.integrator_mode = False
#
# # RUN SYSTEM INITIALIZATION:
# # System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, termination_processing=terminate_initialization)
# # colors_hidden_layer.reinitialize([[0, 0]])
# # words_hidden_layer.reinitialize([[0, 0]])
#
#
# print('colors_hidden_layer after initial trial: ', colors_hidden_layer.output_states.values)
# print('words_hidden_layer after initial trial: ', words_hidden_layer.output_states.values)
# print('response_layer after initial trial: ', response_layer.output_states.values)
# print('task_layer after initial trial: ', task_layer.output_states.values)
#
# # response_layer.integrator_mode = True
# # colors_hidden_layer.integrator_mode = True
# # words_hidden_layer.integrator_mode = True
# # task_layer.integrator_mode = True
#
# # response_layer.reinitialize([[0, 0]])
# print('response_layer after reinitialization trial: ', response_layer.output_states.values)
#
# # System_Conflict_Monitoring.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)
#
# ###--------------------------------------
# # RUN the SYSTEM to initialize:
# ntrials0 = 500
# System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, num_trials=ntrials0)# termination_processing=terminate_trial)
#
# ###------------------------------------
#
# # System_Conflict_Monitoring.show_graph(show_dimensions=pnl.ALL, show_mechanism_structure=True)
#
# # response_layer.output_states[1].function_object.matrix =np.array([[0, -4], [-4, 0]])
#
#
# # response_layer.reinitialize([[-1, -1]])
# response_layer.hetero = np.array([[0.0, -2.0], [-2.0, 0.0]])
# # task_layer.hetero = np.array([[0.0, -1.0,], [-1.0, 0.0]])
#
#
# #####----------- Turn on projections to response layer and from response layer since response layer is silenced in initialization phase
#
# response_color_weights.matrix = np.array([[1.5, 0.0, 0.0],
#                                    [0.0, 1.5, 0.0]])
#
# response_word_weights.matrix  = np.array([[2.5, 0.0, 0.0],
#                                    [0.0, 2.5, 0.0]])
# #
# color_response_weights.matrix = np.array([[1.5, 0.0],
#                                    [0.0, 1.5],
#                                    [0.0, 0.0]])
# word_response_weights.matrix  = np.array([[2.5, 0.0],
#                                    [0.0, 2.5],
#                                    [0.0, 0.0]])
#
# ####----------- RUN SYSTEM for trial input
# ### ----------- Conflict monitoring paper only used color naming (CN) trials
#
# CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
#
# # WR_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# # WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# # WR_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
#
# ntrials = 800
# System_Conflict_Monitoring.run(inputs=CN_control_trial_input, num_trials=ntrials)# termination_processing=terminate_trial)

plt.figure()
plt.plot(ttt_cong, 'k')
plt.plot(www_cong, 'b')
plt.plot(ccc_cong,'r')
plt.plot(rrr_cong, 'g')
# legend = ['task unit 1',
#           'task unit 2',
#           'word units',
#           'word units',
#           'word units',
#           'color units',
#           'color units',
#           'color units',
#           'response units',
#           'response units',
#           'threshold']
# plt.legend(legend)
plt.show()


