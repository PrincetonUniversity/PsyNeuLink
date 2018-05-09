import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# Define Variables ----------------------------------------------------------------------------------------------------
rate = 0.01
inhibition = -2.0
bias = 4.0
threshold = 0.55
settle_trials = 50
prior200 = 20
prior400 = 40
prior600 = 60
prior800 = 80
prior1000 = 100
prior1200 = 120
after200 = 20
after400 = 40
after600 = 60
after800 = 80
after1000 = 100
terminate2 = 230
terminate3 = 250
terminate4 = 270

# Create mechanisms ---------------------------------------------------------------------------------------------------
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(size=3,
                                           function=pnl.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=3,
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
                                            smoothing_factor=0.1,
                                            name='TASK')

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                            function=pnl.Logistic(bias=4.0),
                                            integrator_mode=True,
                                                     hetero=-2.0,
                                           # noise=pnl.NormalDist(mean=0.0, standard_dev=.0).function,
                                            smoothing_factor=0.1, # cohen-huston text says 0.01
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[1, 1, 1]]),
                                                    size=3,
                                           function=pnl.Logistic(bias=4.0),
                                                    hetero=-2,
                                           integrator_mode=True,
                                          # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
                                           smoothing_factor=0.1,
                                           name='WORDS HIDDEN')
#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(size=2,  #Recurrentdefault_variable=np.array([[3.1, 3.1]]),
                                                function=pnl.Logistic(),
                                                hetero=-2.0,
                                                integrator_mode=True,
                                                smoothing_factor=0.1,
                                                name='RESPONSE')

# Log mechanisms ------------------------------------------------------------------------------------------------------
#task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
task_layer.set_log_conditions('InputState-0')

colors_hidden_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('InputState-0')

words_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('InputState-0')

response_layer.set_log_conditions('value')
response_layer.set_log_conditions('InputState-0')
# Connect mechanisms --------------------------------------------------------------------------------------------------
# (note that response layer projections are set to all zero first for initialization

color_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                             [0.0, 1.0, 0.0],
                                                             [0.0, 0.0, 0.0]]))

word_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0],
                                                            [0.0, 0.0, 0.0]]))

task_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))

color_task_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 0.0],
                                                             [4.0, 0.0],
                                                             [4.0, 0.0]]))

task_color_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 4.0, 4.0],
                                                             [0.0, 0.0, 0.0]]))

word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                           [0.0, 4.0],
                                                           [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                           [4.0, 4.0, 4.0]]))

# color_response_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
#                                    [0.0, 0.0],
#                                    [0.0, 0.0]]))
#
# word_response_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
#                                    [0.0, 0.0],
#                                    [0.0, 0.0]]))
#
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))
#
# response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
#                                    [0.0, 1.5, 0.0]]))
#
# response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
#                                    [0.0, 2.5, 0.0]]))

color_response_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
                                   [0.0, 1.5],
                                   [0.0, 0.0]]))
word_response_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
                                   [0.0, 2.5],
                                   [0.0, 0.0]]))
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

# 2nd threshold function
def pass_threshold2(response_layer, thresh, terminate_trial):
    results1 = response_layer.output_states.values[0][0] #red response
    results2 = response_layer.output_states.values[0][1] #green response
    length = response_layer.log.nparray_dictionary()['value'].shape[0]
    # print(results1)
    # print(results2)
    if results1  >= thresh or results2 >= thresh or length ==terminate_trial:
        return True
    return False


terminate_trial = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)
}

terminate_trial2 = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold2, response_layer, threshold, terminate2)
}

terminate_trial3 = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold2, response_layer, threshold, terminate3)
}

terminate_trial4 = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold2, response_layer, threshold, terminate4)
}
# Create test trials function -----------------------------------------------------------------------------------------
# a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a blue color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]
def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):

    trialdict = {
    colors_input_layer: [red_color, green_color, neutral_color],
    words_input_layer: [red_word, green_word, neutral_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
WR_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 1)
CN_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)

CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_congruent_word_first_input = trial_dict(0, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_incongruent_word_first_input = trial_dict(0, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR

WR_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
WR_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR


# ---------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 1200 trials prior ------------------------------------------------------------
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_congruent_word_first_input,num_trials=prior1200)# termination_processing=terminate_trial) # run system with congruent stimulus input until


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_neg1200 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]
ccc_cong_neg1200 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_neg1200 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_neg1200 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


# ---------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 1000 trials prior ------------------------------------------------------------
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_congruent_word_first_input,num_trials=prior1000)# termination_processing=terminate_trial) # run system with congruent stimulus input until


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_neg1000 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]
ccc_cong_neg1000 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_neg1000 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_neg1000 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


# ---------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 800 trials prior ------------------------------------------------------------
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior400)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_congruent_word_first_input,num_trials=prior800)# termination_processing=terminate_trial) # run system with congruent stimulus input until


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_neg800 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]
ccc_cong_neg800 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_neg800 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_neg800 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


# ---------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 600 trials prior ------------------------------------------------------------
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior600)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_congruent_word_first_input,num_trials=prior600)# termination_processing=terminate_trial) # run system with congruent stimulus input until


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_neg600 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]
ccc_cong_neg600 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_neg600 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_neg600 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


# ---------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 400 trials prior ------------------------------------------------------------
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior800)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_congruent_word_first_input,num_trials=prior400)# termination_processing=terminate_trial) # run system with congruent stimulus input until


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_neg400 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]
ccc_cong_neg400 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_neg400 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_neg400 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


# Run congruent trial with word presented 200 trials prior ------------------------------------------------------------
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1000)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))


Bidirectional_Stroop.run(inputs=CN_congruent_word_first_input,num_trials=prior200)# termination_processing=terminate_trial) # run system with congruent stimulus input until


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_neg200 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]
ccc_cong_neg200 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_neg200 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_neg200 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])

#----------------------------------------------------------------------------------------------------------------------
# Run congruent trials with word and color presented at the same time -------------------------------------------------
# Set input projections back to 0 for settling period
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))


Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong0 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]

ccc_cong0 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong0 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong0 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])

#----------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 200 trials after color ------------------------------------------------------
# Set input projections back to 0 for settling period
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_control_trial_input,num_trials=after200)# termination_processing=terminate_trial) # run system with congruent stimulus input until

Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_pos200 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]

ccc_cong_pos200 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_pos200 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_pos200 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])

#----------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 400 trials after color ------------------------------------------------------
# Set input projections back to 0 for settling period
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_control_trial_input,num_trials=after400)# termination_processing=terminate_trial) # run system with congruent stimulus input until

Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_pos400 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]

ccc_cong_pos400 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_pos400 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_pos400 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])

#----------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 600 trials after color ------------------------------------------------------
# Set input projections back to 0 for settling period
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_control_trial_input,termination_processing=terminate_trial2)# termination_processing=terminate_trial) # run system with congruent stimulus input until

Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_pos600 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]

ccc_cong_pos600 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_pos600 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_pos600 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


#----------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 800 trials after color ------------------------------------------------------
# Set input projections back to 0 for settling period
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_control_trial_input,termination_processing=terminate_trial3)# termination_processing=terminate_trial) # run system with congruent stimulus input until

Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_pos800 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]

ccc_cong_pos800 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_pos800 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_pos800 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


#----------------------------------------------------------------------------------------------------------------------
# Run congruent trial with word presented 1000 trials after color ------------------------------------------------------
# Set input projections back to 0 for settling period
response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
Bidirectional_Stroop.run(inputs=CN_initialize_input, num_trials=prior1200)    # run system to settle for 200 trials with congruent stimuli input

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

Bidirectional_Stroop.run(inputs=CN_control_trial_input,termination_processing=terminate_trial4)# termination_processing=terminate_trial) # run system with congruent stimulus input until

Bidirectional_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                # threshold in of of the response layer units is reached

# Store values from run -----------------------------------------------------------------------------------------------
t = task_layer.log.nparray_dictionary('value')    # Log task output from special logistic function
tt = t['value']
n_con = tt.shape[0]
ttt_cong_pos1000 = tt.reshape(n_con,2)

c = colors_hidden_layer.log.nparray_dictionary('value')  # Log color output from special logistic function
cc = c['value']
n_con = cc.shape[0]

ccc_cong_pos1000 = cc.reshape(n_con,3)
w = words_hidden_layer.log.nparray_dictionary('value')   # Log word output from special logistic function
ww = w['value']
n_con = ww.shape[0]

www_cong_pos1000 = ww.reshape(n_con,3)
r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
rr = r['value']
n_con = rr.shape[0]

rrr_cong_pos1000 = rr.reshape(n_con,2)
# Clear log & reinitialize --------------------------------------------------------------------------------------------
response_layer.log.clear_entries(delete_entry=False)
colors_hidden_layer.log.clear_entries(delete_entry=False)
words_hidden_layer.log.clear_entries(delete_entry=False)
task_layer.log.clear_entries(delete_entry=False)

colors_hidden_layer.reinitialize([[0,0,0]])
words_hidden_layer.reinitialize([[0,0,0]])
response_layer.reinitialize([[0,0]])
task_layer.reinitialize([[0,0]])


a = np.repeat(0.55,n_con)

plt.figure()
#
# plt.plot(ttt_cong_neg400, 'k')
# plt.plot(www_cong_neg400, 'b')
# plt.plot(ccc_cong_neg400,'r')
# plt.plot(rrr_cong_neg400, 'lime')
#
# plt.plot(ttt_cong_neg200, 'k')
# plt.plot(www_cong_neg200, 'b')
# plt.plot(ccc_cong_neg200,'r')
# plt.plot(rrr_cong_neg200, 'lime')
#
# plt.plot(ttt_cong0, 'k')
# plt.plot(www_cong0, 'b')
# plt.plot(ccc_cong0,'r')
# plt.plot(rrr_cong0, 'pink')
# #
# plt.plot(ttt_cong_pos200, 'k')
# plt.plot(www_cong_pos200, 'b')
# plt.plot(ccc_cong_pos200,'r')
# plt.plot(rrr_cong_pos200, 'g')
#
# plt.plot(ttt_cong_pos400, 'k')
# plt.plot(www_cong_pos400, 'b')
# plt.plot(ccc_cong_pos400,'r')
# plt.plot(rrr_cong_pos400, 'g')
#
# plt.plot(ttt_cong_pos600, 'k')
# plt.plot(www_cong_pos600, 'b')
# plt.plot(ccc_cong_pos600,'r')
# plt.plot(rrr_cong_pos600, 'g')
# #
# plt.plot(ttt_cong_pos800, 'k')
# plt.plot(www_cong_pos800, 'b')
# plt.plot(ccc_cong_pos800,'r')
# plt.plot(rrr_cong_pos800, 'g')
# # #
# plt.plot(ttt_cong_pos1000, 'k')
# plt.plot(www_cong_pos1000, 'b')
# plt.plot(ccc_cong_pos1000,'r')
# plt.plot(rrr_cong_pos1000, 'g')
# plt.plot(a, 'aqua')

responses = [rrr_cong_neg1200.shape[0],rrr_cong_neg1000.shape[0], rrr_cong_neg800.shape[0], rrr_cong_neg600.shape[0],rrr_cong_neg400.shape[0], rrr_cong_neg200.shape[0], rrr_cong0.shape[0], rrr_cong_pos200.shape[0], rrr_cong_pos400.shape[0], rrr_cong_pos600.shape[0],rrr_cong_pos800.shape[0], rrr_cong_pos1000.shape[0]]
plt.plot(responses)
plt.show()

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