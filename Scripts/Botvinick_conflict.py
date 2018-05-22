import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# This script implements Figure 1 of Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001).
# Conflict monitoring and cognitive control. Psychological Review, 108, 624–652.
# http://dx.doi.org/10.1037/0033-295X.108.3.624

# Figure 1 shows that for incongruent trials the ENERGY computed by a conflict mechanism is highest for incongruent
# trials and similar for congruent and neutral trials.
# This script implements a similar Figure than the original one in the paper. The main difference between the Figure
# implemented here, compared to the original one is that in the original version the units in the response layer got
# silenced with an input of -1. However, the the version implemented here is identical with a version MATLAB
# produced.

# Set Initial Values

threshold = 0.6
rate = 0.01


# SET UP MECHANISMS ----------------------------------------------------------------------------------------------------
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
                                            function=pnl.Logistic(bias=0),
                                            hetero=-2,
                                            integrator_mode=True,
                                            smoothing_factor=0.01,
                                            name='TASK_LAYER')

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                     function=pnl.Logistic(bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                                     integrator_mode=True,
                                                     hetero=-2,
                                                     # noise=pnl.NormalDist(mean=0.0, standard_dev=.0).function,
                                                     smoothing_factor=0.01, # cohen-huston text says 0.01
                                                     name='COLORS_HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                    function=pnl.Logistic(bias=4.0),
                                                    integrator_mode=True,
                                                    hetero=-2,
                                                    # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
                                                    smoothing_factor=0.01,
                                                    name='WORDS_HIDDEN')

#   Response layer, responses: ('red', 'green')
response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                function=pnl.Logistic(),
                                                hetero=-2.0,
                                                integrator_mode=True,
                                                # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
                                                smoothing_factor=0.01,
                                                output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                                                 {pnl.NAME: 'DECISION_ENERGY',
                                                                  pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                                  pnl.FUNCTION: pnl.Stability(
                                                                      default_variable = np.array([0.0, 0.0]),
                                                                      metric = pnl.ENERGY,
                                                                      matrix = np.array([[0.0, -4.0],
                                                                                        [-4.0, 0.0]]))}],
                                                name='RESPONSE',)

#Log ------------------------------------------------------------------------------------------------------------------
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
response_layer.set_log_conditions('value')
response_layer.set_log_conditions('DECISION_ENERGY')

#####-------- Mapping projections (note that response layer projections are set to all zero first for initialization

color_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                             [0.0, 1.0, 0.0],
                                                             [0.0, 0.0, 0.0]]))
2
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

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

color_response_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
                                                                [0.0, 1.5],
                                                                [0.0, 0.0]]))
word_response_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
                                                                [0.0, 2.5],
                                                                [0.0, 0.0]]))
#
# color_response_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
#                                                                 [0.0, 0.0],
#                                                                 [0.0, 0.0]]))
#
# word_response_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
#                                                                 [0.0, 0.0],
#                                                                 [0.0, 0.0]]))
#
# response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
#                                                                 [0.0, 0.0, 0.0]]))
#
# response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
#                                                                 [0.0, 0.0, 0.0]]))

word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                           [0.0, 4.0],
                                                           [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                           [4.0, 4.0, 4.0]]))


# CREATE PATHWAYS ---------------------------------------------------------------------------------------------------
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

# CREATE SYSTEM -------------------------------------------------------------------------------------------------------
System_Conflict_Monitoring = pnl.System(processes=[color_response_process,
                                                   word_response_process,
                                                   task_color_response_process,
                                                   task_word_response_process],
                                        name='CONFLICT MONITORING_SYSTEM')

# System_Conflict_Monitoring.show_graph(show_dimensions=pnl.ALL, show_mechanism_structure=True)

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
    trialdict = {
    colors_input_layer: [red_color, green_color, neutral_color],
    words_input_layer: [red_word, green_word, neutral_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 1)
CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1.0, 0)

###
# RUN the SYSTEM to initialize ----------------------------------------------------------------------------------------
ntrials0 = 500
System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, num_trials=ntrials0)# termination_processing=terminate_trial)

###------------------------------------


# response_layer.output_states[1].function_object.matrix =np.array([[0, -4], [-4, 0]])

# response_layer.output_states[0].value = np.array([0,0])
# response_layer.output_states[1].value = [0]

# response_layer.reinitialize([[-1, -1]])
# response_layer.initial_value = np.array([[0.0, 0.0]])
# response_layer.hetero = np.array([[0.0, -2.0], [-2.0, 0.0]])
# task_layer.hetero = np.array([[0.0, -1.0,], [-1.0, 0.0]])

# Turn on projections to response layer and from response layer since response layer is silenced in initialization phase

# response_color_weights.matrix = np.array([[1.5, 0.0, 0.0],
#                                           [0.0, 1.5, 0.0]])
# response_word_weights.matrix  = np.array([[2.5, 0.0, 0.0],
#                                           [0.0, 2.5, 0.0]])
#
# color_response_weights.matrix = np.array([[1.5, 0.0],
#                                           [0.0, 1.5],
#                                           [0.0, 0.0]])
# word_response_weights.matrix  = np.array([[2.5, 0.0],
#                                           [0.0, 2.5],
#                                           [0.0, 0.0]])

####----------- RUN SYSTEM for CONGRUENT trial input
### ----------- Conflict monitoring paper only used color naming (CN) trials

# CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR

CN_incongruent_trial_input = trial_dict(1, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5) #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5) #red_color, green color, red_word, green word, CN, WR


# WR_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# WR_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR

ntrials = 00
System_Conflict_Monitoring.run(inputs=CN_incongruent_trial_input, num_trials=ntrials)# termination_processing=terminate_trial)
# # #####---------- RUN SYSTEM for INCONGRUENT trial
#
# # reinitialize layers
#
# colors_hidden_layer.reinitialize([[0,0,0]])
# words_hidden_layer.reinitialize([[0,0,0]])
# response_layer.reinitialize([[0,0]])
# task_layer.reinitialize([[0,0]])
# response_layer.hetero = np.array([[0.0, -3.0], [-3.0, 0.0]])
#
# response_color_weights.matrix = np.array([[0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0]])
#
# response_word_weights.matrix  = np.array([[0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0]])
# #
# color_response_weights.matrix = np.array([[0.0, 0.0],
#                                    [0.0, 0.0],
#                                    [0.0, 0.0]])
# word_response_weights.matrix  = np.array([[0.0, 0.0],
#                                    [0.0, 0.0],
#                                    [0.0, 0.0]])
#
# ##-----------------------RUN the SYSTEM to initialize:
# System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, num_trials=ntrials0)# termination_processing=terminate_trial)
#
# ###----------------------Now RUN SYSTEM for incongruent trials
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
#                                           [0.0, 1.5, 0.0]])
#
# response_word_weights.matrix  = np.array([[2.5, 0.0, 0.0],
#                                           [0.0, 2.5, 0.0]])
# #
# color_response_weights.matrix = np.array([[1.5, 0.0],
#                                           [0.0, 1.5],
#                                           [0.0, 0.0]])
# word_response_weights.matrix  = np.array([[2.5, 0.0],
#                                           [0.0, 2.5],
#                                           [0.0, 0.0]])
#
# ####----------- RUN SYSTEM for CONGRUENT trial input
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
# System_Conflict_Monitoring.run(inputs=CN_incongruent_trial_input, num_trials=ntrials)# termination_processing=terminate_trial)

####------- PLOTTING

r = response_layer.log.nparray_dictionary('value')
r2 = response_layer.log.nparray_dictionary('DECISION_ENERGY')
response = r['value']
response = response.reshape(ntrials0+ ntrials,2)
energy = r2['DECISION_ENERGY']

response = response.reshape(ntrials0+ntrials,2)
t = task_layer.log.nparray_dictionary('value')
color = colors_hidden_layer.log.nparray_dictionary('value')
c = color['value']
word = words_hidden_layer.log.nparray_dictionary('value')
w = word['value']
tt = t['value']
ttt = tt.reshape(ntrials0+ ntrials,2)
cc = c.reshape(ntrials0+ ntrials,3)
ww = w.reshape(ntrials0+ ntrials,3)

a = np.repeat(0.6,ntrials0+ ntrials)

plt.figure()
plt.plot(ttt, 'k')
plt.plot(ww, 'b')
plt.plot(cc,'r')
plt.plot(response, 'g')
plt.plot(energy, 'm')
plt.plot(a, 'aqua')
legend = ['task unit 1',
          'task unit 2',
          'word units',
          'word units',
          'word units',
          'color units',
          'color units',
          'color units',
          'response units',
          'response units',
          'energy',
          'threshold']
plt.legend(legend)
plt.show()


#This is a function that acts the same as the builtin ENERGY function below
def my_energy_function(variable):
    matrix = np.array([[0, 2],
                       [2, 0]])
    vT = np.array([[variable[0]],
                    [variable[1]]])
    e = np.matmul(matrix,vT)
    sum = np.dot(variable, e)
    return sum

print('ENERGY AFTER INITIALIZATION: ', my_energy_function([0.5, 0.5]))


###------------

# r = response_layer.log.nparray_dictionary('value')
# r2 = response_layer.log.nparray_dictionary('DECISION_ENERGY')
# response = r['value']
# response = response.reshape(ntrials0*2+ ntrials*2,2)
# energy = r2['DECISION_ENERGY']
#
# response = response.reshape(ntrials0*2+ ntrials*2,2)
# t = task_layer.log.nparray_dictionary('value')
# color = colors_hidden_layer.log.nparray_dictionary('value')
# c = color['value']
# word = words_hidden_layer.log.nparray_dictionary('value')
# w = word['value']
# tt = t['value']
# ttt = tt.reshape(ntrials0*2+ ntrials*2,2)
# cc = c.reshape(ntrials0*2+ ntrials*2,3)
# ww = w.reshape(ntrials0*2+ ntrials*2,3)
#
# a = np.repeat(0.6,ntrials0*2+ ntrials*2)
#
# plt.figure()
# #congruent log
# # plt.plot(ttt[ntrials0:ntrials0+ntrials], 'k') # task units
# # plt.plot(ww[ntrials0:ntrials0+ntrials], 'b') # word units
# # plt.plot(cc[ntrials0:ntrials0+ntrials],'r')   # color units
# plt.plot(response[ntrials0:ntrials0+ntrials], 'g')  #response units
# plt.plot(energy[ntrials0:ntrials0+ntrials], 'm')    #ENEGRY
# plt.plot(a[ntrials0:ntrials0+ntrials], 'aqua')      #threshold
#
# #incongruent log
# # plt.plot(ttt[ntrials0*2+ntrials:ntrials0*2+ntrials*2], 'darkgrey')    #task units
# # plt.plot(ww[ntrials0*2+ntrials:ntrials0*2+ntrials*2], 'lightblue')    #words
# # plt.plot(cc[ntrials0*2+ntrials:ntrials0*2+ntrials*2],'pink')          #color
# plt.plot(response[ntrials0*2+ntrials:ntrials0*2+ntrials*2], 'lime')     # response
# plt.plot(energy[ntrials0*2+ntrials:ntrials0*2+ntrials*2], 'violet')     #energy
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
#           'energy',
#           'threshold']
# plt.legend(legend)
# plt.show()
#
#
# # #This is a function that acts the same as the builtin ENERGY function below
# def my_energy_function(variable):
#     matrix = np.array([[0, 2],
#                        [2, 0]])
#     vT = np.array([[variable[0]],
#                     [variable[1]]])
#     e = np.matmul(matrix,vT)
#     sum = np.dot(variable, e)
#     return sum
#
# print('ENERGY AFTER INITIALIZATION: ', my_energy_function([0.5, 0.5]))