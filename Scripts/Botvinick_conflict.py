import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# LAMBDA = 0.95
# alpha = 11.24
# beta = 9.46

#Conflict  equation:

#C(t+1) = LAMBDA*C(t) +(1-LAMBDA) * (alpha*ENERGY(t) + beta)

# SET UP MECHANISMS
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
#change from Linear to Logistic with control on. Linear for saniti checks
task_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[0, 0]]),
                                            function=pnl.Logistic(bias=1.0),#gain=(1.0, pnl.ControlProjection())),
                                            size=2,
                                            hetero=-2,
                                            integrator_mode=True,
                                            smoothing_factor=0.01,
                                   # function=pnl.Logistic(gain=(1.0, pnl.ControlProjection())),
                                   name='TASK')


#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[3, 3, 3]]),
                                                     size=3,
                                            function=pnl.Logistic(bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                            integrator_mode=True,
                                                     hetero=-2,
                                           # noise=pnl.NormalDist(mean=0.0, standard_dev=.0).function,
                                            smoothing_factor=0.01, # cohen-huston text says 0.01
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[1, 1, 1]]),
                                                    size=3,
                                           function=pnl.Logistic(bias=4.0),
                                                    hetero=-2,
                                           integrator_mode=True,
                                          # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
                                           smoothing_factor=0.01,
                                           name='WORDS HIDDEN')

#Log:
#task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
task_layer.set_log_conditions('InputState-0')

colors_hidden_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('InputState-0')

words_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('InputState-0')

#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(initial_value= np.array([[0.0, 0.0]]),
                                                size=2,  #Recurrentdefault_variable=np.array([[3.1, 3.1]]),
                         function=pnl.Logistic(),
                                                hetero=-2.0,
                                                # auto=1,
                         name='RESPONSE',
                         output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                          {pnl.NAME: 'DECISION_ENERGY',
                                           pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                           pnl.FUNCTION: pnl.Stability(default_variable = np.array([0.0, 0.0]),
                                                                       metric = pnl.ENERGY,
                                                                       # transfer_fct=,
                                                                       matrix = np.array([[0.0, -2.0],
                                                                                        [-2.0, 0.0]]))}],
                                                integrator_mode=True,
                         # noise=pnl.NormalDist(mean=0.0, standard_dev=.05).function,
                         smoothing_factor=0.01)


response_layer.set_log_conditions('value')
response_layer.set_log_conditions('InputState-0')


response_layer.set_log_conditions('DECISION_ENERGY')
# response_layer.set_log_conditions('gain')

# energy_layer = pnl.ProcessingMechanism(size=2,
#                                      function= pnl.Stability(default_variable = np.array([0.0, 0.0]),
#                                                                        metric = pnl.ENERGY,
#                                                                        # transfer_fct=,
#                                                                        matrix = np.array([[0.0],
#                                                                                         [-4.0]])),#pnl.Linear(slope=2.0),
#                                      name='ENERGY')

# energy_layer.set_log_conditions('value')
# variable_scale = pnl.TransferMechanism(function = pnl.Linear(slope = 11.24,
#                                                   intercept= 9.46),
#                                        name='SCALING VARIABLE')
#
# conflict = pnl.ProcessingMechanism(function=pnl.AdaptiveIntegrator(rate = 1-LAMBDA),
#                                    name='CONFLICT COMPUTATION')
#
# conflict.set_log_conditions('value')

#####-------- Mapping projections (note that response layer projections are set to all zero first for initialization

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

# response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
#                                    [0.0, 1.5, 0.0]]))
#
# response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
#                                    [0.0, 2.5, 0.0]]))
#
# color_response_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
#                                    [0.0, 1.5],
#                                    [0.0, 0.0]]))
# word_response_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
#                                    [0.0, 2.5],
#                                    [0.0, 0.0]]))

# # #
color_response_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
                                   [0.0, 0.0],
                                   [0.0, 0.0]]))

word_response_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0],
                                   [0.0, 0.0],
                                   [0.0, 0.0]]))

response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]))

word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                           [0.0, 4.0],
                                                           [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                           [4.0, 4.0, 4.0]]))


#   CREATE PATHWAYS
color_response_process = pnl.Process(pathway=[colors_input_layer,
                                              color_input_weights,
                                              colors_hidden_layer,
                                              color_response_weights,
                                              response_layer,
                                              response_color_weights,
                                              colors_hidden_layer],
                                     name='COLORS_RESPONSE_PROCESS')

# color_control_process = pnl.Process(pathway=[colors_hidden_layer,
#                                              color_task_weights,
#                                              task_layer],
#                                     name='COLORS_TASK_PROCESS')

word_response_process = pnl.Process(pathway=[words_input_layer,
                                             word_input_weights,
                                             words_hidden_layer,
                                             word_response_weights,
                                             response_layer,
                                             response_word_weights,
                                             words_hidden_layer],
                                     name='WORDS_RESPONSE_PROCESS')

# word_control_process = pnl.Process(pathway=[words_hidden_layer,
#                                             word_task_weights,
#                                             task_layer],
#                                    name='WORDS_TASK_PROCESS')

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

# response_color_task_process = pnl.Process(pathway=[response_layer,
#                                                    response_color_weights,
#                                                    colors_hidden_layer])
#
# response_word_task_process = pnl.Process(pathway=[response_layer,
#                                                   response_word_weights,
#                                                   words_hidden_layer])
# response_conflict_process = pnl.Process(pathway=[response_layer]),
                                                 # variable_scale,
                                                 # conflict])

#   CREATE SYSTEM
System_Conflict_Monitoring = pnl.System(processes=[color_response_process,
                                                   word_response_process,
                                                   task_color_response_process,
                                                   task_word_response_process],
                      # controller=pnl.ControlMechanism,
                      #  monitor_for_control=[conflict],
                      #  enable_controller=True,
                       name='FEEDFORWARD_STROOP_SYSTEM')


# LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')

#   CREATE THRESHOLD FUNCTION
#first value of DDM's value is DECISION_VARIABLE
# def pass_threshold(response_layer, thresh):
#     results1 = response_layer.output_states.values[0][0] #red response
#     results2 = response_layer.output_states.values[0][1] #green response
#     # print(results1)
#     # print(results2)
#     if results1  >= thresh or results2 >= thresh:
#         return True
#     # for val in results1:
#     #     if val >= thresh:
#     #         return True
#     return False
# accumulator_threshold = 0.6
#
# terminate_trial = {
#    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, accumulator_threshold)
# }

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):

    trialdict = {
    colors_input_layer: [red_color, green_color, neutral_color],
    words_input_layer: [red_word, green_word, neutral_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
# input just task and run once so system asymptotes
WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 1)
CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0.7, 0)

# Initialize System:
# colors_hidden_layer.integrator_mode = False
# words_hidden_layer.integrator_mode = False
# task_layer.integrator_mode = False
# response_layer.integrator_mode = False

# RUN SYSTEM INITIALIZATION:
# System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, termination_processing=terminate_initialization)
# colors_hidden_layer.reinitialize([[0, 0]])
# words_hidden_layer.reinitialize([[0, 0]])


print('colors_hidden_layer after initial trial: ', colors_hidden_layer.output_states.values)
print('words_hidden_layer after initial trial: ', words_hidden_layer.output_states.values)
print('response_layer after initial trial: ', response_layer.output_states.values)
print('task_layer after initial trial: ', task_layer.output_states.values)

# response_layer.integrator_mode = True
# colors_hidden_layer.integrator_mode = True
# words_hidden_layer.integrator_mode = True
# task_layer.integrator_mode = True

# response_layer.reinitialize([[0, 0]])
print('response_layer after reinitialization trial: ', response_layer.output_states.values)

# System_Conflict_Monitoring.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)

###--------------------------------------
# RUN the SYSTEM to initialize:
ntrials0 = 200
System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, num_trials=ntrials0)# termination_processing=terminate_trial)

###------------------------------------

# System_Conflict_Monitoring.show_graph(show_dimensions=pnl.ALL, show_mechanism_structure=True)

# response_layer.output_states[1].function_object.matrix =np.array([[0, -4], [-4, 0]])

response_layer.output_states[0].value = np.array([0,0])
# response_layer.output_states[1].value = [0]

# response_layer.reinitialize([[-1, -1]])
# response_layer.initial_value = np.array([[0.0, 0.0]])
# response_layer.hetero = np.array([[0.0, -2.0], [-2.0, 0.0]])
# task_layer.hetero = np.array([[0.0, -1.0,], [-1.0, 0.0]])

#####----------- Turn on projections to response layer and from response layer since response layer is silenced in initialization phase

response_color_weights.matrix = np.array([[1.5, 0.0, 0.0],
                                   [0.0, 1.5, 0.0]])

response_word_weights.matrix  = np.array([[2.5, 0.0, 0.0],
                                   [0.0, 2.5, 0.0]])
#
color_response_weights.matrix = np.array([[1.5, 0.0],
                                   [0.0, 1.5],
                                   [0.0, 0.0]])
word_response_weights.matrix  = np.array([[2.5, 0.0],
                                   [0.0, 2.5],
                                   [0.0, 0.0]])

####----------- RUN SYSTEM for CONGRUENT trial input
### ----------- Conflict monitoring paper only used color naming (CN) trials

CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR

# WR_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# WR_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR

ntrials = 0
System_Conflict_Monitoring.run(inputs=CN_congruent_trial_input, num_trials=ntrials)# termination_processing=terminate_trial)
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