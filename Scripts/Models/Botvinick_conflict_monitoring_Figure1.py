import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# This script implements Figure 1 of Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001).
# Conflict monitoring and cognitive control. Psychological Review, 108, 624–652.
# http://dx.doi.org/10.1037/0033-295X.108.3.624

# Figure 1 plots the ENERGY computed by a conflict mechanism. It is highest for incongruent trials,
# and similar for congruent and neutral trials.
# Noise is turned of and for each condition we ran one trial only. A response threshold was not defined. Responses were
# made at the marked * signs in the figure.
# Note that this script implements a slightly different Figure than in the original Figure in the paper.
# However, this implementation is identical with a plot we created with an old MATLAB code which was used for the
# conflict monitoring simulations.

# SET UP MECHANISMS ----------------------------------------------------------------------------------------------------
# Linear input layer
# colors: ('red', 'green'), words: ('RED','GREEN')
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
                                            integration_rate=0.01,
                                            name='TASK_LAYER')

# Hidden layer
# colors: ('red','green', 'neutral') words: ('RED','GREEN', 'NEUTRAL')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                     function=pnl.Logistic(bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                                     integrator_mode=True,
                                                     hetero=-2,
                                                     integration_rate=0.01, # cohen-huston text says 0.01
                                                     name='COLORS_HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                    function=pnl.Logistic(bias=4.0),
                                                    integrator_mode=True,
                                                    hetero=-2,
                                                    integration_rate=0.01,
                                                    name='WORDS_HIDDEN')

#   Response layer, responses: ('red', 'green')
response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                function=pnl.Logistic(),
                                                hetero=-2.0,
                                                integrator_mode=True,
                                                integration_rate=0.01,
                                                output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                                                 {pnl.NAME: 'DECISION_ENERGY',
                                                                  pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                                  pnl.FUNCTION: pnl.Stability(
                                                                      default_variable = np.array([0.0, 0.0]),
                                                                      metric = pnl.ENERGY,
                                                                      matrix = np.array([[0.0, -4.0],
                                                                                        [-4.0, 0.0]]))}],
                                                name='RESPONSE',)

# Log ------------------------------------------------------------------------------------------------------------------
response_layer.set_log_conditions('DECISION_ENERGY')

# Mapping projections---------------------------------------------------------------------------------------------------

color_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                             [0.0, 1.0, 0.0],
                                                             [0.0, 0.0, 1.0]]))

word_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0],
                                                            [0.0, 0.0, 1.0]]))

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

word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                           [0.0, 4.0],
                                                           [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                           [4.0, 4.0, 4.0]]))

# CREATE PATHWAYS -----------------------------------------------------------------------------------------------------
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
                                        reinitialize_mechanisms_when=pnl.Never(),
                                        name='CONFLICT MONITORING_SYSTEM')

response_layer._add_system(System_Conflict_Monitoring, pnl.TERMINAL)
System_Conflict_Monitoring.terminal_mechanisms.append(response_layer)

# System_Conflict_Monitoring.show_graph(show_dimensions=pnl.ALL)#, show_mechanism_structure=True)

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
    trialdict = {
    colors_input_layer: [red_color, green_color, neutral_color],
    words_input_layer: [red_word, green_word, neutral_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)#red_color, green color, red_word, green word, CN, WR
CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0)   #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 1, 1, 0)     #red_color, green color, red_word, green word, CN, WR

Stimulus = [[CN_trial_initialize_input, CN_congruent_trial_input],
            [CN_trial_initialize_input, CN_incongruent_trial_input],
            [CN_trial_initialize_input, CN_control_trial_input]]

ntrials0 = 500
ntrials = 1000
condition = 3
for cond in range(condition):
# RUN the SYSTEM to initialize ----------------------------------------------------------------------------------------
    System_Conflict_Monitoring.run(inputs=Stimulus[cond][0], num_trials=ntrials0)   #run System with initial input
    System_Conflict_Monitoring.run(inputs=Stimulus[cond][1], num_trials=ntrials)    #run System with condition input
    # reinitialize System after condition was run
    colors_hidden_layer.reinitialize([[0,0,0]])
    words_hidden_layer.reinitialize([[0,0,0]])
    response_layer.reinitialize([[0,0]])
    task_layer.reinitialize([[0,0]])

####------- PLOTTING  -------------------------------------------------------------------------------------------------
# Plot energy figure
r2 = response_layer.log.nparray_dictionary('DECISION_ENERGY') #get logged DECISION_ENERGY dictionary
energy = r2['DECISION_ENERGY']                                #save logged DECISION_ENERGY

plt.figure()
x = np.arange(0,1500,1)             # create x-axis length
plt.plot(x, energy[:1500], 'r')     # plot congruent condition
plt.plot(x, energy[1500:3000], 'b') # plot incongruent condition
plt.plot(x, energy[3000:4500], 'g') # plot neutral condition
plt.ylabel('ENERGY')                # add ylabel
plt.xlabel('cycles')                # add x label
legend = ['congruent', 'incongruent', 'neutral']
plt.legend(legend)
plt.show()


