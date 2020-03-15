import matplotlib.pyplot as plt
import numpy as np
import psyneulink as pnl

# This implements the horse race Figure shown in Cohen & Huston (1994).
# Note that noise is turned off and each stimulus is only showed once for each stimulus onset asynchrony.

# Define Variables ----------------------------------------------------------------------------------------------------
import psyneulink.core.components.functions.transferfunctions

rate = 0.1          # The integration rate was changed from 0.01 to 0.1
inhibition = -2.0   # Mutual inhibition across each layer
bias = 4.0          # bias for hidden layer units
threshold = 0.55    # Threshold until a response is made, changed from 0.6 to 0.55
settle_trials = 50  # Time for system to initialize and settle
prior120 = 100      # Cycles needed to be added for stimulus to start

# Different time steps at which the System should end run and start new terminate_processing run
# This is needed for conditions in which the irrelevant condition is like a neutral trial and could already lead to
# a correct response. This is basically the reason why with long positive stimulus onset asynchrony the three
# condistions (congruent, incongruent, neutral) lead to the same reaction time.
terminate2 = 180
terminate3 = 200
terminate4 = 220
terminate5 = 240

# Create mechanisms ---------------------------------------------------------------------------------------------------
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(size=3,
                                           function=psyneulink.core.components.functions.transferfunctions.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=3,
                                          function=psyneulink.core.components.functions.transferfunctions.Linear,
                                          name='WORDS_INPUT')

task_input_layer = pnl.TransferMechanism(size=2,
                                         function=psyneulink.core.components.functions.transferfunctions.Linear,
                                         name='TASK_INPUT')

#   Task layer, tasks: ('name the color', 'read the word')
task_layer = pnl.RecurrentTransferMechanism(size=2,
                                            function=psyneulink.core.components.functions.transferfunctions.Logistic(),
                                            hetero=-2,
                                            integrator_mode=True,
                                            integration_rate=0.1,
                                            name='TASK')

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                     function=psyneulink.core.components.functions.transferfunctions
                                                     .Logistic(x_0=4.0),
                                                     integrator_mode=True,
                                                     hetero=-2.0,
                                                     # noise=pnl.NormalDist(mean=0.0, standard_deviation=.0).function,
                                                     integration_rate=0.1,  # cohen-huston text says 0.01
                                                     name='COLORS HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                    function=psyneulink.core.components.functions.transferfunctions.Logistic(x_0=4.0),
                                                    hetero=-2,
                                                    integrator_mode=True,
                                                    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.05).function,
                                                    integration_rate=0.1,
                                                    name='WORDS HIDDEN')
#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                function=psyneulink.core.components.functions.transferfunctions.Logistic(),
                                                hetero=-2.0,
                                                integrator_mode=True,
                                                integration_rate=0.1,
                                                name='RESPONSE')

# Log mechanisms ------------------------------------------------------------------------------------------------------
#task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
response_layer.set_log_conditions('value')
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

response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0]]))

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

                                  reinitialize_mechanisms_when=pnl.Never(),
                                  name='FEEDFORWARD_STROOP_SYSTEM')

# LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')

Bidirectional_Stroop.show()
# Bidirectional_Stroop.show_graph(show_dimensions=pnl.ALL)#,show_mechanism_structure=pnl.VALUES) # Uncomment to show graph of the system

# Create threshold function -------------------------------------------------------------------------------------------
# context is automatically passed into Conditions, and references the execution context in which they are being run,
# which in this case is simply the Bidirectional_Stroop system
def pass_threshold(response_layer, thresh, context):
    results1 = response_layer.get_output_values(context)[0][0] #red response
    results2 = response_layer.get_output_values(context)[0][1] #green response
    if results1  >= thresh or results2 >= thresh:
        return True
    return False

# 2nd threshold function
def pass_threshold2(response_layer, thresh, terminate, context):
    results1 = response_layer.get_output_values(context)[0][0] #red response
    results2 = response_layer.get_output_values(context)[0][1] #green response
    length = response_layer.log.nparray_dictionary()[context.execution_id]['value'].shape[0]
    if results1  >= thresh or results2 >= thresh:
        return True
    if length ==terminate:
        return True
    return False

# Create different terminate trial conditions --------------------------------------------------------------------------
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
terminate_trial5 = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold2, response_layer, threshold, terminate5)
}

terminate_list = [terminate_trial2,
                  terminate_trial3,
                  terminate_trial4,
                  terminate_trial5]

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
# WR_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 1)
CN_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)

CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_control_word_trial_input = trial_dict(0, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR

CN_congruent_word_first_input = trial_dict(0, 0, 0, 1, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
CN_incongruent_word_first_input = trial_dict(0, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR

# WR_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# WR_control_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR

conditions = 3
runs = 5
runs2 = 4
response_all = []
response_all2 = []

Stimulus = [[CN_initialize_input, CN_congruent_word_first_input, CN_congruent_trial_input, CN_control_trial_input],
            [CN_initialize_input, CN_incongruent_word_first_input, CN_incongruent_trial_input, CN_control_trial_input],
            [CN_initialize_input, CN_control_word_trial_input, CN_control_trial_input, CN_control_trial_input]]


# First "for loop" over conditions
# Second "for loop" over runs
for cond in range(conditions):
# ---------------------------------------------------------------------------------------------------------------------
    # Run congruent trial with word presented 1200 trials prior ------------------------------------------------------------
    for run in range(runs):
        response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                        [0.0, 0.0, 0.0]]))

        response_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                       [0.0, 0.0, 0.0]]))
        Bidirectional_Stroop.run(inputs=Stimulus[cond][0], num_trials=settle_trials)    # run system to settle for 200 trials with congruent stimuli input
        Bidirectional_Stroop.run(inputs=Stimulus[cond][0], num_trials=20 * (run))  # run system to settle for 200 trials with congruent stimuli input

        response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                        [0.0, 1.5, 0.0]]))

        response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                        [0.0, 2.5, 0.0]]))

        Bidirectional_Stroop.run(inputs=Stimulus[cond][1], num_trials=prior120 - (run * 20))# termination_processing=terminate_trial) # run system with congruent stimulus input until
        Bidirectional_Stroop.run(inputs=Stimulus[cond][2], termination_processing=terminate_trial) # run system with congruent stimulus input until
                                                                    # threshold in of of the response layer units is reached

    # Store values from run -----------------------------------------------------------------------------------------------
        r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
        rr = r[Bidirectional_Stroop.name]['value']
        n_r = rr.shape[0]
        rrr = rr.reshape(n_r,2)

        response_all.append(rrr.shape[0])

        # Clear log & reinitialize ----------------------------------------------------------------------------------------
        response_layer.log.clear_entries()
        colors_hidden_layer.log.clear_entries()
        words_hidden_layer.log.clear_entries()
        task_layer.log.clear_entries()

        colors_hidden_layer.reinitialize([[0, 0, 0]], context=Bidirectional_Stroop)
        words_hidden_layer.reinitialize([[0, 0, 0]], context=Bidirectional_Stroop)
        response_layer.reinitialize([[0, 0]], context=Bidirectional_Stroop)
        task_layer.reinitialize([[0, 0]], context=Bidirectional_Stroop)

    print('response_all: ', response_all)

    # Run trials after congruent color was presented ----------------------------------------------------------------------
    for run2 in range(runs2):
        response_color_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                        [0.0, 0.0, 0.0]]))
        response_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                                       [0.0, 0.0, 0.0]]))
        Bidirectional_Stroop.run(inputs=Stimulus[cond][0], num_trials = settle_trials)  # run system to settle for 200 trials with congruent stimuli input
        Bidirectional_Stroop.run(inputs=Stimulus[cond][0], num_trials = prior120)  # run system to settle for 200 trials with congruent stimuli input
        response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                        [0.0, 1.5, 0.0]]))
        response_word_weights = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                       [0.0, 2.5, 0.0]]))
        Bidirectional_Stroop.run(inputs=Stimulus[cond][3], termination_processing=terminate_list[run2])#terminate_list[run2])  # termination_processing=terminate_trial) # run system with congruent stimulus input until

        Bidirectional_Stroop.run(inputs=Stimulus[cond][2], termination_processing=terminate_trial)  # run system with congruent stimulus input until
        # threshold in of of the response layer units is reached
        # Store values from run -----------------------------------------------------------------------------------------------
        r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
        rr = r[Bidirectional_Stroop.name]['value']
        n_r = rr.shape[0]
        rrr = rr.reshape(n_r,2)
        response_all.append(rrr.shape[0])
        # Clear log & reinitialize ------------------------------------------------------------------------------------
        response_layer.log.clear_entries()
        colors_hidden_layer.log.clear_entries()
        words_hidden_layer.log.clear_entries()
        task_layer.log.clear_entries()
        colors_hidden_layer.reinitialize([[0, 0, 0]], context=Bidirectional_Stroop)
        words_hidden_layer.reinitialize([[0, 0, 0]], context=Bidirectional_Stroop)
        response_layer.reinitialize([[0, 0]], context=Bidirectional_Stroop)
        task_layer.reinitialize([[0, 0]], context=Bidirectional_Stroop)

# Plotting ------------------------------------------------------------------------------------------------------------
#compute regression for model
reg = np.dot(response_all, 2) + 123
plt.figure()
# plt.plot(response_all[0:9])
# plt.plot(response_all[9:18])
# plt.plot(response_all[18:27])

stimulus_onset_asynchrony = np.linspace(-400,400,9)
plt.plot(stimulus_onset_asynchrony, reg[0:9], '-^')
plt.plot(stimulus_onset_asynchrony, reg[9:18], '-s')
plt.plot(stimulus_onset_asynchrony, reg[18:27], '-o')
plt.title('stimulus onset asynchrony - horse race model ')
plt.legend(['congruent', 'incongruent', 'neutral'])
plt.ylabel('reaction time in ms')
plt.show()
