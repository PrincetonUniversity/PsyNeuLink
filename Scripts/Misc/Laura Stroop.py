
# coding: utf-8

# In[ ]:


import numpy as np
import psyneulink as pnl


# In[ ]:


# SET UP MECHANISMS
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
import psyneulink.core.components.functions.transferfunctions

colors_input_layer = pnl.TransferMechanism(size=2,
                                           function=psyneulink.core.components.functions.transferfunctions.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=2,
                                          function=psyneulink.core.components.functions.transferfunctions.Linear,
                                          name='WORDS_INPUT')

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')

#   Logistic activation function, Gain = 1.0, Bias = -4.0
#should be randomly distributed noise to the net input of each unit (except input unit)
#should have tau = integration_rate = 0.1
colors_hidden_layer = pnl.TransferMechanism(size=2,
                                            function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),
                                            # function=pnl.Logistic(gain=1.0, offset=-4.0),
                                            integrator_mode=False,
                                            noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0.0, standard_deviation=.01).function,
                                            integration_rate=0.1,
                                            name='COLORS HIDDEN')
#should be randomly distributed noise to the net input of each unit (except input unit)
#should have tau
words_hidden_layer = pnl.TransferMechanism(size=2,
                                           function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),
                                           # function=pnl.Logistic(gain=1.0, offset=-4.0),
                                           integrator_mode=False,
                                           noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0.0, standard_deviation=.01).function,
                                           integration_rate=0.1,
                                           name='WORDS HIDDEN')

#log hidden layer activation
colors_hidden_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('RESULT')

words_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('RESULT')

#   Task layer, tasks: ('name the color', 'read the word')
task_layer = pnl.TransferMechanism(size=2,
                                   function=psyneulink.core.components.functions.transferfunctions.Linear,
                                   name='TASK')

#   Response layer, responses: ('red', 'green')
#tau = 0.1 (here, smoothing factor)
#should be randomly distributed noise to the net input of each unit (except input unit)
response_layer = pnl.TransferMechanism(size=2,
                                       function=psyneulink.core.components.functions.transferfunctions.Logistic,
                                       name='RESPONSE',
                                       integrator_mode=True,
                                       noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0.0, standard_deviation=.01).function,
                                       integration_rate=0.1)
#   Respond red accumulator
#parameters from paper
#alpha = rate of evidence accumlation = 0.1
#sigma = noise = 0.1
#noise will be
# squareroot(time_step_size * noise) * a random sample from a normal distribution
respond_red_accumulator = pnl.IntegratorMechanism(function=psyneulink.core.components.functions.statefulfunctions.integratorfunctions
                                                  .SimpleIntegrator(noise=0.1,
                                                                    rate=0.1),
                                                  name='respond_red_accumulator')
#   Respond green accumulator
respond_green_accumulator = pnl.IntegratorMechanism(function=psyneulink.core.components.functions.statefulfunctions.integratorfunctions.SimpleIntegrator(noise=0.1,
                                                                                                                                                         rate=0.1),
                                                    name='respond_green_accumulator')

#   add logging
response_layer.set_log_conditions('value')
respond_red_accumulator.set_log_conditions('value')
respond_green_accumulator.set_log_conditions('value')


# In[ ]:


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

#   Response units to accumulators
# column 0: response_'red' to respond_red_accumulator
# column 1: response_'green' to respond_red_accumulator
respond_red_differencing_weights = pnl.MappingProjection(matrix=np.matrix([[1.0], [-1.0]]),
                                                         name='RESPOND_RED_WEIGHTS')


#something weird with dimensions we might think that it should be a 1 x 2 matrix
# column 0: response_'red' to respond_green_accumulator
# column 1: response_'green' to respond_green_accumulator
respond_green_differencing_weights = pnl.MappingProjection(matrix=np.matrix([[-1.0], [1.0]]),
                                                           name='RESPOND_GREEN_WEIGHTS')



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


#   evidence accumulation pathway
respond_red_process = pnl.Process(pathway=[response_layer,
                                           respond_red_differencing_weights,
                                           respond_red_accumulator],
                                  name='RESPOND_RED_PROCESS')
respond_green_process = pnl.Process(pathway=[response_layer,
                                             respond_green_differencing_weights,
                                             respond_green_accumulator],
                                    name='RESPOND_GREEN_PROCESS')


# In[ ]:


#   CREATE SYSTEM
my_Stroop = pnl.System(processes=[colors_process,
                                  words_process,
                                  task_CN_process,
                                  task_WR_process,
                                  respond_red_process,
                                  respond_green_process],
                       name='FEEDFORWARD_STROOP_SYSTEM')


# In[ ]:


# test inputs

# no color, no word, no task
notask_noinput_trial_input = {
    colors_input_layer: [0, 0],
    words_input_layer: [0, 0],
    task_layer: [0, 0]
}

# color 'red', word 'RED', task none
notask_congruent_trial_input = {
    colors_input_layer: [1, 0],
    words_input_layer: [1, 0],
    task_layer: [0, 0]
}

# color 'red', word 'RED', task WR
WR_congruent_trial_input = {
    colors_input_layer: [1, 0],
    words_input_layer: [1, 0],
    task_layer: [0, 1]
}

# color 'red', word 'RED', task CN
CN_congruent_trial_input = {
    colors_input_layer: [1, 0],
    words_input_layer: [1, 0],
    task_layer: [1, 0]
}

# color 'red', word 'GREEN', task WR, correct response red
WR_incongruent_trial_input = {
    colors_input_layer: [1, 0],
    words_input_layer: [0, 1],
    task_layer: [0, 1]
}

# color 'red', word 'GREEN', task CN, correct reponse red
CN_incongruent_trial_input = {
    colors_input_layer: [1, 0],
    words_input_layer: [0, 1],
    task_layer: [1, 0]
}

#input just task and run once so system asymptotes
CN_trial_initialize_input = {
    colors_input_layer: [0, 0],
    words_input_layer: [0, 0],
    task_layer: [1, 0]
}

#input just task and run once so system asymptotes
WR_trial_initialize_input = {
    colors_input_layer: [0, 0],
    words_input_layer: [0, 0],
    task_layer: [0, 1]
}



# In[ ]:


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

#   CREATE INITIAL CONDITION FUNCTION *** NOT WORKING
#   want colors_hidden_layer to have low activity if WR is the task
#   or want words_hidden_layer to have low activity if CN is the task
#   in 1990 paper we want 0.01 in non-task relevant pathway hidden units, 0.5 in task relevant pathway hidden units
#   and 0.5 for all of the response units
#   point of clarification: when we initialize, and then present test pattern does the task input stay on?
#   option to set these units to the values you want to reach, initial_value

#want to test this with
# colors_hidden_layer.log.nparray()
# colors_hidden_layer.log.print_entries()


# In[ ]:


my_Stroop.show()
# my_Stroop.show_graph(show_dimensions=pnl.ALL, output_fmt = 'jupyter')


# In[ ]:


#run system once with only task so asymptotes
my_Stroop.run(inputs=CN_trial_initialize_input)
#but didn't want to run accumulators so set those back to zero
respond_green_accumulator.reinitialize(0)
respond_red_accumulator.reinitialize(0)
# now run test trial
my_Stroop.show_graph()
# my_Stroop.show_graph(show_mechanism_structure=pnl.VALUES)
my_Stroop.run(inputs=CN_incongruent_trial_input, termination_processing=terminate_trial)



# In[ ]:


#check out system after a run

print(response_layer.value)

#can check out other parts of the system
# words_hidden_layer.value
# colors_hidden_layer.value

# respond_red_accumulator.value
# respond_green_accumulator.value


# In[ ]:


#if you want to run again you have to reset integrator mechanisms
words_hidden_layer.reinitialize([0,0])
colors_hidden_layer.reinitialize([0,0])
response_layer.reinitialize([0,0])


# In[ ]:


# for example you can run a congruent trial
#run system once with only task so asymptotes
my_Stroop.run(inputs=CN_trial_initialize_input)
#but didn't want to run accumulators so set those back to zero
respond_green_accumulator.reinitialize(0)
respond_red_accumulator.reinitialize(0)
# now run test trial
my_Stroop.run(inputs=CN_congruent_trial_input, termination_processing=terminate_trial)

