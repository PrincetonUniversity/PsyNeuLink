import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# Implements the model by Moeller, Huber, Nuerk & Willmes (2011) : Two-digit number processing: holistic, decomposed
# or hybrid? A computational modelling approach. Psychological Research.

# Create decomposed model:


# Define Variables ----------------------------------------------------------------------------------------------------
threshold = 0.5
tau = 0.01
learning_constant = 0.1
learning_cycles = 1000 # 50000 in paper


# def my_conflict_function(variable):
#     maxi = variable -0.0180
#     new = np.fmax([0],maxi)
#     out = [new[0]*new[1]*500]
#     return out


# User defined functions:
# hyperbolic tangent
def tangent(variable):
    return np.tanh(variable)
# Create mechanisms ---------------------------------------------------------------------------------------------------
# 4 Input layers for color, word, task & bias
# number_input_layer = pnl.TransferMechanism(size=12,
#                                            function=pnl.Linear,
#                                            name='NUMBER_INPUT')

decade_digit_one = pnl.TransferMechanism(size=9,
                                         function= pnl.Linear,
                                         name= 'DECADE_DIGIT_ONE')

decade_digit_two = pnl.TransferMechanism(size=9,
                                         function= pnl.Linear,
                                         name= 'DECADE_DIGIT_TWO')

unit_digit_one = pnl.TransferMechanism(size=9,
                                         function= pnl.Linear,
                                         name= 'unit_DIGIT_ONE')

unit_digit_two = pnl.TransferMechanism(size=9,
                                         function= pnl.Linear,
                                         name= 'unit_DIGIT_TWO')

hidden_decade_layer = pnl.RecurrentTransferMechanism(size=1,
                                                     function= pnl.Linear,
                                                     integrator_mode= True,
                                                     smoothing_factor= tau,
                                                     output_states = [
                                                         {pnl.NAME: 'TANGENT_FUNCTION',
                                                          pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                          pnl.FUNCTION: tangent}],
                                                     name= 'HIDDEN_DECADE_LAYER')

hidden_unit_layer = pnl.RecurrentTransferMechanism(size=1,
                                                     function= pnl.Linear,
                                                     integrator_mode= True,
                                                     smoothing_factor= tau,
                                                     output_states = [
                                                         {pnl.NAME: 'TANGENT_FUNCTION',
                                                          pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                          pnl.FUNCTION: tangent}],
                                                     name= 'HIDDEN_UNIT_LAYER')

output_layer = pnl.RecurrentTransferMechanism(size=2,
                                                     function= pnl.Linear,
                                                     integrator_mode= True,
                                                     smoothing_factor= tau,
                                                     output_states = [
                                                         {pnl.NAME: 'TANGENT_FUNCTION',
                                                          pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                          pnl.FUNCTION: tangent}],
                                                     name= 'OUTPUT_LAYER')

# Log mechanisms ------------------------------------------------------------------------------------------------------
hidden_decade_layer.set_log_conditions('TANGENT_FUNCTION')
hidden_unit_layer.set_log_conditions('TANGENT_FUNCTION')
output_layer.set_log_conditions('value')

# Connect mechanisms --------------------------------------------------------------------------------------------------
random_weights = np.array(np.random.uniform(0,1,9)).reshape(9,1)
random_weights2 = np.array(np.random.uniform(0,1,2)).reshape(1,2)

decade_one_input_to_hidden_decade_weights = pnl.MappingProjection(matrix= random_weights)
decade_two_input_to_hidden_decade_weights = pnl.MappingProjection(matrix=random_weights)
unit_one_input_to_hidden_unit_weights = pnl.MappingProjection(matrix=random_weights)
unit_two_input_to_hidden_unit_weights = pnl.MappingProjection(matrix=random_weights)

hidden_decade_to_output_weights = pnl.MappingProjection(matrix=random_weights2)
hidden_unit_to_output_weights = pnl.MappingProjection(matrix=random_weights2)



# Create pathways -----------------------------------------------------------------------------------------------------
decade_one = pnl.Process(pathway=[decade_digit_one,
                                  decade_one_input_to_hidden_decade_weights,
                                  hidden_decade_layer,
                                  hidden_decade_to_output_weights,
                                  output_layer],
                              # learning_rate=learning_constant,
                         learning=pnl.LEARNING,
                         name= 'DECASE_ONE_PROCESS')

decade_two = pnl.Process(pathway=[decade_digit_two,
                                  decade_two_input_to_hidden_decade_weights,
                                  hidden_decade_layer,
                                  hidden_decade_to_output_weights,
                                  output_layer],
                              # learning_rate=learning_constant,
                         learning=pnl.LEARNING,
                         name= 'DECADE_TWO_PROCESS')

unit_one = pnl.Process(pathway=[unit_digit_one,
                                unit_one_input_to_hidden_unit_weights,
                                hidden_unit_layer,
                                hidden_unit_to_output_weights,
                                output_layer],
                       learning=pnl.LEARNING,
                       # learning_rate=learning_constant,
                       name = 'UNIT_ONE_PROCESS')

unit_two = pnl.Process(pathway=[unit_digit_two,
                                unit_two_input_to_hidden_unit_weights,
                                hidden_unit_layer,
                                hidden_unit_to_output_weights,
                                output_layer],
                       learning=pnl.LEARNING,
                       # learning_rate=learning_constant,
                       name = 'UNIT_TWO_PROCESS')


# Create system -------------------------------------------------------------------------------------------------------
Decomposed_model = pnl.System(processes=[decade_one,
                             decade_two,
                             unit_one,
                             unit_two],
                  name='DECOMPOSED_MODEL')


Decomposed_model.show()
Decomposed_model.show_graph(show_dimensions=pnl.ALL)#,show_mechanism_structure=pnl.VALUES) # Uncomment to show graph of the system


#toy example with numbers 38 - 53 & 42 - 57
first_number = {
    decade_digit_one: [0,0,1,0,0,0,0,0,0],      # thirty
    decade_digit_two: [0,0,0,0,1,0,0,0,0],      # fifty
    unit_digit_one: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # eight
    unit_digit_two: [0,0,1,0,0,0,0,0,0]
}
# second_number = {
#     decade_digit_one: [0,0,0,1,0,0,0,0,0],      # forty
#     decade_digit_two: [0,0,0,0,1,0,0,0,0],      # fifty
#     unit_digit_one: [0, 1, 0, 0, 0, 0, 0, 0, 0],# two
#     unit_digit_two: [0,0,0,0,0,0,1,0,0]         # seven
# }
target_list_dict = {output_layer: [0,1]}
# target2 = [0,1]
Decomposed_model.run(num_trials=5, inputs=first_number, targets=target_list_dict)



# # Create threshold function -------------------------------------------------------------------------------------------
# def pass_threshold(response_layer, thresh):
#     results1 = response_layer.output_states.values[0][0] #red response
#     results2 = response_layer.output_states.values[0][1] #green response
#     # print(results1)
#     # print(results2)
#     if results1  >= thresh or results2 >= thresh:
#         return True
#     return False
#
# terminate_trial = {
#    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)
# }
#
