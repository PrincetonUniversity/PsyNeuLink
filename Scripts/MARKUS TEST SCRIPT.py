import psyneulink as pnl
import numpy as np

# Built 2 layer network

# Create inpute and output layer
input_layer = pnl.TransferMechanism(size = 2,
                                    initial_value= [[0.0,0.0]],
                                    name = 'INPUT LAYER')
input_layer.loggable_items
input_layer.set_log_conditions('value')


output_layer = pnl.TransferMechanism(size= 1,
                                   function = pnl.Logistic,
                                   name = 'OUTPUT LAYER')

# Create weights from input to output and from bias to output layer
#First set the
weight1 = 0.1 #np.random.uniform(0, 0.1)
weight2 = 0.1 #np.random.uniform(0, 0.1)
input_output_weights = np.array([[weight1],
                               [weight2]])

# Create Process from input layer to output layer
model_process = pnl.Process(pathway=[input_layer,
                                    input_output_weights,
                                    output_layer],
                           learning = pnl.ENABLED,
                           target= 1.0
                          )

# Create System by putting the process in the System.
new_system = pnl.System(processes = [model_process])
stim_list_dict = {input_layer:[1,0]}
target_list_dict = {output_layer:[[1]]}

new_system.run(stim_list_dict,
              targets =target_list_dict,
              num_trials= 2,
              learning = True)

print(new_system.results)
new_system.show_graph(show_learning=True)

input_layer.log.nparray()

#