# Musslick et al 2021

This directory contains an implementation of each multitasking simulation in
Musslick et al. Each simulation directory, Simulation [n], contains two scripts:

sim[n].py: the main script that configures (using sim[n]_configure.py), trains, 
and runs the neural network model and runs the analysis on the output data.

sim[n]_configure.py: script that configures the neural network according to the
simulation's parameters. This is run by sim[n].py and does not need to be run 
separately by the user.

### Running the Simulation

Each simulation can be run using its sim[n].py script. The script takes three
arguments:
- graph_input: the number of the bipartite task structure graph to be used
- hidden_arg: the number of hidden layers in the neural network
- silence: whether to silence debug messages while running the simulation

### Simulation Paramters

Each simulation uses the following parameters, defined in its confinguration
script:
- init_scale: scales for initialized random weights
- learning_rate: learning rate
- decay: weight penalization parameter
- bias: weight from bias units to hidden & output units
- thresh: mean-squared error stopping criterion
- hidden_path_size: size of a subset of hidden units that each receive the same 
weights from the task layer
- output_path_size: size of a subset of output units that each receive the same 
weights from the task layer

