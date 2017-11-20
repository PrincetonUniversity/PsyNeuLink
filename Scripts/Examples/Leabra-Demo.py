# The following script briefly demos the LeabraMechanism in PsyNeuLink.
# Before running this, please make sure you are using Python 3.5, and that you have installed the leabra package in
# your Python 3.5 environment.

# Installation notes:
#
# If you see an error such as:
#  "Runtime warning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88"
# then this may be an issue with scipy (or other similar modules such as scikit-learn or sklearn).
#
# To resolve this, if you have pip, then use PyCharm to uninstall scipy (or other packages if they continue
# to give you trouble) and then use "pip install scipy --no-use-wheel". Or, if you can figure out how to get PyCharm
#  to ignore warnings, that's fine too.
#
# More info here: https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate
# -binary-incompatibility

import warnings
warnings.filterwarnings("ignore", message=r".*numpy.dtype size changed.*")
import numpy as np
import random
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     import leabra
from psyneulink.library.mechanisms.processing.leabramechanism import LeabraMechanism, build_network
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.functions.function import Linear
from psyneulink.components.process import Process
from psyneulink.components.system import System
import time

random_seed_value = 1  # feel free to change this value
random.seed(random_seed_value)  # this (also see random.seed below) is to ensure Leabra network is identical below
num_trials = 10  # how many trials should we run?
input_size = 4  # how big is the input layer?
output_size = 2  # how big is the output layer?
hidden_layers = 4  # how many hidden layers are there?
hidden_sizes = [2, 3, 4, 5]  # how big is each hidden layer?
input_pattern = [[0, 1, 3, 4]] * num_trials  # the input
# similar example: input_pattern = [[0, 1, 3, 4]] * int(num_trials/2) + [[0, 0, 0, 0]] * int(num_trials/2)
training_pattern = [[0, 1]] * num_trials  # the training pattern

# the LeabraMechanism of interest
L = LeabraMechanism(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers,
                     hidden_sizes=hidden_sizes, name='L', training_flag=True)


T1 = TransferMechanism(name='T1', size=4, function=Linear)
T2 = TransferMechanism(name='T2', size=2, function=Linear)

p1 = Process(pathway=[T1, L])
proj = MappingProjection(sender=T2, receiver=L.input_states[1])
p2 = Process(pathway=[T2, L])
T2.output_state.efferents[1].matrix = T2.output_state.efferents[1].matrix * 0
s = System(processes=[p1, p2])

print('Running Leabra in PsyNeuLink...')
start_time = time.process_time()
outputs = s.run(inputs={T1: input_pattern.copy(), T2: training_pattern.copy()})
end_time = time.process_time()

print('PNL time to run: ', end_time - start_time, "seconds")
print('PNL Outputs Over Time: ', outputs, type(outputs))
print('PNL Final Output: ', outputs[-1], type(outputs[-1]))


random.seed(random_seed_value)
leabra_net = build_network(n_input=input_size, n_output=output_size, n_hidden=hidden_layers, hidden_sizes=hidden_sizes)
leabra_net.set_inputs({'input_layer': np.zeros(4)})
leabra_net.trial()

def train_network(network, input_pattern, output_pattern):
    """Run one trial on the network"""
    assert len(network.layers[0].units) == len(input_pattern)

    assert len(network.layers[-1].units) == len(output_pattern)
    network.set_inputs({'input_layer': input_pattern})
    network.set_outputs({'output_layer': output_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]

print('\nRunning Leabra in Leabra...')
start_time = time.process_time()
for i in range(num_trials):
    train_network(leabra_net, input_pattern[i].copy(), training_pattern[i])
end_time = time.process_time()
print('Leabra time to run: ', end_time - start_time, "seconds")
print('Leabra Output: ', [unit.act_m for unit in leabra_net.layers[-1].units], type([unit.act_m for unit in leabra_net.layers[-1].units][0]))