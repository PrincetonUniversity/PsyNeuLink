# test script for LeabraMechanism 11/1/17
import copy
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

def train_network(network, input_pattern, output_pattern):
    """Run one trial on the network"""
    assert len(network.layers[0].units) == len(input_pattern)

    assert len(network.layers[-1].units) == len(output_pattern)
    network.set_inputs({'input_layer': input_pattern})
    network.set_outputs({'output_layer': output_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]
def test_network(network, input_pattern):
    assert len(network.layers[0].units) == len(input_pattern)
    network.set_inputs({'input_layer': input_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]


random_seed_value = 1  # feel free to change this value
random.seed(random_seed_value)
num_trials = 10 # how many trials should we run?
input_size = 4  # how big is the input layer?
output_size = 4  # how big is the output layer?
hidden_layers = 4  # how many hidden layers are there?
hidden_sizes = [3, 4, 3, 4]  # how big is each hidden layer?
# input_pattern = [[0, .1, .3, .4]] * num_trials  # the input
input_pattern = [[]] * num_trials
for i in range(num_trials):
    input_pattern[i] = [random.random() for j in range(input_size)]
# similar example: input_pattern = [[0, 1, 3, 4]] * int(num_trials/2) + [[0, 0, 0, 0]] * int(num_trials/2)
# training_pattern = [[0, 1, -1]] * num_trials  # the training pattern
training_pattern = input_pattern
train_flag = True

random.seed(random_seed_value)  # this (and random.seed below) is to ensure Leabra network is identical below
leabra_net = build_network(n_input=input_size, n_output=output_size, n_hidden=hidden_layers,
                           hidden_sizes=hidden_sizes, training_flag=train_flag)
leabra_net2 = copy.deepcopy(leabra_net)

### INSERT: test_network etc. etc.


print('\nRunning Leabra in Leabra...')
start_time = time.process_time()

# test_network(leabra_net, np.zeros(input_size))
# train_network(leabra_net, np.zeros(input_size), np.zeros(output_size))
# train_network(leabra_net, np.zeros(input_size), np.zeros(output_size))
for i in range(num_trials):
    train_network(leabra_net, input_pattern[i], training_pattern[i])
for i in range(num_trials):
    test_network(leabra_net, input_pattern[i])
end_time = time.process_time()

print('Leabra time to run: ', end_time - start_time, "seconds")
print('Leabra Output: ', [unit.act_m for unit in leabra_net.layers[-1].units], type([unit.act_m for unit in leabra_net.layers[-1].units][0]))


# the LeabraMechanism
random.seed(random_seed_value)

L = LeabraMechanism(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers,
                     hidden_sizes=hidden_sizes, name='L', training_flag=train_flag)
# L = LeabraMechanism(leabra_net=leabra_net2, name='L', training_flag=True)


T1 = TransferMechanism(name='T1', size=input_size, function=Linear)
T2 = TransferMechanism(name='T2', size=output_size, function=Linear)

p1 = Process(pathway=[T1, L])
proj = MappingProjection(sender=T2, receiver=L.input_states[1])
p2 = Process(pathway=[T2, proj, L])
s = System(processes=[p1, p2])

print('\nRunning Leabra in PsyNeuLink...')
start_time = time.process_time()
outputs = s.run(inputs={T1: input_pattern.copy(), T2: training_pattern.copy()})
L.training_flag = False
outputs = s.run(inputs={T1: input_pattern.copy(), T2: training_pattern.copy()})
end_time = time.process_time()

print('PNL time to run: ', end_time - start_time, "seconds")
print('PNL Outputs Over Time: ', outputs, type(outputs))
print('PNL Final Output: ', outputs[-1], type(outputs[-1]))

# import matplotlib.pyplot as plt
# out_1 = list(map(lambda x: x[0][0], outputs))
# out_2 = list(map(lambda x: x[0][1], outputs))
# out_3 = list(map(lambda x: x[0][2], outputs))
# out_4 = list(map(lambda x: x[0][3], outputs))
#
# plt.plot(range(len(out_1)), out_1, 'b-')
# plt.plot(range(len(out_2)), out_2, 'r-')
# plt.plot(range(len(out_3)), out_3, 'g-')
# plt.plot(range(len(out_4)), out_4, 'c-')