# script written by Changyan in Sept. 2017 to test the LeabraMechanism: this will probably be rapidly deprecated.
# Feel free to remove it.
import numpy as np
import warnings
import random
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import leabra
from psyneulink.library.mechanisms.processing.leabramechanism import LeabraMechanism, build_network
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.functions.function import Linear
from psyneulink.components.process import process
from psyneulink.components.system import system
import time

random.seed(1)
input_size=4
output_size=2
hidden_layers=4
hidden_sizes=[2, 3, 4, 5]
input_pattern = [[0, 1, 3, 4]]
training_pattern = [[0, 1]]
num_trials = 100

L3 = LeabraMechanism(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers, hidden_sizes=hidden_sizes, name='L', training_flag=True)

T1 = TransferMechanism(name='T1', size=4, function=Linear)
T2 = TransferMechanism(name='T2', size=2, function=Linear)

p1 = process(pathway=[T1, L3])
proj = MappingProjection(sender=T2, receiver=L3.input_states[1])
p2 = process(pathway=[T2, L3])
T2.output_state.efferents[1].matrix = T2.output_state.efferents[1].matrix * 0
s = system(processes=[p1, p2])
start_time = time.process_time()
tmp = s.run(inputs={T1: input_pattern.copy()*num_trials, T2: training_pattern.copy()*num_trials})
print('PNL time to run: ', time.process_time()-start_time)

# begin: optional print statements
print('PNL Output: ', tmp)
print('types within PNL output: ', type(tmp), type(tmp[0]), type(tmp[0][0]), type(tmp[0][0][0]))
print('activities and types: ', L3.function_object.network.layers[-1].activities, type(L3.function_object.network.layers[-1].activities[0]))
unit = [unit.act_m for unit in L3.function_object.network.layers[-1].units]
print('output layer unit values and their types: ', unit, type(unit), type(unit[0]))
print('output state value and type: ', L3.output_state.value, type(L3.output_state.value), type(L3.output_state.value[0]))
print("L3 value and L3 variable: ", L3.value, L3.variable)
# end: optional print statements

random.seed(1)
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
start_time = time.process_time()
for _ in range(num_trials):
    train_network(leabra_net, input_pattern[0].copy(), training_pattern[0])
print('Leabra time to run: ', time.process_time() - start_time)
print('Leabra Output: ', [unit.act_m for unit in leabra_net.layers[-1].units], type([unit.act_m for unit in leabra_net.layers[-1].units][0]))