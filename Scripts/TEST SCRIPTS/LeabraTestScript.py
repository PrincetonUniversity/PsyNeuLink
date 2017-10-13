# script written by Changyan in Sept. 2017 to test the LeabraMechanism: this will probably be rapidly deprecated.
# Feel free to remove it.
import numpy as np
import warnings
import random
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import leabra
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.LeabraMechanism import LeabraMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system

def printWeights(link):
    print('link.pre: ', link.pre)
    print('link.post: ', link.post)
    print('link.wt: ', link.wt)
    print('link.fwt: ', link.fwt)
    print('link.dwt: ', link.dwt)

random.seed(1)  # to avoid the bug with mixed types in statistics._sum

L3 = LeabraMechanism(input_size=4, output_size=2, hidden_layers=4, hidden_sizes=[2, 3, 4, 5], name='L', training_flag=True)

print(L3.function)
print(L3.function_object.network)
printWeights(L3.function_object.network.connections[0].links[0])
printWeights(L3.function_object.network.connections[1].links[1])

L3.execute(input=[[0, 1, 3, 4], [10, 20]])
L3.execute(input=[[0, 1, 3, 4], [10, 20]])
L3.execute(input=[[0, 1, 3, 4], [10, 20]])
L3.execute(input=[[0, 1, 3, 4], [10, 20]])
L3.execute(input=[[0, 1, 3, 4], [10, 20]])

printWeights(L3.function_object.network.connections[0].links[0])
printWeights(L3.function_object.network.connections[1].links[1])
print(L3.value)
print(L3.function_object.network.layers[-1].activities)

T1 = TransferMechanism(name='T1', size=4, function=Linear)
T2 = TransferMechanism(name='T2', size=2, function=Linear)

p1 = process(pathway=[T1, L3])
proj = MappingProjection(sender=T2, receiver=L3.input_states[1])
p2 = process(pathway=[T2, L3])
s = system(processes=[p1, p2])
s.run(inputs={T1: [[0, 1, 3, 4]], T2: [[1, 4]]})
print(L3.function_object.network.layers[-1].activities)
print(L3.output_state.value)

# L = LeabraMechanism(input_size=4, output_size=2, hidden_layers=2, name='L')
#
# print(L.function)
# print(L.function_object.network)
#
# L.execute(input=[[0, 1, 3, 4], [10, 20]])
# print(L.value)
# print(L.function_object.network.layers[-1].activities)
#
# L2 = LeabraMechanism(input_size=4, output_size=2, hidden_layers=4, hidden_sizes=[2, 3, 4, 5], name='L')
#
# print(L2.function)
# print(L2.function_object.network)
#
# L2.execute(input=[[0, 1, 3, 4], [10, 20]])
# print(L2.value)
# print(L2.function_object.network.layers[-1].activities)
# layers = L2.function_object.network.layers
# for i in range(len(layers)):
#     print('layer {} has size {}'.format(i, layers[i].size))