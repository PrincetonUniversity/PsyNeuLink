# script written by Changyan in Sept. 2017 to test the LeabraMechanism: this will probably be rapidly deprecated.
# Feel free to remove it.

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import leabra
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.LeabraMechanism import LeabraMechanism

L = LeabraMechanism(input_size=4, output_size=2, hidden_layers=2, name='L')

print(L.function)
print(L.function_object.network)

L.execute(input=[[0, 1, 3, 4], [10, 20]])
print(L.value)
print(L.function_object.network.layers[-1].activities)

L2 = LeabraMechanism(input_size=4, output_size=2, hidden_layers=4, hidden_sizes=[2, 3, 4, 5], name='L')

print(L2.function)
print(L2.function_object.network)

L2.execute(input=[[0, 1, 3, 4], [10, 20]])
print(L2.value)
print(L2.function_object.network.layers[-1].activities)
layers = L2.function_object.network.layers
for i in range(len(layers)):
    print('layer {} has size {}'.format(i, layers[i].size))

L3 = LeabraMechanism(input_size=4, output_size=2, hidden_layers=4, hidden_sizes=[2, 3, 4, 5], name='L', training_flag=True)

print(L3.function)
print(L3.function_object.network)

L3.execute(input=[[0, 1, 3, 4], [10, 20]])
print(L3.value)
print(L3.function_object.network.layers[-1].activities)