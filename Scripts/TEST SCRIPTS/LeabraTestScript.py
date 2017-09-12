import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import leabra
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.LeabraMechanism import LeabraMechanism

L = LeabraMechanism(input_size=4, output_size=2, hidden_layers=2, name='L')

print(L.function)
print(L.function_object.network)

L.execute(input=[0, 1, 3, 4])
print(L.value)
print(L.function_object.network.layers[-1].activities)  # output? or some other name?