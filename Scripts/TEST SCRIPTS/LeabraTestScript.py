import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import leabra
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.LeabraMechanism import LeabraMechanism

L = LeabraMechanism(input_size=2, output_size=2, hidden_layers=0, name='L')

print(L.function)
print(L.function.network)

L.execute(input=[0, 1])
print(L.value)
print(L.function.network.output)  # output? or some other name?