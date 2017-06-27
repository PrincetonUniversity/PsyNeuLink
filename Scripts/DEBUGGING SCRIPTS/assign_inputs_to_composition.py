
from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism, mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.composition import Composition, CompositionError, MechanismRole
from PsyNeuLink.Globals.Keywords import EXECUTING
from pprint import pprint


comp = Composition()
A = TransferMechanism(function=Linear(slope=5.0))
B = TransferMechanism(function=Linear(slope=5.0))
comp.add_mechanism(A)
comp.add_mechanism(B)
comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
comp.analyze_graph()
print(comp.graph.vertices)
is_origin = True
for v in comp.graph.vertices:
    if isinstance(v.component, Mechanism):
        if is_origin:
            num = v.component.execute(input= 5.0, context=EXECUTING)
            print("=============================================")
            print(num)
            print()
            print()
            # pprint(v.component.__dict__)
            print("=============================================")
            is_origin = False
        else:
            num = v.component.execute(context=EXECUTING)
            print("=============================================")
            print(num)
            print()
            print()
            # pprint(v.component.__dict__)
            print("=============================================")
