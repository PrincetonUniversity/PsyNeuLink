
from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism, mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.composition import Composition, CompositionError, MechanismRole
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.Globals.Keywords import EXECUTING
from pprint import pprint


comp = Composition()
sched = Scheduler(composition=comp)
A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
B = TransferMechanism(function=Linear(slope=5.0))
comp.add_mechanism(A)
comp.add_mechanism(B)
comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
comp.analyze_graph()
inputs_dict = {A:5}
print(comp.graph.vertices)
comp.run(inputs=inputs_dict, scheduler=sched)
# is_origin = comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
# for v in comp.graph.vertices:
#     if isinstance(v.component, Mechanism):
#         if (v.component in is_origin) and (v.component in inputs_dict.keys()):
#             print()
#             num = v.component.execute(input= inputs_dict[v.component], context=EXECUTING)
#             print("=============================================")
#             print(num)
#             print()
#             print()
#             # pprint(v.component.__dict__)
#             print("=============================================")
#         else:
#             num = v.component.execute(context=EXECUTING)
#             print("=============================================")
#             print(num)
#             print()
#             print()
#             # pprint(v.component.__dict__)
#             print("=============================================")
