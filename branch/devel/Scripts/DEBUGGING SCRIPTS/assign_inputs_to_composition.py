
from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism, mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Composition import Composition, CompositionError, MechanismRole
from PsyNeuLink.Globals.Keywords import EXECUTING
from PsyNeuLink.Scheduling.Scheduler import Scheduler
from pprint import pprint


comp = Composition()
A = TransferMechanism(default_variable=1.0, function=Linear(slope=5.0))
B = TransferMechanism(function=Linear(slope=5.0))
comp.add_mechanism(A)
comp.add_mechanism(B)
comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
comp.analyze_graph()
inputs_dict = {A:5}
sched = Scheduler(composition=comp)
comp.run(
    # comment or uncomment the next line to switch between input = 1 [default_variable] and input = 5 [inputs_dict]
    inputs=inputs_dict,
    scheduler=sched
)
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
