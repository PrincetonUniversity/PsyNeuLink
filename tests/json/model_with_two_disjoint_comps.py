import psyneulink as pnl

comp = pnl.Composition(name='comp')
A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')

for m in [A, B]:
    comp.add_node(m)

comp.add_projection(pnl.MappingProjection(), A, B)

comp.scheduler.add_condition_set({
    A: pnl.EveryNPasses(1),
    B: pnl.EveryNCalls(A, 2),
})

comp.termination_processing = {
    pnl.TimeScale.RUN: pnl.AfterNTrials(1),
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(B, 4)
}


comp2 = pnl.Composition(name='comp2')
C = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='C')
D = pnl.TransferMechanism(function=pnl.Logistic, name='D')

for m in [C, D]:
    comp2.add_node(m)

comp2.add_projection(pnl.MappingProjection(), C, D)

comp2.scheduler.add_condition_set({
    C: pnl.EveryNPasses(1),
    D: pnl.EveryNCalls(C, 4),
})

comp2.termination_processing = {
    pnl.TimeScale.RUN: pnl.AfterNTrials(1),
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(D, 8)
}
