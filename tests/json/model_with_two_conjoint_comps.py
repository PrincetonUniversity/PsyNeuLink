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

for m in [A, B]:
    comp2.add_node(m)

comp2.add_projection(pnl.MappingProjection(), A, B)

comp2.scheduler.add_condition_set({
    A: pnl.EveryNPasses(1),
    B: pnl.EveryNAalls(A, 4),
})

comp2.termination_processing = {
    pnl.TimeScale.RUN: pnl.AfterNTrials(1),
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(B, 8)
}
