import psyneulink as pnl

comp = pnl.Composition(name="comp")
A = pnl.TransferMechanism(
    name="A",
    function=pnl.Linear(slope=0.5, intercept=1.0),
    integrator_mode=True,
    integrator_function=pnl.SimpleIntegrator(
        rate=0.5,
        noise=pnl.UniformDist(low=-1.0, high=1.0, seed=0)
    ),
)
B = pnl.TransferMechanism(
    name="B",
    function=pnl.Logistic(gain=0.1),
    integrator_mode=True,
    integration_rate=0.9,
    integrator_function=pnl.AdaptiveIntegrator(
        offset=-1,
        noise=pnl.NormalDist(mean=-1.0, standard_deviation=0.5, seed=0)
    ),
)

C = pnl.TransferMechanism(
    name="C",
    integrator_mode=True,
    integration_rate=0.5,
    integrator_function=pnl.AccumulatorIntegrator(
        noise=pnl.NormalDist(standard_deviation=0.25, seed=0)
    ),
)

D = pnl.TransferMechanism(
    name="D",
    integrator_mode=True,
    integration_rate=0.5,
    integrator_function=pnl.LeakyCompetingIntegrator(
        noise=pnl.UniformDist(low=-0.5, high=0.5, seed=0),
        time_step_size=0.2,
    ),
)

E = pnl.IntegratorMechanism(
    name="E",
    function=pnl.SimpleIntegrator(
        rate=0.5,
        offset=-1,
        noise=pnl.UniformDist(low=-0.25, high=0.5, seed=0)
    )
)

comp.add_linear_processing_pathway([A, B, C, D, E])

comp.scheduler.add_condition_set(
    {
        A: pnl.EveryNPasses(1),
        B: pnl.EveryNCalls(A, 2),
        C: pnl.EveryNCalls(B, 2),
        D: pnl.EveryNCalls(C, 2),
        E: pnl.EveryNCalls(D, 2),
    }
)

comp.termination_processing = {
    pnl.TimeScale.RUN: pnl.AfterNTrials(1),
    pnl.TimeScale.TRIAL: pnl.All(pnl.Not(pnl.BeforeNCalls(E, 5))),
}
