"""Example: distributed PEC maximum-likelihood fitting (DDM).

A normal, serial-looking PEC study. The only difference from a serial fit is
``distributed=True`` plus a ``pec_factory`` in ``distributed_options``. There is
no Dask administration here -- no scheduler, no addresses.

Run it two ways:

  Single node (zero config -- a LocalCluster is created automatically)::

      python study.py

  Multiple nodes (PsyNeuLink forms the cluster from the SLURM allocation)::

      srun -n <workers+2> python -m psyneulink.dask_run study.py

  i.e. rank 0 runs the scheduler, rank 1 runs this script (the driver), and the
  remaining ranks are workers. See submit_dask.slurm for a ready-to-edit batch
  script. Requires the ``psyneulink[dask]`` extra and LLVM execution.

``pec_factory(data) -> (pec, inputs)`` is how a worker rebuilds the model: a live
PEC is never shipped across the wire, so each worker builds and caches its own
from this recipe. Keep it self-contained (imports + literals inside) so it can be
shipped to workers by value.
"""

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)

# Ground-truth DDM parameters used to synthesize the "observed" data, the search
# bounds for the three fitted parameters, and sizing knobs.
TRUE = dict(rate=0.3, threshold=0.6, non_decision_time=0.15)
FIT_BOUNDS = {"rate": (-0.5, 0.5), "threshold": (0.5, 1.0), "non_decision_time": (0.0, 1.0)}
NUM_TRIALS = 50
NUM_ESTIMATES = 4000
INITIAL_SEED = 42


def trial_inputs():
    rng = np.random.default_rng(12345)
    return rng.choice([5.0, -5.0], size=(NUM_TRIALS, 1), p=[0.1, 0.9])


def build_ddm():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=TRUE["rate"], noise=1.0,
            threshold=TRUE["threshold"], non_decision_time=TRUE["non_decision_time"],
            time_step_size=0.001,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def make_observed_data():
    """Synthesize data from TRUE params (replace with your real subject data)."""
    comp, decision = build_ddm()
    comp.run(inputs={decision: trial_inputs()})
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)), columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data


def pec_factory(data):
    """Worker recipe: rebuild a fresh serial PEC + inputs from the observed data.

    Self-contained (imports + literals inside) so Dask can ship it to workers by
    value; workers only need PsyNeuLink importable in the same environment.
    """
    import numpy as np
    import psyneulink as pnl
    from psyneulink.core.components.functions.nonstateful.fitfunctions import (
        PECOptimizationFunction,
    )

    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=0.3, noise=1.0, threshold=0.6,
            non_decision_time=0.15, time_step_size=0.001,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    bounds = {"rate": (-0.5, 0.5), "threshold": (0.5, 1.0), "non_decision_time": (0.0, 1.0)}
    pec = pnl.ParameterEstimationComposition(
        name="pec_worker", nodes=[comp],
        parameters={
            ("rate", decision): np.linspace(*bounds["rate"], 1000),
            ("threshold", decision): np.linspace(*bounds["threshold"], 1000),
            ("non_decision_time", decision): np.linspace(*bounds["non_decision_time"], 1000),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        # Inner optimizer is unused for likelihood-only evaluation but required by
        # the PEC constructor.
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=4000, initial_seed=42,
        same_seed_for_all_parameter_combinations=True,  # common random numbers
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    rng = np.random.default_rng(12345)
    return pec, {comp: rng.choice([5.0, -5.0], size=(50, 1), p=[0.1, 0.9])}


def main():
    import optuna

    data = make_observed_data()
    comp, decision = build_ddm()

    # CMA-ES; with the launcher, max_concurrent_evaluations defaults to the live
    # worker count. Pinning the CMA-ES popsize to the batch keeps the trajectory
    # deterministic. Adjust max_iterations for your evaluation budget.
    optimizer = PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=8),
        max_iterations=480,
        distributed=True,
        distributed_options={"pec_factory": pec_factory},
    )

    pec = pnl.ParameterEstimationComposition(
        name="ddm_mle", nodes=[comp],
        parameters={
            ("rate", decision): np.linspace(*FIT_BOUNDS["rate"], 1000),
            ("threshold", decision): np.linspace(*FIT_BOUNDS["threshold"], 1000),
            ("non_decision_time", decision): np.linspace(*FIT_BOUNDS["non_decision_time"], 1000),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=optimizer,
        num_estimates=NUM_ESTIMATES, initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")

    pec.run(inputs={comp: trial_inputs()})

    print("optimal log-likelihood:", pec.optimal_value)
    for name, val in pec.optimized_parameter_values.items():
        true = TRUE[name.split(".")[-1]]
        print(f"  {name}: recovered={val:.4f}  true={true}")


if __name__ == "__main__":
    main()
