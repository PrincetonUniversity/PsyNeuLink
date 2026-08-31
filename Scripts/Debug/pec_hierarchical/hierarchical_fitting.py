"""Fit a group of participants jointly with `ParameterEstimationComposition`.

Read this one if you have data and want to fit it hierarchically.  It expects a CSV holding
every participant's trials stacked, with a column naming who produced each row::

    subject,decision,response_time
    S00,1,0.512
    S00,0,0.734
    S01,1,0.488

Participants may have different trial counts.  ``make_example_data.py`` writes a synthetic
table in this format if you want something to run against first.

Run in one process::

    python hierarchical_fitting.py --data group_data.csv

Across a single-node cluster::

    python hierarchical_fitting.py --data group_data.csv --distributed --n-workers 4

Across several nodes, using the SLURM launcher::

    srun -n <workers+2> python -m psyneulink.dask_run hierarchical_fitting.py \\
        --data group_data.csv --distributed
"""

import argparse
import time

import numpy as np
import pandas as pd

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction

# Ranges searched for each parameter.  These are also the support of the group model, so every
# participant must be fitted over the same ranges.
FIT_RANGES = {"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)}
FIT_PARAMS = tuple(FIT_RANGES)

NON_DECISION_TIME = 0.15
TIME_STEP_SIZE = 0.01
NUM_ESTIMATES = 300


def build_model(rate=0.0, threshold=0.9, seed=None):
    """A two-alternative drift-diffusion model."""
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=rate,
            noise=1.0,
            threshold=threshold,
            non_decision_time=NON_DECISION_TIME,
            time_step_size=TIME_STEP_SIZE,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    if seed is not None:
        decision.function.parameters.seed.set(int(seed) % (2 ** 32))
    return pnl.Composition(pathways=decision), decision


def trial_inputs(n_trials):
    return np.ones((n_trials, 1))


def load_data(path):
    """Read the stacked trial table.

    The choice column has to be `category` dtype for the likelihood to treat it as discrete;
    a CSV does not carry dtypes, so set it after reading.
    """
    data = pd.read_csv(path)
    data["decision"] = data["decision"].astype("category")
    return data


def participant_pec(data, subject_index=None):
    """Build one participant's model from their trials.

    Defined at module level so that it can be sent to a worker process.  The seed varies with the
    participant: a shared seed would give every participant the same stream of simulation noise,
    which would be absorbed into the group variance rather than averaging out.
    """
    comp, decision = build_model()
    pec = pnl.ParameterEstimationComposition(
        name="participant",
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(*FIT_RANGES[name], 1000) for name in FIT_PARAMS
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=NUM_ESTIMATES,
        initial_seed=100 + (subject_index or 0),
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, {comp: trial_inputs(len(data))}


def report(results):
    """Print what the fit found."""
    print(f"\n{results!r}")
    print("\nobjective by iteration:",
          np.round(results.em_history["objective"].to_numpy(), 2).tolist())

    print("\ngroup estimate:")
    print(results.group_parameters.round(4).to_string())

    print("\nper-participant estimates:")
    print(results.subject_parameters.round(4).to_string())

    # The group variance accounts for both the spread of the estimates and how uncertain each one
    # is. The two agree once the fit has converged, and differ by one update before that.
    spread = results.z_hat.var(axis=0) + results.posterior_variance.mean(axis=0)
    print(f"\n  spread of estimates + mean uncertainty: {spread.round(4)}")
    print(f"  estimated group variance:               {results.sigma.round(4)}"
          f"{'' if results.converged else '   (fit did not converge; these agree once it has)'}")

    failures = int((~results.subject_converged).sum())
    print(f"\n  participant fits that converged: "
          f"{len(results.subject_labels) - failures}/{len(results.subject_labels)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="group_data.csv",
                        help="stacked trial table, as written by make_example_data.py")
    parser.add_argument("--subject-id", default="subject",
                        help="column of the table naming who produced each row")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--distributed", action="store_true",
                        help="fit participants across a Dask cluster")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="size of the cluster created for --distributed")
    args = parser.parse_args()

    data = load_data(args.data)
    print(f"{len(data)} trials from {data[args.subject_id].nunique()} participants")

    distributed_options = {"pec_factory": participant_pec}
    if args.n_workers is not None:
        distributed_options["n_workers"] = args.n_workers

    # The model given here declares which parameters are fitted and which outputs are compared
    # against the data. It is not simulated; each participant's model comes from the factory.
    comp, decision = build_model()
    pec = pnl.ParameterEstimationComposition(
        name="group",
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(*FIT_RANGES[name], 1000) for name in FIT_PARAMS
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": args.subject_id,
            "max_iterations": args.max_iterations,
            "tol": 1e-3,
            "estep_options": {"xatol": 1e-2, "fatol": 5e-2, "maxiter": 120},
        },
        distributed=args.distributed,
        distributed_options=distributed_options,
    )

    started = time.time()
    results = pec.run()
    print(f"fitted in {time.time() - started:.1f}s")

    report(results)


if __name__ == "__main__":
    main()
