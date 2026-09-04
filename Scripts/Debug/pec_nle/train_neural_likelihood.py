"""Train a neural likelihood for a drift-diffusion model, then fit with it.

A neural likelihood is trained once on data simulated from the model, and replaces the
simulate-then-estimate-a-density step for every fit afterwards.

Train an estimator and fit a synthetic dataset with it::

    python train_neural_likelihood.py

Spread training-data generation over a single-node cluster::

    python train_neural_likelihood.py --distributed --n-workers 4

Across several nodes, using the SLURM launcher::

    srun -n <workers+2> python -m psyneulink.dask_run train_neural_likelihood.py --distributed

Training dominates the runtime; the defaults below are small enough to finish in minutes
and are not large enough for a publishable fit.
"""

import argparse
import time

import numpy as np
import pandas as pd

import psyneulink as pnl
from psyneulink import train_neural_likelihood
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction

# Ranges searched when fitting, and the region the estimator is trained over.  Fitting
# outside a trained range is extrapolation, so these are the same by construction.
FIT_RANGES = {"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)}

NON_DECISION_TIME = 0.15
TIME_STEP_SIZE = 0.01
OUTCOME_NAMES = ("decision", "response_time")


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


def build_pec(data, num_estimates=25):
    """Build one model over ``data``.

    Training calls this with a placeholder table to simulate from, and it is the same
    factory contract distributed and hierarchical fitting use.
    """
    comp, decision = build_model()
    pec = pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(*FIT_RANGES[name], 1000) for name in FIT_RANGES
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=num_estimates,
        initial_seed=0,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, {comp: trial_inputs(len(data))}


def simulate_data(n_trials, rate, threshold, seed=0):
    """Trials from the model at known parameters, to fit afterwards."""
    comp, decision = build_model(rate=rate, threshold=threshold, seed=seed)
    comp.run(inputs={decision: trial_inputs(n_trials)})
    outcomes = np.asarray(comp.results, dtype=float).reshape(n_trials, -1)
    data = pd.DataFrame({name: outcomes[:, i] for i, name in enumerate(OUTCOME_NAMES)})
    data["decision"] = data["decision"].astype("category")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-parameter-samples", type=int, default=512)
    parser.add_argument("--n-trials-per-sample", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=400,
                        help="trials in the dataset that is fitted afterwards")
    parser.add_argument("--rate", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--artifact", default="ddm_nle.pt")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()

    distributed_options = None
    if args.distributed:
        distributed_options = {}
        if args.n_workers is not None:
            distributed_options["n_workers"] = args.n_workers

    print("training a neural likelihood", flush=True)
    started = time.time()
    likelihood = train_neural_likelihood(
        build_pec,
        bounds=FIT_RANGES,
        outcome_names=OUTCOME_NAMES,
        n_parameter_samples=args.n_parameter_samples,
        n_trials_per_sample=args.n_trials_per_sample,
        epochs=args.epochs,
        distributed_options=distributed_options,
    )
    likelihood.save(args.artifact)
    print(f"  trained in {(time.time() - started) / 60:.1f} min, "
          f"held-out NLL {likelihood.provenance.val_nll:.4f} per trial", flush=True)
    print(f"  saved to {args.artifact}", flush=True)

    data = simulate_data(args.n_trials, args.rate, args.threshold)
    print(f"\nfitting {len(data)} trials simulated at "
          f"rate={args.rate}, threshold={args.threshold}", flush=True)

    comp, decision = build_model()
    pec = pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(*FIT_RANGES[name], 1000) for name in FIT_RANGES
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=50
        ),
        likelihood_estimator="neural",
        likelihood_estimator_kwargs={"artifact": args.artifact},
    )
    started = time.time()
    pec.run(inputs={comp: trial_inputs(len(data))})
    print(f"  fitted in {time.time() - started:.1f}s", flush=True)

    for name, estimate in pec.optimized_parameter_values.items():
        print(f"    {name:24s} {estimate:.4f}")
    print(f"  log-likelihood {pec.optimal_value:.2f}")


if __name__ == "__main__":
    main()
