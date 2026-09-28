"""Train a neural likelihood across a Dask cluster, then fit data with it.

As train_neural_likelihood.py, but the simulations for training are divided among the workers of a
cluster.  A composition cannot be sent to another process, so each worker builds its own model, by
calling ``build_training_pec``, rather than being given one.

On one machine::

    python train_neural_likelihood_distributed.py --n-workers 4

Across several nodes, using the SLURM launcher::

    srun -n <workers+2> python -m psyneulink.dask_run train_neural_likelihood_distributed.py

Distributing is worthwhile when simulating the model is most of the cost of training: with many
parameter draws, or a model that is slow to simulate.
"""

import argparse
import time

import psyneulink as pnl

from train_neural_likelihood import (
    FIT_RANGES,
    OUTCOME_NAMES,
    build_pec,
    fit,
    report_training,
    simulate_data,
    trial_inputs,
)


def build_training_pec(data):
    """Build the model a worker simulates; ``data`` has one placeholder row per trial."""
    pec, comp = build_pec(data, num_estimates=25, initial_seed=0)
    return pec, {comp: trial_inputs(len(data))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-parameter-samples", type=int, default=512)
    parser.add_argument("--n-trials-per-sample", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=400,
                        help="trials in the dataset that is fitted")
    parser.add_argument("--rate", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--artifact", default="ddm_nle.pt")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="size of the cluster to create; omit under the SLURM launcher")
    args = parser.parse_args()

    distributed_options = {}
    if args.n_workers is not None:
        distributed_options["n_workers"] = args.n_workers

    print("training a neural likelihood", flush=True)
    started = time.time()
    likelihood = pnl.train_neural_likelihood(
        FIT_RANGES,
        OUTCOME_NAMES,
        pec_factory=build_training_pec,
        n_parameter_samples=args.n_parameter_samples,
        n_trials_per_sample=args.n_trials_per_sample,
        epochs=args.epochs,
        distributed_options=distributed_options,
    )
    likelihood.save(args.artifact)
    report_training(likelihood, started, args.artifact)

    data = simulate_data(args.n_trials, args.rate, args.threshold)
    print(f"\nfitting {len(data)} trials simulated at "
          f"rate={args.rate}, threshold={args.threshold}", flush=True)
    fit(data, args.artifact)


if __name__ == "__main__":
    main()
