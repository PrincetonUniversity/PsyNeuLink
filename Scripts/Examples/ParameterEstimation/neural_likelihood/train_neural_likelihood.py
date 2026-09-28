"""Train a neural likelihood for a drift-diffusion model, then fit data with it.

The estimator is trained on data simulated from the model, and is then used in place of simulating
the model when fitting::

    python train_neural_likelihood.py

To divide the simulations for training among the workers of a Dask cluster, see
train_neural_likelihood_distributed.py.

The defaults finish in a few minutes, and are too small for the estimates to be relied on.
"""

import argparse
import time

import numpy as np
import pandas as pd

import psyneulink as pnl

# The ranges fitted, which are also the ranges the estimator is trained over: it is valid only
# within them.
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


def build_pec(data, **kwargs):
    """Build a ParameterEstimationComposition that fits the model's rate and threshold to ``data``."""
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
        **kwargs,
    )
    return pec, comp


def simulate_data(n_trials, rate, threshold, seed=0):
    """Trials from the model at known parameters, to fit afterwards."""
    comp, decision = build_model(rate=rate, threshold=threshold, seed=seed)
    comp.run(inputs={decision: trial_inputs(n_trials)})
    outcomes = np.asarray(comp.results, dtype=float).reshape(n_trials, -1)
    data = pd.DataFrame({name: outcomes[:, i] for i, name in enumerate(OUTCOME_NAMES)})
    data["decision"] = data["decision"].astype("category")
    return data


def report_training(likelihood, started, artifact):
    print(f"  trained in {(time.time() - started) / 60:.1f} min, "
          f"held-out NLL {likelihood.provenance.val_nll:.4f} per trial", flush=True)
    print(f"  saved to {artifact}", flush=True)


def fit(data, artifact):
    """Fit ``data`` with a trained estimator, and print the estimates."""
    pec, comp = build_pec(
        data,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=50
        ),
        likelihood_estimator="neural",
        likelihood_estimator_kwargs={"artifact": artifact},
    )
    started = time.time()
    pec.run(inputs={comp: trial_inputs(len(data))})
    print(f"  fitted in {time.time() - started:.1f}s", flush=True)
    for name, estimate in pec.optimized_parameter_values.items():
        print(f"    {name:24s} {estimate:.4f}")
    print(f"  log-likelihood {pec.optimal_value:.2f}")


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
    args = parser.parse_args()

    data = simulate_data(args.n_trials, args.rate, args.threshold)

    # The model as it would be fitted without an estimator, scored by simulating it. Training
    # simulates it for as many trials as its inputs have, at each set of parameter values.
    pec, comp = build_pec(data, num_estimates=25, initial_seed=0)

    print("training a neural likelihood", flush=True)
    started = time.time()
    likelihood = pnl.train_neural_likelihood(
        FIT_RANGES,
        OUTCOME_NAMES,
        pec=pec,
        inputs={comp: trial_inputs(args.n_trials_per_sample)},
        n_parameter_samples=args.n_parameter_samples,
        epochs=args.epochs,
    )
    likelihood.save(args.artifact)
    report_training(likelihood, started, args.artifact)

    print(f"\nfitting {len(data)} trials simulated at "
          f"rate={args.rate}, threshold={args.threshold}", flush=True)
    fit(data, args.artifact)


if __name__ == "__main__":
    main()
