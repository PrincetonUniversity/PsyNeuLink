"""Sample the posterior of a group of participants with the No-U-Turn sampler.

Read this one if you want intervals taken from the posterior itself, where
``hierarchical_fitting.py`` reports a best estimate with a Gaussian width around it.  Sampling
scores the model with a trained neural likelihood, so the script

1. simulates a group of drift-diffusion participants, or reads ``--data``;
2. trains a neural likelihood for the model, or loads the one an earlier run saved;
3. samples the group and every participant jointly;
4. checks whether the draws can be trusted, and prints estimates with their 95% intervals.

It needs the ``nle`` extra::

    pip install "psyneulink[nle]"

A quick run, to see every step work.  On 8 cores it takes about 15 minutes the first time, most
of it training, and a few minutes once the estimator is saved.  It trains too little and draws
too few samples for the estimates to be relied on::

    python hierarchical_sampling.py

A run of the size meant for inference.  Training is the slow part, and is done once: later runs
load the saved estimator::

    python hierarchical_sampling.py --artifact ddm_nle_large.pt \\
        --n-parameter-samples 8000 --epochs 30 --draws 1000 --warmup 1000 --chains 4

With your own data, in the format ``hierarchical_fitting.py`` describes::

    python hierarchical_sampling.py --data group_data.csv --artifact ddm_nle_large.pt
"""

import argparse
import functools
import os
import time

import numpy as np

import psyneulink as pnl

from hierarchical_fitting import FIT_PARAMS, FIT_RANGES, build_model, load_data, trial_inputs
from make_example_data import make_group_data

OUTCOME_NAMES = ("decision", "response_time")


def participant_pec(likelihood, data, subject_index=None):
    """Build one participant's model from their trials.

    With a trained `likelihood` the model is scored by it; with None it is scored by simulation,
    which is the form the estimator is trained from.
    """
    comp, decision = build_model()
    scoring = {}
    if likelihood is not None:
        scoring = {
            "likelihood_estimator": "neural",
            "likelihood_estimator_kwargs": {"artifact": likelihood},
        }
    participant = pnl.ParameterEstimationComposition(
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
        **scoring,
    )
    return participant, {comp: trial_inputs(len(data))}


def trained_likelihood(args, example_trials):
    """Load the estimator saved at ``args.artifact``, or train one and save it there.

    An estimator is valid for this model over `FIT_RANGES`, whatever data it is later used to
    fit, so it is trained once and reused.
    """
    if os.path.exists(args.artifact):
        print(f"loading the neural likelihood saved at {args.artifact}", flush=True)
        return pnl.NeuralLikelihood.load(args.artifact)

    print("training a neural likelihood", flush=True)
    started = time.time()
    pec, _ = participant_pec(None, example_trials)
    likelihood = pnl.train_neural_likelihood(
        FIT_RANGES,
        OUTCOME_NAMES,
        pec=pec,
        inputs={pec.model: trial_inputs(args.n_trials_per_sample)},
        n_parameter_samples=args.n_parameter_samples,
        epochs=args.epochs,
        seed=args.seed,
    )
    likelihood.save(args.artifact)
    print(f"  trained in {(time.time() - started) / 60:.1f} min, "
          f"held-out NLL {likelihood.metadata.val_nll:.4f} per trial; saved to {args.artifact}",
          flush=True)
    return likelihood


def report(results, theta_true=None):
    """Print whether the draws can be trusted, and then what they say."""
    print(f"\n{results!r}")

    # Whether the chains agree (r_hat near 1) and how many independent draws they are worth
    # (ess), for the group and for each participant; and whether the sampler lost the posterior
    # anywhere (divergences). These come before the estimates: a fit that fails them has not
    # described the posterior.
    print("\nconvergence of the group means and spreads:")
    print(results.convergence.round(3).to_string())
    participants = results.subject_posteriors
    print(f"\nconvergence of the participants' parameters: "
          f"worst r_hat {participants['r_hat'].max():.3f}, "
          f"smallest ess {participants['ess'].min():.0f}")
    diagnostics = results.diagnostics
    print(f"divergences per chain: {diagnostics.divergences.tolist()}")
    print(f"mean acceptance probability per chain: {diagnostics.accept_rate.round(2).tolist()}")
    if not results.converged:
        print("\nNOT CONVERGED. Raise --draws and --warmup if r_hat is above 1.01, and "
              "--target-accept if there are divergences. The estimates below are not to be "
              "relied on.")

    print("\ngroup estimate, with its 95% interval, in the model's units:")
    print(results.group_parameters[["value", "lower_95", "upper_95"]].round(3).to_string())

    print("\nper-participant estimates, with their 95% intervals:")
    table = participants.set_index(["subject", "parameter"])[
        ["theta_mean", "theta_lower_95", "theta_upper_95"]
    ]
    if theta_true is not None:
        table.insert(0, "simulated_from", np.ravel(theta_true))
    print(table.round(3).to_string())

    # Anything else can be read off the draws, which are (chains, draws, participants,
    # parameters): here, how sure the fit is that the first participant's rate is the higher
    # of the first two.
    rate = results.subject_draws[..., FIT_PARAMS.index("rate")]
    first, second = results.subject_labels[:2]
    print(f"\nprobability that {first}'s rate exceeds {second}'s: "
          f"{np.mean(rate[..., 0] > rate[..., 1]):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None,
                        help="stacked trial table to fit; a group is simulated when omitted")
    parser.add_argument("--subject-id", default="subject",
                        help="column of the table naming who produced each row")
    parser.add_argument("--n-participants", type=int, default=6,
                        help="participants in the simulated group")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="trials per participant in the simulated group")
    parser.add_argument("--artifact", default="ddm_nle.pt",
                        help="where the trained estimator is saved, and loaded from if present")
    parser.add_argument("--n-parameter-samples", type=int, default=512)
    parser.add_argument("--n-trials-per-sample", type=int, default=1000,
                        help="trials simulated at each parameter draw in training")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--draws", type=int, default=200, help="draws kept per chain")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the simulated group, training and the sampler")
    args = parser.parse_args()

    theta_true = None
    if args.data is None:
        data, theta_true = make_group_data(args.n_participants, args.n_trials, seed=args.seed)
    else:
        data = load_data(args.data)
    print(f"{len(data)} trials from {data[args.subject_id].nunique()} participants")

    likelihood = trained_likelihood(args, data.drop(columns=args.subject_id))

    # Every participant's model is given the same estimator object, so the sampler scores them
    # all in one call to it.
    group_fit = pnl.ParameterEstimationComposition(
        name="group",
        data=data,
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": args.subject_id,
            "sampler": "nuts",
            "sampler_options": {
                "draws": args.draws,
                "warmup": args.warmup,
                "chains": args.chains,
                "target_accept": args.target_accept,
                # With the same seed, data and estimator, a run gives the same draws.
                "seed": args.seed,
            },
        },
        distributed_options={"pec_factory": functools.partial(participant_pec, likelihood)},
    )

    started = time.time()
    results = group_fit.run()
    print(f"sampled in {time.time() - started:.1f}s")

    report(results, theta_true)


if __name__ == "__main__":
    main()
