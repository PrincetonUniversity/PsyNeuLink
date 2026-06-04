"""The driver: owns the Optuna study and runs the ask/tell loop over Dask.

CMA-ES lives here, in the driver process. Each round it asks a batch of
candidate parameter sets, submits them to Dask workers as log-likelihood
evaluations, gathers the scalar scores, and tells them back to the study.
Workers never touch Optuna.

``run_fit`` takes an already-constructed Dask ``client``, so the same driver
works for a single-node ``LocalCluster`` (run_dask_mle_local.py) and for a
multi-node scheduler (Option B) without change.
"""

import optuna
import pandas as pd
from optuna.distributions import FloatDistribution

from . import config
from .worker import evaluate_loglik


def run_fit(client, data_to_fit, trial_inputs, *, verbose=True):
    """Run the distributed ask/tell MLE fit. Returns the completed Optuna study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Broadcast the immutable inputs/data once so they are not re-serialized on
    # every submit.
    inputs_f = client.scatter(trial_inputs, broadcast=True)
    data_f = client.scatter(data_to_fit, broadcast=True)

    param_order = list(config.FIT_BOUNDS.keys())
    distributions = {
        name: FloatDistribution(lo, hi) for name, (lo, hi) in config.FIT_BOUNDS.items()
    }

    # Single authoritative study in the driver -- no shared storage.
    study = optuna.create_study(
        sampler=optuna.samplers.CmaEsSampler(seed=0), direction="maximize"
    )

    n_batches = config.TOTAL_EVALS // config.BATCH_SIZE
    for b in range(n_batches):
        # Ask a batch of proposals. (BATCH_SIZE need not equal the CMA-ES
        # population; Optuna accumulates tells into generations internally.)
        trials = [study.ask(distributions) for _ in range(config.BATCH_SIZE)]
        futures = [
            client.submit(
                evaluate_loglik,
                [t.params[name] for name in param_order],
                inputs_f, data_f, config.WORKER_CORES,
                pure=False,
            )
            for t in trials
        ]
        values = client.gather(futures)
        for t, v in zip(trials, values):
            study.tell(t, v)

        if verbose:
            # Live worker count -- workers join elastically as their SLURM jobs
            # schedule, so this ramps up over the first few batches.
            n_workers = len(client.scheduler_info()["workers"])
            print(
                f"batch {b + 1:>3}/{n_batches}  workers={n_workers}  "
                f"best_ll={study.best_value:.4f}  "
                f"best={ {k: round(v, 4) for k, v in study.best_params.items()} }",
                flush=True,
            )

    return study


def summarize(study):
    """Print a recovered-vs-true parameter table for the synthetic test."""
    param_order = list(config.FIT_BOUNDS.keys())
    print("\nBest params:", study.best_params)
    records = []
    for name in param_order:
        rec = study.best_params[name]
        pct = 100.0 * abs(config.TRUE_PARAMS[name] - rec) / config.TRUE_PARAMS[name]
        records.append((name, config.TRUE_PARAMS[name], rec, pct))
    print(
        pd.DataFrame(
            records, columns=["Parameter", "True", "Recovered", "Percent Error"]
        )
    )
