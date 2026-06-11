"""Model-agnostic Dask worker + ask/tell driver for the benchmark.

This module is IMPORTABLE (not __main__) so dask-jobqueue workers on other nodes
can resolve ``bench_core.evaluate_loglik`` by reference. It depends only on the
``models`` provider registry, so it never needs to know which model is running.
"""

import time

import optuna
from optuna.distributions import FloatDistribution

import models

# Per-process fallback cache for non-Dask (serial) calls. Inside Dask the cache
# lives on the worker object instead.
_FALLBACK = {}


def evaluate_loglik(model_name, param_values, num_trials, data, num_estimates,
                    worker_cores):
    """One candidate -> one scalar log-likelihood, for any model.

    Rebuilds the composition + inputs + PEC locally from the provider (cached per
    worker), so only model_name + scalars + the data DataFrame cross the wire.
    """
    import psyneulink as pnl
    from dask.distributed import get_worker

    try:
        worker = get_worker()
        cache = getattr(worker, "_pec_cache", None)
    except ValueError:
        worker = None
        cache = _FALLBACK.get("pec")

    if cache is None:
        pnl.set_num_threads(worker_cores)
        provider = models.get(model_name)
        comp = provider.build_comp()
        inputs = provider.make_inputs(comp, num_trials)
        pec = provider.build_pec(comp, data, num_estimates)
        cache = (pec, inputs)
        if worker is not None:
            worker._pec_cache = cache
        else:
            _FALLBACK["pec"] = cache

    pec, inputs = cache
    return float(pec.log_likelihood(*param_values, inputs=inputs))


def run_fit(client, model_name, data, *, num_trials, num_estimates, total_evals,
            batch_size, worker_cores, fit_bounds, verbose=False):
    """Driver: owns one CMA-ES study; ask a batch -> submit -> gather -> tell.

    The CMA-ES population is pinned to ``batch_size`` independently of Dask
    worker count, so a fixed ``n_rounds = total_evals // batch_size`` gives the
    same optimizer trajectory while Dask only changes evaluation concurrency.
    """
    if batch_size < 2:
        raise ValueError(
            "CMA-ES requires popsize/batch_size >= 2."
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Broadcast once, not per submit. hash=False gives this fit a unique key:
    # with content-hashed keys, a SECOND fit on the same data races against the
    # release of the first fit's key and gets "lost dependencies" cancellations.
    data_f = client.scatter(data, broadcast=True, hash=False)
    param_order = list(fit_bounds)
    distributions = {n: FloatDistribution(lo, hi) for n, (lo, hi) in fit_bounds.items()}

    study = optuna.create_study(
        sampler=optuna.samplers.CmaEsSampler(seed=0, popsize=batch_size),
        direction="maximize",
    )

    n_batches = total_evals // batch_size
    for b in range(n_batches):
        trials = [study.ask(distributions) for _ in range(batch_size)]
        futures = [
            client.submit(
                evaluate_loglik,
                model_name,
                [t.params[n] for n in param_order],
                num_trials, data_f, num_estimates, worker_cores,
                pure=False,
            )
            for t in trials
        ]
        values = client.gather(futures)
        for t, v in zip(trials, values):
            study.tell(t, v)
        if verbose:
            n_up = len(client.scheduler_info()["workers"])
            print(f"batch {b + 1:>3}/{n_batches}  workers={n_up}  "
                  f"best_ll={study.best_value:.4f}", flush=True)

    return study
