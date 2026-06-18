"""Model-agnostic Dask worker + ask/tell driver for the benchmark.

This module is IMPORTABLE (not __main__) so dask-srun workers on other nodes
can resolve ``bench_core.evaluate_loglik`` by reference. It depends only on the
``models`` provider registry, so it never needs to know which model is running.
"""

import time

import optuna
from optuna.distributions import FloatDistribution

import models

# Selectable Optuna samplers (see make_sampler). Order = display order.
SAMPLERS = ("cmaes", "tpe", "tpe_noliar", "random", "qmc", "gp", "nsga2")

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
        # The worker re-scores one fixed PEC structure many times, so keep its
        # compiled binary across evals instead of tearing it down per call.
        from psyneulink.core import llvm as pnllvm
        pnllvm.cleanup = lambda *a, **k: None
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


def make_sampler(name, seed, popsize):
    """Build an Optuna sampler by short name, for the synchronous batch ask/tell
    loop used by both the dask driver and the regular baseline.

    ``popsize`` is the per-round batch size (== n_workers in this study). Only
    the generational samplers (cmaes, nsga2) consume it; the rest ignore it and
    simply have ``popsize`` candidates asked before each tell. ``seed`` fixes the
    trajectory so a config is reproducible.

    Sampler-specific knobs reflect the synchronous-batch setting:
      tpe   -- constant_liar=True so the batch's asks (issued before any tell)
               stay diverse instead of collapsing onto one point.
      qmc   -- scramble=True for a randomized-but-reproducible low-discrepancy
               sequence (better coverage than the raw Sobol points).
      gp    -- deterministic_objective=True: PEC with common random numbers is
               deterministic in the params, which the GP can exploit. The GP fit
               is a SERIAL driver cost (not overlapped with workers) that grows
               with trial count, so it is the one sampler whose throughput is not
               worker-bound.
    """
    s = optuna.samplers
    if name == "cmaes":
        return s.CmaEsSampler(seed=seed, popsize=popsize)
    if name == "tpe":
        return s.TPESampler(seed=seed, constant_liar=True)
    if name == "tpe_noliar":
        return s.TPESampler(seed=seed, constant_liar=False)
    if name == "random":
        return s.RandomSampler(seed=seed)
    if name == "qmc":
        return s.QMCSampler(seed=seed, scramble=True)
    if name == "gp":
        return s.GPSampler(seed=seed, deterministic_objective=True)
    if name == "nsga2":
        return s.NSGAIISampler(seed=seed, population_size=popsize)
    raise ValueError(f"unknown sampler '{name}'")


# Samplers that require a population/batch of at least 2 candidates per round.
_NEEDS_POPSIZE_GE_2 = {"cmaes", "nsga2"}


def run_fit(client, model_name, data, *, num_trials, num_estimates, total_evals,
            batch_size, worker_cores, fit_bounds, sampler="cmaes", verbose=False):
    """Driver: owns one Optuna study; ask a batch -> submit -> gather -> tell.

    The population/batch is pinned to ``batch_size`` (== n_workers in this
    study), so a fixed ``n_rounds = total_evals // batch_size`` defines the
    optimizer trajectory while Dask only changes evaluation concurrency. The
    sampler is pluggable (see ``make_sampler``); the ask/tell loop is identical
    for every sampler, which is what lets us compare them on equal footing.
    """
    if sampler in _NEEDS_POPSIZE_GE_2 and batch_size < 2:
        raise ValueError(f"sampler '{sampler}' requires popsize/batch_size >= 2.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Broadcast once, not per submit. hash=False gives this fit a unique key:
    # with content-hashed keys, a SECOND fit on the same data races against the
    # release of the first fit's key and gets "lost dependencies" cancellations.
    data_f = client.scatter(data, broadcast=True, hash=False)
    param_order = list(fit_bounds)
    distributions = {n: FloatDistribution(lo, hi) for n, (lo, hi) in fit_bounds.items()}

    study = optuna.create_study(
        sampler=make_sampler(sampler, seed=0, popsize=batch_size),
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
