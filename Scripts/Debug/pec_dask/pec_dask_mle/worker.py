"""The Dask worker task: one parameter proposal -> one scalar log-likelihood.

Pure evaluator. Builds (and caches per Dask worker) a fresh serial PEC from the
factory, then evaluates. Only param values, the broadcast inputs/data, and a
scalar result cross the wire -- never a constructed PEC.
"""

import psyneulink as pnl
from dask.distributed import get_worker

from .factory import build_pec

# Fallback cache used when called outside a Dask worker (e.g. a serial sanity
# check). Inside Dask the cache is stashed on the worker object instead.
_FALLBACK_CACHE = {}


def evaluate_loglik(param_values, trial_inputs, data_to_fit, worker_cores):
    """Return ``pec.log_likelihood(*param_values)`` for one candidate.

    The PEC is built once per worker and cached, so the model is only
    constructed/compiled a single time and reused for every evaluation.
    """
    try:
        worker = get_worker()
        cache = getattr(worker, "_pec_cache", None)
    except ValueError:
        worker = None
        cache = _FALLBACK_CACHE.get("pec")

    if cache is None:
        # One-time setup: pin LLVM threads to this worker's cores, then build +
        # compile the PEC once.
        pnl.set_num_threads(worker_cores)
        cache = build_pec(data_to_fit)
        if worker is not None:
            worker._pec_cache = cache
        else:
            _FALLBACK_CACHE["pec"] = cache

    pec, comp = cache
    ll = pec.log_likelihood(*param_values, inputs={comp: trial_inputs})
    return float(ll)
