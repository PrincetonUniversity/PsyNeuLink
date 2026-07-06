"""Distributed E-step: one subject MAP per Dask task (M3).

Mirrors the candidate-level worker pattern in fitfunctions.py but distributes over subjects: each
task builds (or reuses) that subject's PEC on a worker and returns its Laplace posterior. The PEC
cache is keyed by ``(fit_id, subject_id)`` because, unlike the candidate-level case, each task uses
a different subject's data. The driver runs the shared M-step, so serial and distributed EM differ
only in how the E-step is executed.
"""

import threading
import uuid

import numpy as np

from transforms import BoundedTransform
from estep import subject_map_estep
from laplace_em import _log_gauss_diag, subject_laplace_objective

_SUBJECT_ESTEP_LOCK = threading.Lock()
_SUBJECT_FALLBACK_CACHE = {}


def _dask_subject_estep(
    pec_factory, subject_id, payload, mu_s, sigma, lower, upper, z0,
    worker_cores, fit_id, estep_kwargs,
):
    """One subject E-step on a worker: cached PEC -> MAP -> diagonal Laplace posterior."""
    with _SUBJECT_ESTEP_LOCK:
        try:
            from dask.distributed import get_worker
            worker = get_worker()
            cache = getattr(worker, "_subject_pec_cache", None)
            if cache is None:
                cache = worker._subject_pec_cache = {}
        except (ImportError, ValueError):
            cache = _SUBJECT_FALLBACK_CACHE

        key = (fit_id, subject_id)
        if key not in cache:
            from psyneulink.core.globals.threads import set_num_threads
            if worker_cores is not None:
                set_num_threads(worker_cores)
            cache[key] = pec_factory(payload)
        pec, inputs = cache[key]

    transform = BoundedTransform(lower, upper)
    sigma = np.asarray(sigma, dtype=float)
    mu_s = np.asarray(mu_s, dtype=float)

    def neg_log_post(z):
        theta = transform.to_natural(z)
        return -float(pec.log_likelihood(*theta, inputs=inputs)) - _log_gauss_diag(z, mu_s, sigma)

    post = subject_map_estep(neg_log_post, z0, prior_variance=sigma, **estep_kwargs)
    return subject_id, post.z_hat, post.variance, post.curvature, post.neg_log_post


def make_distributed_estep_runner(
    client, pec_factory, payloads, lower, upper, *,
    worker_cores=None, fit_id=None,
    method="Nelder-Mead", hessian_step=1e-4, variance_floor=1e-6, estep_options=None,
):
    """E-step runner that submits one task per subject to a Dask client."""
    fit_id = fit_id or uuid.uuid4().hex
    estep_kwargs = dict(
        method=method, hessian_step=hessian_step,
        variance_floor=variance_floor, optimizer_options=estep_options,
    )
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def runner(mu, sigma, prev_z, warm_start):
        n_subjects, n_params = mu.shape
        futures = []
        for s in range(n_subjects):
            z0 = prev_z[s] if warm_start else mu[s]
            futures.append(client.submit(
                _dask_subject_estep, pec_factory, s, payloads[s], mu[s], sigma, lower, upper,
                np.asarray(z0, dtype=float), worker_cores, fit_id, estep_kwargs, pure=False,
            ))

        z_hat = np.empty((n_subjects, n_params))
        variance = np.empty((n_subjects, n_params))
        curvature = np.empty((n_subjects, n_params))
        objective = 0.0
        for sid, zh, var_s, curv, nlp in client.gather(futures):
            z_hat[sid] = zh
            variance[sid] = var_s
            curvature[sid] = curv
            objective += subject_laplace_objective(nlp, var_s, n_params)
        return z_hat, variance, curvature, objective

    return runner
