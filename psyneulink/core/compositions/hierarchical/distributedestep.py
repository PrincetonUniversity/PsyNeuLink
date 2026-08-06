# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ************************************  Distributed E-step  ***********************************************************

"""E-step distributed across a Dask cluster.

One task per participant: build that participant's model on a worker, fit it, return the posterior
summary.  The M-step stays on the driver, so this runner is interchangeable with the in-process one.

Building a participant's model constructs a composition and compiles it, so each worker caches the
models it builds.  That only pays off if a participant keeps landing on the same worker, hence the
pinning after the first iteration; otherwise every worker ends up holding a model for every
participant, which is how a large fit exhausts memory.

Dask is imported inside the functions that need it, so this module can be imported, and its worker
task exercised, without it.
"""

import uuid

import numpy as np

from psyneulink.core.components.functions.nonstateful import fitfunctions as _fitfunctions
from psyneulink.core.compositions.hierarchical.laplaceem import (
    EStepConfig,
    EStepResult,
    log_gauss_diag,
    subject_laplace_objective,
    subject_map_estep,
)
from psyneulink.core.compositions.hierarchical.transforms import BoundedTransform

__all__ = ["make_distributed_estep_runner"]

#: Per-worker cache of participant models, used when no Dask worker context is available.  Keeping
#: it module-level lets the task body be exercised in-process, without a cluster.
_SUBJECT_FALLBACK_CACHE = {}


def _worker_subject_cache(fit_id):
    """Return the cache dict for this worker, creating it if needed."""
    try:
        from dask.distributed import get_worker
        worker = get_worker()
    except (ImportError, ValueError):
        # No worker context: either Dask is absent, or this is running on the driver.
        return _SUBJECT_FALLBACK_CACHE
    cache = getattr(worker, "_hierarchical_subject_cache", None)
    if cache is None:
        cache = worker._hierarchical_subject_cache = {}
    return cache


def _dask_subject_estep(
    pec_factory, subject_index, data_slice, mu_s, sigma, lower, upper, z0, worker_cores,
    fit_id, config,
):
    """Fit one participant on a worker and return their posterior summary.

    The whole call holds the evaluation lock, not just the model construction: two compiled models
    driven at once in a single process is what the lock exists to prevent.
    """
    with _fitfunctions._PEC_EVALUATION_LOCK:
        cache = _worker_subject_cache(fit_id)
        key = (fit_id, subject_index)
        if key not in cache:
            from psyneulink.core.globals.threads import set_num_threads
            if worker_cores is not None:
                set_num_threads(worker_cores)
            cache[key] = pec_factory(data_slice, subject_index)
        pec, inputs = cache[key]

        transform = BoundedTransform(lower=lower, upper=upper)
        sigma = np.asarray(sigma, dtype=float)
        mu_s = np.asarray(mu_s, dtype=float)

        def neg_log_post(z):
            theta = transform.to_natural(z)
            return -float(pec.log_likelihood(*theta, inputs=inputs)) - log_gauss_diag(z, mu_s, sigma)

        post = subject_map_estep(neg_log_post, z0=z0, prior_variance=sigma, config=config)

        worker_address = None
        try:
            from dask.distributed import get_worker
            worker_address = get_worker().address
        except (ImportError, ValueError):
            pass

        return subject_index, post, worker_address


def make_distributed_estep_runner(
    client, pec_factory, data_slices, bounds, *, config=None, worker_cores=None, fit_id=None,
):
    """Build an E-step that fits every participant at once, across a cluster.

    Arguments
    ---------

    client : dask.distributed.Client
        Cluster to submit to.

    pec_factory : callable
        ``pec_factory(data, subject_index=None) -> (pec, inputs)``.  Must be importable on the
        workers, so it has to be defined at module level rather than nested or bound.

    data_slices : sequence of pandas.DataFrame
        One participant's trials each, in participant order.  Only the relevant slice is sent to
        each task.

    bounds : tuple
        ``(lower, upper)`` arrays of the search range, shared by every participant.

    config : EStepConfig : default None
        Settings for each participant's optimization.

    worker_cores : int : default None
        Threads to give the compiled model on a worker.

    fit_id : str : default None
        Identifies this fit in the per-worker model cache.  Reusing a value across calls against the
        same cluster lets the workers keep the models they already built.

    Returns
    -------

    A callable ``runner(mu, sigma, prev_z, warm_start) -> EStepResult``, interchangeable with the
    in-process runner.
    """
    config = config if config is not None else EStepConfig()
    fit_id = fit_id or uuid.uuid4().hex
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    data_slices = list(data_slices)
    # Filled in after the first iteration, so each participant returns to the worker holding
    # their model instead of being rebuilt somewhere else.
    home = {}

    def runner(mu, sigma, prev_z, warm_start):
        n_subjects, n_params = mu.shape
        futures = []
        for s in range(n_subjects):
            z0 = np.asarray(prev_z[s] if warm_start else mu[s], dtype=float)
            submit_kwargs = {"pure": False}
            if s in home:
                # allow_other_workers so a dead worker reschedules rather than hanging; the
                # participant re-pins on the next iteration.
                submit_kwargs.update(workers=[home[s]], allow_other_workers=True)
            futures.append(client.submit(
                _dask_subject_estep, pec_factory, s, data_slices[s], mu[s], sigma,
                lower, upper, z0, worker_cores, fit_id, config, **submit_kwargs,
            ))

        z_hat = np.empty((n_subjects, n_params))
        variance = np.empty((n_subjects, n_params))
        curvature = np.empty((n_subjects, n_params))
        steps = np.empty((n_subjects, n_params))
        subject_objective = np.empty(n_subjects)
        success = np.empty(n_subjects, dtype=bool)
        messages = []

        for subject_index, post, worker_address in client.gather(futures):
            # Results are placed by participant index rather than in completion order, so that a
            # distributed fit and an in-process one agree exactly.
            z_hat[subject_index] = post.z_hat
            variance[subject_index] = post.variance
            curvature[subject_index] = post.curvature
            steps[subject_index] = post.hessian_step
            subject_objective[subject_index] = subject_laplace_objective(
                post.neg_log_post, post.variance, n_params
            )
            success[subject_index] = post.success
            if not post.success:
                messages.append((subject_index, post.message))
            if worker_address is not None:
                home[subject_index] = worker_address

        return EStepResult(
            z_hat=z_hat,
            variance=variance,
            curvature=curvature,
            hessian_step=steps,
            subject_objective=subject_objective,
            success=success,
            messages=tuple(messages),
        )

    return runner
