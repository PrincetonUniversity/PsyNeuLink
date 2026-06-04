"""Dask-distributed PEC maximum-likelihood fitting (ask/tell backend, prototype).

Architecture (see ../PLAN_DASK.md and ../CONCEPTS.md):

  * The DRIVER owns one Optuna study (CMA-ES). Each round it asks a batch of
    candidate parameter sets, submits them to Dask as log-likelihood
    EVALUATIONS, gathers the scalar scores, and tells them back to the study.
  * WORKERS are pure evaluators. Each builds a fresh *serial* PEC from the
    factory (cached per worker) and returns pec.log_likelihood(*params, ...).
    A constructed PEC is never shipped over the wire -- only the factory
    (imported by reference), lightweight payloads, and scalar results are.

Modules:
  config   -- problem/run constants (pure data; no heavy imports)
  factory  -- build_pec, the PEC "recipe" each worker builds locally
  worker   -- evaluate_loglik, the per-worker pure evaluator
  data     -- make_data, synthetic test-harness data generation
  driver   -- run_fit, the ask/tell loop over a Dask client

Submodules are intentionally NOT imported here so that importing `config` on a
worker does not drag in PsyNeuLink.
"""

import warnings

# Silence a noisy FutureWarning from the third-party `graph_scheduler` dependency
# (it assigns `functools.partial(...)` as class attributes, which Python 3.13
# flags as a future method-descriptor change). The fix belongs upstream; we only
# suppress this one message. This runs before any `psyneulink` import in our
# code path (driver and workers both import this package first), so it covers
# every process.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"functools\.partial will be a method descriptor.*",
)
