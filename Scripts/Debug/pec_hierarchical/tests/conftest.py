"""Shared fixtures for the pec_dask_mle correctness suite."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)  # .../pec_hierarchical
# The repo checkout must shadow any pip-installed psyneulink: the prototype
# depends on this branch (e.g. log_likelihood(..., return_sim_data=...),
# set_num_threads).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PKG_ROOT)))
for path in (PKG_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
# Inherited by any spawned Dask worker processes (mirrors the entrypoints).
os.environ["PYTHONPATH"] = (
    REPO_ROOT + os.pathsep + PKG_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
)


@pytest.fixture(autouse=True)
def clear_worker_cache():
    """Every test starts and ends with no serially-cached PEC.

    evaluate_loglik caches the first PEC it builds keyed only by "pec", so a
    PEC built under one test's config must not leak into the next test.
    """
    from pec_dask_mle import worker

    worker._FALLBACK_CACHE.clear()
    yield
    worker._FALLBACK_CACHE.clear()


@pytest.fixture
def small_problem(monkeypatch):
    """Shrink the problem so PEC evaluations are test-sized.

    config attributes are read at call time everywhere (factory, data, driver),
    so monkeypatching the module is sufficient for in-process evaluation and
    for threaded (processes=False) Dask workers. It does NOT reach separate
    worker processes -- distributed tests must use a threaded LocalCluster.
    """
    from pec_dask_mle import config

    monkeypatch.setattr(config, "NUM_TRIALS", 10)
    monkeypatch.setattr(config, "NUM_ESTIMATES", 50)
    monkeypatch.setattr(config, "WORKER_CORES", 1)
    return config
