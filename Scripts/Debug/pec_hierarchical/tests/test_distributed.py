"""Integration tests over a real process-based Dask LocalCluster.

Workers run in separate processes, exactly like the SLURM deployment. (An
in-process threaded cluster is NOT usable here: PNL's LLVM runtime asserts on
cleanup when compiled executions run on multiple threads of one process.)

Because worker processes re-import the real ``config``, monkeypatches only
reach the driver side. That is enough: workers read only NUM_ESTIMATES /
INITIAL_SEED / FIT_BOUNDS, so the tests shrink the driver-side knobs
(NUM_TRIALS for the synthetic data, TOTAL_EVALS / BATCH_SIZE for the loop) and
keep NUM_ESTIMATES at full size on both sides so serial and distributed
evaluations are comparable.

Pins down:
  * distributed evaluation returns exactly what serial evaluation returns for
    the same candidates (which worker scored a candidate must not matter),
  * run_fit completes end-to-end on a real client, with every score finite,
    every candidate in bounds, and the best score equal to the best tell,
  * each worker built and cached its PEC.
"""

import math

import numpy as np
import optuna
import pytest
from dask.distributed import Client, LocalCluster

from pec_dask_mle import config
from pec_dask_mle.data import make_data
from pec_dask_mle.driver import run_fit
from pec_dask_mle.worker import evaluate_loglik

pytestmark = [pytest.mark.dask, pytest.mark.pnl]


@pytest.fixture(scope="module")
def small_data_config():
    """Shrink driver-side knobs only; worker-side config stays full-size."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "NUM_TRIALS", 10)
        yield config


@pytest.fixture(scope="module")
def client():
    # Module-scoped: both tests share the workers, so the second test also
    # exercises PEC-cache reuse across fits.
    with LocalCluster(n_workers=2, threads_per_worker=1) as cluster:
        with Client(cluster) as c:
            yield c


@pytest.fixture(scope="module")
def small_data(small_data_config):
    return make_data()


def test_distributed_scores_match_serial(small_data, client):
    data, inputs = small_data
    candidates = [
        [0.3, 0.6, 0.15],   # true params
        [0.1, 0.8, 0.3],
        [-0.2, 0.55, 0.05],
    ]

    serial = [
        evaluate_loglik(c, inputs, data, worker_cores=config.WORKER_CORES)
        for c in candidates
    ]

    inputs_f = client.scatter(inputs, broadcast=True)
    data_f = client.scatter(data, broadcast=True)
    futures = [
        client.submit(
            evaluate_loglik, c, inputs_f, data_f, config.WORKER_CORES, pure=False
        )
        for c in candidates
    ]
    distributed = client.gather(futures)

    np.testing.assert_allclose(distributed, serial, rtol=1e-10)


def test_run_fit_end_to_end(small_data, client, monkeypatch):
    monkeypatch.setattr(config, "TOTAL_EVALS", 8)
    monkeypatch.setattr(config, "BATCH_SIZE", 4)
    data, inputs = small_data

    study = run_fit(client, data, inputs, verbose=False)

    assert len(study.trials) == 8
    for t in study.trials:
        assert t.state == optuna.trial.TrialState.COMPLETE
        assert math.isfinite(t.value)
        for name, (lo, hi) in config.FIT_BOUNDS.items():
            assert lo <= t.params[name] <= hi
    assert study.best_value == max(t.value for t in study.trials)

    # Every worker that evaluated must hold a cached PEC (built exactly once
    # per worker; the cache is what keeps repeat evaluations cheap).
    def has_cache(dask_worker):
        return hasattr(dask_worker, "_pec_cache")

    assert all(client.run(has_cache).values())
