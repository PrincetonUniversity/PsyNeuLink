"""Correctness tests for the driver's ask/tell loop -- no Dask, no model runs.

A FakeClient stands in for ``dask.distributed.Client`` and evaluates a known
analytic objective synchronously. This isolates the driver's bookkeeping:

  * exactly N_ROUNDS * BATCH_SIZE trials are asked and told,
  * candidate values respect FIT_BOUNDS,
  * parameter vectors are sent to workers in FIT_BOUNDS insertion order,
  * each future's score is told back to the *matching* trial,
  * the scattered inputs/data handles (not the raw objects) are what gets
    submitted, alongside WORKER_CORES,
  * CMA-ES actually maximizes (direction="maximize" is wired correctly).
"""

import math

import optuna
import pytest

from pec_dask_mle import config
from pec_dask_mle import worker
from pec_dask_mle.driver import run_fit

PARAM_ORDER = list(config.FIT_BOUNDS.keys())

# Maximum of the test objective; chosen strictly inside every bound.
TARGET = {"rate": 0.25, "threshold": 0.65, "non_decision_time": 0.2}


def neg_quadratic(param_values):
    """Analytic stand-in for the log-likelihood, maximized at TARGET."""
    return -sum(
        (v - TARGET[name]) ** 2 for name, v in zip(PARAM_ORDER, param_values)
    )


class _Future:
    def __init__(self, value):
        self.value = value


class _ScatterHandle:
    """Opaque stand-in for a Dask future returned by scatter()."""

    def __init__(self, obj):
        self.obj = obj


class FakeClient:
    """Synchronous Client double recording every scatter and submit."""

    def __init__(self, eval_fn):
        self.eval_fn = eval_fn
        self.scatter_calls = []   # (obj, broadcast, handle)
        self.submit_calls = []    # (fn, args, kwargs)

    def scatter(self, obj, broadcast=False):
        handle = _ScatterHandle(obj)
        self.scatter_calls.append((obj, broadcast, handle))
        return handle

    def submit(self, fn, *args, **kwargs):
        self.submit_calls.append((fn, args, kwargs))
        param_values = args[0]
        return _Future(self.eval_fn(param_values))

    def gather(self, futures):
        return [f.value for f in futures]

    def scheduler_info(self):
        return {"workers": {"fake-worker": {}}}


@pytest.fixture
def short_run(monkeypatch):
    monkeypatch.setattr(config, "N_ROUNDS", 4)
    monkeypatch.setattr(config, "BATCH_SIZE", 6)


@pytest.fixture
def fitted(short_run):
    """One short fake fit shared by the bookkeeping assertions."""
    client = FakeClient(neg_quadratic)
    study = run_fit(client, data_to_fit="DATA", trial_inputs="INPUTS", verbose=False)
    return client, study


def test_all_evals_asked_and_told(fitted):
    client, study = fitted
    assert len(study.trials) == 24
    assert len(client.submit_calls) == 24
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_candidates_respect_bounds(fitted):
    _, study = fitted
    for t in study.trials:
        for name, (lo, hi) in config.FIT_BOUNDS.items():
            assert lo <= t.params[name] <= hi


def test_param_vector_sent_in_fit_bounds_order(fitted):
    # Trials are asked in the same order submits are issued, so the i-th
    # submitted vector must be the i-th trial's params in FIT_BOUNDS order.
    client, study = fitted
    for t, (_, args, _) in zip(study.trials, client.submit_calls):
        assert args[0] == [t.params[name] for name in PARAM_ORDER]


def test_scores_told_to_matching_trials(fitted):
    # Each trial's stored value must be the objective of *its own* params --
    # a zip misalignment between futures and trials would break this.
    _, study = fitted
    for t in study.trials:
        expected = neg_quadratic([t.params[name] for name in PARAM_ORDER])
        assert t.value == pytest.approx(expected, abs=1e-12)
        assert math.isfinite(t.value)


def test_submits_use_scattered_handles_and_worker_cores(fitted):
    client, _ = fitted
    assert all(broadcast for _, broadcast, _ in client.scatter_calls)
    scattered = {id(h): obj for obj, _, h in client.scatter_calls}
    assert scattered[id(client.submit_calls[0][1][1])] == "INPUTS"
    assert scattered[id(client.submit_calls[0][1][2])] == "DATA"
    for fn, args, kwargs in client.submit_calls:
        assert fn is worker.evaluate_loglik
        assert args[1] is client.submit_calls[0][1][1]
        assert args[2] is client.submit_calls[0][1][2]
        assert args[3] == config.WORKER_CORES
        assert kwargs.get("pure") is False


def test_cmaes_maximizes_toward_target(monkeypatch):
    monkeypatch.setattr(config, "N_ROUNDS", 20)
    monkeypatch.setattr(config, "BATCH_SIZE", 8)
    client = FakeClient(neg_quadratic)
    study = run_fit(client, data_to_fit=None, trial_inputs=None, verbose=False)

    # If the study minimized (or tells were misrouted), best_value would sit
    # near a bounds corner instead of approaching 0.
    assert study.best_value > -0.01
    for name in PARAM_ORDER:
        assert study.best_params[name] == pytest.approx(TARGET[name], abs=0.15)


def test_rejects_one_trial_cmaes_population(monkeypatch):
    monkeypatch.setattr(config, "N_ROUNDS", 2)
    monkeypatch.setattr(config, "BATCH_SIZE", 1)
    client = FakeClient(neg_quadratic)

    with pytest.raises(ValueError, match="BATCH_SIZE/popsize"):
        run_fit(client, data_to_fit=None, trial_inputs=None, verbose=False)
