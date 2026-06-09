"""Correctness tests for the worker-side evaluation path (serial, real PNL).

These exercise the actual PsyNeuLink PEC, so they pay one LLVM compile per
fresh PEC build. They pin down:

  * synthetic data generation is reproducible and well-formed,
  * the positional order of pec.log_likelihood(*params) matches FIT_BOUNDS --
    the contract the driver relies on when flattening trial params,
  * evaluate_loglik builds the PEC once and caches it,
  * the log-likelihood is deterministic across calls AND across fresh PEC
    builds (the property that makes results independent of which worker /
    scheduling order evaluated a candidate),
  * the likelihood surface discriminates true parameters from distant ones.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pec_dask_mle import config, worker
from pec_dask_mle.data import make_data
from pec_dask_mle.factory import build_pec
from pec_dask_mle.worker import evaluate_loglik

pytestmark = pytest.mark.pnl

TRUE_VALUES = [config.TRUE_PARAMS[name] for name in config.FIT_BOUNDS]


def test_make_data_reproducible(small_problem):
    data1, inputs1 = make_data()
    data2, inputs2 = make_data()
    pd.testing.assert_frame_equal(data1, data2)
    np.testing.assert_array_equal(inputs1, inputs2)


def test_make_data_well_formed(small_problem):
    data, inputs = make_data()
    assert inputs.shape == (small_problem.NUM_TRIALS, 1)
    assert set(np.unique(inputs)) <= {5.0, -5.0}
    assert list(data.columns) == ["decision", "response_time"]
    assert len(data) == small_problem.NUM_TRIALS
    assert data["decision"].dtype == "category"
    assert set(data["decision"].unique()) <= {0.0, 1.0}
    # Every RT includes the non-decision time plus at least one step.
    assert (data["response_time"] > config.TRUE_PARAMS["non_decision_time"]).all()


def test_fit_param_order_matches_pec(small_problem):
    # The driver flattens trial params in FIT_BOUNDS insertion order and the
    # worker splats them positionally into pec.log_likelihood. That is only
    # correct if the PEC's own fit-parameter order agrees.
    data, _ = make_data()
    pec, _ = build_pec(data)
    # fit_param_names are qualified as "<node>.<param>"; the suffix order is
    # the positional order log_likelihood expects.
    unqualified = [n.split(".")[-1] for n in pec.controller.function.fit_param_names]
    assert unqualified == list(config.FIT_BOUNDS)


def test_evaluate_loglik_builds_once_and_is_deterministic(small_problem, monkeypatch):
    data, inputs = make_data()

    builds = []
    real_build = worker.build_pec

    def counting_build(d):
        builds.append(1)
        return real_build(d)

    monkeypatch.setattr(worker, "build_pec", counting_build)

    ll1 = evaluate_loglik(TRUE_VALUES, inputs, data, worker_cores=1)
    ll2 = evaluate_loglik(TRUE_VALUES, inputs, data, worker_cores=1)

    assert len(builds) == 1, "PEC must be built once and cached per worker"
    assert isinstance(ll1, float) and math.isfinite(ll1)
    assert ll1 == ll2


def test_loglik_identical_across_fresh_builds(small_problem):
    # Common random numbers (fixed initial_seed + same_seed_for_all_parameter_
    # combinations) must make the score a pure function of the params, no
    # matter which freshly-built PEC -- i.e. which worker -- evaluates it.
    data, inputs = make_data()

    ll_first = evaluate_loglik(TRUE_VALUES, inputs, data, worker_cores=1)
    worker._FALLBACK_CACHE.clear()  # force a rebuild, as on a different worker
    ll_rebuilt = evaluate_loglik(TRUE_VALUES, inputs, data, worker_cores=1)

    assert ll_first == pytest.approx(ll_rebuilt, rel=1e-10)


def test_loglik_discriminates_true_from_distant_params(small_problem):
    data, inputs = make_data()

    ll_true = evaluate_loglik(TRUE_VALUES, inputs, data, worker_cores=1)
    distant = [-0.4, 0.95, 0.9]  # wrong sign drift, huge threshold and ndt
    ll_distant = evaluate_loglik(distant, inputs, data, worker_cores=1)

    assert ll_true > ll_distant
