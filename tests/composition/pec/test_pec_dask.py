"""Tests for distributed (Dask) PEC fitting.

Three layers, fastest first:

  * **Helpers** -- the module-level Dask helpers in fitfunctions, exercised with
    fakes (no cluster, no PNL run): worker-cores resolution, the map-like used by
    differential_evolution, the per-worker PEC cache, and the missing-Dask guard.
  * **Driver logic** -- the synchronous ask/tell bookkeeping (``_run_ask_tell_rounds``)
    against an analytic objective: exactly ``n_rounds * batch`` trials asked/told in
    order, scores routed to the matching trial, and CMA-ES actually maximizing.
  * **Integration** -- a real process-based ``LocalCluster`` (workers in separate
    processes, like the SLURM deployment) for the headline claims: distributed
    per-candidate log-likelihoods match serial, an end-to-end distributed fit
    completes with each worker caching its PEC, and -- with common random numbers --
    a distributed fit matches the serial fit's optimized values.
"""

import builtins
import math

import numpy as np
import optuna
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful import fitfunctions
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
    _dask_evaluate_loglik,
    _dask_evaluate_loglik_de,
    _dask_map,
    _require_dask,
    _resolve_worker_cores,
    _run_ask_tell_rounds,
)
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
    OptimizationFunctionError,
)

# The cluster fixtures/tests need real Dask; the helper/driver tests above use
# fakes but the whole module imports cleanly either way.
distributed = pytest.importorskip("dask.distributed")


# ---------------------------------------------------------------------------
# Shared DDM problem (top-level so the factory is picklable for Dask workers).
# ---------------------------------------------------------------------------
NUM_TRIALS = 15
NUM_ESTIMATES = 100
INITIAL_SEED = 42
FIT_BOUNDS = {
    "rate": (-0.5, 0.5),
    "threshold": (0.5, 1.0),
    "non_decision_time": (0.0, 1.0),
}
PARAM_ORDER = list(FIT_BOUNDS)
TRUE_PARAMS = dict(rate=0.3, threshold=0.6, non_decision_time=0.15)


def _build_ddm_comp():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=TRUE_PARAMS["rate"],
            noise=1.0,
            threshold=TRUE_PARAMS["threshold"],
            non_decision_time=TRUE_PARAMS["non_decision_time"],
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def _trial_inputs():
    # Deterministic inputs so serial and every worker score against the same drive.
    return np.ones((NUM_TRIALS, 1))


def make_ddm_data():
    """Synthesize observed DDM data once on the driver (consistent across the fit)."""
    comp, decision = _build_ddm_comp()
    comp.run(inputs={decision: _trial_inputs()})
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)), columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data


def build_ddm_pec(data):
    """pec_factory: rebuild a fresh serial PEC + inputs from the observed data.

    Top-level and data-argument-only (no driver-side captured state) so a worker
    can build its own local PEC. Returns ``(pec, inputs)``; never ship a live PEC.
    """
    comp, decision = _build_ddm_comp()
    pec = pnl.ParameterEstimationComposition(
        name="pec_dask_test",
        nodes=[comp],
        parameters={
            ("rate", decision): np.linspace(*FIT_BOUNDS["rate"], 1000),
            ("threshold", decision): np.linspace(*FIT_BOUNDS["threshold"], 1000),
            ("non_decision_time", decision): np.linspace(*FIT_BOUNDS["non_decision_time"], 1000),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=NUM_ESTIMATES,
        initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,  # common random numbers
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, {comp: _trial_inputs()}


# ===========================================================================
# Layer 1: module-level helpers (no cluster, no PNL run)
# ===========================================================================
class _Future:
    def __init__(self, value):
        self.value = value


def test_resolve_worker_cores_explicit():
    assert _resolve_worker_cores({"worker_cores": 7}) == 7


def test_resolve_worker_cores_from_slurm_env(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    assert _resolve_worker_cores({}) == 5
    # Explicit option still wins over the environment.
    assert _resolve_worker_cores({"worker_cores": 2}) == 2


def test_resolve_worker_cores_fallback(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    cores = _resolve_worker_cores({})
    assert isinstance(cores, int) and cores >= 1


def test_dask_map_applies_func_in_order():
    class _MapClient:
        def __init__(self):
            self.kwargs = []

        def submit(self, func, x, **kw):
            self.kwargs.append(kw)
            return _Future(func(x))

        def gather(self, futures):
            return [f.value for f in futures]

    client = _MapClient()
    out = _dask_map(client, lambda x: x * x, [1, 2, 3, 4])
    assert out == [1, 4, 9, 16]
    # Tasks must be impure so identical candidates are not deduplicated.
    assert all(kw.get("pure") is False for kw in client.kwargs)


@pytest.fixture
def clear_fallback_cache():
    fitfunctions._PEC_FALLBACK_CACHE.clear()
    yield
    fitfunctions._PEC_FALLBACK_CACHE.clear()


@pytest.mark.usefixtures("clear_fallback_cache")
def test_dask_evaluate_loglik_builds_once_and_caches():
    calls = {"n": 0}

    class _FakePEC:
        def log_likelihood(self, *params, inputs=None):
            return float(sum(params))

    def factory(data):
        calls["n"] += 1
        assert data == "DATA"
        return _FakePEC(), {"inputs": "x"}

    v1 = _dask_evaluate_loglik(factory, [1.0, 2.0, 3.0], "DATA", worker_cores=None)
    v2 = _dask_evaluate_loglik(factory, [10.0, 20.0, 30.0], "DATA", worker_cores=None)

    assert v1 == 6.0 and v2 == 60.0
    assert isinstance(v1, float)
    # Built exactly once (cached) despite two evaluations.
    assert calls["n"] == 1


@pytest.mark.usefixtures("clear_fallback_cache")
def test_dask_evaluate_loglik_de_sign():
    class _FakePEC:
        def log_likelihood(self, *params, inputs=None):
            return 5.0

    def factory(data):
        return _FakePEC(), None

    # scipy minimizes: maximize -> flip the sign, minimize -> as is.
    assert _dask_evaluate_loglik_de(factory, None, "D", "maximize", [0.0]) == -5.0
    fitfunctions._PEC_FALLBACK_CACHE.clear()
    assert _dask_evaluate_loglik_de(factory, None, "D", "minimize", [0.0]) == 5.0


def test_require_dask_present_returns_module():
    assert _require_dask() is distributed


def test_require_dask_missing_raises(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dask.distributed" or name.startswith("dask"):
            raise ImportError("no dask")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="psyneulink\\[dask\\]"):
        _require_dask()


# ===========================================================================
# Layer 2: driver ask/tell bookkeeping (real optuna study, analytic objective)
# ===========================================================================
_TARGET = {"rate": 0.25, "threshold": 0.65, "non_decision_time": 0.2}


def _neg_quadratic(param_values):
    return -sum((v - _TARGET[n]) ** 2 for n, v in zip(PARAM_ORDER, param_values))


class _SyncEvaluator:
    """Synchronous stand-in for client.submit/gather recording every dispatch."""

    def __init__(self, fn):
        self.fn = fn
        self.submitted = []

    def submit_one(self, param_values):
        self.submitted.append(list(param_values))
        return self.fn(param_values)  # the "future" is just the value

    def gather(self, futures):
        return list(futures)


def _distributions():
    from optuna.distributions import FloatDistribution
    return {n: FloatDistribution(lo, hi) for n, (lo, hi) in FIT_BOUNDS.items()}


def test_ask_tell_exact_count_order_and_routing():
    batch, n_rounds = 6, 4
    study = optuna.create_study(
        sampler=optuna.samplers.CmaEsSampler(seed=0, popsize=batch),
        direction="maximize",
    )
    ev = _SyncEvaluator(_neg_quadratic)
    _run_ask_tell_rounds(study, _distributions(), PARAM_ORDER, batch, n_rounds,
                         ev.submit_one, ev.gather)

    assert len(study.trials) == batch * n_rounds == 24
    assert len(ev.submitted) == 24
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    for t in study.trials:
        for name, (lo, hi) in FIT_BOUNDS.items():
            assert lo <= t.params[name] <= hi
    # i-th submitted vector == i-th trial's params in FIT_BOUNDS order, and each
    # trial's stored value is the objective of its OWN params (no misrouting).
    for t, submitted in zip(study.trials, ev.submitted):
        expected_vec = [t.params[n] for n in PARAM_ORDER]
        assert submitted == expected_vec
        assert t.value == pytest.approx(_neg_quadratic(expected_vec), abs=1e-12)
        assert math.isfinite(t.value)


def test_ask_tell_cmaes_maximizes_toward_target():
    batch, n_rounds = 8, 20
    study = optuna.create_study(
        sampler=optuna.samplers.CmaEsSampler(seed=0, popsize=batch),
        direction="maximize",
    )
    ev = _SyncEvaluator(_neg_quadratic)
    _run_ask_tell_rounds(study, _distributions(), PARAM_ORDER, batch, n_rounds,
                         ev.submit_one, ev.gather)
    assert study.best_value > -0.01
    for name in PARAM_ORDER:
        assert study.best_params[name] == pytest.approx(_TARGET[name], abs=0.15)


# ===========================================================================
# Layer 2b: batch-size / factory resolution (instance methods, no owner)
# ===========================================================================
class _InfoClient:
    def __init__(self, n_workers):
        self._n = n_workers

    def scheduler_info(self):
        return {"workers": {f"w{i}": {} for i in range(self._n)}}


def _opt_func(sampler, **dist_opts):
    return PECOptimizationFunction(
        method=sampler, distributed=True, distributed_options=dist_opts
    )


def test_resolve_batch_size_defaults_to_live_workers():
    opt = _opt_func(optuna.samplers.RandomSampler(seed=0))
    assert opt._resolve_batch_size(_InfoClient(3), opt.method) == 3


def test_resolve_batch_size_explicit_option_wins():
    opt = _opt_func(optuna.samplers.RandomSampler(seed=0), max_concurrent_evaluations=5)
    assert opt._resolve_batch_size(_InfoClient(2), opt.method) == 5


def test_resolve_batch_size_generational_rejects_lt2():
    opt = _opt_func(optuna.samplers.CmaEsSampler(seed=0))
    with pytest.raises(OptimizationFunctionError, match="at least 2"):
        opt._resolve_batch_size(_InfoClient(1), opt.method)


def test_resolve_pec_factory_missing_raises():
    opt = _opt_func(optuna.samplers.RandomSampler(seed=0))
    with pytest.raises(OptimizationFunctionError, match="pec_factory"):
        opt._resolve_pec_factory()


# ===========================================================================
# Layer 2c: PEC-level forwarding of distributed knobs onto the optimizer
# (construction only -- no cluster, no fit)
# ===========================================================================
_FIT_PARAMS_SPEC = {
    "rate": FIT_BOUNDS["rate"],
    "threshold": FIT_BOUNDS["threshold"],
    "non_decision_time": FIT_BOUNDS["non_decision_time"],
}


def _make_pec_with(optimization_function, data, decision, comp, **pec_kwargs):
    return pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(lo, hi, 1000)
            for name, (lo, hi) in _FIT_PARAMS_SPEC.items()
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=optimization_function,
        num_estimates=NUM_ESTIMATES,
        initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,
        **pec_kwargs,
    )


@pytest.mark.composition
def test_pec_forwards_distributed_to_string_optimizer(ddm_data):
    """distributed=True on the PEC + a string method forwards onto the created optimizer."""
    comp, decision = _build_ddm_comp()
    pec = _make_pec_with(
        "differential_evolution", ddm_data, decision, comp,  # Form 2: string method
        name="pec_fwd_string",
        distributed=True,
        distributed_options={"pec_factory": build_ddm_pec, "worker_cores": 1},
    )
    of = pec.controller.function
    assert isinstance(of, PECOptimizationFunction)
    assert of.method == "differential_evolution"
    assert of.distributed is True
    assert of._distributed_options.get("pec_factory") is build_ddm_pec
    assert of._distributed_options.get("worker_cores") == 1


@pytest.mark.composition
def test_pec_passed_optimizer_distributed_is_authoritative(ddm_data):
    """A passed optimizer that already enabled distributed keeps its own options."""
    own_opts = {"pec_factory": build_ddm_pec, "worker_cores": 3}
    optimizer = PECOptimizationFunction(
        method="differential_evolution", max_iterations=1,
        distributed=True, distributed_options=own_opts,
    )
    comp, decision = _build_ddm_comp()
    pec = _make_pec_with(
        optimizer, ddm_data, decision, comp,
        name="pec_fwd_instance",
        distributed=False,                          # PEC off...
        distributed_options={"worker_cores": 99},   # ...must NOT override the instance
    )
    of = pec.controller.function
    assert of.distributed is True
    assert of._distributed_options.get("worker_cores") == 3   # its own, not 99


# ===========================================================================
# Layer 3: integration over a real process-based LocalCluster
# ===========================================================================
@pytest.fixture(scope="module")
def cluster_client():
    # Process workers, threads_per_worker=1: PNL's LLVM runtime asserts on
    # multi-threaded in-process cleanup, so an in-process threaded cluster is
    # not usable here.
    from dask.distributed import Client, LocalCluster
    with LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            client.wait_for_workers(2, timeout=120)
            yield client


@pytest.fixture(scope="module")
def ddm_data():
    return make_ddm_data()


@pytest.mark.composition
def test_distributed_loglik_matches_serial(ddm_data, cluster_client):
    """Which worker scored a candidate must not matter: distributed == serial."""
    candidates = [
        [0.3, 0.6, 0.15],
        [0.1, 0.8, 0.30],
        [-0.2, 0.55, 0.05],
    ]
    serial_pec, inputs = build_ddm_pec(ddm_data)
    serial = [float(serial_pec.log_likelihood(*c, inputs=inputs)) for c in candidates]

    data_f = cluster_client.scatter(ddm_data, broadcast=True, hash=False)
    futures = [
        cluster_client.submit(
            _dask_evaluate_loglik, build_ddm_pec, c, data_f, 1, pure=False
        )
        for c in candidates
    ]
    dist = cluster_client.gather(futures)

    np.testing.assert_allclose(dist, serial, rtol=1e-10)


def _make_driver_pec(data, optimizer, name, crn=True):
    """Build a driver-side DDM PEC wired to ``optimizer`` (LLVM, optional CRN)."""
    comp, decision = _build_ddm_comp()
    pec = pnl.ParameterEstimationComposition(
        name=name,
        nodes=[comp],
        parameters={
            ("rate", decision): np.linspace(*FIT_BOUNDS["rate"], 1000),
            ("threshold", decision): np.linspace(*FIT_BOUNDS["threshold"], 1000),
            ("non_decision_time", decision): np.linspace(*FIT_BOUNDS["non_decision_time"], 1000),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=optimizer,
        num_estimates=NUM_ESTIMATES,
        initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=crn,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, comp


def _distributed_options(cluster_client, **overrides):
    opts = {"client": cluster_client, "pec_factory": build_ddm_pec, "worker_cores": 1}
    opts.update(overrides)
    return opts


@pytest.mark.composition
def test_distributed_fit_end_to_end_and_worker_cache(ddm_data, cluster_client):
    batch = 4
    optimizer = PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=batch),
        max_iterations=batch * 2,
        distributed=True,
        distributed_options=_distributed_options(
            cluster_client, max_concurrent_evaluations=batch
        ),
    )
    pec, comp = _make_driver_pec(ddm_data, optimizer, "pec_dask_e2e")
    pec.run(inputs={comp: _trial_inputs()})

    recovered = pec.optimized_parameter_values
    assert set(recovered) == {f"DDM.{p}" for p in PARAM_ORDER}
    assert all(np.isfinite(v) for v in recovered.values())

    # Every worker that evaluated must hold a cached PEC (built once per worker).
    caches = cluster_client.run(lambda dask_worker: hasattr(dask_worker, "_pec_cache"))
    assert any(caches.values())


@pytest.mark.composition
def test_distributed_fit_matches_serial_with_crn(ddm_data, cluster_client):
    """With common random numbers, distributed optimized values == serial.

    Uses RandomSampler: the serial path drives the study with ``study.optimize``
    (ask-one/tell-one) while the distributed path asks a batch before telling, so
    only tell-order-independent samplers yield identical candidate sequences across
    the two drivers. RandomSampler is such a sampler; stateful samplers (CMA-ES,
    QMC) legitimately explore different points under batched ask/tell, so they are
    not expected to be bit-identical to the serial driver.
    """
    batch, rounds = 4, 2

    def make_optimizer(distributed):
        opts = _distributed_options(cluster_client, max_concurrent_evaluations=batch) \
            if distributed else None
        return PECOptimizationFunction(
            method=optuna.samplers.RandomSampler(seed=0),
            max_iterations=batch * rounds,
            distributed=distributed,
            distributed_options=opts,
        )

    def run_fit(optimizer, name):
        pec, comp = _make_driver_pec(ddm_data, optimizer, name)
        pec.run(inputs={comp: _trial_inputs()})
        return pec.optimized_parameter_values, pec.optimal_value

    serial_params, serial_val = run_fit(make_optimizer(False), "pec_parity_serial")
    dist_params, dist_val = run_fit(make_optimizer(True), "pec_parity_dist")

    # Compare positionally in fit-parameter order: the two PECs are built in one
    # process, so their mechanism names differ by a dedup suffix (DDM vs DDM-1),
    # but the fit order and the values must match bit-for-bit under CRN.
    assert list(serial_params) and list(dist_params)
    np.testing.assert_allclose(
        list(dist_params.values()), list(serial_params.values()), rtol=1e-10
    )
    np.testing.assert_allclose(dist_val, serial_val, rtol=1e-10)


@pytest.mark.composition
def test_distributed_differential_evolution_runs(ddm_data, cluster_client):
    """The distributed differential_evolution path completes and recovers finite params."""
    optimizer = PECOptimizationFunction(
        method="differential_evolution",
        max_iterations=2,
        popsize=4,  # small DE population to keep the test quick
        distributed=True,
        distributed_options=_distributed_options(cluster_client),
    )
    pec, comp = _make_driver_pec(ddm_data, optimizer, "pec_dask_de")
    pec.run(inputs={comp: _trial_inputs()})

    recovered = pec.optimized_parameter_values
    assert len(recovered) == len(PARAM_ORDER)
    assert all(np.isfinite(v) for v in recovered.values())


@pytest.mark.composition
def test_distributed_without_crn_warns(ddm_data, cluster_client):
    """A distributed fit with CRN off is valid but warns about non-reproducibility."""
    batch = 2
    optimizer = PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=batch),
        max_iterations=batch,
        distributed=True,
        distributed_options=_distributed_options(
            cluster_client, max_concurrent_evaluations=batch
        ),
    )
    pec, comp = _make_driver_pec(ddm_data, optimizer, "pec_dask_nocrn", crn=False)
    with pytest.warns(UserWarning, match="common random numbers"):
        pec.run(inputs={comp: _trial_inputs()})
    assert all(np.isfinite(v) for v in pec.optimized_parameter_values.values())
