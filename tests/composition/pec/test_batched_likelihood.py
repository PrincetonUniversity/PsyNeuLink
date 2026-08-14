"""Tests for the batched histogram likelihood (roadmap step 9).

The pure-histogram tests only need ``torch`` (the density estimate is Torch, not
Triton).  The ``plan.log_likelihood`` test additionally needs ``triton`` because
it runs the generated kernels on CPU through Triton's interpreter.
"""

import importlib.util

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    histogram_likelihood,
    histogram_log_likelihood,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is required for the histogram likelihood",
)
requires_triton = pytest.mark.triton_interpreter

pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _numpy_reference(sim, exp, cat_mask, bins, bin_range):
    """Independent numpy histogram density, matching ``histogram_likelihood``."""
    T, S, D = sim.shape
    cat = np.asarray(cat_mask, dtype=bool)
    con_idx = np.flatnonzero(~cat)
    cat_idx = np.flatnonzero(cat)
    edges = []
    for j, d in enumerate(con_idx):
        lo, hi = bin_range[j]
        hi = hi + (hi - lo) * 1e-6
        edges.append(np.linspace(lo, hi, bins + 1))
    out = np.zeros(T)
    for t in range(T):
        match = np.ones(S, dtype=bool)
        for d in cat_idx:
            match &= np.isclose(sim[t, :, d], exp[t, d], atol=1e-6)
        vol = 1.0
        for j, d in enumerate(con_idx):
            interior = edges[j][1:-1]
            match &= np.searchsorted(interior, sim[t, :, d], side="right") == np.searchsorted(
                interior, exp[t, d], side="right"
            )
            vol *= edges[j][1] - edges[j][0]
        out[t] = max(match.sum() / (S * vol), 1e-10)
    return out


@requires_torch
def test_histogram_likelihood_matches_numpy_reference():
    rng = np.random.default_rng(0)
    T, S = 6, 4000
    decisions = rng.choice([-1.0, 1.0], size=(T, S))
    rts = np.abs(rng.normal(0.5, 0.15, size=(T, S)))
    sim = np.stack([decisions, rts], axis=-1)
    exp = np.stack(
        [rng.choice([-1.0, 1.0], size=T), np.abs(rng.normal(0.5, 0.15, size=T))], axis=-1
    )
    cat_mask = [True, False]
    br = [(0.0, 1.5)]

    got = histogram_likelihood(sim, exp, cat_mask, bins=50, bin_range=br)
    ref = _numpy_reference(sim, exp, cat_mask, bins=50, bin_range=br)
    assert got.shape == (T,)
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-9)


@requires_torch
def test_histogram_likelihood_batches_over_lanes():
    """A leading lane axis (e.g. parameter sets) is scored independently."""
    rng = np.random.default_rng(1)
    L, T, S = 3, 5, 3000
    br = [(0.0, 1.5)]
    cat_mask = [True, False]
    exp = np.stack(
        [rng.choice([0.0, 1.0], size=T), np.abs(rng.normal(0.5, 0.15, size=T))], axis=-1
    )
    sim_lanes = np.stack(
        [
            np.stack(
                [rng.choice([0.0, 1.0], size=(T, S)), np.abs(rng.normal(0.5 + 0.1 * i, 0.15, size=(T, S)))],
                axis=-1,
            )
            for i in range(L)
        ],
        axis=0,
    )
    got = histogram_likelihood(sim_lanes, exp, cat_mask, bins=40, bin_range=br)
    assert got.shape == (L, T)
    for i in range(L):
        ref = _numpy_reference(sim_lanes[i], exp, cat_mask, bins=40, bin_range=br)
        assert np.allclose(got[i], ref, rtol=1e-5, atol=1e-9)


@requires_torch
def test_histogram_log_likelihood_peaks_at_matching_distribution():
    """Log-likelihood is higher when the simulated distribution matches the data."""
    rng = np.random.default_rng(2)
    T, S = 8, 5000
    cat_mask = [True, False]
    br = [(0.0, 2.0)]
    exp = np.stack([np.ones(T), np.abs(rng.normal(0.5, 0.15, size=T))], axis=-1)
    sim_good = np.stack(
        [np.ones((T, S)), np.abs(rng.normal(0.5, 0.15, size=(T, S)))], axis=-1
    )
    sim_bad = np.stack(
        [np.ones((T, S)), np.abs(rng.normal(1.2, 0.15, size=(T, S)))], axis=-1
    )
    ll_good = histogram_log_likelihood(sim_good, exp, cat_mask, bins=50, bin_range=br)
    ll_bad = histogram_log_likelihood(sim_bad, exp, cat_mask, bins=50, bin_range=br)
    assert np.ndim(ll_good) == 0
    assert ll_good > ll_bad


@requires_torch
def test_histogram_log_likelihood_include_mask():
    rng = np.random.default_rng(3)
    T, S = 5, 2000
    cat_mask = [True, False]
    br = [(0.0, 1.5)]
    exp = np.stack([rng.choice([0.0, 1.0], size=T), np.abs(rng.normal(0.5, 0.15, T))], axis=-1)
    sim = np.stack(
        [rng.choice([0.0, 1.0], size=(T, S)), np.abs(rng.normal(0.5, 0.15, (T, S)))], axis=-1
    )
    like = histogram_likelihood(sim, exp, cat_mask, bins=40, bin_range=br)
    mask = np.array([True, False, True, True, False])
    ll = histogram_log_likelihood(
        sim, exp, cat_mask, bins=40, bin_range=br, include_mask=mask
    )
    assert np.isclose(ll, np.log(like)[mask].sum(), rtol=1e-5)


@requires_triton
def test_plan_log_likelihood_recovers_ddm_threshold():
    """``plan.log_likelihood`` scores the data-generating threshold highest."""
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.5, threshold=0.2,
            non_decision_time=0.0, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=400)

    n_trials = 4
    inputs = {decision: np.zeros((n_trials, 1))}
    # "Experimental" data generated at the true threshold (0.2).
    exp = plan.run(
        inputs=inputs, parameter_sets=[{"DDM.threshold": 0.2}], num_estimates=1, seed=123
    ).values[0, 0, :, 0, :]

    cat_dims = [True, False]  # decision categorical, RT continuous
    grid = [0.2, 0.4]
    lls = plan.log_likelihood(
        inputs=inputs,
        parameter_sets=[{"DDM.threshold": t} for t in grid],
        num_estimates=150,
        data=exp,
        categorical_dims=cat_dims,
        bins=30,
        seed=7,
    )
    assert lls.shape == (2,)
    assert grid[int(np.argmax(lls))] == 0.2

    # A single parameter set yields a scalar.
    one = plan.log_likelihood(
        inputs=inputs,
        parameter_sets=[{"DDM.threshold": 0.2}],
        num_estimates=150,
        data=exp,
        categorical_dims=cat_dims,
        bins=30,
        seed=7,
    )
    assert np.ndim(one) == 0
