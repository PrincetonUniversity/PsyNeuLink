"""Independent CDF checks for the continuous CSI GPU comparison sampler."""

import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/csi/csi_fit"
sys.path.insert(0, str(DIRECTORY))

pytestmark = [pytest.mark.composition, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_continuous_csi_sampler_matches_independent_first_passage_cdfs(monkeypatch):
    from direct_likelihood.continuous_monte_carlo import simulate_continuous_drift
    from psyneulink.core.batched.numerical.first_passage import MovingBoundaryDDMSolver

    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dt, count = .00025, 2000
        time = (torch.arange(count, dtype=torch.float64) + .5) * dt
        drift = torch.stack((time * 0., .06 + .04 * torch.exp(-time / .2), -.05 + .03 * torch.sin(time * 8.)))
        threshold = torch.tensor([.08, .1, .12], dtype=torch.float64)
        collapse = torch.tensor([0., -.04, -.08], dtype=torch.float64)
        samples = simulate_continuous_drift(drift, threshold, collapse, time_step=dt, estimates=100000, seed=127)
        cutoffs = torch.tensor([.1, .25, .5], dtype=torch.float64)
        reference = MovingBoundaryDDMSolver(time_step=dt, spatial_points=257, native_forward=True).solve_observation_batch(
            drift=drift.repeat_interleave(3, 0), threshold=threshold.repeat_interleave(3),
            collapse_rate=collapse.repeat_interleave(3), interval_low=torch.zeros(9, dtype=torch.float64),
            interval_high=cutoffs.repeat(3), choice=torch.ones(9, dtype=torch.float64))
        expected = torch.stack((reference.lower_probability, reference.upper_probability), -1).reshape(3, 3, 2).numpy()
        actual = np.asarray([[[np.mean((row[:, 0] == choice) & (row[:, 1] <= t + 1.e-7))
                               for choice in (0, 1)] for t in cutoffs.numpy()] for row in samples])
        # Fixed independent random seed; roughly four worst-case Monte Carlo SEs.
        np.testing.assert_allclose(actual, expected, atol=.0065, rtol=0.)
        assert abs(actual[0, -1, 0] - actual[0, -1, 1]) < .0065
        assert np.all(samples[:, :, 1] > 0.)
        assert np.all(samples[:, :, 1] <= .5 + 1.e-7)
        assert np.all(np.isin(samples[:, :, 0], [-1., 0., 1.]))
        # Incomplete paths remain censored, with their original denominator.
        assert np.all(np.mean(samples[:, :, 0] < 0, axis=1) > .1)
    finally:
        torch.set_num_threads(old_threads)


def test_continuous_csi_sampler_rejects_collapsing_domain_and_rng_overflow():
    from direct_likelihood.continuous_monte_carlo import simulate_continuous_drift

    drift = torch.zeros((1, 2000), dtype=torch.float64)
    with pytest.raises(ValueError, match="boundary"):
        simulate_continuous_drift(drift, torch.tensor([.1]), torch.tensor([-.3]), time_step=.001)
    with pytest.raises(ValueError, match="RNG"):
        simulate_continuous_drift(drift, torch.tensor([.1]), torch.tensor([0.]), time_step=.001, estimates=1000000)
