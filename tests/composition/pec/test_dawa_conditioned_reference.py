"""Analytic checks for the independent short-sequence likelihood reference."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


_PATH = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa/dawa_conditioned_reference.py"
_SPEC = importlib.util.spec_from_file_location("dawa_conditioned_reference_test", _PATH)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


def test_observation_kernel_normalizes_at_edges_and_preserves_overflow():
    bins = 3
    edges = reference.observation_edges(bins, (0., 3.))
    centers = (edges[:-1].astype(float) + edges[1:]) / 2.
    samples = np.array([[0., centers[0]], [1., centers[1]], [0., centers[-1]], [1., -1.]])
    summed = sum(reference.numpy_observation_density(
        samples, (choice, center), edges=edges, bins=bins, sigma=.8, alpha_per_estimate=.1,
    ) for choice in (0., 1.) for center in centers) * float(edges[1] - edges[0])
    np.testing.assert_allclose(summed[:3], 1., rtol=1e-14)
    # An overflow simulation assigns only contamination mass to finite bins.
    np.testing.assert_allclose(summed[-1], .6 / 1.6, rtol=1e-14)


def test_observation_kernel_internal_edge_belongs_to_lower_bin():
    edges = reference.observation_edges(3, (0., 3.))
    width = float(edges[1] - edges[0])
    samples = np.array([[0., edges[1]], [0., np.nextafter(edges[1], np.float32(np.inf))]])
    density = reference.numpy_observation_density(samples, [0., width / 2.], bins=3,
                                                 edges=edges, sigma=0., alpha_per_estimate=0.)
    np.testing.assert_array_equal(density, [1. / width, 0.])


def test_full_history_reference_averages_products_and_retains_dependency():
    edges = reference.observation_edges(2, (0., 2.))
    center = float(edges[1]) / 2.
    width = float(edges[1] - edges[0])
    # The two observations are perfectly correlated through the simulated
    # history. Their joint mass is 1/2, whereas the marginal product is 1/4.
    values = np.zeros((2, 4, 4))
    values[:, :, 0] = [0., 0., 1., 1.]
    values[:, :, 1] = center
    values[:, :, 2:] = [[.1, .9], [.3, .7], [.7, .3], [.9, .1]]
    prefixes = reference.full_history_estimate(values, [[0., center], [0., center]], bins=2,
                                               edges=edges, rt_range=(0., 2.), sigma=0., alpha_per_estimate=0.)
    np.testing.assert_allclose(prefixes[0]["joint_likelihood"], .5 / width)
    np.testing.assert_allclose(prefixes[1]["joint_likelihood"], .5 / width ** 2)
    np.testing.assert_allclose(prefixes[1]["conditional_density"], 1. / width)
    np.testing.assert_allclose(prefixes[1]["control_posterior"]["mean"], [.2, .8])
    np.testing.assert_allclose([item["effective_sample_size"] for item in prefixes], [2., 2.])


def test_exact_exported_boundaries_and_neighbors_match_bucketize():
    edges = reference.observation_edges()
    internal = edges[1:-1]
    points = np.stack([np.nextafter(internal, np.float32(-np.inf)), internal,
                       np.nextafter(internal, np.float32(np.inf))], axis=1).reshape(-1)
    samples = np.stack([np.zeros_like(points), points], axis=-1)
    centers = (edges[:-1].astype(float) + edges[1:]) / 2.
    density = np.stack([reference.numpy_observation_density(samples, [0., center], edges=edges,
                                                            sigma=0., alpha_per_estimate=0.)
                        for center in centers])
    expected = torch.bucketize(torch.from_numpy(points), torch.from_numpy(internal)).numpy()
    np.testing.assert_array_equal(density.argmax(axis=0), expected)
    np.testing.assert_array_equal((density > 0).sum(axis=0), 1)
