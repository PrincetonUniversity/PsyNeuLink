"""Finite-difference batching preserves posterior curvature and backend selection."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from psyneulink.core.compositions.hierarchical import distributedestep
from psyneulink.core.compositions.hierarchical.laplaceem import (
    EStepConfig, diagonal_hessian, full_hessian, make_inprocess_estep_runner, subject_map_estep,
)
from psyneulink.core.compositions.hierarchical.subjectlikelihood import PECFactorySubjectLikelihood
from psyneulink.core.compositions.hierarchical.transforms import BoundedTransform


@pytest.mark.parametrize("probe", [diagonal_hessian, full_hessian])
@pytest.mark.parametrize("batch_size", [1, 3, 64])
@pytest.mark.parametrize("cached_center", [False, True])
def test_batched_stencil_matches_analytic_hessian(probe, batch_size, cached_center):
    matrix = np.array([[4., 1., -2.], [1., 3., .5], [-2., .5, 6.]])
    z = np.array([.4, -.6, 1.2])
    steps = np.array([.03, .02, .04])
    scalar_rows, batch_rows = [], []

    def scalar(row):
        scalar_rows.append(row.copy())
        return .5 * row @ matrix @ row

    def batch(rows):
        batch_rows.append(rows.copy())
        return .5 * np.einsum('bi,ij,bj->b', rows, matrix, rows)

    center = .5 * z @ matrix @ z if cached_center else None
    actual = probe(scalar, z, steps, f0=center, func_batch=batch, batch_size=batch_size)
    expected = matrix if probe is full_hessian else np.diag(matrix)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    count = 18 if probe is full_hessian else 6
    assert len(scalar_rows) + sum(map(len, batch_rows)) == count + (not cached_center)
    if batch_size == 1:
        assert not batch_rows
    else:
        assert len(scalar_rows) == (not cached_center)
        assert max(map(len, batch_rows)) <= batch_size
        assert len(batch_rows) == (count + batch_size - 1) // batch_size


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(True), None])
def test_hessian_rejects_invalid_batch_size(value):
    with pytest.raises(ValueError, match="positive integer"):
        EStepConfig(hessian_batch_size=value)


@pytest.mark.parametrize("bad_result", [0., np.zeros((2, 1)), np.zeros(1)])
def test_batch_scores_must_match_parameter_rows(bad_result):
    with pytest.raises(ValueError, match="one score per parameter vector"):
        full_hessian(lambda z: 0., [0.], .1, func_batch=lambda rows: bad_result)


def test_batch_errors_are_not_silently_retried_as_scalar():
    def batch(rows):
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        full_hessian(lambda z: 0., [0.], .1, func_batch=batch)


@pytest.mark.parametrize("mode", [.3, 1e-4])
def test_batching_preserves_curvature_retries_and_prior_fallback(mode):
    def scalar(z):
        return np.inf if z[0] <= 0 else .5 * (z[0] - mode) ** 2 / 4

    batches = []

    def batch(rows):
        batches.append(rows.copy())
        return np.where(rows[:, 0] <= 0, np.inf, .5 * (rows[:, 0] - mode) ** 2 / 4)

    config = EStepConfig(hessian_step=.5, hessian_batch_size=2)
    reference = subject_map_estep(scalar, [mode], [4.], config)
    actual = subject_map_estep(scalar, [mode], [4.], config, neg_log_post_batch=batch)
    assert len(batches) > 1
    for field in ("z_hat", "curvature", "covariance", "hessian_step"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))


class QuadraticPEC:
    """A known likelihood with independent scalar and batch implementations."""

    def __init__(self, subject, backend="triton", simulation=True):
        self.subject = subject
        self.scores_by_simulation = simulation
        self.controller = SimpleNamespace(function=SimpleNamespace(
            fit_param_names=["a", "b"], fit_param_bounds={"a": (-2., 3.), "b": (.1, 2.)},
            batched_backend=backend,
        ))
        self.batches = []
        self.scalars = []

    def log_likelihood(self, *theta, inputs=None):
        assert inputs == self.subject
        self.scalars.append(theta)
        return -float(np.sum((np.asarray(theta) - self.subject * .2) ** 2))

    def log_likelihood_batch(self, theta, inputs=None):
        assert inputs == self.subject
        self.batches.append(theta.copy())
        return -np.sum((theta - self.subject * .2) ** 2, axis=1)


@pytest.mark.parametrize("backend,simulation", [(None, True), ("triton", True), ("triton", False)])
def test_provider_preserves_backend_and_neural_likelihood(backend, simulation):
    models = [QuadraticPEC(s, backend, simulation) for s in range(2)]
    provider = PECFactorySubjectLikelihood(lambda data, subject_index: (models[subject_index], subject_index),
                                          [None, None])
    rows = np.array([[.2, .4], [-.1, .8], [.7, .3]])
    for s, model in enumerate(models):
        np.testing.assert_array_equal(provider.log_likelihood_batch(rows, s), -np.sum((rows - s * .2) ** 2, axis=1))
        assert len(model.batches) == int(backend is not None and simulation)
        assert len(model.scalars) == (0 if model.batches else len(rows))


@pytest.mark.parametrize("curvature", ["full", "diagonal"])
def test_batched_estep_matches_scalar_after_transform_and_prior(curvature):
    models = [QuadraticPEC(s) for s in range(2)]
    provider = PECFactorySubjectLikelihood(lambda data, subject_index: (models[subject_index], subject_index),
                                          [None, None])
    config = EStepConfig(curvature=curvature, hessian_step=.2, hessian_batch_size=3)
    args = (np.array([[.2, -.3], [-.4, .5]]), np.array([.7, 1.3]), np.zeros((2, 2)), True)
    scalar = make_inprocess_estep_runner(provider.log_likelihood, provider.transform, config)(*args)
    batched = make_inprocess_estep_runner(provider.log_likelihood, provider.transform, config,
                                        log_likelihood_batch=provider.log_likelihood_batch)(*args)
    for field in ("z_hat", "curvature", "covariance", "hessian_step", "subject_objective", "success"):
        np.testing.assert_array_equal(getattr(batched, field), getattr(scalar, field))
    assert all(model.batches for model in models)


def test_distributed_worker_batches_the_same_posterior_as_local_runner():
    frame = pd.DataFrame({"rt": [.1]})
    model = QuadraticPEC(1)
    provider = PECFactorySubjectLikelihood(lambda data, subject_index: (model, 1), [frame])
    schema = provider.schema
    config = EStepConfig(hessian_step=.2, hessian_batch_size=3)
    mean, variance = np.array([.2, -.3]), np.array([.7, 1.3])
    transform = BoundedTransform(schema.lower, schema.upper)
    local = make_inprocess_estep_runner(provider.log_likelihood, transform, config)(
        mean[None, :], variance, mean[None, :], True,
    )
    fit_id = "test-hessian-batches"
    try:
        _, worker, _ = distributedestep._dask_subject_estep(
            lambda data, subject_index: (model, 1), 0, frame, mean, variance, schema, mean, None, fit_id, config,
        )
        for field in ("z_hat", "curvature", "covariance", "hessian_step"):
            np.testing.assert_array_equal(getattr(worker, field), getattr(local, field)[0])
        assert [len(rows) for rows in model.batches] == [3, 3, 2]
    finally:
        distributedestep._release_fit_models(fit_id)
