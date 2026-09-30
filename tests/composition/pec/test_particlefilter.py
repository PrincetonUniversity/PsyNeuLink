"""Probability targets and filtering checked against independent small models."""

import numpy as np
import pytest
from scipy.stats import norm

from psyneulink.core.components.functions.nonstateful.particlefilter import (
    ParticleObservationModel,
    ParticleSupportError,
    particle_filter,
    systematic_resample,
)


def binary_filter(data, **kwargs):
    state = np.repeat([0.0, 1.0], 8)[:, None]

    def advance(trial):
        return state

    def resample(ancestors):
        nonlocal state
        state = state[ancestors]

    model = ParticleObservationModel(
        np.array(data)[:, None],
        [True],
        kernel="histogram",
        categorical_values=[[0, 1]],
        **kwargs.pop("observation", {}),
    )
    return particle_filter(advance, resample, model, seed=7, **kwargs)


def test_masked_observation_updates_exact_binary_posterior():
    score, diag, _ = binary_filter(
        [1, 1],
        include_mask=[False, True],
        observation={"contamination_probability": 0.5},
    )
    assert score == pytest.approx(np.log(0.625))
    np.testing.assert_allclose(np.exp(diag["per_trial_log_densities"]), [0.5, 0.625])
    assert diag["effective_sample_size"][0] == pytest.approx(12.8)
    assert diag["contamination_responsibility"][0] == pytest.approx(0.5)
    other, _, _ = binary_filter(
        [0, 1],
        include_mask=[False, True],
        observation={"contamination_probability": 0.5},
    )
    assert other == pytest.approx(np.log(0.375))


def test_zero_support_raises_even_on_unscored_trial():
    assert binary_filter([1, 1])[0] == pytest.approx(np.log(0.5))
    with pytest.raises(ParticleSupportError, match="trial 1"):
        binary_filter([1, 0], include_mask=[False, False])


def test_tiny_positive_contamination_is_not_floored():
    _, diag, _ = binary_filter([1, 0], observation={"contamination_probability": 1e-30})
    assert diag["per_trial_log_densities"][1] == pytest.approx(np.log(5e-31))


@pytest.mark.parametrize("source", [0, 2, 4])
@pytest.mark.parametrize("contamination", [0.0, 0.3])
def test_histogram_observation_integrates_to_one_at_boundaries(source, contamination):
    observations = np.array(
        [[category, (b + 0.5) / 5] for category in [0, 1] for b in range(5)]
    )
    model = ParticleObservationModel(
        observations,
        [True, False],
        kernel="histogram",
        bins=5,
        bin_range=[(0.0, 1.0)],
        smoothing_sigma=0.7,
        contamination_probability=contamination,
        categorical_values=[[0, 1]],
    )
    density = []
    for t in range(len(observations)):
        try:
            density.append(np.exp(model.evaluate([[0.0, (source + 0.5) / 5]], t)[0]))
        except ParticleSupportError:
            density.append(0.0)
    assert sum(density) / 5 == pytest.approx(1.0)


def test_histogram_keeps_overflow_mass():
    observations = np.array([[(b + 0.5) / 5] for b in range(5)])
    model = ParticleObservationModel(
        observations,
        [False],
        kernel="histogram",
        bins=5,
        bin_range=[(0, 1)],
        smoothing_sigma=2.0,
    )
    mass = sum(np.exp(model.evaluate([[0.5], [2.0]], t)[0]) / 5 for t in range(5))
    assert mass == pytest.approx(0.5)


def test_gaussian_mixed_outputs_match_normal_density_and_category_mass():
    model = ParticleObservationModel(
        [[1.0, 0.2, -0.4]], [True, False, False], bandwidth=[0.3, 0.7]
    )
    simulations = np.array([[1.0, 0.1, -0.2], [0.0, 100.0, 100.0], [1.0, 0.4, -0.8]])
    log_density, weights, _ = model.evaluate(simulations, 0)
    expected = np.array(
        [
            norm.pdf(0.2, 0.1, 0.3) * norm.pdf(-0.4, -0.2, 0.7),
            0.0,
            norm.pdf(0.2, 0.4, 0.3) * norm.pdf(-0.4, -0.8, 0.7),
        ]
    )
    assert log_density == pytest.approx(np.log(expected.mean()))
    np.testing.assert_allclose(weights, expected / expected.sum())


def test_gaussian_log_weights_avoid_underflow():
    model = ParticleObservationModel([[100.0]], [False], bandwidth=1.0)
    density, weights, _ = model.evaluate([[0.0], [0.0]], 0)
    assert density == pytest.approx(norm.logpdf(100.0))
    np.testing.assert_array_equal(weights, [0.5, 0.5])


@pytest.mark.parametrize("label", [0.3, 123456.3])
def test_fp32_noninteger_categories_match_without_ambiguous_support(label):
    model = ParticleObservationModel([[label]], [True])
    density, weights, _ = model.evaluate(np.array([[label]], dtype=np.float32), 0)
    assert density == 0.0
    np.testing.assert_array_equal(weights, [1.0])
    model = ParticleObservationModel([[0.3], [0.300000001]], [True])
    assert model.evaluate([[0.3]], 0)[0] == 0.0
    with pytest.raises(ValueError, match="indistinguishable"):
        model.evaluate(np.array([[0.3]], dtype=np.float32), 0)


def test_filter_matches_kalman_predictive_likelihood():
    observed = np.array([0.6, -0.4, 0.1, 1.0])
    transition_sd, observation_sd = 0.4, 0.7
    model = ParticleObservationModel(
        observed[:, None], [False], bandwidth=observation_sd
    )
    rng = np.random.default_rng(17)
    state = np.zeros(60000)

    def advance(trial):
        nonlocal state
        state += rng.normal(0.0, transition_sd, len(state))
        return state[:, None]

    def resample(ancestors):
        nonlocal state
        state = state[ancestors]

    _, diag, _ = particle_filter(advance, resample, model, seed=41)
    mean, variance, expected = 0.0, 0.0, []
    for y in observed:
        variance += transition_sd**2
        expected.append(norm.logpdf(y, mean, np.sqrt(variance + observation_sd**2)))
        gain = variance / (variance + observation_sd**2)
        mean += gain * (y - mean)
        variance *= 1 - gain
    np.testing.assert_allclose(
        diag["per_trial_log_densities"], expected, atol=0.015, rtol=0
    )


def test_systematic_resampling_and_seed_replay():
    np.testing.assert_array_equal(
        systematic_resample(np.ones(4), np.random.default_rng(4)), np.arange(4)
    )
    np.testing.assert_array_equal(
        systematic_resample([0, 0, 1, 0], np.random.default_rng(4)), np.full(4, 2)
    )
    assert (
        binary_filter([1, 0, 1], observation={"contamination_probability": 0.3})[0]
        == binary_filter([1, 0, 1], observation={"contamination_probability": 0.3})[0]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"bandwidth": 0},
        {"contamination_probability": 1},
        {"bins": 0},
        {"smoothing_sigma": -0.1},
        {"bin_range": [(1, 0)]},
        {"kernel": "unknown"},
        {"kernel": "histogram", "bin_range": [(2, 3)]},
    ],
)
def test_invalid_observation_models_rejected(options):
    with pytest.raises(ValueError):
        ParticleObservationModel([[0.0], [1.0]], [False], **options)
