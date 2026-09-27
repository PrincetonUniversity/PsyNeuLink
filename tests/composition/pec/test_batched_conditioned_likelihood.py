"""Observation-model and particle-ancestry tests independent of GPU execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from psyneulink.core.batched.compiler import BatchedSimulationPlan, _systematic_resample
from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION
from psyneulink.core.batched.likelihood import histogram_likelihood, histogram_observation_weights

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.mark.parametrize("sigma", [0., .5, 1.])
def test_pseudocount_density_and_particle_weights_match_histogram(sigma):
    simulated = np.array([[[0., -.1], [0., .01], [1., .5], [0., .99], [1., 1.2]]])
    observed = np.array([[0., .01]])
    options = dict(categorical_dims=[0], bins=5, bin_range=[(0., 1.)],
                   smoothing_sigma=sigma, categorical_cardinalities=[2])
    alpha = .7
    original, _ = histogram_observation_weights(simulated[0], observed, 0, **options)
    weights, density = histogram_observation_weights(simulated[0], observed, 0, pseudocount=alpha, **options)
    reference = histogram_likelihood(simulated, observed, pseudocount=alpha, **options)
    np.testing.assert_allclose(density.numpy(), reference[0], rtol=1e-6)
    np.testing.assert_allclose(weights.numpy(), original.numpy() + alpha / 5, rtol=1e-6)


@pytest.mark.parametrize("source_bin", [0, 2, 4])
@pytest.mark.parametrize("alpha", [0., .5])
def test_source_normalized_observation_kernel_integrates_to_one(source_bin, alpha):
    # Sum every possible observation category/bin for a particle fixed at a
    # boundary or interior source. Target-normalized smoothing fails this check.
    observed = np.array([[category, (bin_index + .5) / 5]
                         for category in [0., 1.] for bin_index in range(5)])
    simulated = np.array([[0., (source_bin + .5) / 5]])
    probabilities = []
    for trial in range(len(observed)):
        _, density = histogram_observation_weights(
            simulated, observed, trial, [0], bins=5, bin_range=[(0., 1.)],
            smoothing_sigma=.7, pseudocount=alpha, categorical_cardinalities=[2],
            source_normalized=True,
        )
        probabilities.append(float(density) * (1.000001 / 5))
    assert sum(probabilities) == pytest.approx(1., abs=2e-7)


def test_uniform_contamination_retains_predictive_ancestry_without_matches():
    weights, density = histogram_observation_weights(
        np.array([[0., .5], [0., .8], [0., .9]]), np.array([[1., .2]]), 0,
        [0], bins=10, bin_range=[(0., 1.)], pseudocount=1.,
        categorical_cardinalities=[2], source_normalized=True,
    )
    np.testing.assert_allclose(weights.numpy(), np.ones(3) / 3)
    assert float(density) == pytest.approx(1. / (23 * .1000001), rel=1e-6)


def test_out_of_range_predictive_mass_is_preserved_as_overflow():
    observed = np.array([[(b + .5) / 5] for b in range(5)])
    simulated = np.array([[.5], [2.]])
    alpha = .4
    mass = 0.
    for trial in range(5):
        _, density = histogram_observation_weights(
            simulated, observed, trial, bins=5, bin_range=[(0., 1.)],
            smoothing_sigma=.7, pseudocount=alpha, source_normalized=True,
        )
        mass += float(density) * 1.000001 / 5
    # Half of the model predictive probability lies outside the histogram.
    # Uniform contamination lies wholly inside; its fraction is 5a/(2+5a).
    assert mass == pytest.approx((1 + 5 * alpha) / (2 + 5 * alpha), rel=1e-6)


@pytest.mark.parametrize("options,match", [
    ({"pseudocount": -.1}, "pseudocount"),
    ({"pseudocount": .1, "categorical_cardinalities": [0]}, "positive integers"),
    ({"pseudocount": .1, "categorical_cardinalities": [1]}, "observed category"),
    ({"pseudocount": .1, "bin_range": [(0., .5)]}, "outside bin_range"),
])
def test_invalid_contamination_support_is_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        histogram_observation_weights(
            np.array([[0., .2], [1., .7]]), np.array([[0., .2], [1., .7]]),
            0, [0], **options,
        )


def _binary_state_plan():
    """Exact two-state persistent process: each response reveals its state."""
    def run(inputs, parameter_sets, num_estimates, *, initial_states, **kwargs):
        if initial_states is None:
            state = (torch.arange(num_estimates) >= num_estimates // 2).float()[None, None, :, None]
            state = state.expand(len(parameter_sets), 1, -1, -1).clone()
        else:
            state = initial_states
        return SimpleNamespace(values=state[:, :, None], metadata={"final_states": state})

    return SimpleNamespace(ir=SimpleNamespace(param_defaults={}, params=()), backend="triton_cpu",
                           kernel_ir=SimpleNamespace(fusion_kind=COEVOLVING_GRAPH_FUSION), run=run)


def _binary_filter(observed, *, candidates=1, **options):
    return BatchedSimulationPlan.conditioned_log_likelihood(
        _binary_state_plan(), {"stimulus": np.ones((len(observed), 1))},
        [{} for _ in range(candidates)], 16, np.asarray(observed, dtype=float)[:, None],
        [0], categorical_cardinalities=[2], seed=9, **options,
    )


def test_masked_observation_conditions_history_and_diagnostics_match_bayes():
    # With alpha=N/2, the observation is uninformative half the time.
    # Prior state P(1)=1/2, posterior P(1 | y1=1)=3/4, next predictive
    # observation P(y2=1 | y1=1) = (3/4)/2 + 1/4 = 5/8.
    score, diagnostics = _binary_filter([1., 1.], pseudocount=8.,
                                        include_mask=[False, True], return_diagnostics=True)
    assert score == pytest.approx(np.log(5 / 8))
    np.testing.assert_allclose(diagnostics["per_trial_densities"], [[[.5, .625]]])
    np.testing.assert_allclose(diagnostics["prior_mixture_fraction"], [[[.5, .4]]])
    assert diagnostics["effective_sample_size"].shape == (1, 1, 2)
    assert diagnostics["effective_sample_size"][0, 0, 0] == pytest.approx(12.8)
    assert not diagnostics["zero_support"].any()
    other = _binary_filter([0., 1.], pseudocount=8., include_mask=[False, True])
    assert other == pytest.approx(np.log(3 / 8))


def test_noiseless_history_update_is_exact_and_zero_support_is_an_error():
    assert _binary_filter([1., 1.]) == pytest.approx(np.log(.5))
    with pytest.raises(ValueError, match="zero particle observation support.*trial 1"):
        _binary_filter([1., 0.], include_mask=[False, False])


def test_common_random_number_filter_replays_and_matches_candidate_batches():
    options = dict(pseudocount=.7, return_diagnostics=True)
    scalar, single = _binary_filter([1., 0., 1., 1.], **options)
    batch, many = _binary_filter([1., 0., 1., 1.], candidates=3, **options)
    np.testing.assert_array_equal(batch, np.repeat(scalar, 3))
    for key in ["per_trial_densities", "effective_sample_size", "prior_mixture_fraction"]:
        np.testing.assert_array_equal(many[key], np.repeat(single[key], 3, axis=0))
    replay, _ = _binary_filter([1., 0., 1., 1.], **options)
    assert replay == scalar


def test_conditioned_density_does_not_floor_small_positive_contamination():
    score, diagnostics = _binary_filter([1., 0.], pseudocount=1e-12, return_diagnostics=True)
    assert diagnostics["per_trial_densities"][0, 0, 1] == pytest.approx(1e-12 / 16, rel=1e-6, abs=0.)
    assert score == pytest.approx(np.log(.5) + np.log(1e-12 / 16), rel=1e-6)


def test_legacy_multinomial_resampling_replays_at_fixed_batch_size():
    kwargs = dict(pseudocount=8., resampling="multinomial")
    assert _binary_filter([1., 0., 1.], **kwargs) == _binary_filter([1., 0., 1.], **kwargs)


def test_systematic_resampling_uniform_and_point_mass():
    normalized = torch.tensor([[.25, .25, .25, .25], [0., 0., 1., 0.]])
    ancestors = _systematic_resample(normalized, generator=torch.Generator().manual_seed(3))
    np.testing.assert_array_equal(ancestors.numpy(), [[0, 1, 2, 3], [2, 2, 2, 2]])


@pytest.mark.triton_gpu
@pytest.mark.parametrize("estimates", [10000, 100000])
def test_gpu_systematic_ancestry_is_invariant_to_candidate_batch_size(estimates):
    # Tiny background weights and sparse supported observations exercise both
    # cancellation-sensitive CDF regions and realistic large particle arrays.
    generator = torch.Generator().manual_seed(91)
    weights = torch.rand((4, 1, estimates), generator=generator)
    weights = torch.where(weights > .93, weights, 1e-5).cuda()
    single = _systematic_resample(
        weights[:1], generator=torch.Generator(device="cuda").manual_seed(39), shared_first_axis=True,
    )
    batched = _systematic_resample(
        weights, generator=torch.Generator(device="cuda").manual_seed(39), shared_first_axis=True,
    )
    torch.testing.assert_close(single, batched[:1], rtol=0., atol=0.)
