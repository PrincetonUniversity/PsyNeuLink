import numpy as np
import pytest

# The sampler is written in torch, because it needs gradients; it is skipped rather than failed
# where torch is absent, as the trained estimators it exists to be run on are.
torch = pytest.importorskip("torch")

from psyneulink.core.compositions.hierarchical.nuts import (  # noqa: E402
    HierarchicalNeuralPosterior,
    NUTSConfig,
    SubjectTerms,
    effective_sample_size,
    potential_scale_reduction,
    run_nuts,
)

# Sampling is many small torch calls; see the fixture for why one thread is the faster choice.
pytestmark = pytest.mark.usefixtures("single_threaded_torch")


# ===========================================================================
# The sampler, against a distribution whose answer is known exactly
# ===========================================================================
MEAN = np.array([1.0, -2.0, 0.5])
COVARIANCE = np.array([[1.0, 0.8, -0.3], [0.8, 1.5, 0.2], [-0.3, 0.2, 0.6]])


def _gaussian_target(mean=MEAN, covariance=COVARIANCE):
    precision = torch.as_tensor(np.linalg.inv(covariance))
    centre = torch.as_tensor(np.asarray(mean, dtype=float))

    def log_prob_grad(q):
        q = q.detach().requires_grad_(True)
        d = q - centre
        value = -0.5 * d @ precision @ d
        gradient, = torch.autograd.grad(value, q)
        return float(value.detach()), gradient.detach()

    return log_prob_grad


def _starts(chains, size, seed=0, scale=2.0):
    generator = np.random.default_rng(seed)
    return [generator.normal(scale=scale, size=size) for _ in range(chains)]


@pytest.fixture(scope="module")
def gaussian_draws():
    config = NUTSConfig(draws=700, warmup=500, chains=4, seed=1)
    return run_nuts(_gaussian_target(), _starts(4, 3), config)


def test_sampler_recovers_the_mean_and_spread(gaussian_draws):
    draws, _ = gaussian_draws
    flat = draws.reshape(-1, 3)
    # Tolerances are several Monte Carlo standard errors wide: a sampler is right on average
    # rather than on any one run, and a test that pinned it tighter would fail on its own noise.
    assert np.allclose(flat.mean(axis=0), MEAN, atol=0.12)
    assert np.allclose(flat.std(axis=0), np.sqrt(np.diag(COVARIANCE)), rtol=0.08)


def test_sampler_recovers_the_correlation(gaussian_draws):
    # The part a sampler that treated the parameters as independent would get wrong.
    draws, _ = gaussian_draws
    flat = draws.reshape(-1, 3)
    expected = COVARIANCE[0, 1] / np.sqrt(COVARIANCE[0, 0] * COVARIANCE[1, 1])
    assert np.isclose(np.corrcoef(flat[:, 0], flat[:, 1])[0, 1], expected, atol=0.06)


def test_sampler_reports_a_clean_run_on_a_smooth_target(gaussian_draws):
    draws, diagnostics = gaussian_draws
    assert diagnostics.divergences.sum() == 0
    assert diagnostics.max_depth_hits.sum() == 0
    assert diagnostics.warnings == ()
    assert np.all(diagnostics.accept_rate > 0.6)
    assert np.all(diagnostics.step_size > 0)
    assert draws.shape == (4, 700, 3)


def test_chains_agree_with_each_other(gaussian_draws):
    draws, _ = gaussian_draws
    assert np.all(potential_scale_reduction(draws) < 1.01)


def test_sampler_refuses_a_starting_point_with_no_density():
    def log_prob_grad(q):
        return -np.inf, torch.zeros_like(q)

    from psyneulink.core.compositions.hierarchical.nuts import NUTSError
    with pytest.raises(NUTSError, match="no posterior density"):
        run_nuts(log_prob_grad, _starts(1, 2), NUTSConfig(draws=2, warmup=2, chains=1))


def test_sampler_checks_it_was_given_one_start_per_chain():
    from psyneulink.core.compositions.hierarchical.nuts import NUTSError
    with pytest.raises(NUTSError, match="2 chain"):
        run_nuts(_gaussian_target(), _starts(3, 3), NUTSConfig(chains=2, draws=2, warmup=2))


@pytest.mark.parametrize("kwargs, match", [
    ({"draws": 0}, "at least 1"),
    ({"warmup": 0}, "at least 1"),
    ({"chains": 0}, "at least 1"),
    ({"target_accept": 1.0}, "strictly between"),
    ({"target_accept": 0.0}, "strictly between"),
    ({"max_tree_depth": 0}, "at least 1"),
    ({"beta_prior_sd": 0.0}, "must be positive"),
])
def test_config_rejects_settings_that_cannot_be_run(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NUTSConfig(**kwargs)


# ===========================================================================
# Convergence diagnostics
# ===========================================================================
def test_rhat_is_near_one_when_chains_explore_the_same_thing():
    generator = np.random.default_rng(0)
    draws = generator.normal(size=(4, 500, 2))
    assert np.all(potential_scale_reduction(draws) < 1.02)


def test_rhat_catches_chains_that_disagree():
    # The failure R-hat exists to catch: each chain is well behaved on its own.
    generator = np.random.default_rng(0)
    draws = np.stack([
        generator.normal(loc=0.0, size=(500, 1)),
        generator.normal(loc=5.0, size=(500, 1)),
    ])
    assert potential_scale_reduction(draws)[0] > 1.5


def test_ess_of_independent_draws_is_near_the_draw_count():
    generator = np.random.default_rng(0)
    draws = generator.normal(size=(4, 1000, 2))
    assert np.all(effective_sample_size(draws) > 2500)


def test_ess_falls_when_draws_are_correlated():
    # A random walk retains most of the previous draw, so a run of it is worth far fewer
    # independent draws than its length.
    generator = np.random.default_rng(0)
    steps = generator.normal(scale=0.1, size=(4, 1000, 1))
    walk = np.cumsum(steps, axis=1)
    assert effective_sample_size(walk)[0] < 200


def test_diagnostics_need_enough_draws_to_split():
    with pytest.raises(ValueError, match="at least four draws"):
        potential_scale_reduction(np.zeros((2, 3, 1)))


# ===========================================================================
# The hierarchical posterior
#
# A stub estimator with a Gaussian density stands in for a trained one: it has
# the same interface and a known answer, so the model built on top of it can be
# checked without training anything.
# ===========================================================================
LOWER = np.array([-2.0, 0.2])
UPPER = np.array([2.0, 1.8])


class _GaussianEstimator:
    """Scores each trial as ``N(outcome | theta, noise)``, one row of theta per trial."""

    def __init__(self, noise=0.25):
        self.noise = noise
        self.calls = 0

    def trial_log_prob(self, theta, outcomes, trial_features=None, *, encoded=False):
        self.calls += 1
        d = outcomes - theta
        return -0.5 * (d / self.noise).pow(2).sum(dim=1)


def _make_group(n_subjects=20, n_trials=30, seed=0, noise=0.25, share_estimator=True):
    """A group drawn from a known population, and the terms that score it."""
    generator = np.random.default_rng(seed)
    beta_z = np.array([0.4, -0.3])
    scale = np.array([0.6, 0.45])
    z = generator.normal(beta_z, scale, size=(n_subjects, 2))
    theta = LOWER + (UPPER - LOWER) / (1.0 + np.exp(-z))

    shared = _GaussianEstimator(noise)
    terms = []
    for s in range(n_subjects):
        observed = theta[s] + generator.normal(0.0, noise, size=(n_trials, 2))
        terms.append(SubjectTerms(
            likelihood=shared if share_estimator else _GaussianEstimator(noise),
            outcomes=torch.as_tensor(observed, dtype=torch.float64),
        ))
    return terms, beta_z, scale, theta


@pytest.fixture(scope="module")
def group():
    return _make_group()


def test_posterior_gradient_matches_finite_differences(group):
    # The sampler is only as right as its gradient: a wrong one moves smoothly to the wrong
    # place, which nothing downstream would reveal.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="full")
    q = torch.as_tensor(posterior.initial_points(1, seed=3)[0])
    _, gradient = posterior.log_prob_grad(q)

    step = 1e-5
    probed = [0, 1, posterior._beta_end, posterior._scale_end,
              posterior._off_end, posterior.size - 1]
    for index in probed:
        forward, backward = q.clone(), q.clone()
        forward[index] += step
        backward[index] -= step
        difference = (float(posterior.log_prob(forward))
                      - float(posterior.log_prob(backward))) / (2.0 * step)
        assert np.isclose(float(gradient[index]), difference, rtol=1e-4, atol=1e-4)


def test_posterior_layout_accounts_for_every_parameter(group):
    terms, *_ = group
    diagonal = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    full = HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="full")
    n_subjects, n_params = len(terms), 2

    # intercept means, log scales, and one standard normal vector per participant
    assert diagonal.size == n_params + n_params + n_subjects * n_params
    # full adds the below-diagonal entries of the covariance factor, and nothing else
    assert full.size == diagonal.size + n_params * (n_params - 1) // 2

    q = torch.as_tensor(full.initial_points(1, seed=0)[0])
    beta, log_scale, off_diagonal, raw = full.unpack(q)
    assert beta.shape == (1, n_params)
    assert log_scale.shape == (n_params,)
    assert off_diagonal.shape == (1,)
    assert raw.shape == (n_subjects, n_params)


def test_diagonal_posterior_has_no_off_diagonal_to_move(group):
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    q = torch.as_tensor(posterior.initial_points(1, seed=0)[0])
    _, _, off_diagonal, _ = posterior.unpack(q)
    assert off_diagonal is None
    covariance = posterior.group_covariance(q).numpy()
    assert np.allclose(covariance, np.diag(np.diag(covariance)))


def test_group_covariance_is_a_covariance_wherever_the_sampler_goes(group):
    # The factor is sampled rather than the matrix, so every point maps to a usable covariance
    # and the sampler never has to be kept out of a region.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="full")
    generator = np.random.default_rng(0)
    for _ in range(20):
        q = torch.as_tensor(generator.normal(scale=3.0, size=posterior.size))
        covariance = posterior.group_covariance(q).numpy()
        assert np.allclose(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_parameters_stay_inside_the_search_range(group):
    # The transform is what keeps the model's parameters valid while the sampler moves an
    # unbounded position, so it has to hold wherever the sampler goes, not merely nearby.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="full")
    generator = np.random.default_rng(1)
    for scale in (1.0, 8.0, 40.0):
        for _ in range(20):
            q = torch.as_tensor(generator.normal(scale=scale, size=posterior.size))
            theta = posterior.to_natural(posterior.subject_z(q)).numpy()
            assert np.all(theta >= LOWER) and np.all(theta <= UPPER)


def test_the_transform_saturates_only_far_outside_a_plausible_fit(group):
    # Beyond about 37 in unconstrained units the transform rounds to the bound exactly, where
    # the gradient is zero and the sampler would stop moving. That is recorded here rather than
    # guarded against: it is unreachable at a group scale any real fit has, and a fit whose
    # group scale is that large is reported as not converged for other reasons first.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="full")
    generator = np.random.default_rng(1)
    for start in posterior.initial_points(20, seed=0):
        q = torch.as_tensor(start + generator.normal(scale=0.5, size=posterior.size))
        assert float(posterior.subject_z(q).abs().max()) < 37.0
        theta = posterior.to_natural(posterior.subject_z(q)).numpy()
        assert np.all(theta > LOWER) and np.all(theta < UPPER)

    extreme = torch.full((posterior.size,), 50.0, dtype=torch.float64)
    saturated = posterior.to_natural(posterior.subject_z(extreme)).numpy()
    assert np.all(saturated == UPPER)


def test_participants_sharing_an_estimator_are_scored_together(group):
    terms, *_ = group
    shared = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    assert len(shared._batches) == 1

    separate, *_ = _make_group(n_subjects=4, share_estimator=False)
    assert len(HierarchicalNeuralPosterior(separate, LOWER, UPPER)._batches) == 4


def test_scoring_together_gives_what_scoring_apart_gives():
    # The batching is an optimization, so it has to be invisible in the answer.
    shared_terms, *_ = _make_group(n_subjects=6, seed=2, share_estimator=True)
    separate_terms, *_ = _make_group(n_subjects=6, seed=2, share_estimator=False)
    shared = HierarchicalNeuralPosterior(shared_terms, LOWER, UPPER, covariance="full")
    separate = HierarchicalNeuralPosterior(separate_terms, LOWER, UPPER, covariance="full")
    q = torch.as_tensor(shared.initial_points(1, seed=5)[0])
    assert np.isclose(float(shared.log_prob(q)), float(separate.log_prob(q)))


def test_posterior_rejects_a_design_matrix_of_the_wrong_height(group):
    terms, *_ = group
    with pytest.raises(ValueError, match="one row per participant"):
        HierarchicalNeuralPosterior(terms, LOWER, UPPER, design_matrix=np.ones((3, 1)))


def test_posterior_rejects_an_unknown_covariance(group):
    terms, *_ = group
    with pytest.raises(ValueError, match="covariance must be one of"):
        HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="banded")


def test_sampling_recovers_the_group_it_was_generated_from():
    """The end the whole module exists for: draws that describe the population."""
    terms, beta_z, scale, theta_true = _make_group(n_subjects=25, n_trials=40, seed=4)
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER, covariance="full")
    config = NUTSConfig(draws=250, warmup=250, chains=2, seed=11)
    draws, diagnostics = run_nuts(
        posterior.log_prob_grad, posterior.initial_points(2, seed=2), config
    )
    assert diagnostics.divergences.sum() == 0

    flat = torch.as_tensor(draws.reshape(-1, posterior.size))
    means = flat[:, :posterior._beta_end].numpy()
    assert np.allclose(means.mean(axis=0), beta_z, atol=0.25)

    # Every tenth draw: the point is the estimate, and reading all of them back costs more
    # than it adds.
    sampled = np.stack([
        posterior.to_natural(posterior.subject_z(flat[i])).numpy()
        for i in range(0, len(flat), 10)
    ])
    rmse = float(np.sqrt(np.mean((sampled.mean(axis=0) - theta_true) ** 2)))
    assert rmse < 0.1
