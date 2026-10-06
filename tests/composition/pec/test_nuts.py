import numpy as np
import pytest

# Sampling needs gradients, and so torch and Pyro; it is skipped rather than failed where they are
# absent, as the trained estimators it exists to be run on are.
torch = pytest.importorskip("torch")
pytest.importorskip("pyro")

from psyneulink.core.compositions.hierarchical.nuts import (  # noqa: E402
    HierarchicalNeuralPosterior,
    NUTSConfig,
    SubjectTerms,
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

    def log_prob(q):
        d = q - centre
        return -0.5 * d @ precision @ d

    return log_prob


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
    # What a sampler that treated the parameters as independent would miss.
    draws, _ = gaussian_draws
    flat = draws.reshape(-1, 3)
    expected = COVARIANCE[0, 1] / np.sqrt(COVARIANCE[0, 0] * COVARIANCE[1, 1])
    assert np.isclose(np.corrcoef(flat[:, 0], flat[:, 1])[0, 1], expected, atol=0.06)


def test_sampler_reports_a_clean_run_on_a_smooth_target(gaussian_draws):
    draws, diagnostics = gaussian_draws
    assert diagnostics.divergences.sum() == 0
    assert diagnostics.warnings == ()
    assert np.all(diagnostics.accept_rate > 0.6)
    assert np.all(diagnostics.step_size > 0)
    assert draws.shape == (4, 700, 3)


def test_chains_agree_with_each_other(gaussian_draws):
    from pyro.ops.stats import split_gelman_rubin
    draws, _ = gaussian_draws
    assert np.all(split_gelman_rubin(torch.as_tensor(draws)).numpy() < 1.01)


def test_one_seed_samples_the_same_draws():
    config = NUTSConfig(draws=20, warmup=20, chains=1, seed=3)
    first, _ = run_nuts(_gaussian_target(), _starts(1, 3), config)
    second, _ = run_nuts(_gaussian_target(), _starts(1, 3), config)
    np.testing.assert_array_equal(first, second)


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
])
def test_config_rejects_settings_that_cannot_be_run(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NUTSConfig(**kwargs)


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
            parameter_index=torch.arange(2).expand(n_trials, 2),
        ))
    return terms, beta_z, scale, theta


@pytest.fixture(scope="module")
def group():
    return _make_group()


def test_posterior_gradient_matches_finite_differences(group):
    # A wrong gradient would send the sampler to the wrong place without any error.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    q = torch.as_tensor(posterior.initial_points(1, seed=3)[0]).requires_grad_(True)
    gradient, = torch.autograd.grad(posterior.log_prob(q), q)
    q = q.detach()

    step = 1e-5
    probed = [0, 1, posterior._beta_end, posterior._scale_end, posterior.size - 1]
    for index in probed:
        forward, backward = q.clone(), q.clone()
        forward[index] += step
        backward[index] -= step
        difference = (float(posterior.log_prob(forward))
                      - float(posterior.log_prob(backward))) / (2.0 * step)
        assert np.isclose(float(gradient[index]), difference, rtol=1e-4, atol=1e-4)


def test_posterior_layout_accounts_for_every_parameter(group):
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    n_subjects, n_params = len(terms), 2

    # intercept means, log scales, and one standard normal vector per participant
    assert posterior.size == n_params + n_params + n_subjects * n_params
    q = torch.as_tensor(posterior.initial_points(1, seed=0)[0])
    beta, log_scale, raw = posterior.unpack(q)
    assert beta.shape == (1, n_params)
    assert log_scale.shape == (n_params,)
    assert raw.shape == (n_subjects, n_params)


def test_parameters_stay_inside_the_search_range(group):
    # The transform is what keeps the model's parameters valid while the sampler moves an
    # unbounded position, so it has to hold wherever the sampler goes, not merely nearby.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    generator = np.random.default_rng(1)
    for scale in (1.0, 8.0, 40.0):
        for _ in range(20):
            q = torch.as_tensor(generator.normal(scale=scale, size=posterior.size))
            theta = posterior.to_natural(posterior.subject_z(q)).numpy()
            assert np.all(theta >= LOWER) and np.all(theta <= UPPER)


def test_the_transform_saturates_only_far_outside_a_plausible_fit(group):
    # Beyond about 37 in unconstrained units the transform rounds to the bound, where the
    # gradient is zero. Starting points, and any plausible group scale, stay well inside that.
    terms, *_ = group
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
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
    # Scoring participants together only saves time; it must not change the result.
    shared_terms, *_ = _make_group(n_subjects=6, seed=2, share_estimator=True)
    separate_terms, *_ = _make_group(n_subjects=6, seed=2, share_estimator=False)
    shared = HierarchicalNeuralPosterior(shared_terms, LOWER, UPPER)
    separate = HierarchicalNeuralPosterior(separate_terms, LOWER, UPPER)
    q = torch.as_tensor(shared.initial_points(1, seed=5)[0])
    assert np.isclose(float(shared.log_prob(q)), float(separate.log_prob(q)))


def test_posterior_rejects_a_design_matrix_of_the_wrong_height(group):
    terms, *_ = group
    with pytest.raises(ValueError, match="one row per participant"):
        HierarchicalNeuralPosterior(terms, LOWER, UPPER, design_matrix=np.ones((3, 1)))


def test_sampling_recovers_the_group_it_was_generated_from():
    """Draws for a group simulated from a known population describe that population."""
    terms, beta_z, scale, theta_true = _make_group(n_subjects=25, n_trials=40, seed=4)
    posterior = HierarchicalNeuralPosterior(terms, LOWER, UPPER)
    config = NUTSConfig(draws=250, warmup=250, chains=2, seed=11)
    draws, diagnostics = run_nuts(
        posterior.log_prob, posterior.initial_points(2, seed=2), config
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
