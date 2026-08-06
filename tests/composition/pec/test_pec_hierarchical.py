"""Tests for hierarchical parameter estimation.

Layered fastest first.  This layer covers the parameter transforms and the per-participant E-step,
checked against a Gaussian model whose posterior is available in closed form.  It needs neither
PsyNeuLink nor a cluster, so it runs under a plain ``[dev]`` install.
"""

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from psyneulink.core.compositions.hierarchical.laplaceem import (
    DEFAULT_HESSIAN_STEP_SCALE,
    EStepConfig,
    EStepResult,
    HierarchicalEMWarning,
    diagonal_hessian,
    fit_laplace_em,
    log_gauss_diag,
    make_inprocess_estep_runner,
    subject_map_estep,
)
from psyneulink.core.compositions.hierarchical.transforms import (
    BoundedTransform,
    IdentityTransform,
)


# ===========================================================================
# A Gaussian hierarchical model with closed-form answers.
#
# z_s     ~ N(beta_true, diag(sigma_true))        group random effect
# y_{s,i} ~ N(z_s, tau^2 I),  i = 1..n_obs        observations
#
# The per-participant likelihood is Gaussian in z, so the posterior under a
# Gaussian prior is conjugate and exact -- which is what lets the numerical
# E-step be checked against a known answer, with no PsyNeuLink involved.
# ===========================================================================
LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class ToyHierarchicalModel:
    ybar: np.ndarray     # (n_subjects, n_params) per-participant means
    ss: np.ndarray       # (n_subjects,) within-participant sums of squares
    n_obs: int
    tau: float
    z_true: np.ndarray   # (n_subjects, n_params) ground truth

    @property
    def n_subjects(self):
        return self.ybar.shape[0]

    @property
    def n_params(self):
        return self.ybar.shape[1]

    @classmethod
    def generate(cls, n_subjects, n_params, beta_true, sigma_true, tau, n_obs, rng):
        beta_true = np.asarray(beta_true, float)
        sigma_true = np.asarray(sigma_true, float)
        z_true = rng.normal(beta_true, np.sqrt(sigma_true), size=(n_subjects, n_params))
        ybar = np.empty((n_subjects, n_params))
        ss = np.empty(n_subjects)
        for s in range(n_subjects):
            y = rng.normal(z_true[s], tau, size=(n_obs, n_params))
            ybar[s] = y.mean(axis=0)
            ss[s] = np.sum((y - ybar[s]) ** 2)
        return cls(ybar=ybar, ss=ss, n_obs=n_obs, tau=tau, z_true=z_true)

    def log_likelihood_s(self, theta, s):
        """Exact log-likelihood of participant ``s``'s data at ``theta`` (== z here)."""
        theta = np.asarray(theta, float)
        quad = self.ss[s] + self.n_obs * np.sum((theta - self.ybar[s]) ** 2)
        norm = 0.5 * self.n_obs * self.n_params * (LOG_2PI + 2.0 * np.log(self.tau))
        return -0.5 * quad / self.tau ** 2 - norm

    def closed_form_posterior(self, mu, sigma):
        """Exact posteriors ``N(z_hat, V)`` given prior means ``mu`` and variances ``sigma``."""
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        lam_lik = self.n_obs / self.tau ** 2
        lam_prior = 1.0 / sigma
        lam_post = lam_lik + lam_prior
        z_hat = (lam_lik * self.ybar + lam_prior * mu) / lam_post
        variance = np.broadcast_to(1.0 / lam_post, self.ybar.shape).copy()
        return z_hat, variance


def _make_toy(seed=1, n_subjects=50, n_params=2, n_obs=25):
    rng = np.random.default_rng(seed)
    return ToyHierarchicalModel.generate(
        n_subjects, n_params, beta_true=[0.5, -1.0], sigma_true=[0.4, 0.9],
        tau=1.0, n_obs=n_obs, rng=rng,
    )


# ===========================================================================
# Transforms
# ===========================================================================
def test_bounded_transform_roundtrip():
    t = BoundedTransform(lower=[0.0, -2.0], upper=[1.0, 3.0])
    theta = np.array([0.2, 1.5])
    z = t.to_unconstrained(theta)
    assert np.allclose(t.to_natural(z), theta, atol=1e-10)


def test_bounded_transform_respects_bounds():
    t = BoundedTransform(lower=[0.0], upper=[1.0])
    # Moderate z stays strictly interior; extreme z saturates within the closed box.
    interior = t.to_natural(np.array([-20.0, 0.0, 20.0]))
    assert np.all(interior > 0.0) and np.all(interior < 1.0)
    saturated = t.to_natural(np.array([-50.0, 50.0]))
    assert np.all(saturated >= 0.0) and np.all(saturated <= 1.0)
    assert saturated[0] < 1e-6 and saturated[1] > 1.0 - 1e-6


def test_bounded_transform_saturates_without_overflow():
    # Saturating inputs must reach the bounds without overflowing on the way.
    t = BoundedTransform(lower=[0.0], upper=[1.0])
    with np.errstate(over="raise", under="raise"):
        out = t.to_natural(np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 1.0


def test_dtheta_dz_matches_numerical():
    t = BoundedTransform(lower=[0.0, 1.0], upper=[2.0, 5.0])
    z = np.array([0.3, -0.7])
    h = 1e-6
    num = np.array([
        (t.to_natural(z + h * e)[k] - t.to_natural(z - h * e)[k]) / (2 * h)
        for k, e in enumerate(np.eye(2))
    ])
    assert np.allclose(t.dtheta_dz(z), num, rtol=1e-5)


def test_bounded_transform_rejects_bad_bounds():
    with pytest.raises(ValueError, match="upper bounds must exceed lower bounds"):
        BoundedTransform(lower=[0.0, 2.0], upper=[1.0, 2.0])
    with pytest.raises(ValueError, match="same shape"):
        BoundedTransform(lower=[0.0, 1.0], upper=[1.0])


def test_identity_transform_is_the_identity():
    t = IdentityTransform()
    theta = np.array([-3.0, 0.0, 2.5])
    assert np.allclose(t.to_natural(theta), theta)
    assert np.allclose(t.to_unconstrained(theta), theta)
    assert np.allclose(t.dtheta_dz(theta), np.ones_like(theta))


# ===========================================================================
# Finite-difference curvature
# ===========================================================================
def test_diagonal_hessian_of_quadratic():
    # f(z) = 0.5 * sum a_k (z_k - c_k)^2 has diagonal Hessian a.
    a = np.array([2.0, 5.0, 0.7])
    c = np.array([1.0, -3.0, 0.5])
    f = lambda z: 0.5 * np.sum(a * (z - c) ** 2)  # noqa: E731
    diag = diagonal_hessian(f, np.array([0.0, 0.0, 0.0]), step=1e-3)
    assert np.allclose(diag, a, rtol=1e-5)


def test_diagonal_hessian_accepts_per_dimension_step():
    a = np.array([2.0, 5.0])
    f = lambda z: 0.5 * np.sum(a * z ** 2)  # noqa: E731
    diag = diagonal_hessian(f, np.zeros(2), step=np.array([1e-3, 1e-2]))
    assert np.allclose(diag, a, rtol=1e-5)


def test_diagonal_hessian_rejects_bad_step():
    f = lambda z: float(np.sum(z ** 2))  # noqa: E731
    with pytest.raises(ValueError, match="step must be scalar or of shape"):
        diagonal_hessian(f, np.zeros(3), step=np.array([1e-3, 1e-3]))
    with pytest.raises(ValueError, match="step must be positive"):
        diagonal_hessian(f, np.zeros(2), step=0.0)


# ===========================================================================
# E-step against the closed-form posterior
# ===========================================================================
def test_estep_matches_closed_form_posterior():
    model = _make_toy()
    beta = np.array([0.3, -0.8])
    sigma = np.array([0.5, 0.7])
    s = 7

    def neg_log_post(z):
        return -model.log_likelihood_s(z, s) - log_gauss_diag(z, beta, sigma)

    post = subject_map_estep(neg_log_post, z0=beta, prior_variance=sigma)
    z_cf, v_cf = model.closed_form_posterior(np.tile(beta, (model.n_subjects, 1)), sigma)
    assert np.allclose(post.z_hat, z_cf[s], atol=1e-5)
    assert np.allclose(post.variance, v_cf[s], rtol=1e-4)


def test_estep_reports_optimizer_outcome():
    # The optimizer's verdict must reach the caller: a participant whose fit failed
    # cannot be allowed to feed the group update silently.
    model = _make_toy()
    sigma = np.array([0.5, 0.7])

    def neg_log_post(z):
        return -model.log_likelihood_s(z, 0) - log_gauss_diag(z, np.zeros(2), sigma)

    ok = subject_map_estep(neg_log_post, z0=np.zeros(2), prior_variance=sigma)
    assert ok.success is True
    assert isinstance(ok.message, str)

    capped = subject_map_estep(
        neg_log_post, z0=np.zeros(2), prior_variance=sigma,
        config=EStepConfig(optimizer_options={"maxiter": 1}),
    )
    assert capped.success is False


def test_estep_falls_back_to_prior_where_data_are_uninformative():
    # A flat objective has no curvature to invert; the posterior should report the
    # prior rather than a spuriously tight interval.
    sigma = np.array([0.25, 4.0])
    post = subject_map_estep(lambda z: 0.0, z0=np.zeros(2), prior_variance=sigma)
    assert np.allclose(post.variance, sigma)


# ===========================================================================
# The finite-difference step is derived, and recorded
# ===========================================================================
def test_hessian_step_is_derived_from_the_prior_variance():
    sigma = np.array([0.36, 0.04])
    step = EStepConfig().resolve_hessian_step(sigma)
    assert np.allclose(step, DEFAULT_HESSIAN_STEP_SCALE * np.sqrt(sigma))
    # Pin a concrete value, so the rule is checked against arithmetic and not
    # merely against its own implementation.
    assert np.isclose(step[0], 0.15)


def test_explicit_hessian_step_overrides_the_derived_one():
    sigma = np.array([0.36, 0.04])
    assert np.allclose(EStepConfig(hessian_step=1e-3).resolve_hessian_step(sigma), 1e-3)
    per_dim = np.array([1e-3, 1e-2])
    assert np.allclose(EStepConfig(hessian_step=per_dim).resolve_hessian_step(sigma), per_dim)


def test_estep_records_the_step_it_used():
    # Recorded so that a fit can be audited, rather than the step being an
    # invisible choice made inside the E-step.
    sigma = np.array([0.36, 0.04])
    post = subject_map_estep(lambda z: float(np.sum(z ** 2)), z0=np.zeros(2), prior_variance=sigma)
    assert np.allclose(post.hessian_step, DEFAULT_HESSIAN_STEP_SCALE * np.sqrt(sigma))


def test_estep_config_is_immutable():
    # One config object is shared by every participant and by the group update, so it
    # must not be mutable from any of them.
    config = EStepConfig()
    with pytest.raises(FrozenInstanceError):
        config.variance_floor = 1.0


# ===========================================================================
# EM driver
# ===========================================================================
def _closed_form_em(model, max_iter=500, tol=1e-12):
    """Exact EM for the toy model, using its conjugate posterior instead of an optimizer."""
    beta = np.zeros(model.n_params)
    sigma = np.ones(model.n_params)
    for _ in range(max_iter):
        mu = np.tile(beta, (model.n_subjects, 1))
        z_hat, variance = model.closed_form_posterior(mu, sigma)
        beta_new = z_hat.mean(axis=0)
        sigma_new = np.mean((z_hat - beta_new) ** 2 + variance, axis=0)
        converged = max(
            np.max(np.abs(beta_new - beta)), np.max(np.abs(sigma_new - sigma))
        ) < tol
        beta, sigma = beta_new, sigma_new
        if converged:
            break
    return beta, sigma


def _fit_toy(model, **kwargs):
    runner = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())
    kwargs.setdefault("max_iterations", 300)
    kwargs.setdefault("tol", 1e-8)
    return fit_laplace_em(runner, model.n_subjects, model.n_params, **kwargs)


def test_em_matches_closed_form_em():
    model = _make_toy(seed=3, n_subjects=120)
    beta_cf, sigma_cf = _closed_form_em(model)
    result = _fit_toy(model)
    assert np.allclose(result.beta.ravel(), beta_cf, atol=5e-3)
    assert np.allclose(result.sigma, sigma_cf, atol=5e-3)


def test_em_beta_equals_mean_ybar_at_convergence():
    # With an intercept-only design the group mean is the mean of the per-participant data means.
    model = _make_toy(seed=4, n_subjects=150)
    result = _fit_toy(model)
    assert np.allclose(result.beta.ravel(), model.ybar.mean(axis=0), atol=5e-3)


def test_em_recovers_ground_truth():
    model = _make_toy(seed=5, n_subjects=400, n_obs=40)
    result = _fit_toy(model, tol=1e-7)
    assert np.allclose(result.beta.ravel(), [0.5, -1.0], atol=0.15)
    assert np.allclose(result.sigma, [0.4, 0.9], atol=0.2)


def test_em_objective_nondecreasing():
    model = _make_toy(seed=6, n_subjects=100)
    result = _fit_toy(model, max_iterations=100)
    obj = np.array([h["objective"] for h in result.history])
    assert np.all(np.diff(obj) > -1e-6)


def test_em_history_pairs_each_objective_with_the_estimate_that_produced_it():
    # A history entry must describe one consistent state: the group estimate used for
    # that iteration's E-step, and the objective that E-step returned. The first entry
    # therefore carries the initial estimate, not the result of the first update.
    model = _make_toy(seed=7, n_subjects=40)
    init_beta = np.array([[0.11, -0.22]])
    init_sigma = np.array([0.7, 1.3])
    result = _fit_toy(model, max_iterations=3, tol=0.0,
                      init_beta=init_beta, init_sigma=init_sigma)

    assert np.allclose(result.history[0]["beta"], init_beta)
    assert np.allclose(result.history[0]["sigma"], init_sigma)
    # Each subsequent entry carries the estimate the previous entry's update produced.
    assert not np.allclose(result.history[1]["beta"], init_beta)


def test_em_result_describes_the_group_estimate_it_returns():
    # The participant-level results must correspond to the returned group estimate,
    # not to the one from the iteration before it.
    model = _make_toy(seed=8, n_subjects=30)
    result = _fit_toy(model, max_iterations=4, tol=0.0)

    runner = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())
    recomputed = runner(
        np.ones((model.n_subjects, 1)) @ result.beta, result.sigma, result.z_hat, True
    )
    assert np.allclose(recomputed.z_hat, result.z_hat, atol=1e-6)
    assert np.isclose(recomputed.objective, result.objective, rtol=1e-10)


def test_em_warns_and_records_when_a_participant_fails():
    model = _make_toy(seed=9, n_subjects=6)
    base = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())

    def failing_runner(mu, sigma, prev_z, warm_start):
        out = base(mu, sigma, prev_z, warm_start)
        out.success[1] = False
        out.messages = ((1, "did not converge"),)
        return out

    with pytest.warns(HierarchicalEMWarning, match="did not converge"):
        result = fit_laplace_em(failing_runner, model.n_subjects, model.n_params,
                                max_iterations=2, tol=0.0)

    assert result.subject_converged[1] is np.False_ or not result.subject_converged[1]
    assert result.history[0]["n_subject_failures"] == 1


def test_em_objective_is_summed_in_participant_order():
    # Independent of completion order, so an in-process and a distributed E-step agree.
    r = EStepResult(
        z_hat=np.zeros((3, 2)), variance=np.ones((3, 2)), curvature=np.ones((3, 2)),
        hessian_step=np.ones((3, 2)), subject_objective=np.array([1.5, -2.0, 0.25]),
        success=np.ones(3, dtype=bool), messages=(),
    )
    assert np.isclose(r.objective, float(np.sum(np.array([1.5, -2.0, 0.25]))))


def test_em_rejects_mismatched_design_matrix():
    model = _make_toy(seed=10, n_subjects=8)
    runner = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())
    with pytest.raises(ValueError, match="one row per participant"):
        fit_laplace_em(runner, model.n_subjects, model.n_params,
                       design_matrix=np.ones((3, 1)), max_iterations=1)
