"""M1 tests: transforms, finite-difference Hessian, E-step vs closed form, and EM correctness.

Run with:
    .venv/bin/pytest Scripts/Debug/pec_hierarchical/core/test_laplace_em.py -o addopts="-q"
"""

import numpy as np
import pytest

from estep import subject_map_estep
from hessian import diagonal_hessian
from laplace_em import fit_laplace_em, _log_gauss_diag
from toy_model import ToyHierarchicalModel
from transforms import BoundedTransform, IdentityTransform


# --- transforms ---------------------------------------------------------------------------

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


def test_dtheta_dz_matches_numerical():
    t = BoundedTransform(lower=[0.0, 1.0], upper=[2.0, 5.0])
    z = np.array([0.3, -0.7])
    h = 1e-6
    num = np.array([
        (t.to_natural(z + h * e)[k] - t.to_natural(z - h * e)[k]) / (2 * h)
        for k, e in enumerate(np.eye(2))
    ])
    assert np.allclose(t.dtheta_dz(z), num, rtol=1e-5)


# --- hessian ------------------------------------------------------------------------------

def test_diagonal_hessian_of_quadratic():
    # f(z) = 0.5 * sum a_k (z_k - c_k)^2 has diagonal Hessian a.
    a = np.array([2.0, 5.0, 0.7])
    c = np.array([1.0, -3.0, 0.5])
    f = lambda z: 0.5 * np.sum(a * (z - c) ** 2)
    diag = diagonal_hessian(f, np.array([0.0, 0.0, 0.0]), step=1e-3)
    assert np.allclose(diag, a, rtol=1e-5)


# --- E-step vs closed-form posterior ------------------------------------------------------

def _make_toy(seed=1, n_subjects=50, n_params=2, n_obs=25):
    rng = np.random.default_rng(seed)
    return ToyHierarchicalModel.generate(
        n_subjects, n_params, beta_true=[0.5, -1.0], sigma_true=[0.4, 0.9],
        tau=1.0, n_obs=n_obs, rng=rng,
    )


def test_estep_matches_closed_form_posterior():
    model = _make_toy()
    beta = np.array([0.3, -0.8])
    sigma = np.array([0.5, 0.7])
    s = 7
    mu_s = beta

    def neg_log_post(z):
        return -model.log_likelihood_s(z, s) - _log_gauss_diag(z, mu_s, sigma)

    post = subject_map_estep(neg_log_post, z0=mu_s)
    z_cf, v_cf = model.closed_form_posterior(np.tile(beta, (model.n_subjects, 1)), sigma)
    assert np.allclose(post.z_hat, z_cf[s], atol=1e-5)
    assert np.allclose(post.variance, v_cf[s], rtol=1e-4)


# --- EM correctness vs an exact closed-form EM reference ----------------------------------

def _closed_form_em(model, max_iter=500, tol=1e-12):
    n_params = model.n_params
    beta = np.zeros(n_params)
    sigma = np.ones(n_params)
    for _ in range(max_iter):
        mu = np.tile(beta, (model.n_subjects, 1))
        z_hat, variance = model.closed_form_posterior(mu, sigma)
        beta_new = z_hat.mean(axis=0)
        sigma_new = np.mean((z_hat - beta_new) ** 2 + variance, axis=0)
        if max(np.max(np.abs(beta_new - beta)), np.max(np.abs(sigma_new - sigma))) < tol:
            beta, sigma = beta_new, sigma_new
            break
        beta, sigma = beta_new, sigma_new
    return beta, sigma


def test_em_matches_closed_form_em():
    model = _make_toy(seed=3, n_subjects=120)
    beta_cf, sigma_cf = _closed_form_em(model)
    result = fit_laplace_em(
        model.log_likelihood_s, model.n_subjects, model.n_params,
        IdentityTransform(), max_em_iterations=300, em_tol=1e-8,
    )
    assert np.allclose(result.beta.ravel(), beta_cf, atol=5e-3)
    assert np.allclose(result.sigma, sigma_cf, atol=5e-3)


def test_em_beta_equals_mean_ybar_at_convergence():
    # At convergence the intercept-only group mean equals the mean of per-subject data means.
    model = _make_toy(seed=4, n_subjects=150)
    result = fit_laplace_em(
        model.log_likelihood_s, model.n_subjects, model.n_params,
        IdentityTransform(), max_em_iterations=300, em_tol=1e-8,
    )
    assert np.allclose(result.beta.ravel(), model.ybar.mean(axis=0), atol=5e-3)


def test_em_recovers_ground_truth():
    model = _make_toy(seed=5, n_subjects=400, n_obs=40)
    result = fit_laplace_em(
        model.log_likelihood_s, model.n_subjects, model.n_params,
        IdentityTransform(), max_em_iterations=300, em_tol=1e-7,
    )
    assert np.allclose(result.beta.ravel(), [0.5, -1.0], atol=0.15)
    assert np.allclose(result.sigma, [0.4, 0.9], atol=0.2)


def test_em_objective_nondecreasing():
    model = _make_toy(seed=6, n_subjects=100)
    result = fit_laplace_em(
        model.log_likelihood_s, model.n_subjects, model.n_params,
        IdentityTransform(), max_em_iterations=100, em_tol=1e-8,
    )
    obj = np.array([h["objective"] for h in result.history])
    assert np.all(np.diff(obj) > -1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-o", "addopts=-q"]))
