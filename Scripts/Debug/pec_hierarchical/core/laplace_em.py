"""Empirical-Bayes Laplace EM driver for hierarchical parameter estimation.

The group model places, on each subject's unconstrained parameter vector ``z_s``, a Gaussian
random effect ``z_s ~ N(X_s beta, diag(sigma))``. ``beta`` are group point estimates and
``sigma`` the diagonal group variances. Each iteration:

  E-step: per subject, MAP of ``log_likelihood_s(theta(z)) + log N(z | X_s beta, diag(sigma))``
          plus a diagonal Laplace posterior ``N(z_hat_s, V_s)``.
  M-step: ``beta`` by least squares on the posterior means; per-parameter variance
          ``sigma_k = mean_s[(z_hat_s,k - mu_s,k)^2 + V_s,k]`` with a floor and optional damping.

The likelihood is injected, so the same driver fits the closed-form toy model and the real
LLVM-backed PEC likelihood. v1 is diagonal covariance only.
"""

from dataclasses import dataclass, field

import numpy as np

from estep import subject_map_estep

LOG_2PI = np.log(2.0 * np.pi)


def _log_gauss_diag(z, mean, var):
    """Log density of a diagonal Gaussian, summed over dimensions."""
    return -0.5 * np.sum((z - mean) ** 2 / var + LOG_2PI + np.log(var))


def subject_laplace_objective(neg_log_post, variance, n_params):
    """Per-subject Laplace marginal log-likelihood from the floored posterior variance."""
    return -neg_log_post + 0.5 * n_params * LOG_2PI + 0.5 * np.sum(np.log(variance))


def make_serial_estep_runner(
    log_likelihood_s, transform, *,
    method="Nelder-Mead", hessian_step=1e-4, variance_floor=1e-6, estep_options=None,
):
    """E-step runner that maximizes each subject in turn in the current process."""
    def runner(mu, sigma, prev_z, warm_start):
        n_subjects, n_params = mu.shape
        z_hat = np.empty((n_subjects, n_params))
        variance = np.empty((n_subjects, n_params))
        curvature = np.empty((n_subjects, n_params))
        objective = 0.0
        for s in range(n_subjects):
            mu_s = mu[s]

            def neg_log_post(z, s=s, mu_s=mu_s):
                theta = transform.to_natural(z)
                return -float(log_likelihood_s(theta, s)) - _log_gauss_diag(z, mu_s, sigma)

            z0 = prev_z[s] if warm_start else mu_s
            post = subject_map_estep(
                neg_log_post, z0, method=method, hessian_step=hessian_step,
                variance_floor=variance_floor, prior_variance=sigma, optimizer_options=estep_options,
            )
            z_hat[s] = post.z_hat
            variance[s] = post.variance
            curvature[s] = post.curvature
            objective += subject_laplace_objective(post.neg_log_post, post.variance, n_params)
        return z_hat, variance, curvature, objective

    return runner


@dataclass
class EMResult:
    beta: np.ndarray            # (n_predictors, n_params)
    sigma: np.ndarray           # (n_params,) diagonal variances
    z_hat: np.ndarray           # (n_subjects, n_params) posterior means (unconstrained)
    variance: np.ndarray        # (n_subjects, n_params) posterior variances (unconstrained)
    theta_hat: np.ndarray       # (n_subjects, n_params) MAP in natural units
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)


def fit_laplace_em(
    log_likelihood_s,
    n_subjects,
    n_params,
    transform,
    design_matrix=None,
    *,
    max_em_iterations=50,
    em_tol=1e-4,
    damping=0.0,
    variance_floor=1e-6,
    init_beta=None,
    init_sigma=None,
    estep_method="Nelder-Mead",
    hessian_step=1e-4,
    estep_options=None,
    warm_start=True,
    estep_runner=None,
):
    """Run Laplace EM and return group/subject estimates with an iteration history.

    ``log_likelihood_s(theta, subject)`` returns the scalar log-likelihood of subject
    ``subject``'s data at natural parameters ``theta``. The E-step is delegated to
    ``estep_runner`` (serial by default); a distributed runner gives identical M-step behavior.
    """
    X = np.ones((n_subjects, 1)) if design_matrix is None else np.asarray(design_matrix, float)
    n_pred = X.shape[1]

    beta = np.zeros((n_pred, n_params)) if init_beta is None else np.array(init_beta, float)
    sigma = np.ones(n_params) if init_sigma is None else np.array(init_sigma, float)

    if estep_runner is None:
        estep_runner = make_serial_estep_runner(
            log_likelihood_s, transform, method=estep_method, hessian_step=hessian_step,
            variance_floor=variance_floor, estep_options=estep_options,
        )

    mu = X @ beta                       # (n_subjects, n_params) prior means
    prev_z = mu.copy()                  # warm-start seed for the first E-step

    history = []
    converged = False
    z_hat = prev_z
    variance = np.tile(sigma, (n_subjects, 1))

    for it in range(max_em_iterations):
        mu = X @ beta

        # E-step: independent per-subject MAP + Laplace posterior under current (beta, sigma).
        z_hat, variance, curvature, objective = estep_runner(mu, sigma, prev_z, warm_start)
        prev_z = z_hat

        # M-step: group means by least squares, diagonal variances by posterior moments.
        beta_new = np.linalg.lstsq(X, z_hat, rcond=None)[0]
        resid = z_hat - X @ beta_new
        sigma_new = np.mean(resid ** 2 + variance, axis=0)
        sigma_new = np.maximum(sigma_new, variance_floor)

        if damping > 0.0:
            beta_new = (1.0 - damping) * beta_new + damping * beta
            sigma_new = (1.0 - damping) * sigma_new + damping * sigma

        delta = max(
            float(np.max(np.abs(beta_new - beta))),
            float(np.max(np.abs(sigma_new - sigma))),
        )
        beta, sigma = beta_new, sigma_new

        history.append({
            "iter": it,
            "objective": float(objective),
            "beta": beta.copy(),
            "sigma": sigma.copy(),
            "delta": delta,
        })

        if delta < em_tol:
            converged = True
            break

    theta_hat = np.array([transform.to_natural(z_hat[s]) for s in range(n_subjects)])

    return EMResult(
        beta=beta,
        sigma=sigma,
        z_hat=z_hat,
        variance=variance,
        theta_hat=theta_hat,
        n_iter=len(history),
        converged=converged,
        history=history,
    )
