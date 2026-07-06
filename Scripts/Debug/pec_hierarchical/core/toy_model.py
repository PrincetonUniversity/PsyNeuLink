"""Gaussian hierarchical toy model with a closed-form likelihood and posterior.

Used to validate the Laplace EM machinery against exact answers, with no PNL involved.

Generative model (unconstrained space, identity transform):

    z_s         ~ N(beta_true, diag(sigma_true))            group random effect
    y_{s,i}     ~ N(z_s, tau^2 I),  i = 1..n_obs            per-subject observations

The per-subject likelihood is Gaussian in ``z`` with mean ``ybar_s`` and precision
``n_obs / tau^2``, so the subject posterior under a Gaussian prior is available in closed form
for cross-checking the numerical E-step and M-step.
"""

from dataclasses import dataclass

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class ToyHierarchicalModel:
    ybar: np.ndarray     # (n_subjects, n_params) per-subject means
    ss: np.ndarray       # (n_subjects,) within-subject sums of squares
    n_obs: int
    tau: float
    z_true: np.ndarray   # (n_subjects, n_params) sampled subject effects (ground truth)

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
        """Log-likelihood of subject ``s``'s data at natural parameters ``theta`` (== z)."""
        theta = np.asarray(theta, float)
        quad = self.ss[s] + self.n_obs * np.sum((theta - self.ybar[s]) ** 2)
        norm = 0.5 * self.n_obs * self.n_params * (LOG_2PI + 2.0 * np.log(self.tau))
        return -0.5 * quad / self.tau ** 2 - norm

    def closed_form_posterior(self, mu, sigma):
        """Exact subject posteriors ``N(z_hat, V)`` given prior means ``mu`` and variances ``sigma``."""
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        lam_lik = self.n_obs / self.tau ** 2
        lam_prior = 1.0 / sigma
        lam_post = lam_lik + lam_prior
        z_hat = (lam_lik * self.ybar + lam_prior * mu) / lam_post
        variance = np.broadcast_to(1.0 / lam_post, self.ybar.shape).copy()
        return z_hat, variance
