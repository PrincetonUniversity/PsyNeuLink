"""Smoke run of Laplace EM on the Gaussian hierarchical toy model.

Recovers known group parameters from synthetic data and prints the EM history alongside the
closed-form posterior, so the machinery can be eyeballed before the LLVM PEC likelihood is wired in.
"""

import numpy as np

from laplace_em import fit_laplace_em
from toy_model import ToyHierarchicalModel
from transforms import IdentityTransform


def main():
    rng = np.random.default_rng(0)
    beta_true = np.array([0.5, -1.0])
    sigma_true = np.array([0.4, 0.9])
    tau = 1.0
    n_subjects = 200
    n_obs = 20

    model = ToyHierarchicalModel.generate(
        n_subjects, beta_true.size, beta_true, sigma_true, tau, n_obs, rng
    )

    result = fit_laplace_em(
        model.log_likelihood_s,
        n_subjects=n_subjects,
        n_params=beta_true.size,
        transform=IdentityTransform(),
        max_em_iterations=100,
        em_tol=1e-5,
    )

    print(f"converged={result.converged} in {result.n_iter} iters")
    print("objective by iter:", [round(h["objective"], 2) for h in result.history[-5:]])
    print("beta_true :", beta_true)
    print("beta_hat  :", result.beta.ravel())
    print("sigma_true:", sigma_true)
    print("sigma_hat :", result.sigma)

    # Cross-check the final E-step against the exact posterior at the fitted group params.
    mu = np.tile(result.beta.ravel(), (n_subjects, 1))
    z_cf, v_cf = model.closed_form_posterior(mu, result.sigma)
    print("max |z_hat - z_closed_form|:", np.max(np.abs(result.z_hat - z_cf)))
    print("max |V    - V_closed_form|:", np.max(np.abs(result.variance - v_cf)))


if __name__ == "__main__":
    main()
