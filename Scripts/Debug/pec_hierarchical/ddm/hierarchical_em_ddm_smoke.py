"""Serial hierarchical EM on a synthetic multi-subject DDM (M2).

Draws subject parameters from a group Gaussian (in unconstrained space), simulates each subject's
DDM data, then recovers the group (beta, sigma) with Laplace EM over the real LLVM PEC likelihood.
First checks a single-subject MAP, then runs the full serial EM.
"""

import os
import sys
import time
import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core",):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
from ddm_subjects import generate_group_data, build_serial_subjects, transform, FIT_PARAMS
from pec_likelihood import PECGroupLikelihood
from estep import subject_map_estep
from laplace_em import fit_laplace_em


def main(n_subjects=8, num_trials=30, num_estimates=100, max_em_iterations=10, seed=0):
    rng = np.random.default_rng(seed)
    tf = transform()

    beta_z_true = np.array([0.3, -0.2])
    sigma_z_true = np.array([0.4, 0.4])

    print(f"generating {n_subjects} subjects ({num_trials} trials, {num_estimates} estimates) ...")
    t0 = time.time()
    group = generate_group_data(
        n_subjects, beta_z_true, sigma_z_true, num_trials, rng, num_estimates=num_estimates
    )
    subjects = build_serial_subjects(group["payloads"])
    like = PECGroupLikelihood(subjects, tf)
    print(f"  built in {time.time() - t0:.1f}s; fit params = {like.fit_param_names}")

    # Single-subject MAP sanity check (flat prior == MLE).
    def neg_ll0(z):
        return -like.log_likelihood_s(tf.to_natural(z), 0)

    z0 = tf.to_unconstrained(group["theta_true"][0])
    post0 = subject_map_estep(neg_ll0, np.zeros(len(FIT_PARAMS)))
    print("subject 0 theta_true:", group["theta_true"][0])
    print("subject 0 theta_mle :", tf.to_natural(post0.z_hat))

    # Serial EM. Coarse per-iteration MAP tolerance keeps each E-step cheap.
    estep_options = {"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200}
    print("running serial EM ...")
    t0 = time.time()
    result = fit_laplace_em(
        like.log_likelihood_s,
        n_subjects=n_subjects,
        n_params=len(FIT_PARAMS),
        transform=tf,
        max_em_iterations=max_em_iterations,
        em_tol=1e-3,
        variance_floor=1e-3,
        hessian_step=1e-3,
        estep_options=estep_options,
    )
    print(f"  EM done in {time.time() - t0:.1f}s; converged={result.converged} in {result.n_iter} iters")
    print("objective by iter:", [round(h["objective"], 1) for h in result.history])
    # EM recovers the sample-level MLE; with few subjects the sample stats differ from the
    # population values, so report both for an honest comparison.
    z_true = group["z_true"]
    print("beta_z  pop / sample / hat:", beta_z_true, "/", z_true.mean(axis=0), "/", result.beta.ravel())
    print("sigma_z pop / sample / hat:", sigma_z_true, "/", z_true.var(axis=0), "/", result.sigma)

    theta_true = group["theta_true"]
    print("per-subject natural MAP recovery (rmse):",
          np.sqrt(np.mean((result.theta_hat - theta_true) ** 2, axis=0)))


if __name__ == "__main__":
    main()
