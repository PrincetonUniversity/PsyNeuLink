"""Hierarchical recovery on the decay-Q bandit + DDM: exact-EM referee vs KDE-EM.

The clamped-history design makes the exact conditional likelihood available (rl_model), so this
sequential model gets a referee'd comparison: exact-EM and exact no-pooling MLE run serially in
seconds; the KDE simulation likelihood runs distributed over subjects like the other studies.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PARENT, "core") not in sys.path:
    sys.path.append(os.path.join(_PARENT, "core"))

import rl_model as rm
import rl_subjects as rs
from estep import subject_map_estep
from laplace_em import fit_laplace_em, make_serial_estep_runner
from dask_estep import make_distributed_estep_runner

ESTEP_OPTIONS = {"xatol": 1e-2, "fatol": 5e-2, "maxiter": 200}
NAT_MEANS = np.array([0.3, 3.0, 0.7, 0.22])   # alpha, beta, threshold, ndt
SIGMA_Z = np.array([0.3, 0.3, 0.3, 0.3])


def _report(label, theta_hat, theta_true):
    rmse = np.sqrt(np.mean((theta_hat - theta_true) ** 2, axis=0))
    corr = [np.corrcoef(theta_hat[:, k], theta_true[:, k])[0, 1]
            for k in range(theta_true.shape[1])]
    print(f"  {label:12s} RMSE = " + "/".join(f"{v:.3f}" for v in rmse)
          + "   corr = " + "/".join(f"{v:.3f}" for v in corr))


def main(n_subjects=24, num_trials=240, num_estimates=1000, max_em_iterations=12,
         n_workers=8, worker_cores=1, seed=11, out_dir=None, skip_kde=False):
    rng = np.random.default_rng(seed)
    tf = rm.transform()
    lower, upper = rm.fit_bounds()
    n_params = len(rm.FIT_PARAMS)
    beta_z = tf.to_unconstrained(NAT_MEANS)

    group = rm.generate_group_data(n_subjects, beta_z, SIGMA_Z, num_trials, rng)
    theta_true = group["theta_true"]
    subs = group["subjects"]
    print(f"{n_subjects} subjects x {num_trials} trials generated")

    def exact_loglik(theta, s):
        return rm.exact_log_likelihood(theta, subs[s]["choices"], subs[s]["rts"], subs[s]["rewards"])

    results = {}

    t0 = time.time()
    r = fit_laplace_em(exact_loglik, n_subjects, n_params, tf, max_em_iterations=50,
                       em_tol=1e-4, variance_floor=1e-4, hessian_step=0.05)
    print(f"exact-EM: {time.time() - t0:.1f}s, converged={r.converged} in {r.n_iter} iters")
    results["exact-EM"] = r.theta_hat

    t0 = time.time()
    flat = make_serial_estep_runner(exact_loglik, tf, hessian_step=0.05, variance_floor=1e-4)
    z_mle, _, _, _ = flat(np.zeros((n_subjects, n_params)), np.full(n_params, 25.0),
                          np.zeros((n_subjects, n_params)), False)
    results["exact no-pool"] = np.array([tf.to_natural(z) for z in z_mle])
    print(f"exact no-pool MLE: {time.time() - t0:.1f}s")

    if not skip_kde:
        from dask.distributed import LocalCluster, Client
        payloads = rs.subject_payloads(group, num_estimates)
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        try:
            runner = make_distributed_estep_runner(
                client, rs.rl_pec_factory, payloads, lower, upper,
                worker_cores=worker_cores, hessian_step=0.15, variance_floor=1e-4,
                estep_options=ESTEP_OPTIONS,
            )
            t0 = time.time()
            r_kde = fit_laplace_em(None, n_subjects=n_subjects, n_params=n_params, transform=tf,
                                   max_em_iterations=max_em_iterations, em_tol=1e-3,
                                   variance_floor=1e-4, hessian_step=0.15, estep_runner=runner)
            print(f"KDE-EM: {time.time() - t0:.1f}s, converged={r_kde.converged} in {r_kde.n_iter} iters")
            results["KDE-EM"] = r_kde.theta_hat
        finally:
            try:
                client.close(); cluster.close()
            except Exception as e:
                print(f"(ignored dask shutdown error: {type(e).__name__})")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "rl_result.pkl"), "wb") as fh:
            pickle.dump({"theta_true": theta_true, "z_true": group["z_true"],
                         "results": results, "fit_params": list(rm.FIT_PARAMS),
                         "schedule": group["schedule"], "subjects": subs}, fh)

    print(f"\nrecovery ({', '.join(rm.FIT_PARAMS)}):")
    for label, th in results.items():
        _report(label, th, theta_true)
    print("  truth SD     = " + "/".join(f"{v:.3f}" for v in theta_true.std(axis=0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_subjects", type=int, default=24)
    p.add_argument("--num_trials", type=int, default=240)
    p.add_argument("--num_estimates", type=int, default=1000)
    p.add_argument("--max_em_iterations", type=int, default=12)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--worker_cores", type=int, default=1)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--skip_kde", action="store_true")
    a = p.parse_args()
    main(a.n_subjects, a.num_trials, a.num_estimates, a.max_em_iterations,
         a.n_workers, a.worker_cores, a.seed, a.out_dir, a.skip_kde)
    sys.stdout.flush()
    os._exit(0)