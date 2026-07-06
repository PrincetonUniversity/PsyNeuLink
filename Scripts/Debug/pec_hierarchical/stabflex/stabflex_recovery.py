"""Hierarchical recovery for the Stability-Flexibility composition (distributed Laplace EM).

Reuses the composition-agnostic EM/Dask machinery with the stab-flex per-subject model. Generates
a synthetic multi-subject dataset with known group parameters, fits the 4-parameter hierarchy
distributed over subjects, and reports natural-space group recovery. Parametrized so the same
driver serves a quick validation and the large overnight study.
"""

import os
import pickle
import sys
import time

import numpy as np
from dask.distributed import LocalCluster, Client

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core",):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
import stabflex_subjects as sf
from laplace_em import fit_laplace_em
from dask_estep import make_distributed_estep_runner

ESTEP_OPTIONS = {"xatol": 1e-2, "fatol": 5e-2, "maxiter": 150}

# Plausible group means (natural units) for gain, slope, threshold, non_decision_time.
NAT_MEANS = np.array([3.0, 0.015, 0.12, 0.22])
SIGMA_Z = np.array([0.30, 0.30, 0.30, 0.30])


def save(out_dir, group, result, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "theta_true": group["theta_true"], "z_true": group["z_true"],
        "beta_z": group["beta_z"], "sigma_z": group["sigma_z"],
        "theta_hat": result.theta_hat, "beta": result.beta, "sigma": result.sigma,
        "history": result.history, "fit_params": list(sf.FIT_PARAMS),
    }
    if extra:
        payload.update(extra)
    with open(os.path.join(out_dir, "stabflex_result.pkl"), "wb") as fh:
        pickle.dump(payload, fh)


def _report(label, theta_hat, theta_true):
    print(f"{label} (gain, slope, threshold, ndt):")
    print("  mean true:", np.round(theta_true.mean(0), 4))
    print("  mean hat :", np.round(theta_hat.mean(0), 4))
    print("  SD   true:", np.round(theta_true.std(0), 4))
    print("  SD   hat :", np.round(theta_hat.std(0), 4))
    corr = [np.corrcoef(theta_hat[:, k], theta_true[:, k])[0, 1]
            for k in range(theta_true.shape[1])]
    print("  per-subject corr:", np.round(corr, 3))
    rmse = np.sqrt(np.mean((theta_hat - theta_true) ** 2, axis=0))
    print("  per-subject RMSE:", np.round(rmse, 4))


def main(n_subjects=4, num_trials=100, num_estimates=1000, max_em_iterations=3,
         n_workers=4, worker_cores=2, seed=11, out_dir=None):
    import uuid

    rng = np.random.default_rng(seed)
    tf = sf.transform()
    lower, upper = sf.fit_bounds()
    beta_z = tf.to_unconstrained(NAT_MEANS)
    n_params = len(sf.FIT_PARAMS)

    print(f"generating {n_subjects} subjects ({num_trials} trials, {num_estimates} est) ...")
    t0 = time.time()
    group = sf.generate_group_data(n_subjects, beta_z, SIGMA_Z, num_trials, rng, num_estimates)
    print(f"  generated in {time.time() - t0:.1f}s")
    theta_true = group["theta_true"]

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"  dask: {n_workers} workers x {worker_cores} core(s)")
    fit_id = uuid.uuid4().hex
    runner_kwargs = dict(worker_cores=worker_cores, fit_id=fit_id, hessian_step=0.15,
                         variance_floor=1e-4, estep_options=ESTEP_OPTIONS)
    try:
        runner = make_distributed_estep_runner(
            client, sf.stabflex_pec_factory, group["payloads"], lower, upper, **runner_kwargs,
        )
        t0 = time.time()
        result = fit_laplace_em(
            None, n_subjects=n_subjects, n_params=n_params, transform=tf,
            max_em_iterations=max_em_iterations, em_tol=1e-3, variance_floor=1e-4,
            hessian_step=0.15, estep_runner=runner,
        )
        print(f"  EM done in {time.time() - t0:.1f}s; converged={result.converged} in {result.n_iter} iters")
        if out_dir:
            save(out_dir, group, result)

        # No-pooling baseline: one flat-prior E-step (independent per-subject MLE) using the same
        # fit_id, so workers reuse their cached PECs. Quantifies what partial pooling buys.
        t0 = time.time()
        flat_sigma = np.full(n_params, 25.0)
        z_mle, _, _, _ = runner(np.zeros((n_subjects, n_params)), flat_sigma,
                                np.zeros((n_subjects, n_params)), False)
        theta_mle = np.array([tf.to_natural(z_mle[s]) for s in range(n_subjects)])
        print(f"  MLE baseline done in {time.time() - t0:.1f}s")
        if out_dir:
            save(out_dir, group, result, extra={"theta_mle": theta_mle, "z_mle": z_mle})
    finally:
        # Best-effort teardown: a hung worker shutdown must not discard a finished fit.
        try:
            client.close()
            cluster.close()
        except Exception as e:
            print(f"(ignored dask shutdown error: {type(e).__name__}: {e})")

    print("objective by iter:", [round(h["objective"], 1) for h in result.history])
    _report("HIERARCHICAL group recovery in NATURAL units", result.theta_hat, theta_true)
    _report("NO-POOLING MLE baseline in NATURAL units", theta_mle, theta_true)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_subjects", type=int, default=4)
    p.add_argument("--num_trials", type=int, default=100)
    p.add_argument("--num_estimates", type=int, default=1000)
    p.add_argument("--max_em_iterations", type=int, default=3)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--worker_cores", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    main(a.n_subjects, a.num_trials, a.num_estimates, a.max_em_iterations,
         a.n_workers, a.worker_cores, a.seed, a.out_dir)
    # Results are saved and printed; skip interpreter teardown so lingering dask nanny
    # threads cannot turn a successful fit into a failed batch job.
    sys.stdout.flush()
    os._exit(0)
