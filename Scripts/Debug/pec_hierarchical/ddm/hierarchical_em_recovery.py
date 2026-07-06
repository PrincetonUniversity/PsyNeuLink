"""Robust group-parameter recovery for the hierarchical PEC (distributed EM).

Generates a synthetic multi-subject DDM with known group (beta_z, sigma_z), fits it with the
Dask-distributed Laplace EM, and reports recovery of the group means and SDs against both the
population truth and the finite-sample statistics. The synthetic dataset is saved so the same
data can later be fit with HSSM for a cross-tool comparison.
"""

import sys
import os
import pickle
import time

import numpy as np
import pandas as pd
from dask.distributed import LocalCluster, Client

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core",):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
from ddm_subjects import (
    generate_group_data, ddm_pec_factory, transform, fit_bounds, FIT_PARAMS,
)
from laplace_em import fit_laplace_em
from dask_estep import make_distributed_estep_runner

ESTEP_OPTIONS = {"xatol": 1e-3, "fatol": 1e-2, "maxiter": 250}


def save_dataset(group, out_dir):
    """Persist the stacked trial table, per-subject truth, and group truth for HSSM reuse."""
    os.makedirs(out_dir, exist_ok=True)
    inputs = group["trial_inputs"].ravel()
    frames = []
    for s, payload in enumerate(group["payloads"]):
        df = payload["data"].copy()
        df["subject"] = s
        df["stimulus"] = inputs[: len(df)]
        frames.append(df)
    stacked = pd.concat(frames, ignore_index=True)
    stacked.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    with open(os.path.join(out_dir, "truth.pkl"), "wb") as fh:
        pickle.dump({
            "theta_true": group["theta_true"], "z_true": group["z_true"],
            "beta_z": group["beta_z"], "sigma_z": group["sigma_z"],
            "fit_params": list(FIT_PARAMS), "bounds": fit_bounds(),
        }, fh)
    return stacked


def main(n_subjects=40, num_trials=70, num_estimates=120, max_em_iterations=10,
         n_workers=8, seed=11, out_dir=None):
    rng = np.random.default_rng(seed)
    tf = transform()
    lower, upper = fit_bounds()
    beta_z_true = np.array([0.0, 0.0])
    sigma_z_true = np.array([0.36, 0.36])

    print(f"generating {n_subjects} subjects ({num_trials} trials, {num_estimates} estimates) ...")
    t0 = time.time()
    group = generate_group_data(n_subjects, beta_z_true, sigma_z_true, num_trials, rng, num_estimates)
    if out_dir:
        save_dataset(group, out_dir)
    print(f"  generated in {time.time() - t0:.1f}s")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"  dask: {n_workers} workers, {max(1, 8 // n_workers)} core(s) each")
    try:
        runner = make_distributed_estep_runner(
            client, ddm_pec_factory, group["payloads"], lower, upper,
            worker_cores=max(1, 8 // n_workers), hessian_step=0.15, variance_floor=1e-5,
            estep_options=ESTEP_OPTIONS,
        )
        t0 = time.time()
        result = fit_laplace_em(
            None, n_subjects=n_subjects, n_params=len(FIT_PARAMS), transform=tf,
            max_em_iterations=max_em_iterations, em_tol=1e-3, variance_floor=1e-5,
            hessian_step=0.15, estep_runner=runner,
        )
        print(f"  EM done in {time.time() - t0:.1f}s; converged={result.converged} in {result.n_iter} iters")
    finally:
        client.close()
        cluster.close()

    z_true = group["z_true"]
    print("objective by iter:", [round(h["objective"], 1) for h in result.history])
    print()
    print("group MEAN (beta_z):")
    print("  population :", beta_z_true)
    print("  sample     :", z_true.mean(axis=0))
    print("  recovered  :", result.beta.ravel())
    print("group SD (sqrt sigma_z):")
    print("  population :", np.sqrt(sigma_z_true))
    print("  sample     :", z_true.std(axis=0))
    print("  recovered  :", np.sqrt(result.sigma))
    print()
    # sigma = var(z_hat) + mean(V): shows whether z_hat is over-shrunk or V is underestimated.
    print("variance decomposition (z-space):")
    print("  var(z_hat)      :", result.z_hat.var(axis=0))
    print("  mean(V)         :", result.variance.mean(axis=0))
    print("  var(z_hat)+meanV:", result.z_hat.var(axis=0) + result.variance.mean(axis=0))
    print("  sigma_hat       :", result.sigma)
    print("  var(z_true)     :", z_true.var(axis=0))
    print()
    # Natural-param group recovery (rate, threshold) -- the interpretable, HDDM-comparable view.
    theta_true = group["theta_true"]
    print("group recovery in NATURAL units (rate, threshold):")
    print("  mean true / hat:", theta_true.mean(axis=0), "/", result.theta_hat.mean(axis=0))
    print("  SD   true / hat:", theta_true.std(axis=0), "/", result.theta_hat.std(axis=0))
    print()
    rmse = np.sqrt(np.mean((result.theta_hat - group["theta_true"]) ** 2, axis=0))
    print("per-subject natural-param RMSE:", rmse)
    corr = [np.corrcoef(result.z_hat[:, k], z_true[:, k])[0, 1] for k in range(len(FIT_PARAMS))]
    print("per-subject z_hat vs z_true correlation:", np.round(corr, 3))


if __name__ == "__main__":
    main()
