"""Fit PEC-EM on the exported HSSM dataset so both tools fit identical trials.

Loads the shared-filesystem dataset.csv, rebuilds per-subject PEC payloads, runs the
Dask-distributed Laplace EM, and saves the per-subject natural-parameter estimates for comparison
against HSSM. Run in the PNL venv on a compute node.
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from dask.distributed import LocalCluster, Client

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core",):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
from ddm_subjects import ddm_pec_factory, transform, fit_bounds, trial_inputs_for, FIT_PARAMS
from laplace_em import fit_laplace_em
from dask_estep import make_distributed_estep_runner

ESTEP_OPTIONS = {"xatol": 1e-3, "fatol": 1e-2, "maxiter": 250}


def load_payloads(csv, num_estimates=120):
    d = pd.read_csv(csv)
    payloads = []
    for s, g in d.groupby("subj_idx", sort=True):
        df = pd.DataFrame({"decision": g["decision"].to_numpy(), "response_time": g["rt"].to_numpy()})
        df["decision"] = df["decision"].astype("category")
        # Inputs are reconstructed from the canonical coherence pattern; verify alignment.
        ti = trial_inputs_for(len(df)).ravel()
        assert np.allclose(ti, g["stimulus"].to_numpy()), f"stimulus order mismatch for subject {s}"
        payloads.append({"data": df, "seed": 100 + int(s),
                         "num_estimates": num_estimates, "num_trials": len(df)})
    return payloads


def pooled_mle(data_dir, num_estimates=120, llvm_threads=8):
    """Complete-pooling baseline: one shared parameter set, MLE on all subjects' data stacked.

    The opposite failure mode from no-pooling: no individual differences, group variance assumed
    zero, and a potentially biased group mean (the stacked data is a mixture across subjects).
    """
    import pandas as pd
    from psyneulink.core.globals.threads import set_num_threads
    from ddm_subjects import make_subject_pec, trial_inputs_for
    from estep import subject_map_estep

    tf = transform()
    payloads = load_payloads(f"{data_dir}/dataset.csv")
    stacked = pd.concat([p["data"] for p in payloads], ignore_index=True)
    stacked["decision"] = stacked["decision"].astype("category")
    num_trials = payloads[0]["num_trials"]
    inputs = np.vstack([trial_inputs_for(num_trials)] * len(payloads))

    set_num_threads(llvm_threads)
    pec, comp = make_subject_pec(stacked, num_estimates=num_estimates, initial_seed=100)

    def neg_ll(z):
        return -float(pec.log_likelihood(*tf.to_natural(z), inputs={comp: inputs}))

    t0 = time.time()
    post = subject_map_estep(neg_ll, np.zeros(len(FIT_PARAMS)), hessian_step=0.15,
                             optimizer_options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 150})
    theta_pooled = tf.to_natural(post.z_hat)
    print(f"pooled MLE done in {time.time() - t0:.1f}s  theta={np.round(theta_pooled, 4)}")
    np.save(f"{data_dir}/pec_theta_pooled.npy", theta_pooled)
    print("saved", f"{data_dir}/pec_theta_pooled.npy")


def mle_baseline(data_dir, n_workers=8):
    """No-pooling baseline: one flat-prior E-step (independent per-subject MLE) on the saved data."""
    tf = transform()
    lower, upper = fit_bounds()
    payloads = load_payloads(f"{data_dir}/dataset.csv")
    n_subjects = len(payloads)
    n_params = len(FIT_PARAMS)

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    try:
        runner = make_distributed_estep_runner(
            client, ddm_pec_factory, payloads, lower, upper,
            worker_cores=max(1, 8 // n_workers), hessian_step=0.15, variance_floor=1e-5,
            estep_options=ESTEP_OPTIONS,
        )
        t0 = time.time()
        z_mle, _, _, _ = runner(np.zeros((n_subjects, n_params)), np.full(n_params, 25.0),
                                np.zeros((n_subjects, n_params)), False)
        print(f"MLE baseline done in {time.time() - t0:.1f}s")
    finally:
        try:
            client.close()
            cluster.close()
        except Exception as e:
            print(f"(ignored dask shutdown error: {type(e).__name__}: {e})")

    theta_mle = np.array([tf.to_natural(z_mle[s]) for s in range(n_subjects)])
    np.save(f"{data_dir}/pec_theta_mle.npy", theta_mle)
    print("saved", f"{data_dir}/pec_theta_mle.npy")


def main(data_dir, n_workers=8):
    tf = transform()
    lower, upper = fit_bounds()
    payloads = load_payloads(f"{data_dir}/dataset.csv")
    with open(f"{data_dir}/truth.pkl", "rb") as fh:
        truth = pickle.load(fh)
    theta_true = truth["theta_true"]
    n_subjects = len(payloads)

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    try:
        runner = make_distributed_estep_runner(
            client, ddm_pec_factory, payloads, lower, upper,
            worker_cores=max(1, 8 // n_workers), hessian_step=0.15, variance_floor=1e-5,
            estep_options=ESTEP_OPTIONS,
        )
        t0 = time.time()
        result = fit_laplace_em(
            None, n_subjects=n_subjects, n_params=len(FIT_PARAMS), transform=tf,
            max_em_iterations=10, em_tol=1e-3, variance_floor=1e-5, hessian_step=0.15,
            estep_runner=runner,
        )
        print(f"EM done in {time.time() - t0:.1f}s; converged={result.converged} in {result.n_iter} iters")
    finally:
        client.close()
        cluster.close()

    np.save(f"{data_dir}/pec_theta_hat.npy", result.theta_hat)
    names = FIT_PARAMS
    print("PEC-EM recovery on saved data (rate, threshold):")
    for k, name in enumerate(names):
        print(f"  {name}: mean true/hat = {theta_true[:,k].mean():.3f}/{result.theta_hat[:,k].mean():.3f}"
              f"   SD true/hat = {theta_true[:,k].std():.3f}/{result.theta_hat[:,k].std():.3f}"
              f"   corr = {np.corrcoef(result.theta_hat[:,k], theta_true[:,k])[0,1]:.3f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data_dir = args[0] if args else "/scratch/gpfs/JDC/ap9344/hssm_compare"
    if "--mle" in sys.argv:
        mle_baseline(data_dir)
    elif "--pooled" in sys.argv:
        pooled_mle(data_dir)
    else:
        main(data_dir)
