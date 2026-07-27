"""Stab-flex KDE-EM matrix: {fastkde, gaussian} x {local, pooled} vs NLE-EM.

Reuses PNL's fast LLVM simulation (pec.log_likelihood(..., return_sim_data=True)) to make the
draws, then scores with stabflex_density_policy -- so gaussian KDE and condition-pooling are added
WITHOUT touching the frozen shared E-step core. Runs on the same 24-subject x 120-trial seed-11
dataset as the trained NLE net (stabflex_nle/study_data.pkl), so the NLE-EM rung is comparable.

  run --config {fastkde_local,fastkde_pooled,gaussian_local,gaussian_pooled} --num_estimates N
  aggregate         # join the 4 configs + NLE-EM + truth into metrics.csv + a figure
"""
import argparse
import os
import sys
import time
import uuid

import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("core", "stabflex", "nle"):
    p = os.path.join(_PARENT, _d)
    if p not in sys.path:
        sys.path.append(p)

import stabflex_subjects as sf
import stabflex_nle as sn
import stabflex_density_policy as pol
from transforms import BoundedTransform
from estep import subject_map_estep
from laplace_em import _log_gauss_diag, subject_laplace_objective, fit_laplace_em

NLE_DIR = "/scratch/gpfs/JDC/ap9344/stabflex_nle"
OUT_DIR = "/scratch/gpfs/JDC/ap9344/stabflex_kde_matrix"
NUM_TRIALS = sn.NUM_TRIALS  # 120 -- matches the trained net's dataset
ESTEP_OPTIONS = {"xatol": 1e-2, "fatol": 5e-2, "maxiter": 90}  # capped for the scaled matrix


def _load_study():
    import pickle
    with open(f"{NLE_DIR}/study_data.pkl", "rb") as fh:
        study = pickle.load(fh)
    return study["frames"], np.asarray(study["theta_true"], dtype=float)


# --- distributed E-step task (top-level & picklable; one cached PEC per worker) ----------------
def _kde_subject_estep(subject_id, obs_dec, obs_rt, features, mu_s, sigma, lower, upper, z0,
                       estimator, scope, num_estimates, num_trials, worker_cores, fit_id,
                       estep_kwargs, bin_edges):
    try:
        from dask.distributed import get_worker
        holder = get_worker()
    except (ImportError, ValueError):
        import stabflex_kde_matrix as holder
    cache = getattr(holder, "_matrix_pec_cache", None)
    if cache is None or cache[0] != fit_id:
        from psyneulink.core.globals.threads import set_num_threads
        if worker_cores is not None:
            set_num_threads(worker_cores)
        pec, inputs = sf.stabflex_pec_factory(sn._dummy_payload(num_trials, num_estimates))
        cache = (fit_id, pec, inputs)
        holder._matrix_pec_cache = cache
    _, pec, inputs = cache

    groups = pol.make_groups(scope, features, num_trials, bin_edges=bin_edges)
    tf = BoundedTransform(np.asarray(lower, float), np.asarray(upper, float))
    sigma = np.asarray(sigma, float)
    mu_s = np.asarray(mu_s, float)

    def neg_log_post(z):
        theta = tf.to_natural(z)
        _, sim = pec.log_likelihood(*np.asarray(theta, float), inputs=inputs, return_sim_data=True)
        ll = pol.score_cube(sim, obs_dec, obs_rt, groups, estimator)
        return -ll - _log_gauss_diag(z, mu_s, sigma)

    post = subject_map_estep(neg_log_post, z0, prior_variance=sigma, **estep_kwargs)
    return subject_id, post.z_hat, post.variance, post.curvature, post.neg_log_post


def make_matrix_estep_runner(client, frames, features, lower, upper, *, estimator, scope,
                             num_estimates, num_trials, worker_cores, bin_edges=None):
    fit_id = uuid.uuid4().hex
    estep_kwargs = dict(method="Nelder-Mead", hessian_step=0.15,
                        variance_floor=1e-4, optimizer_options=ESTEP_OPTIONS)
    obs = [(f["decision"].to_numpy(float).astype(int), f["response_time"].to_numpy(float))
           for f in frames]

    def runner(mu, sigma, prev_z, warm_start):
        n_subjects, n_params = mu.shape
        futures = []
        for s in range(n_subjects):
            z0 = prev_z[s] if warm_start else mu[s]
            futures.append(client.submit(
                _kde_subject_estep, s, obs[s][0], obs[s][1], features, mu[s], sigma,
                lower, upper, np.asarray(z0, float), estimator, scope, num_estimates,
                num_trials, worker_cores, fit_id, estep_kwargs, bin_edges, pure=False))
        z_hat = np.empty((n_subjects, n_params))
        variance = np.empty((n_subjects, n_params))
        curvature = np.empty((n_subjects, n_params))
        objective = 0.0
        for sid, zh, var_s, curv, nlp in client.gather(futures):
            z_hat[sid] = zh
            variance[sid] = var_s
            curvature[sid] = curv
            objective += subject_laplace_objective(nlp, var_s, n_params)
        return z_hat, variance, curvature, objective

    return runner


def cmd_run(args):
    from dask.distributed import LocalCluster, Client
    os.makedirs(OUT_DIR, exist_ok=True)
    estimator, scope = pol.parse_config(args.config)
    frames, theta_true = _load_study()
    n_subjects, n_params = theta_true.shape
    features = sn.trial_features(NUM_TRIALS)
    lower, upper = sf.fit_bounds()
    tf = sf.transform()

    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1)
    client = Client(cluster)
    runner = make_matrix_estep_runner(
        client, frames, features, lower, upper, estimator=estimator, scope=scope,
        num_estimates=args.num_estimates, num_trials=NUM_TRIALS, worker_cores=args.worker_cores)

    t0 = time.time()
    result = fit_laplace_em(None, n_subjects=n_subjects, n_params=n_params, transform=tf,
                            max_em_iterations=args.max_em_iterations, em_tol=1e-3,
                            variance_floor=1e-4, hessian_step=0.15, estep_runner=runner)
    dt = time.time() - t0
    print(f"[{args.config}] EM {dt:.1f}s; converged={result.converged} in {result.n_iter} iters")
    print(f"[{args.config}] objective by iter:",
          [round(h["objective"], 1) for h in result.history])
    np.save(f"{OUT_DIR}/theta_{args.config}_b{args.num_estimates}.npy", result.theta_hat)
    rmse = np.sqrt(np.mean((result.theta_hat - theta_true) ** 2, axis=0))
    print(f"[{args.config}] RMSE = " + "/".join(f"{v:.4f}" for v in rmse))
    try:
        client.close(); cluster.close()
    except Exception as e:
        print(f"(ignored dask shutdown: {type(e).__name__})")


def cmd_aggregate(args):
    import glob
    import pandas as pd
    frames, theta_true = _load_study()
    methods = {}
    for f in sorted(glob.glob(f"{OUT_DIR}/theta_*_b{args.num_estimates}.npy")):
        name = os.path.basename(f)[len("theta_"):-len(f"_b{args.num_estimates}.npy")]
        methods[name] = np.load(f)
    nle_path = f"{NLE_DIR}/stabflex_theta_neural_em.npy"
    if os.path.exists(nle_path):
        methods["nle_em"] = np.load(nle_path)

    rows = []
    print(f"\nrecovery on 24 subjects x {NUM_TRIALS} trials  (params: {', '.join(sf.FIT_PARAMS)})")
    truth_sd = theta_true.std(axis=0)
    for name, th in methods.items():
        rmse = np.sqrt(np.mean((th - theta_true) ** 2, axis=0))
        corr = [np.corrcoef(th[:, k], theta_true[:, k])[0, 1] for k in range(theta_true.shape[1])]
        print(f"  {name:16s} RMSE " + "/".join(f"{v:.4f}" for v in rmse)
              + "   r " + "/".join(f"{v:.2f}" for v in corr))
        for k, p in enumerate(sf.FIT_PARAMS):
            rows.append(dict(method=name, parameter=p, rmse=rmse[k], corr=corr[k],
                             norm_rmse=rmse[k] / truth_sd[k], truth_sd=truth_sd[k]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/matrix_metrics.csv", index=False)
    print(f"\nwrote {OUT_DIR}/matrix_metrics.csv")
    _figure(df)


def _figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [m for m in ("fastkde_local", "fastkde_pooled", "gaussian_local",
                         "gaussian_pooled", "nle_em") if m in set(df.method)]
    cols = {"fastkde_local": "#E69F00", "fastkde_pooled": "#D55E00", "gaussian_local": "#56B4E9",
            "gaussian_pooled": "#0072B2", "nle_em": "#009E73"}
    params = list(dict.fromkeys(df.parameter))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(params)); w = 0.8 / len(order)
    for i, m in enumerate(order):
        sub = df[df.method == m].set_index("parameter").loc[params]
        ax.bar(x + i * w - 0.4 + w / 2, sub["norm_rmse"], w, label=m, color=cols.get(m, "#888"))
    ax.set_xticks(x); ax.set_xticklabels(params)
    ax.set_ylabel("RMSE / truth SD  (lower = better)")
    ax.set_title("Stab-flex recovery: KDE estimator x pooling vs NLE-EM")
    ax.axhline(1.0, color="#999", lw=1, ls=":")
    ax.legend(frameon=False, ncol=len(order), fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/matrix_recovery.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_DIR}/matrix_recovery.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True, choices=pol.CONFIGS)
    r.add_argument("--num_estimates", type=int, default=600)
    r.add_argument("--max_em_iterations", type=int, default=6)
    r.add_argument("--n_workers", type=int, default=8)
    r.add_argument("--worker_cores", type=int, default=4)
    a = sub.add_parser("aggregate")
    a.add_argument("--num_estimates", type=int, default=1500)
    args = p.parse_args()
    {"run": cmd_run, "aggregate": cmd_aggregate}[args.cmd](args)
