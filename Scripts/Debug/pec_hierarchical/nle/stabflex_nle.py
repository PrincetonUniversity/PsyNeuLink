"""Neural likelihood for the Stability-Flexibility composition — the generalization test.

Unlike the DDM, stab-flex has no analytical likelihood, four fitted parameters, and trials that
are exchangeable only conditional on trial inputs (task cue, stimulus pair, switch vs repeat,
correct response). The net conditions on (theta, trial features); training data comes from
distributed PsyNeuLink simulation at parameters sampled over the fit box. The only referees are
held-out simulation NLL and hierarchical-EM recovery against the KDE-EM results on the same
24-subject dataset (stabflex_result.pkl).

Subcommands: gen (distributed sim of training configs + regeneration of the study dataset),
train, em. Run on a compute node.
"""

import os
import sys
import argparse
import pickle
import time

import numpy as np
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core", "stabflex"):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
import neural_likelihood as nl
from transforms import BoundedTransform
from laplace_em import fit_laplace_em

OUT_DIR = "/scratch/gpfs/JDC/ap9344/stabflex_nle"
SMALL_RESULT = "/scratch/gpfs/JDC/ap9344/stabflex_hier_small/stabflex_result.pkl"
NUM_TRIALS = 120           # matches the 24-subject small study
N_COMP = 8
N_FEATURES = 5             # task, s1, s2, switch, correct


def trial_features(num_trials):
    """Input-derived per-trial features for the shared trial sequence."""
    import stabflex_subjects as sf
    task, stim, _cue, correct = sf._trial_sequence(num_trials)
    task = np.array([t[0] - t[1] for t in task], float)          # [1,0] -> +1, [0,1] -> -1
    stim = np.array(stim, float)                                  # (T, 2) of +/-1
    switch = np.ones(num_trials)
    switch[1:] = np.where(task[1:] == task[:-1], -1.0, 1.0)
    correct = np.array(correct, float)
    return np.column_stack([task, stim[:, 0], stim[:, 1], switch, correct])


def _dummy_payload(num_trials, num_estimates):
    """Placeholder observed data: the PEC needs it to construct, but only sim_data is used."""
    import pandas as pd
    df = pd.DataFrame({"decision": np.tile([0.0, 1.0], num_trials)[:num_trials],
                       "response_time": np.full(num_trials, 0.8)})
    df["decision"] = df["decision"].astype("category")
    return {"data": df, "seed": 7, "num_estimates": num_estimates, "num_trials": num_trials}


def _sim_config(theta, num_trials, num_estimates, fit_id):
    """One training config on a worker: sweep theta through a worker-cached PEC and return the
    simulated outcomes. Rebuilding a composition per config leaks PNL/LLVM state until dask
    pauses the worker, so the PEC (whose job is parameter sweeping) is built once and reused.

    No lock: workers run single-threaded, and this function is pickled by value (it lives in
    __main__), so it must not reference module-level unpicklables. The cache holder is resolved
    at call time: the dask worker object, or the module itself outside dask.
    """
    import stabflex_subjects as sf
    try:
        from dask.distributed import get_worker
        holder = get_worker()
    except (ImportError, ValueError):
        import stabflex_nle
        holder = stabflex_nle
    cache = getattr(holder, "_gen_pec_cache", None)
    if cache is None or cache[0] != fit_id:
        from psyneulink.core.globals.threads import set_num_threads
        set_num_threads(1)
        pec, inputs = sf.stabflex_pec_factory(_dummy_payload(num_trials, num_estimates))
        cache = (fit_id, pec, inputs)
        holder._gen_pec_cache = cache
    _, pec, inputs = cache
    _, sim = pec.log_likelihood(*np.asarray(theta, float), inputs=inputs, return_sim_data=True)
    # sim: (num_trials, num_estimates, 2) with outcomes (decision, rt)
    return (np.asarray(theta, float),
            sim[:, :, 0].astype(np.int8), sim[:, :, 1].astype(np.float32))


def cmd_gen(args):
    import os
    from dask.distributed import LocalCluster, Client
    import stabflex_subjects as sf
    from stabflex_recovery import NAT_MEANS, SIGMA_Z

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(1)
    lo, hi = sf.fit_bounds()
    thetas = rng.uniform(lo, hi, size=(args.configs, len(sf.FIT_PARAMS)))

    import uuid
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    fit_id = uuid.uuid4().hex
    t0 = time.time()
    futures = [client.submit(_sim_config, thetas[i], NUM_TRIALS, args.estimates, fit_id, pure=False)
               for i in range(args.configs)]
    rows = client.gather(futures)
    print(f"simulated {args.configs} configs x {NUM_TRIALS} trials x {args.estimates} estimates "
          f"in {time.time() - t0:.1f}s")

    theta_rows = np.stack([r[0] for r in rows])
    choice = np.stack([r[1] for r in rows])   # (configs, trials, estimates)
    rt = np.stack([r[2] for r in rows])
    print("choice values:", np.unique(choice), "| rt range:", rt.min().round(3), rt.max().round(3))
    np.savez(f"{OUT_DIR}/train.npz", theta=theta_rows, choice=choice, rt=rt,
             features=trial_features(NUM_TRIALS))

    # Regenerate the 24-subject study data (same seed path as the small study) and save it.
    rng11 = np.random.default_rng(11)
    tf = sf.transform()
    beta_z = tf.to_unconstrained(NAT_MEANS)
    group = sf.generate_group_data(24, beta_z, SIGMA_Z, NUM_TRIALS, rng11, num_estimates=1)
    with open(SMALL_RESULT, "rb") as fh:
        saved = pickle.load(fh)
    assert np.allclose(group["theta_true"], saved["theta_true"]), "seed path drifted from small study"
    with open(f"{OUT_DIR}/study_data.pkl", "wb") as fh:
        pickle.dump({"frames": [p["data"] for p in group["payloads"]],
                     "theta_true": group["theta_true"]}, fh)
    print("study data regenerated and verified against stabflex_result.pkl")

    try:
        client.close(); cluster.close()
    except Exception as e:
        print(f"(ignored dask shutdown error: {type(e).__name__})")


def _flatten(dataset):
    theta, choice, rt, feats = (dataset["theta"], dataset["choice"], dataset["rt"],
                                dataset["features"])
    n_cfg, n_tr, n_est = choice.shape
    cond = np.concatenate([
        np.repeat(theta, n_tr * n_est, axis=0),
        np.tile(np.repeat(feats, n_est, axis=0), (n_cfg, 1)),
    ], axis=1)
    return cond, choice.ravel().astype(np.int64), rt.ravel().astype(float)


def cmd_train(args):
    d = np.load(f"{OUT_DIR}/train.npz")
    cond, choice, rt = _flatten(d)
    print(f"training rows: {cond.shape[0]}  cond dims: {cond.shape[1]}")
    torch.manual_seed(0)
    net = nl.DDMMixtureNet(n_comp=N_COMP, n_inputs=cond.shape[1])
    t0 = time.time()
    # The time argument is the raw rt: ndt is a fitted parameter, so the net models log rt directly.
    nl.fit(net, cond, choice, rt, epochs=args.epochs)
    print(f"trained in {time.time() - t0:.1f}s")
    torch.save(net.state_dict(), f"{OUT_DIR}/stabflex_mdn.pt")
    print("saved", f"{OUT_DIR}/stabflex_mdn.pt")


def cmd_em(args):
    import stabflex_subjects as sf
    with open(f"{OUT_DIR}/study_data.pkl", "rb") as fh:
        study = pickle.load(fh)
    frames, theta_true = study["frames"], study["theta_true"]
    n_subjects, n_params = theta_true.shape
    feats = trial_features(NUM_TRIALS)

    net = nl.DDMMixtureNet(n_comp=N_COMP, n_inputs=n_params + N_FEATURES)
    net.load_state_dict(torch.load(f"{OUT_DIR}/stabflex_mdn.pt", weights_only=True))
    net.eval()

    def neural_loglik(theta, s):
        df = frames[s]
        n = len(df)
        cond = np.concatenate([np.tile(np.asarray(theta, float), (n, 1)), feats[:n]], axis=1)
        return float(net.cond_log_density(cond, df["decision"].to_numpy(float).astype(int),
                                          df["response_time"].to_numpy(float)).sum())

    lo, hi = sf.fit_bounds()
    tf = BoundedTransform(lo, hi)
    t0 = time.time()
    r = fit_laplace_em(neural_loglik, n_subjects, n_params, tf, max_em_iterations=50,
                       em_tol=1e-4, variance_floor=1e-4, hessian_step=0.05)
    print(f"neural EM: {time.time() - t0:.1f}s, converged={r.converged} in {r.n_iter} iters")
    np.save(f"{OUT_DIR}/stabflex_theta_neural_em.npy", r.theta_hat)

    with open(SMALL_RESULT, "rb") as fh:
        saved = pickle.load(fh)
    results = {"neural-EM": r.theta_hat, "KDE-EM": saved["theta_hat"],
               "no-pool MLE": saved["theta_mle"]}
    print(f"\nrecovery on the 24-subject stab-flex dataset ({', '.join(sf.FIT_PARAMS)}):")
    for label, th in results.items():
        rmse = np.sqrt(np.mean((th - theta_true) ** 2, axis=0))
        corr = [np.corrcoef(th[:, k], theta_true[:, k])[0, 1] for k in range(n_params)]
        print(f"  {label:12s} RMSE = " + "/".join(f"{v:.3f}" for v in rmse)
              + "   corr = " + "/".join(f"{v:.3f}" for v in corr))
    print("  truth SD     = " + "/".join(f"{v:.3f}" for v in theta_true.std(axis=0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--configs", type=int, default=4000)
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--estimates", type=int, default=6)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    sub.add_parser("em")
    args = p.parse_args()
    {"gen": cmd_gen, "train": cmd_train, "em": cmd_em}[args.cmd](args)
