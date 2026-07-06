"""Neural-likelihood study: train the DDM surrogate, validate it, and run hierarchical EM with it.

Subcommands (run on a compute node):
  train    simulate a training set, fit the mixture density net, save a checkpoint
  compare  pointwise + subject-level likelihood comparison: neural vs analytical vs KDE, with timing
  em       hierarchical EM on the saved 40-subject dataset with the neural and the analytical
           likelihoods; recovery table against truth and the saved KDE-EM / no-pooling estimates

The saved dataset and estimates come from export_hssm_dataset.py / fit_saved_dataset.py.
"""

import os
import sys
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core", "ddm"):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
import neural_likelihood as nl
from transforms import BoundedTransform
from laplace_em import fit_laplace_em

DATA_DIR = "/scratch/gpfs/JDC/ap9344/hssm_compare"
CKPT = f"{DATA_DIR}/neural_ddm_mdn.pt"
BOUNDS = ((-1.5, 1.5), (0.3, 1.5))  # rate, threshold (matches ddm_subjects.FIT_RANGES)


def load_subjects(data_dir):
    d = pd.read_csv(f"{data_dir}/dataset.csv")
    frames = []
    for _, g in d.groupby("subj_idx", sort=True):
        frames.append(pd.DataFrame({
            "decision": g["decision"].to_numpy(float),
            "response_time": g["rt"].to_numpy(float),
        }))
    stimuli = d.loc[d.subj_idx == 0, "stimulus"].to_numpy(float)
    with open(f"{data_dir}/truth.pkl", "rb") as fh:
        truth = pickle.load(fh)
    return frames, stimuli, truth


def cmd_train(args):
    rng = np.random.default_rng(0)
    t0 = time.time()
    drift, thresh, choice, fpt = nl.make_training_set(args.configs, args.trials, rng)
    print(f"training set: {drift.size} rows (mirror-augmented) in {time.time() - t0:.1f}s")

    # simulator vs analytical density: total log-density of simulated draws should match
    lp = nl.wfpt_logpdf(fpt[:200000] + nl.NDT, choice[:200000], drift[:200000], thresh[:200000])
    print(f"analytic log-density of simulated draws: mean {lp.mean():.4f} (finite check: {np.isfinite(lp).mean():.4f})")

    torch.manual_seed(0)
    net = nl.DDMMixtureNet()
    t0 = time.time()
    nl.fit(net, np.stack([drift, thresh], axis=1), choice, fpt, epochs=args.epochs)
    print(f"trained in {time.time() - t0:.1f}s")
    torch.save(net.state_dict(), CKPT)
    print("saved", CKPT)


def _load_net():
    net = nl.DDMMixtureNet()
    net.load_state_dict(torch.load(CKPT, weights_only=True))
    net.eval()
    return net


def cmd_compare(args):
    net = _load_net()
    frames, stimuli, truth = load_subjects(DATA_DIR)
    theta_true = truth["theta_true"]

    # Pointwise: per-trial log-densities across all subjects at their true parameters.
    lp_net, lp_exact = [], []
    for s, df in enumerate(frames):
        rate, thr = theta_true[s]
        drift = rate * stimuli[: len(df)]
        c = df["decision"].to_numpy(float).astype(int)
        rt = df["response_time"].to_numpy(float)
        lp_net.append(net.trial_log_density(drift, np.full(len(df), thr), c, rt))
        lp_exact.append(nl.wfpt_logpdf(rt, c, drift, thr))
    lp_net, lp_exact = np.concatenate(lp_net), np.concatenate(lp_exact)
    ok = lp_exact > nl.LOG_ZERO + 1
    print(f"pointwise (n={ok.sum()}): corr(neural, exact) = {np.corrcoef(lp_net[ok], lp_exact[ok])[0,1]:.4f}"
          f"   mean|diff| = {np.abs(lp_net[ok] - lp_exact[ok]).mean():.4f} nats"
          f"   bias = {(lp_net[ok] - lp_exact[ok]).mean():+.4f}")

    # Subject-total LL profiles vs the KDE simulation likelihood for one subject.
    from fit_saved_dataset import load_payloads
    from ddm_subjects import ddm_pec_factory
    from pec_likelihood import PECGroupLikelihood

    s = args.subject
    payloads = load_payloads(f"{DATA_DIR}/dataset.csv")
    pec, inputs = ddm_pec_factory(payloads[s])
    kde = PECGroupLikelihood([{"pec": pec, "inputs": inputs}], transform=None)
    df, (rate_true, thr_true) = frames[s], theta_true[s]
    c = df["decision"].to_numpy(float).astype(int)
    rt = df["response_time"].to_numpy(float)

    def totals(rate, thr):
        drift = rate * stimuli[: len(df)]
        t0 = time.time(); a = nl.wfpt_logpdf(rt, c, drift, thr).sum(); ta = time.time() - t0
        t0 = time.time(); m = net.trial_log_density(drift, np.full(len(df), thr), c, rt).sum(); tm = time.time() - t0
        t0 = time.time(); k = kde.log_likelihood_s([rate, thr], 0); tk = time.time() - t0
        return (a, m, k), (ta, tm, tk)

    for name, grid, fixed in (("rate", np.linspace(*BOUNDS[0], 21), thr_true),
                              ("threshold", np.linspace(*BOUNDS[1], 21), rate_true)):
        rows, times = [], []
        for g in grid:
            vals, ts = totals(g, fixed) if name == "rate" else totals(fixed, g)
            rows.append(vals); times.append(ts)
        rows = np.array(rows)
        true_val = rate_true if name == "rate" else thr_true
        print(f"\nsubject {s} {name} profile (true {true_val:.3f}; other param fixed at truth):")
        for label, col in zip(("exact", "neural", "KDE"), rows.T):
            print(f"  {label:7s} argmax = {grid[np.argmax(col)]:.3f}   LL(true-nearest) = "
                  f"{col[np.argmin(np.abs(grid - true_val))]:.2f}")
        t = np.array(times).mean(axis=0)
        print(f"  per-call seconds: exact {t[0]:.4f}   neural {t[1]:.4f}   KDE {t[2]:.3f}")


def cmd_em(args):
    frames, stimuli, truth = load_subjects(DATA_DIR)
    theta_true = truth["theta_true"]
    n_subjects, n_params = theta_true.shape
    tf = BoundedTransform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

    net = _load_net()
    neural = nl.NeuralGroupLikelihood(net, frames, stimuli)

    def exact_loglik(theta, s):
        df = frames[s]
        drift = float(theta[0]) * stimuli[: len(df)]
        return float(nl.wfpt_logpdf(df["response_time"].to_numpy(float),
                                    df["decision"].to_numpy(float).astype(int),
                                    drift, float(theta[1])).sum())

    results = {}
    for label, ll in (("neural", neural.log_likelihood_s), ("exact", exact_loglik)):
        t0 = time.time()
        r = fit_laplace_em(ll, n_subjects, n_params, tf, max_em_iterations=50, em_tol=1e-4,
                           variance_floor=1e-4, hessian_step=0.05)
        print(f"{label} EM: {time.time() - t0:.1f}s, converged={r.converged} in {r.n_iter} iters")
        results[label] = r.theta_hat
        np.save(f"{DATA_DIR}/pec_theta_{label}_em.npy", r.theta_hat)

    results["KDE-EM"] = np.load(f"{DATA_DIR}/pec_theta_hat.npy")
    results["no-pool MLE"] = np.load(f"{DATA_DIR}/pec_theta_mle.npy")

    print("\nrecovery on the same 40-subject dataset (rate, threshold):")
    for label, th in results.items():
        rmse = np.sqrt(np.mean((th - theta_true) ** 2, axis=0))
        corr = [np.corrcoef(th[:, k], theta_true[:, k])[0, 1] for k in range(n_params)]
        sd = th.std(axis=0)
        print(f"  {label:12s} RMSE = {rmse[0]:.3f}/{rmse[1]:.3f}   corr = {corr[0]:.3f}/{corr[1]:.3f}"
              f"   SD = {sd[0]:.3f}/{sd[1]:.3f}")
    sd_t = theta_true.std(axis=0)
    print(f"  {'truth':12s} SD = {sd_t[0]:.3f}/{sd_t[1]:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--configs", type=int, default=6000)
    t.add_argument("--trials", type=int, default=256)
    t.add_argument("--epochs", type=int, default=25)
    c = sub.add_parser("compare")
    c.add_argument("--subject", type=int, default=0)
    sub.add_parser("em")
    args = p.parse_args()
    {"train": cmd_train, "compare": cmd_compare, "em": cmd_em}[args.cmd](args)
