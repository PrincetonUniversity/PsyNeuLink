"""Neural likelihood for the decay-Q bandit + DDM — history-feature conditioning.

The sequential-structure test: the MDN models p(choice_t, rt_t | theta, history features), with
theta-free features (lagged one-hot reward signals) standing in for the running Q state. Training
data comes from the pure-numpy population simulator (no PNL needed), so gen + train + EM run in
minutes. Compared against the exact-EM referee and KDE-EM rows saved by rl_recovery.py.

Subcommands: gen, train, em, all.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core", "nle"):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)

import rl_model as rm
import neural_likelihood as nl
from laplace_em import fit_laplace_em

OUT_DIR = "/scratch/gpfs/JDC/ap9344/rl_hier"
N_LAGS = 10
N_COMP = 8


def simulate_population(thetas, schedule, rng):
    """Vectorized generative runs for many parameter configs sharing one reward schedule."""
    n_cfg = thetas.shape[0]
    n_tr = len(schedule)
    alpha, beta = thetas[:, 0], thetas[:, 1]
    thresh, ndt = thetas[:, 2], thetas[:, 3]
    q = np.zeros((n_cfg, 2))
    choices = np.empty((n_cfg, n_tr), dtype=np.int8)
    rts = np.empty((n_cfg, n_tr), dtype=np.float32)
    rewards = np.empty((n_cfg, n_tr), dtype=np.float32)
    for t in range(n_tr):
        drift = beta * (q[:, 1] - q[:, 0])
        c, fpt, valid = nl.simulate_ddm(drift, thresh, rng)
        c = np.where(valid, c, rng.integers(0, 2, n_cfg))
        fpt = np.where(valid, fpt, 20.0)
        choices[:, t] = c
        rts[:, t] = fpt + ndt
        rewards[:, t] = rng.random(n_cfg) < schedule[t, c]
        signal = np.zeros((n_cfg, 2))
        signal[np.arange(n_cfg), c] = rewards[:, t]
        q = (1.0 - alpha[:, None]) * q + alpha[:, None] * signal
    return choices, rts, rewards


def history_features(choices, rewards, n_lags=N_LAGS):
    """Theta-free per-trial features: one-hot reward signals for the previous n_lags trials.

    choices/rewards: (n_seq, n_trials). Returns (n_seq, n_trials, 2 * n_lags).
    """
    n_seq, n_tr = choices.shape
    sig = np.zeros((n_seq, n_tr, 2), dtype=np.float32)
    idx = np.arange(n_seq)[:, None]
    sig[idx, np.arange(n_tr)[None, :], choices.astype(int)] = rewards
    feats = np.zeros((n_seq, n_tr, 2 * n_lags), dtype=np.float32)
    for k in range(1, n_lags + 1):
        feats[:, k:, 2 * (k - 1):2 * k] = sig[:, :-k]
    return feats


def cmd_gen(configs=4000, seed=1):
    rng = np.random.default_rng(seed)
    lo, hi = rm.fit_bounds()
    thetas = rng.uniform(lo, hi, size=(configs, len(rm.FIT_PARAMS)))
    schedule = rm.reward_schedule(240, np.random.default_rng(11))  # matches the recovery study
    t0 = time.time()
    choices, rts, rewards = simulate_population(thetas, schedule, rng)
    print(f"simulated {configs} configs x {len(schedule)} trials in {time.time() - t0:.1f}s")
    feats = history_features(choices, rewards)
    np.savez(f"{OUT_DIR}/rl_train.npz", theta=thetas, choice=choices, rt=rts, features=feats)
    print("saved", f"{OUT_DIR}/rl_train.npz")


def cmd_train(epochs=25):
    d = np.load(f"{OUT_DIR}/rl_train.npz")
    theta, choice, rt, feats = d["theta"], d["choice"], d["rt"], d["features"]
    n_cfg, n_tr = choice.shape
    cond = np.concatenate([np.repeat(theta, n_tr, axis=0),
                           feats.reshape(n_cfg * n_tr, -1)], axis=1)
    print(f"training rows: {cond.shape[0]}  cond dims: {cond.shape[1]}")
    torch.manual_seed(0)
    net = nl.DDMMixtureNet(n_comp=N_COMP, n_inputs=cond.shape[1])
    t0 = time.time()
    nl.fit(net, cond, choice.ravel().astype(np.int64), rt.ravel().astype(float), epochs=epochs)
    print(f"trained in {time.time() - t0:.1f}s")
    torch.save(net.state_dict(), f"{OUT_DIR}/rl_mdn.pt")
    print("saved", f"{OUT_DIR}/rl_mdn.pt")


def cmd_em():
    with open(f"{OUT_DIR}/rl_result.pkl", "rb") as fh:
        saved = pickle.load(fh)
    subs, theta_true = saved["subjects"], saved["theta_true"]
    n_subjects = len(subs)
    n_params = len(rm.FIT_PARAMS)
    tf = rm.transform()

    net = nl.DDMMixtureNet(n_comp=N_COMP, n_inputs=n_params + 2 * N_LAGS)
    net.load_state_dict(torch.load(f"{OUT_DIR}/rl_mdn.pt", weights_only=True))
    net.eval()

    subj_feats = [history_features(s["choices"][None, :].astype(np.int8),
                                   s["rewards"][None, :].astype(np.float32))[0] for s in subs]

    def neural_loglik(theta, s):
        n = len(subs[s]["choices"])
        cond = np.concatenate([np.tile(np.asarray(theta, float), (n, 1)), subj_feats[s]], axis=1)
        return float(net.cond_log_density(cond, subs[s]["choices"].astype(int),
                                          subs[s]["rts"]).sum())

    t0 = time.time()
    r = fit_laplace_em(neural_loglik, n_subjects, n_params, tf, max_em_iterations=50,
                       em_tol=1e-4, variance_floor=1e-4, hessian_step=0.05)
    print(f"neural EM: {time.time() - t0:.1f}s, converged={r.converged} in {r.n_iter} iters")

    results = dict(saved["results"])
    results["neural-EM"] = r.theta_hat
    with open(f"{OUT_DIR}/rl_result.pkl", "wb") as fh:
        saved["results"] = results
        pickle.dump(saved, fh)

    print(f"\nrecovery ({', '.join(rm.FIT_PARAMS)}):")
    for label, th in results.items():
        rmse = np.sqrt(np.mean((th - theta_true) ** 2, axis=0))
        corr = [np.corrcoef(th[:, k], theta_true[:, k])[0, 1] for k in range(n_params)]
        print(f"  {label:14s} RMSE = " + "/".join(f"{v:.3f}" for v in rmse)
              + "   corr = " + "/".join(f"{v:.3f}" for v in corr))
    print("  truth SD       = " + "/".join(f"{v:.3f}" for v in theta_true.std(axis=0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["gen", "train", "em", "all"])
    a = p.parse_args()
    if a.cmd in ("gen", "all"):
        cmd_gen()
    if a.cmd in ("train", "all"):
        cmd_train()
    if a.cmd in ("em", "all"):
        cmd_em()