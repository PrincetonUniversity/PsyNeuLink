"""What happens to hierarchical fitting when its assumption about the group is wrong?

Fitting many participants together assumes they are all drawn from one bell-shaped
distribution, and each participant's estimate is pulled toward the group average by an
amount based on that assumption. Recovery studies normally generate participants from a
bell curve too, which is the case this method handles best. This study generates them from
other population shapes instead and measures the damage.

Population shapes (all given the SAME total spread, so only the shape differs):
  matched       one bell curve                    <- what the fitter assumes
  bimodal       two separate subgroups
  heavy         bell curve with heavy tails       <- produces outliers
  contaminated  mostly one bell curve, 10% much more spread out
  uniform       flat, no central tendency

Methods compared:
  nopool        each participant fitted alone, no borrowing between them
  hier_k1       standard version: one bell curve shared by everyone
  hier_k2       models the group as two bell curves and tries to find the subgroups
                without being told which participant belongs to which

Average error is reported, but the more informative measure is error split by how unusual
each participant is, because an average hides damage done to the unusual ones.

FINDINGS (8 datasets, 24 participants, 30 trials each; see summary.csv):

  - Pooling only does real work when each participant's own data is weak. At 250 trials
    per participant the estimates barely moved (spread preserved to within 2%) and the
    population shape made no measurable difference. At 30 trials the pull toward the group
    average is real and the comparison becomes meaningful. Results below are the 30-trial
    case.

  - Getting the shape wrong does not break the method on average. Fitting participants
    together beat fitting them separately by 11-13% for every shape tested, including the
    ones that violate the assumption most.

  - The cost lands on the unusual participants instead. With heavy tails or a contaminated
    population, participants far from the group average had ~26% more error than typical
    ones, against ~5% when the assumption held. Average error hides this completely: the
    heavy-tailed case had the LOWEST average error of any shape while carrying the worst
    penalty for its outliers. Even so, those participants were still better off than being
    fitted alone.

  - Automatic subgroup discovery works, provided the subgroups are far enough apart to be
    distinguishable. How far apart they are matters more than anything else:

      population        gap between subgroups   detected?   error vs single-group fit
      matched           none                    no (right)  unchanged
      bimodal           2.7 within-group spreads  no        3% better
      bimodal_wide      6.1 within-group spreads  YES       24% better

    With clearly separated subgroups the two-component version found them without being
    told they existed, model selection correctly asked for two groups, and error dropped
    24%. On a population with no subgroups it correctly reported almost no gap and cost
    nothing. It only failed on subgroups overlapping so heavily that calling them separate
    groups is questionable in the first place.

    An earlier version of this note said subgroup discovery did not work. That conclusion
    came from testing only the heavily-overlapping case and was wrong.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _d in ("core", "nle"):
    p = os.path.join(_PARENT, _d)
    if p not in sys.path:
        sys.path.append(p)

import neural_likelihood as nl
from transforms import BoundedTransform
from hessian import diagonal_hessian

OUT_DIR = "/scratch/gpfs/JDC/ap9344/ddm_prior_misspec"
STIMULI = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
RATE_RANGE, THRESH_RANGE = (-1.5, 1.5), (0.28, 1.55)
GROUP_MEAN = np.array([0.0, 0.9])
BASE_SD = np.array([0.6, 0.6])          # z-space between-subject SD (total, all shapes)
SHAPES = ("matched", "bimodal", "bimodal_wide", "heavy", "contaminated", "uniform")
METHODS = ("nopool", "hier_k1", "hier_k2")
NM_OPTS = {"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400}


def fit_box():
    return (np.array([RATE_RANGE[0], THRESH_RANGE[0]]),
            np.array([RATE_RANGE[1], THRESH_RANGE[1]]))


# ---------------- truth generators (all variance-matched) ----------------
def draw_subjects(shape, n, beta, sd, rng):
    k = len(beta)
    if shape == "matched":
        return rng.normal(beta, sd, size=(n, k))
    if shape == "bimodal":
        # centres 2*0.8/0.6 = 2.7 within-group spreads apart: subgroups that genuinely
        # overlap, which is the hard case for telling them apart.
        d, w = 0.8 * sd, 0.6 * sd            # d^2 + w^2 = sd^2 keeps total spread fixed
        lab = rng.integers(0, 2, size=n)
        centers = np.where(lab[:, None] == 0, beta - d, beta + d)
        return rng.normal(centers, w)
    if shape == "bimodal_wide":
        # same total spread, but centres 6.1 within-group spreads apart: cleanly separated
        # subgroups. If automatic discovery cannot work here it cannot work anywhere.
        d = 0.95 * sd
        w = np.sqrt(np.maximum(sd ** 2 - d ** 2, 1e-12))
        lab = rng.integers(0, 2, size=n)
        centers = np.where(lab[:, None] == 0, beta - d, beta + d)
        return rng.normal(centers, w)
    if shape == "heavy":
        return beta + (sd / np.sqrt(3.0)) * rng.standard_t(3, size=(n, k))
    if shape == "contaminated":
        a = sd / np.sqrt(1.8)                # 0.9a^2 + 0.1(3a)^2 = 1.8a^2
        scale = np.where(rng.random((n, 1)) < 0.1, 3.0 * a, a)
        return rng.normal(beta, scale)
    if shape == "uniform":
        h = sd * np.sqrt(3.0)
        return rng.uniform(beta - h, beta + h, size=(n, k))
    raise ValueError(shape)


def generate_data(theta_true, stim, rng, dt=0.01):
    n_s, n_t = theta_true.shape[0], len(stim)
    choice = np.empty((n_s, n_t), dtype=np.int8)
    rt = np.empty((n_s, n_t))
    for s in range(n_s):
        drift = theta_true[s, 0] * stim
        thr = np.full(n_t, theta_true[s, 1])
        c, f, ok = nl.simulate_ddm(drift, thr, rng, dt=dt)
        for _ in range(50):
            if ok.all():
                break
            i = np.flatnonzero(~ok)
            c[i], f[i], ok[i] = nl.simulate_ddm(drift[i], thr[i], rng, dt=dt)
        good = np.isfinite(f)
        f = np.where(good, f, np.median(f[good]))
        choice[s], rt[s] = c, f + nl.NDT
    return choice, rt


# ---------------- E-step: per-subject MAP under an arbitrary log-prior ----------------
def subject_map(loglik_s, log_prior, z0, n_params):
    def neg(z):
        return -loglik_s(z) - log_prior(z)
    res = minimize(neg, z0, method="Nelder-Mead", options=NM_OPTS)
    z_hat = res.x
    curv = diagonal_hessian(neg, z_hat, step=0.05, f0=float(res.fun))
    with np.errstate(divide="ignore"):
        var = np.where(curv > 0, 1.0 / np.where(curv > 0, curv, 1.0), 25.0)
    return z_hat, np.clip(var, 1e-6, 25.0), float(res.fun)


def _log_gauss(z, mu, var):
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + (z - mu) ** 2 / var))


# ---------------- hierarchical EM with a K-component mixture prior ----------------
def fit_hier(loglik_factory, n_subjects, n_params, K, max_iter=25, seed=0, tol=1e-4):
    """K=1 is standard hierarchical EM. K>1 discovers subgroups automatically."""
    rng = np.random.default_rng(seed)
    mu = np.tile(np.zeros(n_params), (K, 1))
    if K > 1:                                    # break symmetry
        mu += rng.normal(0, 0.3, size=(K, n_params))
    var = np.ones((K, n_params)) * 1.0
    pi = np.full(K, 1.0 / K)
    z = np.zeros((n_subjects, n_params))
    V = np.ones((n_subjects, n_params))
    prev = None

    for it in range(max_iter):
        # ---- E-step: each subject's MAP under the CURRENT mixture prior ----
        def make_prior():
            def lp(zz):
                return logsumexp([np.log(pi[k] + 1e-300) + _log_gauss(zz, mu[k], var[k])
                                  for k in range(K)])
            return lp
        prior = make_prior()
        obj = 0.0
        for s in range(n_subjects):
            z[s], V[s], nlp = subject_map(loglik_factory(s), prior, z[s], n_params)
            obj += -nlp + 0.5 * np.sum(np.log(V[s]))
        # ---- M-step: responsibilities, then variance-corrected weighted moments ----
        logr = np.array([[np.log(pi[k] + 1e-300) + _log_gauss(z[s], mu[k], var[k])
                          for k in range(K)] for s in range(n_subjects)])
        logr -= logsumexp(logr, axis=1, keepdims=True)
        r = np.exp(logr)
        Nk = r.sum(axis=0) + 1e-12
        pi = Nk / n_subjects
        for k in range(K):
            w = r[:, k:k + 1]
            mu[k] = (w * z).sum(axis=0) / Nk[k]
            var[k] = np.maximum(((w * ((z - mu[k]) ** 2 + V)).sum(axis=0)) / Nk[k], 1e-4)
        if prev is not None and abs(obj - prev) < tol * max(1.0, abs(prev)):
            break
        prev = obj
    n_free = K * (2 * n_params) + (K - 1)
    bic = -2 * obj + n_free * np.log(n_subjects)
    return dict(z=z.copy(), var=V.copy(), mu=mu.copy(), comp_var=var.copy(),
                pi=pi.copy(), resp=r.copy(), objective=obj, bic=bic, n_iter=it + 1)


def fit_nopool(loglik_factory, n_subjects, n_params):
    z = np.zeros((n_subjects, n_params))
    V = np.ones((n_subjects, n_params))
    flat = lambda zz: _log_gauss(zz, np.zeros(n_params), np.full(n_params, 25.0))
    for s in range(n_subjects):
        z[s], V[s], _ = subject_map(loglik_factory(s), flat, z[s], n_params)
    return dict(z=z, var=V)


# ---------------- driver ----------------
def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    lo, hi = fit_box()
    tf = BoundedTransform(lo, hi)
    beta_z = tf.to_unconstrained(GROUP_MEAN)
    rng0 = np.random.default_rng(args.seed)
    stim = rng0.choice(STIMULI, size=args.num_trials)

    rows, subj_rows, clusters = [], [], []
    for shape in args.shapes:
        for d in range(args.datasets):
            trng = np.random.default_rng(hash((shape, d, args.seed)) % (2**31))
            z_true = draw_subjects(shape, args.n_subjects, beta_z, BASE_SD, trng)
            theta_true = np.array([tf.to_natural(z_true[s]) for s in range(args.n_subjects)])
            choice, rt = generate_data(theta_true, stim, trng)

            def factory(s):
                def ll(z):
                    th = tf.to_natural(z)
                    return float(nl.wfpt_logpdf(rt[s], choice[s], th[0] * stim, th[1]).sum())
                return ll

            for method in args.methods:
                t0 = time.time()
                if method == "nopool":
                    res = fit_nopool(factory, args.n_subjects, 2)
                else:
                    K = int(method.split("k")[1])
                    res = fit_hier(factory, args.n_subjects, 2, K,
                                   max_iter=args.em_iters, seed=d)
                th = np.array([tf.to_natural(res["z"][s]) for s in range(args.n_subjects)])
                err = np.sqrt(np.mean((th - theta_true) ** 2, axis=1))
                rmse = float(np.sqrt(np.mean((th - theta_true) ** 2)))
                sd_ratio = float(np.mean(th.std(axis=0) / theta_true.std(axis=0)))
                # atypicality = distance of the subject's TRUE z from the overall mean
                atyp = np.linalg.norm((z_true - z_true.mean(axis=0)) / BASE_SD, axis=1)
                rows.append(dict(shape=shape, dataset=d, method=method, rmse=rmse,
                                 sd_ratio=sd_ratio, seconds=time.time() - t0,
                                 bic=res.get("bic"), n_iter=res.get("n_iter")))
                for s in range(args.n_subjects):
                    subj_rows.append(dict(shape=shape, dataset=d, method=method,
                                          subject=s, err=float(err[s]),
                                          atypicality=float(atyp[s])))
                if method == "hier_k2":
                    clusters.append(dict(shape=shape, dataset=d,
                                         pi=res["pi"].tolist(),
                                         mu=res["mu"].tolist(),
                                         separation=float(np.linalg.norm(res["mu"][0] - res["mu"][1]))))
            print(f"  {shape:13s} ds{d} done ({time.time()-t0:.0f}s last fit)", flush=True)

    json.dump(dict(rows=rows, subjects=subj_rows, clusters=clusters),
              open(f"{OUT_DIR}/results.json", "w"), indent=1)
    summarize(rows, subj_rows, clusters, args)


def summarize(rows, subj_rows, clusters, args):
    import pandas as pd
    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(subj_rows)
    df.to_csv(f"{OUT_DIR}/summary.csv", index=False)
    sdf.to_csv(f"{OUT_DIR}/subjects.csv", index=False)

    print("\n" + "=" * 78)
    print("MEAN RMSE by truth shape x method   (lower = better)")
    print("=" * 78)
    piv = df.pivot_table(index="shape", columns="method", values="rmse", aggfunc="mean")
    piv = piv.reindex(index=[s for s in SHAPES if s in piv.index],
                      columns=[m for m in METHODS if m in piv.columns])
    print(piv.round(4).to_string())

    print("\nsd_ratio (estimated spread / true spread; 1.0 = faithful, <1 = over-shrunk)")
    piv2 = df.pivot_table(index="shape", columns="method", values="sd_ratio", aggfunc="mean")
    piv2 = piv2.reindex(index=[s for s in SHAPES if s in piv2.index],
                        columns=[m for m in METHODS if m in piv2.columns])
    print(piv2.round(3).to_string())

    print("\nWHO PAYS: mean error for TYPICAL vs ATYPICAL subjects (atypicality split at median)")
    med = sdf.atypicality.median()
    sdf["grp"] = np.where(sdf.atypicality > med, "atypical", "typical")
    piv3 = sdf.pivot_table(index=["shape", "grp"], columns="method", values="err", aggfunc="mean")
    print(piv3.round(4).to_string())

    if clusters:
        cdf = pd.DataFrame(clusters)
        print("\nMIXTURE PRIOR: discovered cluster separation (z-space) by truth shape")
        print(cdf.groupby("shape")["separation"].agg(["mean", "std"]).round(3).to_string())
        print("(bimodal truth has components at +/-0.8*0.6 => true separation ~0.96)")

    b = df[df.method.isin(["hier_k1", "hier_k2"])]
    if len(b) and b.bic.notna().any():
        print("\nBIC (lower = preferred): does the data ASK for two clusters?")
        pb = b.pivot_table(index="shape", columns="method", values="bic", aggfunc="mean")
        pb["prefers"] = np.where(pb.get("hier_k2", np.inf) < pb.get("hier_k1", np.inf),
                                 "K=2", "K=1")
        print(pb.round(1).to_string())
    print(f"\nwrote {OUT_DIR}/summary.csv, subjects.csv, results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", nargs="+", default=list(SHAPES))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--datasets", type=int, default=6)
    p.add_argument("--n_subjects", type=int, default=24)
    p.add_argument("--num_trials", type=int, default=250)
    p.add_argument("--em_iters", type=int, default=25)
    p.add_argument("--seed", type=int, default=777)
    run(p.parse_args())
