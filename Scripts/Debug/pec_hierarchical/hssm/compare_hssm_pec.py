"""Compare HSSM (full-Bayes) and PEC-EM (empirical-Bayes Laplace) on the same synthetic DDM.

Reads the HSSM InferenceData, the PEC-EM per-subject estimates, and the ground truth, then reports
group-level mean/SD recovery and per-subject agreement. Mapping (ssm-simulators convention,
symmetric +/-a boundaries): HSSM v_coef == rate, a == threshold, z=0.5, t=0.15 fixed.

Run in the HSSM conda env.
"""

import os
import pickle
import sys

import numpy as np
import arviz as az


def main(data_dir):
    idata = az.from_netcdf(f"{data_dir}/hssm_idata.nc")
    post = idata.posterior
    with open(f"{data_dir}/truth.pkl", "rb") as fh:
        truth = pickle.load(fh)
    theta_true = truth["theta_true"]          # (S, 2): rate, threshold
    pec = np.load(f"{data_dir}/pec_theta_hat.npy")  # (S, 2)
    mle_path = f"{data_dir}/pec_theta_mle.npy"
    mle = np.load(mle_path) if os.path.exists(mle_path) else None  # no-pooling baseline
    pooled_path = f"{data_dir}/pec_theta_pooled.npy"
    pooled = np.load(pooled_path) if os.path.exists(pooled_path) else None  # complete pooling

    def pm(name):
        return float(post[name].mean())

    # Group-level estimates.
    hssm_group = {
        "rate_mean": pm("v_stimulus"),
        "rate_sd": pm("v_stimulus|participant_id_sigma"),
        "thr_mean": pm("a_Intercept"),
        "thr_sd": pm("a_1|participant_id_sigma"),
    }

    # Per-subject estimates = group term + random effect (posterior means).
    re_v = post["v_stimulus|participant_id"].mean(dim=("chain", "draw")).values.ravel()
    re_a = post["a_1|participant_id"].mean(dim=("chain", "draw")).values.ravel()
    hssm_rate = hssm_group["rate_mean"] + re_v
    hssm_thr = hssm_group["thr_mean"] + re_a

    true_rate, true_thr = theta_true[:, 0], theta_true[:, 1]
    pec_rate, pec_thr = pec[:, 0], pec[:, 1]

    def line(label, true, pec_v, hssm_v):
        print(f"  {label:18s} true={true:7.3f}   PEC-EM={pec_v:7.3f}   HSSM={hssm_v:7.3f}")

    print("=" * 64)
    print("GROUP-LEVEL RECOVERY (true / PEC-EM / HSSM)")
    print("=" * 64)
    line("rate mean", true_rate.mean(), pec_rate.mean(), hssm_group["rate_mean"])
    line("rate SD", true_rate.std(), pec_rate.std(), hssm_group["rate_sd"])
    line("threshold mean", true_thr.mean(), pec_thr.mean(), hssm_group["thr_mean"])
    line("threshold SD", true_thr.std(), pec_thr.std(), hssm_group["thr_sd"])
    print("  (PEC SD = spread of point estimates; HSSM SD = hierarchical sigma posterior mean)")

    print()
    print("PER-SUBJECT AGREEMENT (correlation)")
    def corr(a, b):
        return np.corrcoef(a, b)[0, 1]
    print(f"  rate:      PEC-vs-true={corr(pec_rate, true_rate):.3f}  "
          f"HSSM-vs-true={corr(hssm_rate, true_rate):.3f}  PEC-vs-HSSM={corr(pec_rate, hssm_rate):.3f}")
    print(f"  threshold: PEC-vs-true={corr(pec_thr, true_thr):.3f}  "
          f"HSSM-vs-true={corr(hssm_thr, true_thr):.3f}  PEC-vs-HSSM={corr(pec_thr, hssm_thr):.3f}")

    print()
    print("PER-SUBJECT RMSE vs true")
    print(f"  rate:      PEC={np.sqrt(np.mean((pec_rate-true_rate)**2)):.3f}  "
          f"HSSM={np.sqrt(np.mean((hssm_rate-true_rate)**2)):.3f}")
    print(f"  threshold: PEC={np.sqrt(np.mean((pec_thr-true_thr)**2)):.3f}  "
          f"HSSM={np.sqrt(np.mean((hssm_thr-true_thr)**2)):.3f}")

    if mle is not None:
        mle_rate, mle_thr = mle[:, 0], mle[:, 1]
        print()
        print("POOLING LADDER: no-pooling MLE -> PEC-EM (partial) -> HSSM (full Bayes)")
        for label, true, cols in (
            ("rate", true_rate, (mle_rate, pec_rate, hssm_rate)),
            ("threshold", true_thr, (mle_thr, pec_thr, hssm_thr)),
        ):
            rmses = [np.sqrt(np.mean((c - true) ** 2)) for c in cols]
            corrs = [corr(c, true) for c in cols]
            sds = [c.std() for c in cols]
            print(f"  {label:9s} RMSE MLE/PEC/HSSM = {rmses[0]:.3f}/{rmses[1]:.3f}/{rmses[2]:.3f}"
                  f"   corr = {corrs[0]:.3f}/{corrs[1]:.3f}/{corrs[2]:.3f}"
                  f"   SD = {sds[0]:.3f}/{sds[1]:.3f}/{sds[2]:.3f} (true {true.std():.3f})")

    if pooled is not None:
        print()
        print("COMPLETE POOLING (one shared MLE on stacked data; no per-subject estimates, SD=0)")
        for k, label in enumerate(("rate", "threshold")):
            true_k = theta_true[:, k]
            print(f"  {label:9s} group mean: true={true_k.mean():.3f}  pooled={pooled[k]:.3f}"
                  f"   (PEC-EM={pec[:, k].mean():.3f}, HSSM={[hssm_group['rate_mean'], hssm_group['thr_mean']][k]:.3f})")
            print(f"            per-subject RMSE if applied to everyone: "
                  f"{np.sqrt(np.mean((pooled[k] - true_k) ** 2)):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/scratch/gpfs/JDC/ap9344/hssm_compare")
