"""Export the well-conditioned synthetic DDM dataset for an HSSM cross-check.

Regenerates the exact data the PEC-EM recovery used (same seed/config) and writes it, in HSSM
format (rt, response, subj_idx, stimulus), to a shared-filesystem directory so the HSSM env can
read it. Run in the PNL venv on a compute node (writes must land on /scratch, not node-local /tmp).
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

from ddm_subjects import generate_group_data, FIT_PARAMS


def main(out_dir, n_subjects=40, num_trials=100, seed=11):
    rng = np.random.default_rng(seed)
    beta_z = np.array([0.0, 0.0])
    sigma_z = np.array([0.36, 0.36])
    group = generate_group_data(n_subjects, beta_z, sigma_z, num_trials, rng, num_estimates=1)

    inputs = group["trial_inputs"].ravel()
    rows = []
    for s, payload in enumerate(group["payloads"]):
        df = payload["data"]
        rows.append(pd.DataFrame({
            "subj_idx": s,
            "decision": df["decision"].astype(float).values,
            "rt": df["response_time"].astype(float).values,
            "stimulus": inputs[: len(df)],
        }))
    data = pd.concat(rows, ignore_index=True)

    decisions = sorted(data["decision"].unique())
    # Upper boundary -> response 1, lower -> 0.
    data["response"] = (data["decision"] == decisions[-1]).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    data.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    with open(os.path.join(out_dir, "truth.pkl"), "wb") as fh:
        pickle.dump({
            "theta_true": group["theta_true"], "z_true": group["z_true"],
            "beta_z": beta_z, "sigma_z": sigma_z, "fit_params": list(FIT_PARAMS),
            "fixed_non_decision_time": 0.15, "noise": 1.0,
            "mapping": "HSSM v = v_coef*stimulus (v_coef=rate); a = 2*threshold; t=0.15; z=0.5",
        }, fh)

    print("decision unique values:", decisions, "-> response coding {0,1}")
    print("rows:", len(data), "| subjects:", data.subj_idx.nunique())
    print("stimulus levels:", sorted(data.stimulus.unique()))
    print("rt: min=%.3f max=%.3f mean=%.3f" % (data.rt.min(), data.rt.max(), data.rt.mean()))
    print("response mean (overall):", round(data.response.mean(), 3))
    print("theta_true mean/SD:", group["theta_true"].mean(0).round(3), group["theta_true"].std(0).round(3))
    print("saved to", out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/scratch/gpfs/JDC/ap9344/hssm_compare")
