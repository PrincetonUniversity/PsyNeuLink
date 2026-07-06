"""Fit the exported synthetic DDM with HSSM (hierarchical regression DDM) for cross-comparison.

Runs in the isolated HSSM conda env (NOT the PNL venv). Mapping to the PNL/PEC generative model:
    v = v_coef * stimulus   (v_coef per subject == PNL drift rate)
    a = boundary separation == 2 * PNL threshold
    z = 0.5 (PNL starting_value 0, symmetric bounds), t = 0.15 (PNL non_decision_time), both fixed.
Saves the InferenceData and prints group-level + per-subject summaries.
"""

import sys
import numpy as np
import pandas as pd
import hssm
import arviz as az


def build_model(df):
    return hssm.HSSM(
        data=df, model="ddm", loglik_kind="analytical", p_outlier=0.0,
        include=[
            {"name": "v", "formula": "v ~ 0 + stimulus + (0 + stimulus | participant_id)"},
            {"name": "a", "formula": "a ~ 1 + (1 | participant_id)"},
            {"name": "t", "prior": 0.15},
            {"name": "z", "prior": 0.5},
        ],
    )


def load(data_dir, subset=None):
    d = pd.read_csv(f"{data_dir}/dataset.csv")
    df = d.rename(columns={"subj_idx": "participant_id"})[
        ["rt", "response", "participant_id", "stimulus"]
    ].copy()
    df["response"] = np.where(df["response"] == 1, 1, -1)
    if subset is not None:
        df = df[df.participant_id < subset].copy()
    return df


def main(data_dir, draws=1000, tune=1000, chains=4, cores=4, subset=None, sampler=None,
         target_accept=0.9):
    df = load(data_dir, subset=subset)
    print(f"fitting HSSM on {df.participant_id.nunique()} subjects, {len(df)} trials")
    model = build_model(df)

    kwargs = dict(draws=draws, tune=tune, chains=chains, cores=cores,
                  target_accept=target_accept)
    if sampler:
        kwargs["sampler"] = sampler
    idata = model.sample(**kwargs)

    print("=== posterior data_vars ===")
    print(list(idata.posterior.data_vars))
    print("=== summary (group-level terms) ===")
    group_vars = [v for v in idata.posterior.data_vars
                  if ("sigma" in v) or v in ("v_stimulus", "a_Intercept")]
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(az.summary(idata, var_names=group_vars))

    out = f"{data_dir}/hssm_idata{'_sub' if subset else ''}.nc"
    idata.to_netcdf(out)
    print("saved", out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--sampler", default=None)
    p.add_argument("--target_accept", type=float, default=0.9)
    a = p.parse_args()
    main(a.data_dir, a.draws, a.tune, a.chains, a.cores, a.subset, a.sampler, a.target_accept)
