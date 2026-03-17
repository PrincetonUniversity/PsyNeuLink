from __future__ import annotations

import itertools
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.run_probabilistic import (
    run_human_choices,
    run_model_choices,
    run_random_choices,
)
from scripts.defaults import STATE_INTEGRATION_RATES, TIME_RETRIEVAL_WEIGHTS, PARAMS
from scripts.utils import project_root

# ============================================================
# Config
# ============================================================

RUN_TYPE = "model"  # "model", "random", "human"

MODEL_BASED_NESS_LIST = [0.0]
CHOICE_BIAS_LIST = [PARAMS.choice_bias]
CHOICE_TEMPERATURE_LIST = [PARAMS.choice_temperature]

METRICS = ["cosine_similarity"]
TIME_DRIFT_NOISE_LIST = [PARAMS.time_drift_noise]
N_BASE_TRIALS_LIST = [200]

OUT_DIR = project_root() / Path("results/two_step/stay_probability")


# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rf(x: float, nd: int = 12) -> float:
    return float(np.round(float(x), nd))


def job_uid() -> str:
    return f"{uuid.uuid4().hex}"


def compute_stay_stats_from_prev(trial_log) -> np.ndarray:
    rc, rr, uc, ur = [], [], [], []

    for tr in trial_log:
        stay = tr.get("stay", None)
        if stay is None:
            continue

        reward = tr["prev_reward"]
        transition = tr["prev_transition"]

        if reward == 1 and transition == "common":
            rc.append(stay)
        elif reward == 1 and transition == "rare":
            rr.append(stay)
        elif reward == 0 and transition == "common":
            uc.append(stay)
        elif reward == 0 and transition == "rare":
            ur.append(stay)

    return np.asarray(
        [
            np.mean(rc) if rc else np.nan,
            np.mean(rr) if rr else np.nan,
            np.mean(uc) if uc else np.nan,
            np.mean(ur) if ur else np.nan,
        ],
        dtype=float,
    )


# ============================================================
# Simulation (one participant)
# ============================================================

def simulate_one_participant(
        *,
        run_type: str,
        metric: str,
        n_base_trials: int,
        ir: float,
        tw: float,
        mb: float,
        choice_bias: float,
        choice_temperature: float,
        time_drift_noise: float,
        common_prob: float = 0.7,
) -> np.ndarray:
    if run_type == "random":
        trial_log = run_random_choices(
            state_integration_rate=ir,
            time_retrieval_weight=tw,
            model_based_ness=mb,
            metric=metric,
            n_base_trials=n_base_trials,
            common_prob=common_prob,
        )
        return compute_stay_stats_from_prev(trial_log)

    if run_type == "model":
        trial_log = run_model_choices(
            state_integration_rate=ir,
            time_retrieval_weight=tw,
            model_based_ness=mb,
            metric=metric,
            n_base_trials=n_base_trials,
            common_prob=common_prob,
            choice_bias=choice_bias,
            choice_temperature=choice_temperature,
            time_drift_noise=time_drift_noise,
        )
        return compute_stay_stats_from_prev(trial_log)

    if run_type == "human":
        human_data = pd.read_csv("./data/twostep.csv")
        ids = pd.unique(human_data["sub"]).tolist()
        _id = random.choice(ids)
        subj_data = human_data[human_data["sub"] == _id]
        trial_log = run_human_choices(
            state_integration_rate=ir,
            time_retrieval_weight=tw,
            model_based_ness=mb,
            metric=metric,
            subj_data=subj_data,
        )
        return compute_stay_stats_from_prev(trial_log)

    raise ValueError(f"Invalid RUN_TYPE: {run_type!r}")


# ============================================================
# Condition enumeration
# ============================================================

@dataclass(frozen=True)
class Cond:
    run_type: str
    metric: str
    time_drift_noise: float
    n_base_trials: int
    choice_bias: float
    choice_temperature: float
    mb: float
    ir: float
    tw: float


def all_conditions() -> List[Cond]:
    out: List[Cond] = []
    for metric, noise, nbt, cb, ct, mb, ir, tw in itertools.product(
            METRICS,
            TIME_DRIFT_NOISE_LIST,
            N_BASE_TRIALS_LIST,
            CHOICE_BIAS_LIST,
            CHOICE_TEMPERATURE_LIST,
            MODEL_BASED_NESS_LIST,
            STATE_INTEGRATION_RATES,
            TIME_RETRIEVAL_WEIGHTS,
    ):
        out.append(
            Cond(
                run_type=str(RUN_TYPE),
                metric=str(metric),
                time_drift_noise=rf(noise),
                n_base_trials=int(nbt),
                choice_bias=rf(cb),
                choice_temperature=rf(ct),
                mb=rf(mb),
                ir=rf(ir),
                tw=rf(tw),
            )
        )
    return out


def output_path(run_type) -> Path:
    ensure_dir(OUT_DIR)
    return OUT_DIR / run_type / f"{job_uid()}.csv"


def main() -> None:
    conditions = all_conditions()

    rows = []
    for c in todo:
        rc, rr, uc, ur = simulate_one_participant(
            run_type=c.run_type,
            metric=c.metric,
            n_base_trials=c.n_base_trials,
            ir=c.ir,
            tw=c.tw,
            mb=c.mb,
            choice_bias=c.choice_bias,
            choice_temperature=c.choice_temperature,
            time_drift_noise=c.time_drift_noise,
        )
        rows.append(
            dict(
                run_type=c.run_type,
                metric=c.metric,
                time_drift_noise=c.time_drift_noise,
                n_base_trials=c.n_base_trials,
                choice_bias=c.choice_bias,
                choice_temperature=c.choice_temperature,
                mb=c.mb,
                ir=c.ir,
                tw=c.tw,
                RC=float(rc),
                RR=float(rr),
                UC=float(uc),
                UR=float(ur),
            )
        )
    df = pd.DataFrame(rows)
    out = output_path(RUN_TYPE)
    df.to_csv(out)


if __name__ == "__main__":
    main()
