from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src import run  # your revaluation runner
from scripts.defaults import STATE_INTEGRATION_RATES, TIME_RETRIEVAL_WEIGHTS, PARAMS
from scripts.utils import project_root

DOT_PRODUCT = "dot_product"
COSINE_SIMILARITY = "cosine_similarity"

# ============================================================
# Config
# ============================================================

METRICS = ["cosine_similarity"]
MODEL_BASED_NESS_LIST = [0.0]
N_TRIALS_LIST = [200]

# how many independent runs per condition *within this job*
RUNS_PER_CONDITION = 1  # bump to 5/10 if you want fewer Slurm jobs

OUT_DIR = project_root() / Path("results/revaluation")

# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rf(x: float, nd: int = 12) -> float:
    return float(np.round(float(x), nd))


def job_uid() -> str:
    return uuid.uuid4().hex


def as_1d_float_array(x) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


def summarize_reval(rr: np.ndarray, rt: np.ndarray) -> Tuple[float, float, int, int]:
    rr = as_1d_float_array(rr)
    rt = as_1d_float_array(rt)

    rr_n = int(np.isfinite(rr).sum())
    rt_n = int(np.isfinite(rt).sum())

    rr_mean = float(np.nanmean(rr)) if rr_n > 0 else float("nan")
    rt_mean = float(np.nanmean(rt)) if rt_n > 0 else float("nan")

    return rr_mean, rt_mean, rr_n, rt_n


# ============================================================
# Condition enumeration
# ============================================================

@dataclass(frozen=True)
class Cond:
    metric: str
    mb: float
    n_trials: int
    ir: float
    tw: float


def all_conditions() -> List[Cond]:
    out: List[Cond] = []
    for metric, mb, n_trials, ir, tw in itertools.product(
        METRICS,
        MODEL_BASED_NESS_LIST,
        N_TRIALS_LIST,
        STATE_INTEGRATION_RATES,
        TIME_RETRIEVAL_WEIGHTS,
    ):
        out.append(
            Cond(
                metric=str(metric),
                mb=rf(mb),
                n_trials=int(n_trials),
                ir=rf(ir),
                tw=rf(tw),
            )
        )
    return out


def output_path() -> Path:
    ensure_dir(OUT_DIR)
    # one CSV per job
    return OUT_DIR / f"{job_uid()}.csv"


# ============================================================
# Simulation (one run)
# ============================================================

def run_one_revaluation(
    *,
    n_trials: int,
    metric: str,
    mb: float,
    ir: float,
    tw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    data = run.run(
        n_trials,
        metric=metric,
        model_based_ness=mb,
        state_integration_rate=ir,
        time_retrieval_weight=tw,
    )
    rr = data.get("reval_scores_reward", None)
    rt = data.get("reval_scores_transition", None)
    return as_1d_float_array(rr), as_1d_float_array(rt)


# ============================================================
# Main
# ============================================================

def main() -> None:
    conditions = all_conditions()

    rows = []
    for c in conditions:
        for rep in range(int(RUNS_PER_CONDITION)):
            rr, rt = run_one_revaluation(
                n_trials=c.n_trials,
                metric=c.metric,
                mb=c.mb,
                ir=c.ir,
                tw=c.tw,
            )
            rr_mean, rt_mean, rr_n, rt_n = summarize_reval(rr, rt)

            rows.append(
                dict(
                    metric=c.metric,
                    mb=c.mb,
                    n_trials=c.n_trials,
                    ir=c.ir,
                    tw=c.tw,
                    rep_in_job=int(rep),

                    reward_reval_mean=float(rr_mean),
                    transition_reval_mean=float(rt_mean),

                    # optional sanity check columns (keep or delete)
                    reward_samples_n=int(rr_n),
                    trans_samples_n=int(rt_n),
                )
            )

    df = pd.DataFrame(rows)
    out = output_path()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()