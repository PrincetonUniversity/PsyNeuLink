import sys

import numpy as np
import pandas as pd

from src import run
from scripts.utils import project_root, get_seed_from_array_task_id

SIR = .5
TR = .3
EGO_TEMPERATURE = 1.
EGO_THRESHOLD = .01

SIMULATIONS_REVAL = 1000

REWARD_TARGET = .5199
TRANSITION_TARGET = .4503

NOISE_RANGE = (0.03, 0.05)
N = 31

NOISE_LIST = np.linspace(NOISE_RANGE[0], NOISE_RANGE[1], N)


def run_reval_once(time_drift_noise) -> tuple[float, float]:
    reward_scores = []
    transition_scores = []

    for s in range(SIMULATIONS_REVAL):
        data = run.run(
            1,
            metric="cosine_similarity",
            model_based_ness=0,
            time_noise=time_drift_noise,
            state_integration_rate=SIR,
            time_retrieval_weight=TR,
            ego_softmax_temperature=EGO_TEMPERATURE,
            ego_softmax_threshold=EGO_THRESHOLD,
            seed=s
        )

        reward_scores.append(np.array(data["reval_scores_reward"]).mean())
        transition_scores.append(np.array(data["reval_scores_transition"]).mean())

    return float(np.mean(reward_scores))/18., float(np.mean(transition_scores))/18.


def main():
    if len(sys.argv) < 2:
        raise ValueError("Noise value must be provided")
    noise_index = int(sys.argv[1])

    out_root = project_root()
    seed = get_seed_from_array_task_id()

    root = out_root / "results" / "revaluation" / (
        f"fit_time_noise_"
        f"sir_{SIR:.3f}_"
        f"tw_{TR:.3f}_"
        f"ego_temp_{EGO_TEMPERATURE:.3f}_"
        f"ego_threshold_{EGO_THRESHOLD:.3f}"
    )

    root.mkdir(parents=True, exist_ok=True)

    noise = NOISE_LIST[noise_index]

    out_file = root / f"noise_{noise:.5f}_seed_{seed}.csv"

    # skip already computed runs
    if out_file.exists():
        print("Skipping existing:", out_file)
        return
    r, t = run_reval_once(noise)

    mse = (r - REWARD_TARGET) ** 2 + (t - TRANSITION_TARGET) ** 2

    df = pd.DataFrame([{
        "time_drift_noise": noise,
        "seed": seed,
        "reval_reward_mean": r,
        "reval_transition_mean": t,
        "target_reward": REWARD_TARGET,
        "target_transition": TRANSITION_TARGET,
        "mse": mse,
        "state_integration_rate": SIR,
        "time_retrieval_weight": TR,
        "ego_temperature": EGO_TEMPERATURE,
        "ego_threshold": EGO_THRESHOLD
    }])

    df.to_csv(out_file, index=False)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
