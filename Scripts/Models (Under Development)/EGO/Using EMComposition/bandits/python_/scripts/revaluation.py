import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src import run

from scripts.defaults import PARAMS, TIME_RETRIEVAL_WEIGHTS, STATE_INTEGRATION_RATES, get_folder
from scripts.utils import project_root, get_seed_from_array_task_id


def run_reval_experiment(*, out_root: Path, **kwargs) -> None:
    _root, _folder = get_folder(**kwargs)

    root = out_root / "results" / "revaluation" / "data" / _root / _folder
    root.mkdir(parents=True, exist_ok=True)

    path = root / f"seed_{kwargs['seed']}.csv"
    if path.exists():
        return

    data = run.run(
        1,
        metric='cosine_similarity',
        model_based_ness=0,
        time_noise=kwargs['time_drift_noise'],
        state_integration_rate=kwargs['state_integration_rate'],
        time_retrieval_weight=kwargs['time_retrieval_weight'],
        ego_softmax_temperature=kwargs['ego_temperature'],
        ego_softmax_threshold=kwargs['ego_threshold'],
        seed=kwargs['seed'],
    )

    res = dict(
        reval_scores_reward=np.array(data['reval_scores_reward']).mean(),
        reval_scores_transition=np.array(data['reval_scores_transition']).mean(),
        er_baseline_1=np.array(data['estimated_reward_state_1_baseline']).mean(),
        er_baseline_2=np.array(data['estimated_reward_state_2_baseline']).mean(),
        er_reward_reval_1=np.array(data['estimated_reward_state_1_reward_reval']).mean(),
        er_reward_reval_2=np.array(data['estimated_reward_state_2_reward_reval']).mean(),
        er_transition_reval_1=np.array(data['estimated_reward_state_1_transition_reval']).mean(),
        er_transition_reval_2=np.array(data['estimated_reward_state_2_transition_reval']).mean(),
    )
    res.update(kwargs)

    reval_data = pd.DataFrame([res])

    reval_data.to_csv(path, index=False)


def main():
    out_root = project_root()
    seed = get_seed_from_array_task_id()

    combinations = itertools.product(STATE_INTEGRATION_RATES, TIME_RETRIEVAL_WEIGHTS)

    for sir, trw in combinations:
        args = PARAMS.copy()
        args.update(
            state_integration_rate=sir,
            time_retrieval_weight=trw,
            seed=seed,
        )
        run_reval_experiment(out_root=out_root, **args)


if __name__ == '__main__':
    main()
