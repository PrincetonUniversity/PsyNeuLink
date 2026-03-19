import itertools
from tqdm import tqdm

import pandas as pd
from pathlib import Path

from src.run_probabilistic import run_model_choices
from scripts.defaults import PARAMS, TIME_RETRIEVAL_WEIGHTS, STATE_INTEGRATION_RATES, get_folder
from scripts.utils import project_root, get_seed_from_array_task_id


def run_model_choices_with_store(*, out_root: Path, **kwargs) -> None:
    _root, _folder = get_folder(**kwargs)

    root = out_root / "results" / "two_step" / "data" / _root / _folder
    root.mkdir(parents=True, exist_ok=True)

    path = root / f"seed_{kwargs['seed']}.csv"
    if path.exists():
        return

    trial_log = run_model_choices(**kwargs)
    pd.DataFrame(trial_log).to_csv(path, index=False)


def main() -> None:
    out_root = project_root()

    seed = get_seed_from_array_task_id()

    combinations = itertools.product(STATE_INTEGRATION_RATES, TIME_RETRIEVAL_WEIGHTS)

    for sir, trw in tqdm(combinations):
        args = PARAMS.copy()
        args.update(
            state_integration_rate=sir,
            time_retrieval_weight=trw,
            seed=seed,
        )
        run_model_choices_with_store(out_root=out_root, **args)


if __name__ == '__main__':
    main()
