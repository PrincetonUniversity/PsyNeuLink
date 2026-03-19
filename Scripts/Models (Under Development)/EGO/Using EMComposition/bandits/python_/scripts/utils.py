import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_seed_from_array_task_id() -> int:
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if task_id_str is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not set. Run via sbatch --array, or set it manually.")

    return int(task_id_str)
