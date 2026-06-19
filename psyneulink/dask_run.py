"""Launcher for distributed PEC fitting: ``python -m psyneulink.dask_run study.py``.

Run a PEC study script across a Dask cluster formed from a SLURM allocation.

Usage (inside an allocation, or via sbatch)::

    srun -n <N> python -m psyneulink.dask_run study.py [study args...]

with ``N = workers + 2``. ``dask_jobqueue.slurm.SLURMRunner`` assigns rank roles
by ``SLURM_PROCID``: rank 0 runs the scheduler, rank 1 runs the driver, and
ranks 2..N-1 run workers.

``study.py`` builds a ``ParameterEstimationComposition`` with ``distributed=True``
and calls ``pec.run(...)``. This module registers the launcher client for the
driver rank.

The number of workers is the SLURM allocation (``N - 2``); per-worker LLVM threads
default to ``$SLURM_CPUS_PER_TASK``. All Dask imports are lazy so importing
``psyneulink`` never requires Dask.
"""

import os
import runpy
import sys


def _scheduler_file_path():
    """A scheduler file on a shared filesystem that every rank can read.

    Honors ``PSYNEULINK_DASK_SCHEDULER_FILE`` if set; otherwise a name unique to
    this SLURM job+step under the current working directory.
    """
    override = os.environ.get("PSYNEULINK_DASK_SCHEDULER_FILE")
    if override:
        return override
    job = os.environ.get("SLURM_JOB_ID", "local")
    step = os.environ.get("SLURM_STEP_ID", "0")
    return os.path.join(os.getcwd(), f".psyneulink_dask_scheduler_{job}_{step}.json")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(
            "usage: python -m psyneulink.dask_run study.py [study args...]"
        )
    study_py, study_args = argv[0], argv[1:]
    if not os.path.exists(study_py):
        raise SystemExit(f"study script not found: {study_py}")

    try:
        from dask.distributed import Client
        from dask_jobqueue.slurm import SLURMRunner
    except ImportError as e:
        raise SystemExit(
            "python -m psyneulink.dask_run requires Dask and dask-jobqueue. "
            "Install with `pip install psyneulink[dask]`."
        ) from e

    from psyneulink.core.components.functions.nonstateful.fitfunctions import (
        _set_active_launcher_client,
    )

    scheduler_file = _scheduler_file_path()
    # One Dask thread per worker; per-eval LLVM threads come from worker_cores.
    worker_options = {"nthreads": 1, "memory_limit": 0}

    # Only rank 1 executes the study body.
    with SLURMRunner(scheduler_file=scheduler_file, worker_options=worker_options) as runner:
        with Client(runner) as client:
            _set_active_launcher_client(client)
            sys.argv = [study_py, *study_args]
            try:
                runpy.run_path(study_py, run_name="__main__")
            finally:
                _set_active_launcher_client(None)
                client.shutdown()


if __name__ == "__main__":
    main()
