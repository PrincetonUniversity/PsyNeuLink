"""Launcher for distributed PEC fitting: ``python -m psyneulink.dask_run study.py``.

Run a normal, serial-looking PEC study script across a multi-node Dask cluster
that PsyNeuLink forms from a SLURM allocation -- the user does no Dask
administration (no scheduler to start, no addresses to pass).

Usage (inside an allocation, or via sbatch)::

    srun -n <N> python -m psyneulink.dask_run study.py [study args...]

with ``N = workers + 2``. ``dask_jobqueue.slurm.SLURMRunner`` assigns rank roles
by ``SLURM_PROCID``: rank 0 runs the scheduler, rank 1 runs the driver (this
script's body, i.e. the study), and ranks 2..N-1 run workers. Scheduler/worker
ranks block until the driver finishes, then all ranks exit together.

``study.py`` is an ordinary PEC script: it builds a ``ParameterEstimationComposition``
with ``distributed=True`` (and a ``pec_factory`` in ``distributed_options``) and
calls ``pec.run(...)``. ``distributed=True`` auto-detects the launcher-formed
cluster via the active-launcher client this module registers -- no connection
info appears in the study.

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
    this SLURM job+step under the current working directory (typically shared
    scratch when a study is submitted), so sequential steps never read a stale,
    dead scheduler's address.
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
    # One task slot per worker (nthreads=1); per-eval LLVM threads come from
    # worker_cores. memory_limit=0 disables Dask's memory-based pausing -- the
    # SLURM allocation governs memory.
    worker_options = {"nthreads": 1, "memory_limit": 0}

    # Only the driver rank (SLURM_PROCID 1) executes the body below; the scheduler
    # and worker ranks block inside the runner context until the driver exits.
    with SLURMRunner(scheduler_file=scheduler_file, worker_options=worker_options) as runner:
        with Client(runner) as client:
            _set_active_launcher_client(client)
            # Run study.py as if invoked directly: sys.argv and __name__=="__main__".
            sys.argv = [study_py, *study_args]
            try:
                runpy.run_path(study_py, run_name="__main__")
            finally:
                _set_active_launcher_client(None)
                # Tear down scheduler + workers so every rank exits the step.
                client.shutdown()


if __name__ == "__main__":
    main()
