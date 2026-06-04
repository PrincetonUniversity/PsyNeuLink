"""Multi-node entrypoint (Option A): run the Dask PEC MLE fit via dask-jobqueue.

Each Dask worker is its own SLURM job (one worker = one job), so workers trickle
in as nodes free up -- well suited to a busy partition where reserving N whole
nodes at once would queue for a long time.

The DRIVER hosts the Dask scheduler: lightweight (it only coordinates; the PEC
simulations run on the workers) but long-lived. So run it inside a SMALL
allocation, NOT on the della login node (the "A2" pattern):

    # tiny driver allocation -- just the scheduler + ask/tell loop. Use >61 min
    # so it (and the workers) land in the `short` QOS, and longer than
    # config.WORKER_WALLTIME so it outlasts the workers:
    salloc --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --time=02:30:00 --partition=cpu
    .venv/bin/python3 Scripts/Debug/pec_hierarchical/run_dask_mle_jobqueue.py

dask-jobqueue submits the worker jobs itself from inside that allocation. The
driver allocation's walltime must OUTLAST the worker walltime (config.WORKER_
WALLTIME). run_fit() is identical to the single-node path -- only the client
construction differs.
"""

import os
import sys

# Make `pec_dask_mle` importable in this driver process. Remote workers get it
# via the PYTHONPATH export in job_script_prologue below.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from pec_dask_mle import config
from pec_dask_mle.data import make_data
from pec_dask_mle.driver import run_fit, summarize


def build_cluster():
    """One SLURM job per Dask worker, each reserving WORKER_CORES cpus.

    The key trick to avoid oversubscription:

      * ``cores=1``  -> Dask gives the worker nthreads=1, so it runs exactly ONE
        evaluation at a time (never two tasks fighting for the same cores).
      * ``job_cpu=WORKER_CORES`` -> but SLURM still reserves WORKER_CORES cpus,
        which evaluate_loglik() hands to LLVM via set_num_threads(WORKER_CORES).

    Net per worker: 1 evaluation x WORKER_CORES LLVM threads = WORKER_CORES cores,
    fully used, none oversubscribed.
    """
    return SLURMCluster(
        queue=config.SLURM_PARTITION,
        cores=1,                       # nthreads=1 -> one task per worker
        processes=1,                   # one worker process per job
        job_cpu=config.WORKER_CORES,   # but reserve WORKER_CORES cpus for LLVM
        memory=config.WORKER_MEMORY,
        walltime=config.WORKER_WALLTIME,
        # Pin scheduler + workers to InfiniBand instead of letting Dask guess the
        # IP via an (unreachable on compute nodes) probe to 8.8.8.8. ib0 is the
        # fast fabric and is cross-node reachable on every node. If ib0 ever
        # misbehaves, "eno8303" (the 172.17/16 ethernet) is a proven fallback.
        interface=config.SLURM_INTERFACE,
        # Remote workers run under this driver's interpreter (sys.executable) by
        # default, so the venv's deps are already present; we only need to add
        # our package directory to their import path.
        job_script_prologue=[f"export PYTHONPATH={HERE}:$PYTHONPATH"],
    )


def main():
    data_to_fit, trial_inputs = make_data()

    cluster = build_cluster()
    print("---- generated worker job script ----")
    print(cluster.job_script())
    print("-------------------------------------", flush=True)

    cluster.scale(jobs=config.N_WORKERS)  # submit N worker jobs

    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)
    print(
        f"Submitted {config.N_WORKERS} worker jobs; waiting for the first to start "
        f"(squeue to watch)...",
        flush=True,
    )
    # Start as soon as one worker is up; the rest join elastically as their jobs
    # schedule. Raise after 10 min if the partition never gives us a worker.
    client.wait_for_workers(1, timeout=600)
    n_up = len(client.scheduler_info()["workers"])
    print(f"{n_up} worker(s) connected; starting fit.", flush=True)

    try:
        study = run_fit(client, data_to_fit, trial_inputs)
        summarize(study)
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
