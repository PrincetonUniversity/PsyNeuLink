"""Local entrypoint: run the Dask PEC MLE fit on a single-node LocalCluster.

Run via salloc:

    salloc --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=8G --time=00:30:00 --partition=cpu
    .venv/bin/python3 Scripts/Debug/pec_hierarchical/run_dask_mle_local.py

This validates the driver/worker ask/tell loop on one node. For true multi-node,
see the benchmark's dask-srun mode (benchmark/bench.py + slurm/run_config.slurm),
which launches the scheduler + workers as SLURM ranks inside one allocation.
"""

import os
import sys

# Make the `pec_dask_mle` package importable on BOTH the driver and the spawned
# Dask workers. sys.path covers this process; PYTHONPATH is inherited by the
# LocalCluster worker processes (and, later, by srun-launched workers when the
# Option B SLURM script exports the same PYTHONPATH).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from dask.distributed import Client, LocalCluster

from pec_dask_mle import config
from pec_dask_mle.data import make_data
from pec_dask_mle.driver import run_fit, summarize


def main():
    data_to_fit, trial_inputs = make_data()

    cluster = LocalCluster(n_workers=config.N_WORKERS, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    try:
        study = run_fit(client, data_to_fit, trial_inputs)
        summarize(study)
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
