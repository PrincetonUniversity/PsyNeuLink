#!/bin/bash
# Plain SBATCH template for multi-node distributed PEC fitting.
#
# PsyNeuLink forms the Dask cluster from the SLURM ranks of a single srun step:
# rank 0 = scheduler, rank 1 = driver (runs study.py), ranks 2+ = workers. So the
# number of workers is (total tasks - 2). Edit --nodes / --ntasks-per-node /
# --cpus-per-task and the paths below; no scheduler, address, or Dask admin needed.
#
# Example below: 2 nodes x 5 tasks = 10 ranks -> 8 workers, 4 LLVM threads each.
#SBATCH --job-name=pec_dask
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# --- edit these two paths ---
REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink          # repo root (has the venv)
STUDY=$REPO/Scripts/Debug/pec_dask/study.py        # your distributed=True study

PY=$REPO/.venv/bin/python3

# Run all ranks from a shared filesystem so they agree on the scheduler file
# (it is written in the working directory by the launcher).
cd $REPO

# worker_cores defaults to $SLURM_CPUS_PER_TASK inside PsyNeuLink, mirroring
# --cpus-per-task above. --distribution=block keeps rank 0/1 (scheduler/driver)
# together and spreads workers across the allocation.
srun --distribution=block $PY -m psyneulink.dask_run "$STUDY"

# The scheduler removes its own scheduler file on exit; sweep any leftover.
rm -f $REPO/.psyneulink_dask_scheduler_${SLURM_JOB_ID}_*.json
