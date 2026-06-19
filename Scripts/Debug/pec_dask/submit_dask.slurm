#!/bin/bash
# Plain SBATCH template for multi-node distributed PEC fitting.
#
# One srun step: rank 0 = scheduler, rank 1 = driver, ranks 2+ = workers.
# Edit --nodes / --ntasks-per-node / --cpus-per-task and the paths below.
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

# --- edit paths ---
REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
STUDY=$REPO/Scripts/Debug/pec_dask/study.py

PY=$REPO/.venv/bin/python3

cd $REPO

# worker_cores defaults to $SLURM_CPUS_PER_TASK.
srun --distribution=block $PY -m psyneulink.dask_run "$STUDY"

rm -f $REPO/.psyneulink_dask_scheduler_${SLURM_JOB_ID}_*.json
