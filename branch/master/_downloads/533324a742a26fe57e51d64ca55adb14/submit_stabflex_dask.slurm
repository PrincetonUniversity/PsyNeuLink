#!/bin/bash
# Multi-node distributed PEC fit of the Stability-Flexibility model.
#
# One srun step: rank 0 = scheduler, rank 1 = driver, ranks 2+ = workers.
# Size the allocation so workers match MAX_CONCURRENT in the study.
#
# Example below: 3 nodes x 6 tasks = 18 ranks -> 16 workers, 4 LLVM threads each,
# which matches MAX_CONCURRENT=16 in stability_flexibility_dask.py.
#SBATCH --job-name=stabflex_dask
#SBATCH --partition=cpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out

# --- edit paths ---
REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
SF_DIR=$REPO/Scripts/Debug/stability_flexibility
STUDY=$SF_DIR/stability_flexibility_dask.py

PY=$REPO/.venv/bin/python3

# Workers import stability_flexibility.py.
export PYTHONPATH=$SF_DIR:$PYTHONPATH

cd $SF_DIR

# worker_cores defaults to $SLURM_CPUS_PER_TASK.
srun --distribution=block $PY -m psyneulink.dask_run "$STUDY"

rm -f $SF_DIR/.psyneulink_dask_scheduler_${SLURM_JOB_ID}_*.json
