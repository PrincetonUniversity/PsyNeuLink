#!/bin/bash
# Multi-node distributed PEC fit of the Stability-Flexibility model.
#
# PsyNeuLink forms the Dask cluster from the SLURM ranks of a single srun step:
# rank 0 = scheduler, rank 1 = driver (runs the study), ranks 2+ = workers. So the
# number of workers is (total tasks - 2). Size the allocation so workers match the
# study's MAX_CONCURRENT (the CMA-ES population / candidates per round): edit
# --nodes / --ntasks-per-node / --cpus-per-task and the paths below.
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

# --- edit these two paths ---
REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink                       # repo root (has the venv)
SF_DIR=$REPO/Scripts/Debug/stability_flexibility              # study + model live here
STUDY=$SF_DIR/stability_flexibility_dask.py

PY=$REPO/.venv/bin/python3

# Workers import the study's pec_factory, which references make_stab_flex /
# generate_trial_sequence from stability_flexibility.py *by name*, so that module
# must be importable on every rank. srun propagates this environment to all tasks.
export PYTHONPATH=$SF_DIR:$PYTHONPATH

# Run all ranks from a shared filesystem so they agree on the scheduler file
# (it is written in the working directory by the launcher).
cd $SF_DIR

# worker_cores defaults to $SLURM_CPUS_PER_TASK inside PsyNeuLink, mirroring
# --cpus-per-task above. --distribution=block keeps rank 0/1 (scheduler/driver)
# together and spreads workers across the allocation.
srun --distribution=block $PY -m psyneulink.dask_run "$STUDY"

# The scheduler removes its own scheduler file on exit; sweep any leftover.
rm -f $SF_DIR/.psyneulink_dask_scheduler_${SLURM_JOB_ID}_*.json
