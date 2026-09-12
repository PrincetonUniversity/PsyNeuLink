#!/bin/bash
# SBATCH template for multi-node hierarchical fitting.
#
# One srun step: rank 0 = scheduler, rank 1 = driver, ranks 2+ = workers. Each worker fits
# whole participants, so there is no benefit to more workers than participants; size the
# allocation by the group, not by the trial count.
#
# Example below: 2 nodes x 5 tasks = 10 ranks -> 8 workers, 4 LLVM threads each.
#SBATCH --job-name=pec_hierarchical
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

# --- edit paths ---
REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
EXAMPLES=$REPO/Scripts/Debug/pec_hierarchical
DATA=$EXAMPLES/group_data.csv

PY=$REPO/.venv/bin/python3

cd $REPO

# Synthetic data to fit. Skip this step and point --data at your own table instead.
$PY "$EXAMPLES/make_example_data.py" --n-participants 24 --n-trials 60 \
    --data-out "$DATA"

# worker_cores defaults to $SLURM_CPUS_PER_TASK.
srun --distribution=block $PY -m psyneulink.dask_run "$EXAMPLES/hierarchical_fitting.py" \
    --data "$DATA" --distributed

rm -f $REPO/.psyneulink_dask_scheduler_${SLURM_JOB_ID}_*.json
