#!/bin/bash
# One config via dask-srun, NON-exclusive and unpinned -- a quick performance
# check, NOT a fair hardware-pinned benchmark (use run_config.slurm for that).
# --ntasks/--cpus-per-task/--job-name and the config (NW WC MODEL SAMPLER NE NT
# POP TE WT OUT) are passed at submit time by the caller; SAMPLER defaults to cmaes.
#SBATCH --partition=cpu
#SBATCH --time=01:30:00          # >61min -> short QOS; post-fix each config is minutes
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/benchmark/logs/%x_%j.out

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
export PYTHONPATH=$BD:$(dirname $BD):${PYTHONPATH:-}
cd $REPO

# SLURM placed (NW+2)*WC cores wherever it found them (shared); block dist ->
# PROCID 0,1 = scheduler+driver, 2+ = workers, same as the pinned path.
srun --exact --distribution=block --cpu-bind=cores --ntasks=$((NW + 2)) --cpus-per-task=$WC \
    $PY $BD/bench.py --model $MODEL --mode dask-srun --sampler ${SAMPLER:-cmaes} \
    --n-workers $NW --worker-cores $WC --optimizer-popsize $POP \
    --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 \
    --worker-timeout $WT --out $OUT

rm -f $REPO/slurm_logs/scheduler_${SLURM_JOB_ID}_*.json   # safety net (scheduler self-cleans)
