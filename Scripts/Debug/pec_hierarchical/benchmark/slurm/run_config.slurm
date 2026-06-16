#!/bin/bash
# ONE config of the PEC benchmark (the pinned/exclusive building block used by
# the sweep_*.sh launchers). Launches a fresh Dask cluster (scheduler + NW
# workers, WC cores each) all at once inside this allocation via a single srun
# step, runs the fit, tears down. The allocation is sized to exactly this
# config's cores so nothing sits idle -- do not run directly; a sweep launcher
# (sweep_core_grid.sh, sweep_samplers.sh) computes --nodes/--job-name and passes
# the config via --export, including SAMPLER (defaults to cmaes if unset).
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --mem=180G           # ~all of a 190G r3c node (Della rejects --mem=0)
#SBATCH --constraint=cascade
# Pin to the 64 identical 32c/190G r3c nodes (the "cascade" feature also tags
# bigmem/icelake/sapphire/40-core oddballs, so exclude those explicitly) -- a
# homogeneous, exclusive node makes every config's timing comparable.
#SBATCH --exclude=della-h12n[1-18],della-h17n[1-3],della-i13n[1-25]
#SBATCH --time=02:00:00      # >61min keeps it in `short` QOS, not the capped `test` QOS
#SBATCH --output=/scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/benchmark/logs/%x_%j.out

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
export PYTHONPATH=$BD:$(dirname $BD):${PYTHONPATH:-}   # workers import bench_core/models/pec_dask_mle

cd $REPO

# Block distribution: ranks fill node-by-node, so SLURM_PROCID 0,1
# (scheduler + driver, co-located) land on the first node and workers fill the
# rest. SLURMRunner maps roles by PROCID; no hostfile needed when the job is
# sized to a single config. srun does not inherit sbatch --cpus-per-task
# (Slurm >= 22.05), so -c is passed explicitly; --exact + --cpu-bind=cores
# cgroup-confine each worker to exactly its WC cores.
srun --exact --distribution=block --cpu-bind=cores \
    --ntasks=$((NW + 2)) --cpus-per-task=$WC \
    $PY $BD/bench.py --model $MODEL --mode dask-srun --sampler ${SAMPLER:-cmaes} \
    --n-workers $NW --worker-cores $WC --optimizer-popsize $POP \
    --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 \
    --worker-timeout $WT --out $OUT

rm -f $REPO/slurm_logs/scheduler_${SLURM_JOB_ID}_*.json   # safety net (scheduler self-cleans)
