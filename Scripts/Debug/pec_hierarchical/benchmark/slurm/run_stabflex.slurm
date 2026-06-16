#!/bin/bash
# One proper parallel PEC fit of the stability-flexibility model via dask-srun,
# plus a no-dask single-process baseline at identical settings to measure the
# parallelization uplift. NOT a benchmark: no --exclusive / no hardware pinning
# (those exist only to make benchmark timings comparable). --nodes=3 forces the
# dask fit to span nodes -> doubles as the dask-srun "multinode verify".
#SBATCH --job-name=pec_stabflex
#SBATCH --partition=cpu
#SBATCH --nodes=3                # force multinode for the dask step
#SBATCH --ntasks=10             # 8 workers + scheduler + driver
#SBATCH --cpus-per-task=8       # LLVM threads per worker (heavy model -> threading helps)
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00         # >61min -> short QOS; measured ~0.4-0.5s/eval
#SBATCH --output=/scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/benchmark/logs/%x_%j.out

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
OUT=$BD/results/stabflex.jsonl
export PYTHONPATH=$BD:$(dirname $BD):${PYTHONPATH:-}   # workers import bench_core/models/pec_dask_mle

cd $REPO
mkdir -p $BD/results $BD/logs
rm -f "$OUT"

COMMON="--model stabflex --optimizer-popsize 24 --num-estimates 2000 \
        --num-trials 50 --total-evals 960 --reps 1 --out $OUT"

# 1) dask-srun parallel fit (primary; verifies multinode on the heavy model).
#    block distribution -> PROCID 0,1 = scheduler+driver, 2+ = workers.
srun --exact --distribution=block --cpu-bind=cores --ntasks=10 --cpus-per-task=8 \
    $PY $BD/bench.py $COMMON --mode dask-srun \
    --n-workers 8 --worker-cores 8 --worker-timeout 300
rm -f $REPO/slurm_logs/scheduler_${SLURM_JOB_ID}_*.json   # safety net (scheduler self-cleans)

# 2) no-dask single-process baseline, identical fit, matched per-eval threading
#    (--worker-cores 8) so the only difference vs the dask run is concurrency.
srun --exact --ntasks=1 --cpus-per-task=8 \
    $PY $BD/bench.py $COMMON --mode regular --worker-cores 8

echo "==== RESULT (speedup = regular loop_s / dask loop_s) ===="
$PY $BD/summarize.py "$OUT"
cat "$OUT"
