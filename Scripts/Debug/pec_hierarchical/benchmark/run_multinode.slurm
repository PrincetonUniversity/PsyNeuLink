#!/bin/bash
# Multi-node benchmark: the DRIVER runs here (lightweight: hosts the scheduler +
# ask/tell loop) and dask-jobqueue submits the WORKER jobs across nodes. The
# driver allocation is tiny but must outlast the worker jobs (config.WORKER_
# WALLTIME) -- hence 3h here. >61 min -> `short` QOS.
#SBATCH --job-name=pec_bench_mn
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ap9344@princeton.edu

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
OUT=$BD/results_multinode.jsonl
MODEL=ddm   # which model provider to benchmark (ddm | stabflex)
NE=8000
TE=960     # fixed eval budget for EVERY config (apples-to-apples: same total
           # work; rounds scale as TE/popsize). Divisible by all n_workers used.
NT=250     # observed data trials per fit (bump for precision; also makes sims heavier)
WT=3600   # per-config worker-start timeout (s): wait up to 1h for all workers

cd $REPO
rm -f "$OUT"

# Each config builds its own SLURMCluster, scales, runs, tears down. Ordered by
# worker_cores ASCENDING: small (1-2 core) workers backfill fast on a busy
# partition, so the cheap-to-schedule data lands first; the fat 16-core workers
# (hard to place) run last, where a timeout costs the least.

# --- 16-core-total controls: same splits as the single-node sweep, but across
# --- nodes. The delta vs single-node = network/serialization overhead.
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 16 --worker-cores 1  --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 8  --worker-cores 2  --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 4  --worker-cores 4  --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 2  --worker-cores 8  --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT

# --- scale-out beyond one node's 16 cores, with cheap 1-core workers ---
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 32 --worker-cores 1  --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT

# --- fat-worker scale-out (16 cores/worker): hard to place, so run LAST ---
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 4  --worker-cores 16 --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 8  --worker-cores 16 --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT

echo "==== SUMMARY ===="
$PY $BD/summarize.py $OUT
