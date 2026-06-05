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
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
OUT=$BD/results_multinode.jsonl
NE=4000
TE=240

cd $REPO
rm -f "$OUT"

# Each config builds its own SLURMCluster, scales, runs, tears down. Workers may
# queue on a busy partition; --worker-timeout guards that wait.
# 4x4 = 16 cores across nodes (control vs the single-node 4x4 LocalCluster point)
$PY $BD/bench.py --mode dask-jobqueue --n-workers 4 --worker-cores 4  --num-estimates $NE --total-evals $TE --reps 1 --worker-timeout 1200 --out $OUT
# scale beyond one node's cores
$PY $BD/bench.py --mode dask-jobqueue --n-workers 4 --worker-cores 16 --num-estimates $NE --total-evals $TE --reps 1 --worker-timeout 1200 --out $OUT
$PY $BD/bench.py --mode dask-jobqueue --n-workers 8 --worker-cores 16 --num-estimates $NE --total-evals $TE --reps 1 --worker-timeout 1800 --out $OUT

echo "==== SUMMARY ===="
$PY $BD/summarize.py $OUT
