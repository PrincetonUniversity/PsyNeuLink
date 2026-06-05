#!/bin/bash
# Single-node benchmark: regular PEC vs the Dask local sweep at a fixed 16-core
# budget (W workers x 16/W cores each). One sbatch job; each config is a fresh
# python process for clean isolation. >61 min walltime -> `short` QOS.
#SBATCH --job-name=pec_bench_sn
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
OUT=$BD/results_singlenode.jsonl
NE=4000
TE=240
REPS=2

cd $REPO
rm -f "$OUT"

# regular baseline: serial trials, all 16 cores for estimate-level threading
$PY $BD/bench.py --mode regular     --worker-cores 16 --num-estimates $NE --total-evals $TE --reps $REPS --out $OUT

# dask-local sweep at fixed 16 cores: trade trial-parallel vs estimate-parallel
$PY $BD/bench.py --mode dask-local --n-workers 1  --worker-cores 16 --num-estimates $NE --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --mode dask-local --n-workers 2  --worker-cores 8  --num-estimates $NE --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --mode dask-local --n-workers 4  --worker-cores 4  --num-estimates $NE --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --mode dask-local --n-workers 8  --worker-cores 2  --num-estimates $NE --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --mode dask-local --n-workers 16 --worker-cores 1  --num-estimates $NE --total-evals $TE --reps $REPS --out $OUT

echo "==== SUMMARY ===="
$PY $BD/summarize.py $OUT
