#!/bin/bash
# Single-node benchmark: regular PEC vs the Dask local sweep at a fixed 16-core
# budget (W workers x 16/W cores each). One sbatch job; each config is a fresh
# python process for clean isolation. >61 min walltime -> `short` QOS.
#SBATCH --job-name=pec_bench_sn
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # the benchmark budget; the node is still whole-charged
# Exclusive + pinned to the same homogeneous pool as run_config.slurm so
# single-node and multi-node rows are comparable (no co-tenants, same CPU).
#SBATCH --exclusive
#SBATCH --mem=180G   # ~all of a 190G r3c node (Della rejects --mem=0)
#SBATCH --constraint=cascade
#SBATCH --exclude=della-h12n[1-18],della-h17n[1-3],della-i13n[1-25]
#SBATCH --time=1-12:00:00   # 36h -> `medium` QOS (>24h); ample for the ~10h sweep
#SBATCH --output=/scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/benchmark/logs/%x_%j.out
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ap9344@princeton.edu

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
OUT=$BD/results/singlenode.jsonl
mkdir -p $BD/results $BD/logs
MODEL=ddm   # which model provider to benchmark (ddm | stabflex)
NE=8000
TE=960     # fixed eval budget for EVERY config (apples-to-apples: same total
           # work; rounds = TE/POP, independent of worker count)
POP=32     # fixed CMA-ES population; >= max n-workers keeps workers busy
NT=200     # observed data trials per fit (bump for precision; also makes sims heavier)
REPS=1

cd $REPO
rm -f "$OUT"

# regular baseline: serial trials, all 16 cores for estimate-level threading
$PY $BD/bench.py --model $MODEL --mode regular     --worker-cores 16 --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps $REPS --out $OUT

# dask-local sweep at fixed 16 cores, fixed optimizer path, and fixed eval budget.
$PY $BD/bench.py --model $MODEL --mode dask-local --n-workers 2  --worker-cores 8  --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-local --n-workers 4  --worker-cores 4  --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-local --n-workers 8  --worker-cores 2  --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps $REPS --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-local --n-workers 16 --worker-cores 1  --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps $REPS --out $OUT

echo "==== SUMMARY ===="
$PY $BD/summarize.py $OUT
