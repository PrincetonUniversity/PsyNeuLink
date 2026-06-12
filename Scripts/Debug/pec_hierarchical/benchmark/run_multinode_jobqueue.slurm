#!/bin/bash
# TODO(remove-jobqueue): delete this whole file once the provisioning-overhead
# numbers are recorded -- dask-srun (run_multinode.slurm) is the replacement.
#
# Two representative dask-jobqueue configs, kept ONLY to quantify the
# provisioning overhead dask-srun eliminates (total_s - compile_s - loop_s).
# Runs from a tiny separate driver job: doing this inside the exclusive 5-node
# allocation would idle 160 cores while worker jobs queue. Worker jobs land on
# unpinned, possibly heterogeneous nodes -- these are overhead rows, NOT fair
# timing rows.
#SBATCH --job-name=pec_bench_mn_jq
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ap9344@princeton.edu

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
RESDIR=$BD/results_multinode.d                    # same dir as the dask-srun sweep
OUT=$RESDIR/results_jobqueue.jsonl                # so summarize.py globs it alongside
MODEL=ddm
NE=8000
TE=960
POP=32
NT=250
WT=3600   # worker JOBS queue independently: wait up to 1h per config

cd $REPO
mkdir -p $RESDIR

$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 16 --worker-cores 1  --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT
$PY $BD/bench.py --model $MODEL --mode dask-jobqueue --n-workers 4  --worker-cores 16 --optimizer-popsize $POP --num-estimates $NE --num-trials $NT --total-evals $TE --reps 1 --worker-timeout $WT --out $OUT

echo "==== SUMMARY (srun sweep + jobqueue overhead rows) ===="
$PY $BD/summarize.py $RESDIR/*.jsonl
