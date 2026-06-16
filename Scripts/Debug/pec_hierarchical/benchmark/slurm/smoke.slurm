#!/bin/bash
# Throwaway smoke test for the dask-srun MECHANICS only: two tiny back-to-back
# configs across 2 nodes (validates hostfile placement, fast worker startup,
# scheduler-file hygiene between steps, and clean teardown). Deliberately NOT
# exclusive/pinned -- those only matter for the real benchmark's fair timing,
# and they push this into a slow, capped QOS. Shared small nodes backfill in
# seconds.
# 3 nodes: node 0 = scheduler+driver, nodes 1-2 = workers, so the hostfile
# round-robin actually spreads workers across 2 nodes (NW_NODES=2). Generous
# cores/node so srun --exact fits every config's per-node cpu demand.
#SBATCH --job-name=pec_srun_smoke
#SBATCH --partition=cpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/benchmark/logs/%x_%j.out

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
OUT=$BD/results/smoke.jsonl

cd $REPO
export PYTHONPATH=$BD:$(dirname $BD):$PYTHONPATH
mkdir -p $BD/results $BD/logs
rm -f "$OUT"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NW_NODES=$((${#NODES[@]} - 1))

run_cfg () {
  local NW=$1 WC=$2 HF=$BD/hostfile_$$
  {
    echo "${NODES[0]}"; echo "${NODES[0]}"
    for ((i = 0; i < NW; i++)); do echo "${NODES[$((1 + i % NW_NODES))]}"; done
  } > "$HF"
  SLURM_HOSTFILE=$HF srun --exact --distribution=arbitrary --cpu-bind=cores \
      --ntasks=$((NW + 2)) --cpus-per-task=$WC \
      $PY $BD/bench.py --model ddm --mode dask-srun \
      --n-workers $NW --worker-cores $WC --optimizer-popsize 32 \
      --num-estimates 200 --num-trials 50 --total-evals 64 --reps 1 \
      --worker-timeout 120 --out $OUT
  rm -f "$HF"
}

echo "nodes: ${NODES[*]}"
time run_cfg 4 2
time run_cfg 8 1
echo "==== RESULTS ===="
cat "$OUT"
