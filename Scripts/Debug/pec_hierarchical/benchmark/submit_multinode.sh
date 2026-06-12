#!/bin/bash
# Submit the PEC core-scaling sweep as INDEPENDENT, right-sized jobs -- one per
# config. Each job requests only the cores its geometry needs (NW workers +
# scheduler + driver, WC cores each) and releases them the instant it finishes,
# instead of one big allocation holding 5 nodes idle through the small configs.
# A queued job consumes nothing, and the driver is co-scheduled with its
# workers, so there is no idle-driver wait either. Each config writes its own
# results file; a dependent summary job merges them when all have finished.
#
# Run from a login node:  ./submit_multinode.sh
set -euo pipefail

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_hierarchical/benchmark
RESDIR=$BD/results_multinode.d        # one JSONL per config; summarize globs them
CORES_PER_NODE=32                     # the pinned r3c nodes

MODEL=ddm
NE=8000     # simulations per likelihood evaluation
TE=960      # fixed eval budget for EVERY config (rounds = TE/POP)
POP=32      # fixed CMA-ES population (>= max NW keeps workers busy)
NT=250      # observed-data trials per fit
WT=300      # worker-start timeout (s); srun workers come up in seconds

mkdir -p "$RESDIR"
rm -f "$RESDIR"/*.jsonl

ids=()
submit () {   # submit NW WC
  local NW=$1 WC=$2
  local ranks=$((NW + 2))                       # workers + scheduler + driver
  local rpn=$((CORES_PER_NODE / WC))            # whole tasks per node
  local nodes=$(( (ranks + rpn - 1) / rpn ))    # ceil(ranks / rpn)
  local out="$RESDIR/results_${NW}x${WC}.jsonl"
  local jid
  jid=$(sbatch --parsable --nodes="$nodes" --job-name="pec_mn_${NW}x${WC}" \
        --export=ALL,NW=$NW,WC=$WC,MODEL=$MODEL,NE=$NE,TE=$TE,POP=$POP,NT=$NT,WT=$WT,OUT="$out" \
        "$BD/run_one.slurm")
  printf '  %-6s %2d core(s)/worker -> %d node(s)  job %s\n' "${NW}x${WC}" "$WC" "$nodes" "$jid"
  ids+=("$jid")
}

# Grid sweep: cores-per-worker WC x worker-count NW. Bounds:
#   - WC <= 8        : more cores/worker is redundant for these sims
#   - NW*WC <= 128   : worker-core budget
#   - NW <= POP      : the optimizer proposes only POP candidates per round, so
#                      more workers than POP would idle every round
MAX_CORES=128
echo "submitting PEC core-scaling grid (model=$MODEL, NE=$NE, TE=$TE, POP=$POP, NT=$NT):"
for WC in 1 2 4 8; do
  for NW in 1 2 4 8 16 32; do
    (( NW * WC > MAX_CORES )) && continue
    (( NW > POP )) && continue
    submit "$NW" "$WC"
  done
done

# Merge once every config job has finished (afterany: summarize whatever landed,
# even if some config failed). One core, runs after the deps clear.
dep=$(IFS=:; echo "${ids[*]}")
sum=$(sbatch --parsable --dependency="afterany:$dep" --kill-on-invalid-dep=yes \
      --job-name=pec_mn_summary --partition=cpu --time=00:10:00 --mem=4G \
      --output="$BD/pec_mn_summary_%j.out" \
      --wrap="$PY $BD/summarize.py $RESDIR/*.jsonl")
echo "summary: job $sum (runs after all configs; see pec_mn_summary_*.out)"
echo "live results land in $RESDIR/ -- summarize anytime with:"
echo "  $PY $BD/summarize.py $RESDIR/*.jsonl"
