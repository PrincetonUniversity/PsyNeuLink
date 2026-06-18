#!/bin/bash
# Per-sampler worker x core grid. For each Optuna sampler, sweep the (NW x WC)
# grid with the optimizer popsize/batch pinned to NW -- i.e. the "popsize =
# num_workers" regime CMA-ES likes (every worker evaluates exactly one candidate
# per round, no idle workers, no serial multiplexing). Eval budget (TE) is fixed,
# so rounds = TE/NW: bigger NW means a bigger batch but fewer optimizer updates.
#
# Each config records BOTH throughput (evals/s) and recovery error (max_pct_err),
# so the cross-sampler comparison shows whether popsize=num_workers generalises:
#   cmaes  -- quality degrades as NW(=popsize) grows (fewer generations at fixed TE)
#   tpe    -- tolerates large batches (constant_liar keeps the batch diverse)
#   random -- batch-invariant: more workers = pure throughput, no quality cost
#   qmc    -- batch-invariant like random (low-discrepancy space-filling)
#   gp     -- keeps quality, but the serial GP refit on the driver caps throughput
#             regardless of worker count (workers idle while the GP is fit)
#
# WC only changes per-eval LLVM threads (throughput), never the optimizer
# trajectory (which depends on NW alone) -- so the quality signal lives on the NW
# axis and the WC axis maps the throughput surface.
#
# Defaults to the pinned/exclusive run_config.slurm (comparable timing, whole-node
# per config). Overridable via env:
#   SAMPLERS="cmaes gp"            -- subset of samplers
#   RUNNER=run_config_quick.slurm  -- shared/non-exclusive (backfills, only charges
#                                     the cores used; evals/s noisier but cheaper)
#
# Run from a login node:  ./slurm/sweep_samplers.sh
set -euo pipefail

REPO=/scratch/gpfs/JDC/ap9344/PsyNeuLink
PY=$REPO/.venv/bin/python3
BD=$REPO/Scripts/Debug/pec_dask/benchmark
RESDIR=$BD/results/samplers.d         # one JSONL per (sampler, NW, WC)
CORES_PER_NODE=32                     # the pinned r3c nodes
RUNNER="${RUNNER:-run_config.slurm}"  # pinned by default; run_config_quick.slurm = shared

MODEL=ddm
SAMPLERS="${SAMPLERS:-cmaes tpe random qmc gp}"
NE=4000     # simulations per likelihood eval (lighter than the 8000 core-grid so
            # 5 samplers x ~19 configs stays tractable; relative behaviour holds)
TE=480      # fixed eval budget for EVERY config (rounds = TE/NW)
NT=250      # observed-data trials per fit
WT=300      # worker-start timeout (s); srun workers come up in seconds
MAX_CORES=128

mkdir -p "$RESDIR" "$BD/logs"
rm -f "$RESDIR"/*.jsonl

ids=()
submit () {   # submit SAMPLER NW WC
  local S=$1 NW=$2 WC=$3
  local ranks=$((NW + 2))                       # workers + scheduler + driver
  local out="$RESDIR/results_${S}_${NW}x${WC}.jsonl"
  local exp="ALL,NW=$NW,WC=$WC,MODEL=$MODEL,SAMPLER=$S,NE=$NE,TE=$TE,POP=$NW,NT=$NT,WT=$WT,OUT=$out"
  local jid
  if [[ "$RUNNER" == run_config_quick.slurm ]]; then
    # shared/non-exclusive: SLURM places (NW+2) tasks of WC cores anywhere.
    jid=$(sbatch --parsable --ntasks="$ranks" --cpus-per-task="$WC" \
          --job-name="pec_${S}_${NW}x${WC}" --export="$exp" "$BD/slurm/$RUNNER")
    printf '  %-8s %-6s (shared)  job %s\n' "$S" "${NW}x${WC}" "$jid"
  else
    # pinned/exclusive: size the allocation to whole nodes (ceil(ranks / tasks-per-node)).
    local rpn=$((CORES_PER_NODE / WC))
    local nodes=$(( (ranks + rpn - 1) / rpn ))
    jid=$(sbatch --parsable --nodes="$nodes" \
          --job-name="pec_${S}_${NW}x${WC}" --export="$exp" "$BD/slurm/$RUNNER")
    printf '  %-8s %-6s -> %d node(s)  job %s\n' "$S" "${NW}x${WC}" "$nodes" "$jid"
  fi
  ids+=("$jid")
}

# popsize = NW, and CMA-ES/NSGA-II need popsize >= 2, so the grid starts at NW=2.
echo "submitting per-sampler worker x core grid (popsize=NW, model=$MODEL, NE=$NE, TE=$TE, NT=$NT)"
echo "samplers: $SAMPLERS"
for S in $SAMPLERS; do
  for WC in 1 2 4 8; do
    for NW in 2 4 8 16 32; do
      (( NW * WC > MAX_CORES )) && continue
      submit "$S" "$NW" "$WC"
    done
  done
done

# Summarize + plot once every config has finished (afterany: use whatever landed).
dep=$(IFS=:; echo "${ids[*]}")
sum=$(sbatch --parsable --dependency="afterany:$dep" --kill-on-invalid-dep=yes \
      --job-name=pec_samplers_summary --partition=cpu --time=00:15:00 --mem=4G \
      --output="$BD/logs/pec_samplers_summary_%j.out" \
      --wrap="$PY $BD/summarize.py $RESDIR/*.jsonl && $PY $BD/plot_samplers.py")
echo "submitted ${#ids[@]} config jobs + summary job $sum"
echo "live results land in $RESDIR/ -- summarize/plot anytime with:"
echo "  $PY $BD/summarize.py $RESDIR/*.jsonl"
echo "  $PY $BD/plot_samplers.py"
