#!/bin/bash
#SBATCH --job-name=csi-direct-full
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-30%20
#SBATCH --output=logs/csi-direct-full-%A_%a.out
#SBATCH --error=logs/csi-direct-full-%A_%a.err

set -euo pipefail

# Refit all 30 direct-likelihood cells from compact-recovery job 1212:
#   5 truth profiles x 3 synthetic sequences x 2 data generators.
# The original synthetic CSV and generation metadata are copied verbatim so
# this experiment changes only the direct optimizer configuration.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
SOURCE_ROOT="${CSI_SOURCE_RESULTS_ROOT:-$SCRIPT_DIR/audit/compact-recovery-1212}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/direct-optimizer-full-${SLURM_ARRAY_JOB_ID:-mapping}}"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 30 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 30; got $TASK_ID" >&2
    exit 1
fi

TRUTH_PROFILES=(
    baseline
    low-control-fast
    high-control
    strong-collapse
    condition-contrast
)
SIMULATION_SEEDS=(17 43 79)
GENERATORS=(continuous gpu-1ms)

OFFSET=$((TASK_ID - 1))
GENERATOR_INDEX=$((OFFSET % 2))
SEED_INDEX=$(((OFFSET / 2) % 3))
TRUTH_INDEX=$((OFFSET / 6))

TRUTH_PROFILE="${TRUTH_PROFILES[TRUTH_INDEX]}"
SIMULATION_SEED="${SIMULATION_SEEDS[SEED_INDEX]}"
GENERATOR="${GENERATORS[GENERATOR_INDEX]}"
OPTIMIZER_SEED=$((1000 + TRUTH_INDEX * 100 + SEED_INDEX))
SOURCE_LABEL="truth-${TRUTH_PROFILE}__${GENERATOR}__direct__seed${SIMULATION_SEED}__likelihood1"
RUN_LABEL="${SOURCE_LABEL}__optimizer-v2"
SOURCE_DIR="$SOURCE_ROOT/$SOURCE_LABEL"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL"

command=(
    "$PYTHON" -u "$DRIVER" run-cell
    --truth-profile "$TRUTH_PROFILE"
    --generator "$GENERATOR"
    --fitter direct
    --data "$DATA_FILE"
    --subject 1
    --simulation-seed "$SIMULATION_SEED"
    --optimizer-seed "$OPTIMIZER_SEED"
    --likelihood-seed 100000
    --likelihood-replicate 1
    --direct-starts 8
    --direct-max-iterations 200
    --direct-random-start-candidates 32
    --direct-polish-restarts 2
    --direct-polish-iterations 150
    --direct-coordinate-step 0.01
    --direct-coordinate-levels 4
    --direct-coordinate-rounds 12
    --direct-coordinate-cycles 2
    --output-dir "$OUTPUT_DIR"
    --run-label "$RUN_LABEL"
)

echo "Array task: $TASK_ID"
echo "Truth profile: $TRUTH_PROFILE"
echo "Generator: $GENERATOR"
echo "Simulation seed: $SIMULATION_SEED"
echo "Optimizer seed: $OPTIMIZER_SEED"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    exit 0
fi

for required in "$PYTHON" "$DRIVER" "$DATA_FILE" \
    "$SOURCE_DIR/synthetic_data.csv" "$SOURCE_DIR/generation.json"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cp "$SOURCE_DIR/synthetic_data.csv" "$OUTPUT_DIR/synthetic_data.csv"
cp "$SOURCE_DIR/generation.json" "$OUTPUT_DIR/generation.json"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Experiment source hashes:"
sha256sum \
    "$DRIVER" \
    "$SCRIPT_DIR/../direct_likelihood/fit.py" \
    "$SCRIPT_DIR/csi_direct_optimizer_full_gb300.slurm"

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
