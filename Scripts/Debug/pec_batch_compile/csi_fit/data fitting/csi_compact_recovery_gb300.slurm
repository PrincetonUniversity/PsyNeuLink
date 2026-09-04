#!/bin/bash
#SBATCH --job-name=csi-compact-recovery
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=1-64%20
#SBATCH --output=logs/csi-compact-recovery-%A_%a.out
#SBATCH --error=logs/csi-compact-recovery-%A_%a.err

set -euo pipefail

# Compact crossed recovery study:
#
# Tasks 1-60:
#   5 truth profiles x 3 simulated sequences x
#   2 generators (continuous, GPU 1 ms) x 2 fitters (direct, GPU 1 ms).
#
# Tasks 61-64:
#   Repeat the baseline/seed-17 GPU fits for both generators with two new
#   likelihood seeds. Optimizer seeds remain fixed, isolating Monte Carlo
#   objective variability.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/compact-recovery-${SLURM_ARRAY_JOB_ID:-mapping}}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for required in "$PYTHON" "$DRIVER" "$DATA_FILE"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 64 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 64; got $TASK_ID" >&2
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
FITTERS=(direct gpu-1ms)
SUBJECT="${CSI_TEMPLATE_SUBJECT:-1}"
LIKELIHOOD_REPLICATE=1

if (( TASK_ID <= 60 )); then
    OFFSET=$((TASK_ID - 1))
    FITTER_INDEX=$((OFFSET % 2))
    GENERATOR_INDEX=$(((OFFSET / 2) % 2))
    SEED_INDEX=$(((OFFSET / 4) % 3))
    TRUTH_INDEX=$((OFFSET / 12))
else
    # Two additional likelihood seeds for each baseline generator. These four
    # cells all use the same synthetic seed and optimizer seed as main tasks
    # 2 and 4.
    OFFSET=$((TASK_ID - 61))
    LIKELIHOOD_REPLICATE=$((2 + OFFSET / 2))
    GENERATOR_INDEX=$((OFFSET % 2))
    FITTER_INDEX=1
    SEED_INDEX=0
    TRUTH_INDEX=0
fi

TRUTH_PROFILE="${TRUTH_PROFILES[TRUTH_INDEX]}"
SIMULATION_SEED="${SIMULATION_SEEDS[SEED_INDEX]}"
GENERATOR="${GENERATORS[GENERATOR_INDEX]}"
FITTER="${FITTERS[FITTER_INDEX]}"
OPTIMIZER_SEED=$((1000 + TRUTH_INDEX * 100 + SEED_INDEX))
LIKELIHOOD_SEED=$((LIKELIHOOD_REPLICATE * 100000 + TRUTH_INDEX * 10000 + SIMULATION_SEED))
RUN_LABEL="truth-${TRUTH_PROFILE}__${GENERATOR}__${FITTER}__seed${SIMULATION_SEED}__likelihood${LIKELIHOOD_REPLICATE}"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL"

command=(
    "$PYTHON" -u "$DRIVER" run-cell
    --truth-profile "$TRUTH_PROFILE"
    --generator "$GENERATOR"
    --fitter "$FITTER"
    --data "$DATA_FILE"
    --subject "$SUBJECT"
    --simulation-seed "$SIMULATION_SEED"
    --optimizer-seed "$OPTIMIZER_SEED"
    --likelihood-seed "$LIKELIHOOD_SEED"
    --likelihood-replicate "$LIKELIHOOD_REPLICATE"
    --generation-maximum-time 12
    --gpu-fit-maximum-time 12
    --gpu-num-estimates 100000
    --gpu-max-iterations 5000
    --gpu-smoothing-sigma 0.5
    --gpu-pseudocount 0.1
    --gpu-parameter-batch-size 11
    --direct-starts 2
    --direct-max-iterations 100
    --direct-random-start-candidates 32
    --output-dir "$OUTPUT_DIR"
    --run-label "$RUN_LABEL"
)

echo "Array task: $TASK_ID"
echo "Truth profile: $TRUTH_PROFILE"
echo "Generator: $GENERATOR"
echo "Fitter: $FITTER"
echo "Simulation seed: $SIMULATION_SEED"
echo "Optimizer seed: $OPTIMIZER_SEED"
echo "Likelihood seed: $LIKELIHOOD_SEED"
echo "Likelihood replicate: $LIKELIHOOD_REPLICATE"
echo "Template subject: $SUBJECT"
echo "Output: $OUTPUT_DIR"

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Git status:"
git -C "$REPO_ROOT" status --short
echo "Experiment source hashes:"
sha256sum "$DRIVER" "$SCRIPT_DIR/csi_compact_recovery_gb300.slurm"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
