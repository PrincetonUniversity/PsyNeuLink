#!/bin/bash
#SBATCH --job-name=csi-direct-rescue
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-3
#SBATCH --output=logs/csi-direct-rescue-%A_%a.out
#SBATCH --error=logs/csi-direct-rescue-%A_%a.err

set -euo pipefail

# Rerun the three job-1212 continuous/direct cells whose recovered point
# scored below the known generating truth. The original synthetic datasets
# are reused exactly; only the optimizer configuration changes.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
SOURCE_ROOT="${CSI_SOURCE_RESULTS_ROOT:-$SCRIPT_DIR/audit/compact-recovery-1212}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/direct-optimizer-rescue-${SLURM_ARRAY_JOB_ID:-mapping}}"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
case "$TASK_ID" in
    1)
        TRUTH_PROFILE="low-control-fast"
        SIMULATION_SEED=79
        OPTIMIZER_SEED=1102
        ;;
    2)
        TRUTH_PROFILE="condition-contrast"
        SIMULATION_SEED=17
        OPTIMIZER_SEED=1400
        ;;
    3)
        TRUTH_PROFILE="high-control"
        SIMULATION_SEED=43
        OPTIMIZER_SEED=1201
        ;;
    *)
        echo "SLURM_ARRAY_TASK_ID must be between 1 and 3; got $TASK_ID" >&2
        exit 1
        ;;
esac

SOURCE_LABEL="truth-${TRUTH_PROFILE}__continuous__direct__seed${SIMULATION_SEED}__likelihood1"
RUN_LABEL="${SOURCE_LABEL}__optimizer-rescue"
SOURCE_DIR="$SOURCE_ROOT/$SOURCE_LABEL"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL"

command=(
    "$PYTHON" -u "$DRIVER" run-cell
    --truth-profile "$TRUTH_PROFILE"
    --generator continuous
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
sha256sum "$DRIVER" "$SCRIPT_DIR/csi_direct_optimizer_rescue_gb300.slurm"

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
