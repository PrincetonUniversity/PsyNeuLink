#!/bin/bash
#SBATCH --job-name=csi-sim-audit
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-3%3
#SBATCH --output=logs/csi-sim-audit-%A_%a.out
#SBATCH --error=logs/csi-sim-audit-%A_%a.err

set -euo pipefail

# Prepared for TigerAI but intentionally not submitted. This audits complete
# real participant input sequences at representative fitted parameter vectors.
# The comparison calls LLVMRun directly and does not exercise its likelihood.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_gpu_llvm_simulation_audit.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
PARAMETER_ROOT="${CSI_PARAMETER_ROOT:-$SCRIPT_DIR/../direct_likelihood/fits}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/gpu-llvm-real-${SLURM_ARRAY_JOB_ID:-mapping}}"

SUBJECTS=(1 4 7)
PARAMETER_FILES=(
    direct_fit_subject_1_fine_mesh.json
    direct_fit_subject_4_scaled_default_start1.json
    direct_fit_subject_7_scaled_default_start1.json
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 3 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 3; got $TASK_ID" >&2
    exit 1
fi
INDEX=$((TASK_ID - 1))
SUBJECT="${SUBJECTS[INDEX]}"
PARAMETER_FILE="$PARAMETER_ROOT/${PARAMETER_FILES[INDEX]}"
OUTPUT="$RESULTS_ROOT/subject-${SUBJECT}.json"

for required in "$PYTHON" "$DRIVER" "$DATA_FILE" "$PARAMETER_FILE"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done

command=(
    "$PYTHON" -u "$DRIVER"
    --data-file "$DATA_FILE"
    --subject "$SUBJECT"
    --parameter-file "$PARAMETER_FILE"
    --gpu-replicates 200
    --llvm-replicates 200
    --max-steps 5000
    --gpu-seed "$((200000 + SUBJECT))"
    --llvm-seed "$((300000 + SUBJECT))"
    --output "$OUTPUT"
)

echo "Array task: $TASK_ID"
echo "Subject: $SUBJECT"
echo "Parameters: $PARAMETER_FILE"
echo "Output: $OUTPUT"

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$RESULTS_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
