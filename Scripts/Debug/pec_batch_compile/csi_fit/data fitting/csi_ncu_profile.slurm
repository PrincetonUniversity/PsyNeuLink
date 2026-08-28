#!/bin/bash
#SBATCH --job-name=csi-ncu
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ncu-%j.out
#SBATCH --error=logs/ncu-%j.err

set -euo pipefail

# Submit from the "data fitting" directory after creating ./logs.  The first
# matching launch is the one-parameter Optuna startup evaluation; skip it and
# profile the next launch, which contains a full 11-member CMA-ES population.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
NCU="$(command -v ncu || true)"

SUBJECT="${CSI_SUBJECT_ID:-1}"
NUM_ESTIMATES="${CSI_PROFILE_NUM_ESTIMATES:-100000}"
RESULTS_DIR="$SCRIPT_DIR/logs/ncu-${SLURM_JOB_ID}"
PROFILE_BASE="$RESULTS_DIR/csi-kernel"
PROFILE_REPORT="$PROFILE_BASE.ncu-rep"
FIT_OUTPUT="$RESULTS_DIR/fit-output"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$DRIVER" ]]; then
    echo "Fit driver not found: $DRIVER" >&2
    exit 1
fi
if [[ ! -f "$DATA_FILE" ]]; then
    echo "Behavioral data not found: $DATA_FILE" >&2
    exit 1
fi
if [[ -z "$NCU" ]]; then
    echo "Nsight Compute CLI (ncu) was not found in PATH." >&2
    exit 1
fi

if [[ -r /proc/driver/nvidia/params ]]; then
    PROFILING_ADMIN_ONLY="$(awk -F': ' '$1 == "RmProfilingAdminOnly" { print $2 }' /proc/driver/nvidia/params)"
    echo "RmProfilingAdminOnly: ${PROFILING_ADMIN_ONLY:-unknown}"
    if [[ "$PROFILING_ADMIN_ONLY" == "1" ]]; then
        echo "GPU performance counters are still restricted on this node." >&2
        exit 1
    fi
fi

mkdir -p "$RESULTS_DIR" "$FIT_OUTPUT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Participant: ${SUBJECT}"
echo "Num estimates: ${NUM_ESTIMATES}"
echo "Profile: ${PROFILE_REPORT}"
"$NCU" --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo "Collecting one full-population simulation-kernel launch"
"$NCU" \
    --target-processes all \
    --kernel-name 'regex:pnl_batched_coevolving_graph_kernel' \
    --launch-skip 1 \
    --launch-count 1 \
    --replay-mode kernel \
    --set full \
    --apply-rules yes \
    --clock-control none \
    --force-overwrite \
    --export "$PROFILE_BASE" \
    "$PYTHON" -u "$DRIVER" \
        --backend triton \
        --subject-id "$SUBJECT" \
        --data-file "$DATA_FILE" \
        --num-estimates "$NUM_ESTIMATES" \
        --max-iterations 12 \
        --max-steps 1200 \
        --bins 100 \
        --smoothing-sigma 0.5 \
        --pseudocount 0.1 \
        --parameter-batch-size 11 \
        --cpu-count "${SLURM_CPUS_PER_TASK:-8}" \
        --optimizer-seed 1 \
        --simulation-seed 1 \
        --triton-block-size 32 \
        --triton-num-warps 1 \
        --skip-posterior-predictive \
        --output-dir "$FIT_OUTPUT" \
        --run-label "ncu_${NUM_ESTIMATES}estimates"

if [[ ! -f "$PROFILE_REPORT" ]]; then
    echo "Nsight Compute report was not produced: $PROFILE_REPORT" >&2
    exit 1
fi

echo "Exporting text and CSV summaries"
if ! "$NCU" --import "$PROFILE_REPORT" --page details > "$RESULTS_DIR/csi-kernel-details.txt"; then
    echo "Warning: unable to export the details page; the .ncu-rep is intact." >&2
fi
if ! "$NCU" --import "$PROFILE_REPORT" --page raw --csv > "$RESULTS_DIR/csi-kernel-raw.csv"; then
    echo "Warning: unable to export raw CSV; the .ncu-rep is intact." >&2
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "Profile report: ${PROFILE_REPORT}"
