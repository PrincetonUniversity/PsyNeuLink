#!/bin/bash
#SBATCH --job-name=csi-fit-audit
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1-86%20
#SBATCH --output=logs/csi-fit-audit-%A_%a.out
#SBATCH --error=logs/csi-fit-audit-%A_%a.err

set -euo pipefail

# Submit from the "data fitting" directory after ensuring that ./logs exists.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="$SCRIPT_DIR/data_to_fit_study3.csv"
PRODUCTION_FITS="$SCRIPT_DIR/fits"
RESULTS_ROOT="$SCRIPT_DIR/audit/independent-fit-${SLURM_ARRAY_JOB_ID}"

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Targeted audit set:
#   late improvements: 4 9 27 41 52 54
#   poor predictive checks: 1 2 29 43 50
#   boundary estimates: 7 16 67 72
#   controls: 10 20 40 60 80
AUDIT_SUBJECTS=(1 2 4 7 9 10 16 20 27 29 40 41 43 50 52 54 60 67 72 80)
LATE_SUBJECTS=(4 9 27 41 52 54)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODE=""
SUBJECT=""
OPTIMIZER_SEED=1
FIT_BUDGET=5000
RUN_LABEL=""

if (( TASK_ID <= 20 )); then
    # Rescore the existing production seed-1, 5k candidate.
    MODE="rescore-existing"
    SUBJECT="${AUDIT_SUBJECTS[TASK_ID - 1]}"
    RUN_LABEL="baseline_seed1_5k"
elif (( TASK_ID <= 80 )); then
    # Three additional CMA-ES seeds for each audit participant.
    OFFSET=$((TASK_ID - 21))
    SUBJECT_INDEX=$((OFFSET / 3))
    OPTIMIZER_SEED=$((2 + OFFSET % 3))
    SUBJECT="${AUDIT_SUBJECTS[SUBJECT_INDEX]}"
    RUN_LABEL="optimizer_seed${OPTIMIZER_SEED}_5k"
    MODE="fit-and-rescore"
else
    # A longer seed-1 fit for participants that improved late.
    SUBJECT="${LATE_SUBJECTS[TASK_ID - 81]}"
    FIT_BUDGET=10000
    RUN_LABEL="optimizer_seed1_10k"
    MODE="fit-and-rescore"
fi

OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL/sub${SUBJECT}"
mkdir -p "$OUTPUT_DIR"

FIT_BASENAME="expectation_3.2_real_sequences_single_csi_leak12_sub${SUBJECT}_triton.csv"
if [[ "$MODE" == "rescore-existing" ]]; then
    CANDIDATE_FILE="$PRODUCTION_FITS/$FIT_BASENAME"
else
    CANDIDATE_FILE="$OUTPUT_DIR/$FIT_BASENAME"
fi

# Use the same five validation streams for every candidate belonging to a
# participant, while keeping streams distinct between participants.
VALIDATION_BASE=$((200000 + SUBJECT * 10))
VALIDATION_SEEDS=(
    $((VALIDATION_BASE + 1))
    $((VALIDATION_BASE + 2))
    $((VALIDATION_BASE + 3))
    $((VALIDATION_BASE + 4))
    $((VALIDATION_BASE + 5))
)

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Mode: ${MODE}"
echo "Participant: ${SUBJECT}"
echo "Run label: ${RUN_LABEL}"
echo "Optimizer seed: ${OPTIMIZER_SEED}"
echo "Fitting simulation seed: 1"
echo "Fit budget: ${FIT_BUDGET}"
echo "Validation seeds: ${VALIDATION_SEEDS[*]}"
echo "Results: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [[ "$MODE" == "fit-and-rescore" ]]; then
    echo "Starting fit"
    "$PYTHON" -u "$DRIVER" \
        --backend triton \
        --subject-id "$SUBJECT" \
        --data-file "$DATA_FILE" \
        --num-estimates 10000 \
        --max-iterations "$FIT_BUDGET" \
        --max-steps 1200 \
        --bins 100 \
        --parameter-batch-size 11 \
        --cpu-count "${SLURM_CPUS_PER_TASK:-8}" \
        --optimizer-seed "$OPTIMIZER_SEED" \
        --simulation-seed 1 \
        --triton-block-size 32 \
        --triton-num-warps 1 \
        --posterior-predictive-simulations 100 \
        --posterior-predictive-seed "$((100000 + SUBJECT))" \
        --output-dir "$OUTPUT_DIR" \
        --run-label "$RUN_LABEL"
fi

if [[ ! -f "$CANDIDATE_FILE" ]]; then
    echo "Candidate fit file not found: $CANDIDATE_FILE" >&2
    exit 1
fi

echo "Starting fixed-parameter likelihood rescoring"
"$PYTHON" -u "$DRIVER" \
    --backend triton \
    --subject-id "$SUBJECT" \
    --data-file "$DATA_FILE" \
    --num-estimates 50000 \
    --max-iterations 1 \
    --max-steps 1200 \
    --bins 100 \
    --parameter-batch-size 11 \
    --cpu-count "${SLURM_CPUS_PER_TASK:-8}" \
    --optimizer-seed "$OPTIMIZER_SEED" \
    --simulation-seed 1 \
    --triton-block-size 32 \
    --triton-num-warps 1 \
    --skip-posterior-predictive \
    --rescore-parameter-file "$CANDIDATE_FILE" \
    --rescore-simulation-seeds "${VALIDATION_SEEDS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --run-label "$RUN_LABEL"

echo "Finished: $(date --iso-8601=seconds)"
