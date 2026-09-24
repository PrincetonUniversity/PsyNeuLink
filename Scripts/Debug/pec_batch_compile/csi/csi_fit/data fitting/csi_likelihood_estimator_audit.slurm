#!/bin/bash
#SBATCH --job-name=csi-like-audit
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-86%20
#SBATCH --output=logs/csi-like-audit-%A_%a.out
#SBATCH --error=logs/csi-like-audit-%A_%a.err

set -euo pipefail

# Submit from the "data fitting" directory after creating ./logs.  By default,
# this reuses the 86 candidates produced by csi_independent_fit_audit.slurm for
# job 568.  Override that location when needed, for example:
#
#   CSI_SOURCE_AUDIT_ROOT="$PWD/audit/independent-fit-123" sbatch \
#       csi_likelihood_estimator_audit.slurm

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="$SCRIPT_DIR/data_to_fit_study3.csv"
PRODUCTION_FITS="$SCRIPT_DIR/fits"
SOURCE_AUDIT_ROOT="${CSI_SOURCE_AUDIT_ROOT:-$SCRIPT_DIR/audit/independent-fit-568}"
RESULTS_ROOT="$SCRIPT_DIR/audit/likelihood-estimator-${SLURM_ARRAY_JOB_ID}"

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

AUDIT_SUBJECTS=(1 2 4 7 9 10 16 20 27 29 40 41 43 50 52 54 60 67 72 80)
LATE_SUBJECTS=(4 9 27 41 52 54)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SUBJECT=""
SOURCE_LABEL=""

if (( TASK_ID <= 20 )); then
    SUBJECT="${AUDIT_SUBJECTS[TASK_ID - 1]}"
    SOURCE_LABEL="baseline_seed1_5k"
elif (( TASK_ID <= 80 )); then
    OFFSET=$((TASK_ID - 21))
    SUBJECT_INDEX=$((OFFSET / 3))
    OPTIMIZER_SEED=$((2 + OFFSET % 3))
    SUBJECT="${AUDIT_SUBJECTS[SUBJECT_INDEX]}"
    SOURCE_LABEL="optimizer_seed${OPTIMIZER_SEED}_5k"
else
    SUBJECT="${LATE_SUBJECTS[TASK_ID - 81]}"
    SOURCE_LABEL="optimizer_seed1_10k"
fi

FIT_BASENAME="expectation_3.2_real_sequences_single_csi_leak12_sub${SUBJECT}_triton.csv"
if [[ "$SOURCE_LABEL" == "baseline_seed1_5k" ]]; then
    CANDIDATE_FILE="$PRODUCTION_FITS/$FIT_BASENAME"
else
    CANDIDATE_FILE="$SOURCE_AUDIT_ROOT/$SOURCE_LABEL/sub${SUBJECT}/$FIT_BASENAME"
fi

if [[ ! -f "$CANDIDATE_FILE" ]]; then
    echo "Candidate fit file not found: $CANDIDATE_FILE" >&2
    exit 1
fi

# The 50k configurations form a small factorial experiment that separates the
# effects of smoothing and pseudocounts.  The paired 200k runs test convergence
# of the legacy and preferred hybrid estimators rather than assuming that lower
# seed variance alone implies lower bias.
CONFIG_LABELS=(
    raw_50k
    pc05_50k
    s05_50k
    s05_pc01_50k
    s05_pc05_50k
    s10_pc05_50k
    raw_200k
    s05_pc05_200k
)
SMOOTHING_SIGMAS=(0.0 0.0 0.5 0.5 0.5 1.0 0.0 0.5)
PSEUDOCOUNTS=(0.0 0.5 0.0 0.1 0.5 0.5 0.0 0.5)
NUM_ESTIMATES=(50000 50000 50000 50000 50000 50000 200000 200000)
NUM_SEEDS=(5 5 5 5 5 5 3 3)

VALIDATION_BASE=$((300000 + SUBJECT * 10))

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Participant: ${SUBJECT}"
echo "Source label: ${SOURCE_LABEL}"
echo "Candidate: ${CANDIDATE_FILE}"
echo "Results: ${RESULTS_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

for INDEX in "${!CONFIG_LABELS[@]}"; do
    CONFIG_LABEL="${CONFIG_LABELS[INDEX]}"
    SIGMA="${SMOOTHING_SIGMAS[INDEX]}"
    ALPHA="${PSEUDOCOUNTS[INDEX]}"
    ESTIMATES="${NUM_ESTIMATES[INDEX]}"
    SEED_COUNT="${NUM_SEEDS[INDEX]}"
    VALIDATION_SEEDS=()
    for ((SEED_INDEX = 1; SEED_INDEX <= SEED_COUNT; SEED_INDEX++)); do
        VALIDATION_SEEDS+=("$((VALIDATION_BASE + SEED_INDEX))")
    done

    RUN_LABEL="${SOURCE_LABEL}__${CONFIG_LABEL}"
    OUTPUT_DIR="$RESULTS_ROOT/$CONFIG_LABEL/$SOURCE_LABEL/sub${SUBJECT}"
    mkdir -p "$OUTPUT_DIR"

    echo
    echo "Configuration: ${CONFIG_LABEL}"
    echo "Smoothing sigma: ${SIGMA}"
    echo "Pseudocount: ${ALPHA}"
    echo "Num estimates: ${ESTIMATES}"
    echo "Validation seeds: ${VALIDATION_SEEDS[*]}"

    "$PYTHON" -u "$DRIVER" \
        --backend triton \
        --subject-id "$SUBJECT" \
        --data-file "$DATA_FILE" \
        --num-estimates "$ESTIMATES" \
        --max-iterations 1 \
        --max-steps 1200 \
        --bins 100 \
        --smoothing-sigma "$SIGMA" \
        --pseudocount "$ALPHA" \
        --parameter-batch-size 11 \
        --cpu-count "${SLURM_CPUS_PER_TASK:-8}" \
        --triton-block-size 32 \
        --triton-num-warps 1 \
        --skip-posterior-predictive \
        --rescore-parameter-file "$CANDIDATE_FILE" \
        --rescore-simulation-seeds "${VALIDATION_SEEDS[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --run-label "$RUN_LABEL"
done

echo "Finished: $(date --iso-8601=seconds)"
