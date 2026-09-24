#!/bin/bash
#SBATCH --job-name=csi-smooth-fit
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=1-80%20
#SBATCH --output=logs/csi-smooth-fit-%A_%a.out
#SBATCH --error=logs/csi-smooth-fit-%A_%a.err

set -euo pipefail

# Submit from the "data fitting" directory after creating ./logs.  Tasks 1-60
# fit 20 selected participants from three independent optimizer/simulation
# seeds.  Tasks 61-80 rescore the best candidate from the earlier independent
# fit audit as a control.  Every candidate is scored with the same three 200k
# validation streams for its participant.
#
# The control candidates default to independent-fit job 568.  Override that
# location if the earlier audit has a different job ID, for example:
#
#   CSI_SOURCE_AUDIT_ROOT="$PWD/audit/independent-fit-123" sbatch \
#       csi_smoothed_fit_audit.slurm

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="$SCRIPT_DIR/data_to_fit_study3.csv"
PRODUCTION_FITS="$SCRIPT_DIR/fits"
SOURCE_AUDIT_ROOT="${CSI_SOURCE_AUDIT_ROOT:-$SCRIPT_DIR/audit/independent-fit-568}"
RESULTS_ROOT="$SCRIPT_DIR/audit/smoothed-fit-${SLURM_ARRAY_JOB_ID}"

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

# Best candidate for each participant under the sigma=0.5, pseudocount=0.1
# estimator audit.  Entries correspond positionally to AUDIT_SUBJECTS.
CONTROL_LABELS=(
    optimizer_seed3_5k
    optimizer_seed4_5k
    optimizer_seed1_10k
    optimizer_seed4_5k
    baseline_seed1_5k
    optimizer_seed2_5k
    baseline_seed1_5k
    optimizer_seed4_5k
    optimizer_seed1_10k
    optimizer_seed2_5k
    baseline_seed1_5k
    optimizer_seed1_10k
    baseline_seed1_5k
    baseline_seed1_5k
    optimizer_seed3_5k
    optimizer_seed1_10k
    optimizer_seed2_5k
    baseline_seed1_5k
    baseline_seed1_5k
    optimizer_seed2_5k
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODE=""
SUBJECT=""
OPTIMIZER_SEED=1
SIMULATION_SEED=1
SOURCE_LABEL=""
RUN_LABEL=""

if (( TASK_ID <= 60 )); then
    OFFSET=$((TASK_ID - 1))
    SUBJECT_INDEX=$((OFFSET / 3))
    OPTIMIZER_SEED=$((1 + OFFSET % 3))
    SIMULATION_SEED="$OPTIMIZER_SEED"
    SUBJECT="${AUDIT_SUBJECTS[SUBJECT_INDEX]}"
    MODE="fit-and-rescore"
    RUN_LABEL="smooth_s05_pc01_optimizer_seed${OPTIMIZER_SEED}_simulation_seed${SIMULATION_SEED}_15k"
    OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL/sub${SUBJECT}"
else
    SUBJECT_INDEX=$((TASK_ID - 61))
    SUBJECT="${AUDIT_SUBJECTS[SUBJECT_INDEX]}"
    SOURCE_LABEL="${CONTROL_LABELS[SUBJECT_INDEX]}"
    MODE="rescore-control"
    RUN_LABEL="${SOURCE_LABEL}__control_s05_pc01_200k"
    OUTPUT_DIR="$RESULTS_ROOT/control/sub${SUBJECT}"
fi

mkdir -p "$OUTPUT_DIR"

FIT_BASENAME="expectation_3.2_real_sequences_single_csi_leak12_sub${SUBJECT}_triton.csv"
if [[ "$MODE" == "fit-and-rescore" ]]; then
    CANDIDATE_FILE="$OUTPUT_DIR/$FIT_BASENAME"
elif [[ "$SOURCE_LABEL" == "baseline_seed1_5k" ]]; then
    CANDIDATE_FILE="$PRODUCTION_FITS/$FIT_BASENAME"
else
    CANDIDATE_FILE="$SOURCE_AUDIT_ROOT/$SOURCE_LABEL/sub${SUBJECT}/$FIT_BASENAME"
fi

# Pair all candidates for a participant on identical validation streams.  The
# streams are separate from those used during fitting and in the earlier audit.
VALIDATION_BASE=$((400000 + SUBJECT * 10))
VALIDATION_SEEDS=(
    $((VALIDATION_BASE + 1))
    $((VALIDATION_BASE + 2))
    $((VALIDATION_BASE + 3))
)

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Mode: ${MODE}"
echo "Participant: ${SUBJECT}"
echo "Run label: ${RUN_LABEL}"
echo "Optimizer seed: ${OPTIMIZER_SEED}"
echo "Fitting simulation seed: ${SIMULATION_SEED}"
echo "Validation seeds: ${VALIDATION_SEEDS[*]}"
echo "Candidate: ${CANDIDATE_FILE}"
echo "Results: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [[ "$MODE" == "fit-and-rescore" ]]; then
    echo "Starting smoothed fit"
    "$PYTHON" -u "$DRIVER" \
        --backend triton \
        --subject-id "$SUBJECT" \
        --data-file "$DATA_FILE" \
        --num-estimates 10000 \
        --max-iterations 15000 \
        --max-steps 1200 \
        --bins 100 \
        --smoothing-sigma 0.5 \
        --pseudocount 0.1 \
        --parameter-batch-size 11 \
        --cpu-count "${SLURM_CPUS_PER_TASK:-8}" \
        --optimizer-seed "$OPTIMIZER_SEED" \
        --simulation-seed "$SIMULATION_SEED" \
        --triton-block-size 32 \
        --triton-num-warps 1 \
        --posterior-predictive-simulations 100 \
        --posterior-predictive-seed "$((500000 + SUBJECT * 10 + OPTIMIZER_SEED))" \
        --output-dir "$OUTPUT_DIR" \
        --run-label "$RUN_LABEL"
fi

if [[ ! -f "$CANDIDATE_FILE" ]]; then
    echo "Candidate fit file not found: $CANDIDATE_FILE" >&2
    exit 1
fi

echo "Starting paired 200k fixed-parameter likelihood rescoring"
"$PYTHON" -u "$DRIVER" \
    --backend triton \
    --subject-id "$SUBJECT" \
    --data-file "$DATA_FILE" \
    --num-estimates 200000 \
    --max-iterations 1 \
    --max-steps 1200 \
    --bins 100 \
    --smoothing-sigma 0.5 \
    --pseudocount 0.1 \
    --parameter-batch-size 11 \
    --cpu-count "${SLURM_CPUS_PER_TASK:-8}" \
    --optimizer-seed "$OPTIMIZER_SEED" \
    --simulation-seed "$SIMULATION_SEED" \
    --triton-block-size 32 \
    --triton-num-warps 1 \
    --skip-posterior-predictive \
    --rescore-parameter-file "$CANDIDATE_FILE" \
    --rescore-simulation-seeds "${VALIDATION_SEEDS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --run-label "$RUN_LABEL"

echo "Finished: $(date --iso-8601=seconds)"
