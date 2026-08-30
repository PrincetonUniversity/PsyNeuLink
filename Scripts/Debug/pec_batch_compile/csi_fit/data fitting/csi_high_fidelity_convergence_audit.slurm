#!/bin/bash
#SBATCH --job-name=csi-hifi-fit
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=1-48%20
#SBATCH --output=logs/csi-hifi-fit-%A_%a.out
#SBATCH --error=logs/csi-hifi-fit-%A_%a.err

set -euo pipefail

# Submit from the "data fitting" directory after creating ./logs.  The array
# crosses six participants, four CMA-ES optimizer seeds, and two fitting
# simulation seeds (6 * 4 * 2 = 48 fits).  Each candidate is independently
# rescored on the same five one-million-estimate streams for its participant.
# Posterior prediction is deferred until the independently validated winners
# have been selected.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
RESULTS_ROOT="$SCRIPT_DIR/audit/high-fidelity-fit-${SLURM_ARRAY_JOB_ID}"

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

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 1 || TASK_ID > 48 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 48; got ${TASK_ID}" >&2
    exit 1
fi

AUDIT_SUBJECTS=(1 4 27 29 50 60)
OFFSET=$((TASK_ID - 1))
SUBJECT_INDEX=$((OFFSET / 8))
WITHIN_SUBJECT=$((OFFSET % 8))
OPTIMIZER_SEED=$((1 + WITHIN_SUBJECT / 2))
SIMULATION_SEED=$((1 + WITHIN_SUBJECT % 2))
SUBJECT="${AUDIT_SUBJECTS[SUBJECT_INDEX]}"

NUM_ESTIMATES=100000
MAX_ITERATIONS=15000
VALIDATION_NUM_ESTIMATES=1000000
RUN_LABEL="high_fidelity_100k_optimizer_seed${OPTIMIZER_SEED}_simulation_seed${SIMULATION_SEED}_15k"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL/sub${SUBJECT}"
FIT_BASENAME="expectation_3.2_real_sequences_single_csi_leak12_sub${SUBJECT}_triton.csv"
CANDIDATE_FILE="$OUTPUT_DIR/$FIT_BASENAME"

# Every candidate for one participant is evaluated with common random numbers.
# These streams are distinct from fitting and all previous validation audits.
VALIDATION_BASE=$((600000 + SUBJECT * 10))
VALIDATION_SEEDS=(
    $((VALIDATION_BASE + 1))
    $((VALIDATION_BASE + 2))
    $((VALIDATION_BASE + 3))
    $((VALIDATION_BASE + 4))
    $((VALIDATION_BASE + 5))
)

echo "Array task: ${TASK_ID}"
echo "Participant: ${SUBJECT}"
echo "Optimizer seed: ${OPTIMIZER_SEED}"
echo "Fitting simulation seed: ${SIMULATION_SEED}"
echo "Validation seeds: ${VALIDATION_SEEDS[*]}"
echo "Run label: ${RUN_LABEL}"
echo "Results: ${OUTPUT_DIR}"

# Mapping-only mode makes the complete array layout testable without a GPU or
# writing results.  Slurm jobs leave this unset.
if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Fit estimates: ${NUM_ESTIMATES}"
echo "Maximum iterations: ${MAX_ITERATIONS}"
echo "Validation estimates per seed: ${VALIDATION_NUM_ESTIMATES}"
echo "Posterior prediction: deferred"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo "Starting high-fidelity fit"
/usr/bin/time -v "$PYTHON" -u "$DRIVER" \
    --backend triton \
    --subject-id "$SUBJECT" \
    --data-file "$DATA_FILE" \
    --num-estimates "$NUM_ESTIMATES" \
    --max-iterations "$MAX_ITERATIONS" \
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
    --output-dir "$OUTPUT_DIR" \
    --run-label "$RUN_LABEL"

if [[ ! -f "$CANDIDATE_FILE" ]]; then
    echo "Candidate fit file not found: $CANDIDATE_FILE" >&2
    exit 1
fi

echo "Starting paired one-million-estimate rescoring"
"$PYTHON" -u "$DRIVER" \
    --backend triton \
    --subject-id "$SUBJECT" \
    --data-file "$DATA_FILE" \
    --num-estimates "$VALIDATION_NUM_ESTIMATES" \
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
