#!/bin/bash
#SBATCH --job-name=csi-fit-100k
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/csi-fit-100k-%j.out
#SBATCH --error=logs/csi-fit-100k-%j.err

set -euo pipefail

# Submit from the "data fitting" directory after creating ./logs.  Defaults to
# participant 1 and optimizer/simulation seed 1.  Override those independently:
#
#   sbatch --export=ALL,CSI_SUBJECT_ID=52,CSI_OPTIMIZER_SEED=2,CSI_SIMULATION_SEED=1 \
#       csi_fit_100k_benchmark.slurm

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"

SUBJECT="${CSI_SUBJECT_ID:-1}"
OPTIMIZER_SEED="${CSI_OPTIMIZER_SEED:-1}"
SIMULATION_SEED="${CSI_SIMULATION_SEED:-1}"
NUM_ESTIMATES=100000
MAX_ITERATIONS="${CSI_MAX_ITERATIONS:-15000}"

RUN_LABEL="benchmark_100k_optimizer_seed${OPTIMIZER_SEED}_simulation_seed${SIMULATION_SEED}_${MAX_ITERATIONS}iter"
RESULTS_ROOT="$SCRIPT_DIR/audit/fit-100k-benchmark-${SLURM_JOB_ID}"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL/sub${SUBJECT}"

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

mkdir -p "$OUTPUT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Participant: ${SUBJECT}"
echo "Optimizer seed: ${OPTIMIZER_SEED}"
echo "Simulation seed: ${SIMULATION_SEED}"
echo "Num estimates: ${NUM_ESTIMATES}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Posterior prediction: skipped"
echo "Results: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo "Starting 100k-estimate fit"
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

echo "Finished: $(date --iso-8601=seconds)"
