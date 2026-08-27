#!/bin/bash
#SBATCH --job-name=csi-fit-array
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1-97%8
#SBATCH --output=logs/csi-fit-%A_%a.out
#SBATCH --error=logs/csi-fit-%A_%a.err

set -euo pipefail

# Submit this script from the data fitting directory. SLURM_SUBMIT_DIR remains
# stable even though Slurm executes a copied script from its spool directory.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="$SCRIPT_DIR/data_to_fit_study3.csv"

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

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

fit_command=(
    "$PYTHON" -u "$DRIVER"
    --backend triton
    --subject_id "$SLURM_ARRAY_TASK_ID"
    --data_file "$DATA_FILE"
    --num_estimates 10000
    --max_iterations 5000
    --max_steps 1200
    --bins 100
    --parameter_batch_size 11
    --cpu_count "${SLURM_CPUS_PER_TASK:-8}"
    --seed 1
    --triton-block-size 32
    --triton-num-warps 1
    --posterior-predictive-simulations 100
    --posterior-predictive-seed "$((100000 + SLURM_ARRAY_TASK_ID))"
)

echo "Starting participant fit"
if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${fit_command[@]}"
else
    "${fit_command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
