#!/bin/bash
#SBATCH --job-name=csi-gpu1ms-recovery
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=1-384%20
#SBATCH --output=logs/csi-gpu1ms-recovery-%A_%a.out
#SBATCH --error=logs/csi-gpu1ms-recovery-%A_%a.err

set -euo pipefail

# GPU 1 ms counterpart to csi_direct_comprehensive_recovery_gb300.slurm:
#   128 continuous synthetic datasets x 3 likelihood seeds = 384 fits.
#
# Each likelihood repeat reuses the exact synthetic CSV and generation
# metadata from direct-recovery job 1345.  The optimizer seed is held fixed
# within a dataset, while the Monte Carlo likelihood seed changes.  This
# isolates likelihood-estimator variability in the three-repeat comparison.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
GPU_FIT_DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
SOURCE_ROOT="${CSI_SOURCE_RESULTS_ROOT:-$SCRIPT_DIR/audit/direct-comprehensive-recovery-1345}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/gpu1ms-comprehensive-recovery-${SLURM_ARRAY_JOB_ID:-mapping}}"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 384 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 384; got $TASK_ID" >&2
    exit 1
fi

REGIME_PROFILES=(
    baseline
    low-control-fast
    high-control
    strong-collapse
    condition-contrast
)
REGIME_SEEDS=(17 43 79 101 137 181 223 269 307 349)
OAT_PROFILES=(
    oat-gain-ni-low oat-gain-ni-high
    oat-gain-rr-low oat-gain-rr-high
    oat-gain-rf-low oat-gain-rf-high
    oat-csi-low oat-csi-high
    oat-threshold-ni-low oat-threshold-ni-high
    oat-threshold-rr-low oat-threshold-rr-high
    oat-threshold-rf-low oat-threshold-rf-high
    oat-collapse-ni-low oat-collapse-ni-high
    oat-collapse-rr-low oat-collapse-rr-high
    oat-collapse-rf-low oat-collapse-rf-high
    oat-ndt-ni-low oat-ndt-ni-high
    oat-ndt-rr-low oat-ndt-rr-high
    oat-ndt-rf-low oat-ndt-rf-high
)
OAT_SEEDS=(17 137 269)

OFFSET=$((TASK_ID - 1))
BASE_TASK=$((OFFSET / 3 + 1))
LIKELIHOOD_REPLICATE=$((OFFSET % 3 + 1))

if (( BASE_TASK <= 50 )); then
    BASE_OFFSET=$((BASE_TASK - 1))
    PROFILE_INDEX=$((BASE_OFFSET / 10))
    SEED_INDEX=$((BASE_OFFSET % 10))
    TRUTH_PROFILE="${REGIME_PROFILES[PROFILE_INDEX]}"
    SIMULATION_SEED="${REGIME_SEEDS[SEED_INDEX]}"
    STUDY_BLOCK="regime-replication"
else
    BASE_OFFSET=$((BASE_TASK - 51))
    PROFILE_INDEX=$((BASE_OFFSET / 3))
    SEED_INDEX=$((BASE_OFFSET % 3))
    TRUTH_PROFILE="${OAT_PROFILES[PROFILE_INDEX]}"
    SIMULATION_SEED="${OAT_SEEDS[SEED_INDEX]}"
    STUDY_BLOCK="one-at-a-time"
fi

OPTIMIZER_SEED=$((20000 + BASE_TASK))
LIKELIHOOD_SEED=$((LIKELIHOOD_REPLICATE * 100000 + BASE_TASK))
SOURCE_LABEL="truth-${TRUTH_PROFILE}__continuous__direct__seed${SIMULATION_SEED}__optimizer${OPTIMIZER_SEED}"
RUN_LABEL="truth-${TRUTH_PROFILE}__continuous__gpu-1ms__seed${SIMULATION_SEED}__optimizer${OPTIMIZER_SEED}__likelihood${LIKELIHOOD_REPLICATE}"
SOURCE_DIR="$SOURCE_ROOT/$SOURCE_LABEL"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL"

command=(
    "$PYTHON" -u "$DRIVER" run-cell
    --truth-profile "$TRUTH_PROFILE"
    --generator continuous
    --fitter gpu-1ms
    --data "$DATA_FILE"
    --subject 1
    --simulation-seed "$SIMULATION_SEED"
    --continuous-simulation-time-step 0.0005
    --generation-maximum-time 12.0
    --optimizer-seed "$OPTIMIZER_SEED"
    --likelihood-seed "$LIKELIHOOD_SEED"
    --likelihood-replicate "$LIKELIHOOD_REPLICATE"
    --gpu-fit-maximum-time 12.0
    --gpu-num-estimates 100000
    --gpu-max-iterations 5000
    --gpu-bins 100
    --gpu-smoothing-sigma 0.5
    --gpu-pseudocount 0.1
    --gpu-parameter-batch-size 11
    --triton-block-size 32
    --triton-num-warps 1
    --output-dir "$OUTPUT_DIR"
    --run-label "$RUN_LABEL"
)

echo "Array task: $TASK_ID"
echo "Base dataset: $BASE_TASK"
echo "Study block: $STUDY_BLOCK"
echo "Truth profile: $TRUTH_PROFILE"
echo "Simulation seed: $SIMULATION_SEED"
echo "Optimizer seed: $OPTIMIZER_SEED"
echo "Likelihood replicate: $LIKELIHOOD_REPLICATE"
echo "Likelihood seed: $LIKELIHOOD_SEED"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    exit 0
fi

for required in \
    "$PYTHON" \
    "$DRIVER" \
    "$GPU_FIT_DRIVER" \
    "$DATA_FILE" \
    "$SOURCE_DIR/synthetic_data.csv" \
    "$SOURCE_DIR/generation.json"; do
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

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Experiment source hashes:"
sha256sum \
    "$DRIVER" \
    "$GPU_FIT_DRIVER" \
    "$SCRIPT_DIR/csi_gpu1ms_comprehensive_recovery_gb300.slurm"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
