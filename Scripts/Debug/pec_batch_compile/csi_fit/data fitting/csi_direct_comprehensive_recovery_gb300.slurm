#!/bin/bash
#SBATCH --job-name=csi-direct-recovery
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-128%20
#SBATCH --output=logs/csi-direct-recovery-%A_%a.out
#SBATCH --error=logs/csi-direct-recovery-%A_%a.err

set -euo pipefail

# Comprehensive continuous-generation/direct-likelihood recovery study:
#   tasks   1-50: five multivariate regimes x ten simulated datasets
#   tasks 51-128: low/high perturbations of 13 parameters x three datasets

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/direct-comprehensive-recovery-${SLURM_ARRAY_JOB_ID:-mapping}}"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 128 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 128; got $TASK_ID" >&2
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

if (( TASK_ID <= 50 )); then
    OFFSET=$((TASK_ID - 1))
    PROFILE_INDEX=$((OFFSET / 10))
    SEED_INDEX=$((OFFSET % 10))
    TRUTH_PROFILE="${REGIME_PROFILES[PROFILE_INDEX]}"
    SIMULATION_SEED="${REGIME_SEEDS[SEED_INDEX]}"
    STUDY_BLOCK="regime-replication"
else
    OFFSET=$((TASK_ID - 51))
    PROFILE_INDEX=$((OFFSET / 3))
    SEED_INDEX=$((OFFSET % 3))
    TRUTH_PROFILE="${OAT_PROFILES[PROFILE_INDEX]}"
    SIMULATION_SEED="${OAT_SEEDS[SEED_INDEX]}"
    STUDY_BLOCK="one-at-a-time"
fi

OPTIMIZER_SEED=$((20000 + TASK_ID))
RUN_LABEL="truth-${TRUTH_PROFILE}__continuous__direct__seed${SIMULATION_SEED}__optimizer${OPTIMIZER_SEED}"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL"

command=(
    "$PYTHON" -u "$DRIVER" run-cell
    --truth-profile "$TRUTH_PROFILE"
    --generator continuous
    --fitter direct
    --data "$DATA_FILE"
    --subject 1
    --simulation-seed "$SIMULATION_SEED"
    --continuous-simulation-time-step 0.0005
    --generation-maximum-time 12.0
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
echo "Study block: $STUDY_BLOCK"
echo "Truth profile: $TRUTH_PROFILE"
echo "Simulation seed: $SIMULATION_SEED"
echo "Optimizer seed: $OPTIMIZER_SEED"
echo "Output: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    exit 0
fi

for required in "$PYTHON" "$DRIVER" "$DATA_FILE"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Experiment source hashes:"
sha256sum \
    "$DRIVER" \
    "$SCRIPT_DIR/../direct_likelihood/fit.py" \
    "$SCRIPT_DIR/csi_direct_comprehensive_recovery_gb300.slurm"

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "${command[@]}"
else
    "${command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
