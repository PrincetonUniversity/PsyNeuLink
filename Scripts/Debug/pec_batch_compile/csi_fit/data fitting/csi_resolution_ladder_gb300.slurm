#!/bin/bash
#SBATCH --job-name=csi-resolution
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-99%20
#SBATCH --output=logs/csi-resolution-%A_%a.out
#SBATCH --error=logs/csi-resolution-%A_%a.err

set -euo pipefail

# Prepared for review only. Do not submit before the investigator approves the
# seed count, recovery budget, and one-GPU allocation for every array task.
#
# Tasks 1-90 estimate the generative resolution effect:
#   30 seeds x (continuous, GPU 1 ms, GPU 10 ms).
# Tasks 91-99 complete the recovery matrix:
#   GPU 1 ms generation x 3 anchor seeds x (direct, GPU 1 ms, GPU 10 ms).
# Together with fitting-readiness job 1072, the recovery cells form a
# 3-generator x 3-fitter x 3-seed crossed comparison.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/resolution-ladder-${SLURM_ARRAY_JOB_ID:-mapping}}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for required in "$PYTHON" "$DRIVER" "$DATA_FILE"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 99 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 99; got $TASK_ID" >&2
    exit 1
fi

GENERATORS=(continuous gpu-1ms gpu-10ms)
FITTERS=(direct gpu-1ms gpu-10ms)
SEEDS=(
    17 43 79 101 127 149 173 197 223 251
    277 307 337 367 397 431 463 499 541 577
    613 653 691 733 773 811 853 887 929 971
)
ANCHOR_SEEDS=(17 43 79)
OPTIMIZER_SEEDS=(100 101 102)
SUBJECT="${CSI_TEMPLATE_SUBJECT:-1}"

if (( TASK_ID <= 90 )); then
    OFFSET=$((TASK_ID - 1))
    GENERATOR_INDEX=$((OFFSET / 30))
    SEED_INDEX=$((OFFSET % 30))
    GENERATOR="${GENERATORS[GENERATOR_INDEX]}"
    SIMULATION_SEED="${SEEDS[SEED_INDEX]}"
    OUTPUT_DIR="$RESULTS_ROOT/generation/$GENERATOR/seed-$SIMULATION_SEED"
    command=(
        "$PYTHON" -u "$DRIVER" generate
        --generator "$GENERATOR"
        --data "$DATA_FILE"
        --subject "$SUBJECT"
        --simulation-seed "$SIMULATION_SEED"
        --generation-maximum-time 12
        --output "$OUTPUT_DIR/synthetic_data.csv"
        --metadata-output "$OUTPUT_DIR/generation.json"
    )
    DESCRIPTION="generation: $GENERATOR, seed $SIMULATION_SEED"
else
    OFFSET=$((TASK_ID - 91))
    SEED_INDEX=$((OFFSET / 3))
    FITTER_INDEX=$((OFFSET % 3))
    SIMULATION_SEED="${ANCHOR_SEEDS[SEED_INDEX]}"
    OPTIMIZER_SEED="${OPTIMIZER_SEEDS[SEED_INDEX]}"
    FITTER="${FITTERS[FITTER_INDEX]}"
    LIKELIHOOD_SEED=$((100000 + SIMULATION_SEED))
    RUN_LABEL="gpu-1ms__${FITTER}__seed${SIMULATION_SEED}"
    OUTPUT_DIR="$RESULTS_ROOT/recovery/$RUN_LABEL"
    command=(
        "$PYTHON" -u "$DRIVER" run-cell
        --generator gpu-1ms
        --fitter "$FITTER"
        --data "$DATA_FILE"
        --subject "$SUBJECT"
        --simulation-seed "$SIMULATION_SEED"
        --optimizer-seed "$OPTIMIZER_SEED"
        --likelihood-seed "$LIKELIHOOD_SEED"
        --generation-maximum-time 12
        --gpu-fit-maximum-time 12
        --gpu-num-estimates 100000
        --gpu-max-iterations 5000
        --gpu-smoothing-sigma 0.5
        --gpu-pseudocount 0.1
        --gpu-parameter-batch-size 11
        --direct-starts 2
        --direct-max-iterations 100
        --direct-random-start-candidates 32
        --output-dir "$OUTPUT_DIR"
        --run-label "$RUN_LABEL"
    )
    DESCRIPTION="recovery: GPU 1 ms -> $FITTER, seed $SIMULATION_SEED"
fi

echo "Array task: $TASK_ID"
echo "Experiment: $DESCRIPTION"
echo "Template subject: $SUBJECT"
echo "Output: $OUTPUT_DIR"

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
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
