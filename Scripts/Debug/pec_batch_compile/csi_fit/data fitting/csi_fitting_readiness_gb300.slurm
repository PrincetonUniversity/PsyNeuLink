#!/bin/bash
#SBATCH --job-name=csi-recovery
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-18%18
#SBATCH --output=logs/csi-recovery-%A_%a.out
#SBATCH --error=logs/csi-recovery-%A_%a.err

set -euo pipefail

# This file is intentionally only a submission template. Do not submit it
# until its account/partition/GRES settings and experiment size have been
# reviewed with the investigator.
#
# The 18 array tasks cross:
#   3 data seeds x 2 generators (continuous, GPU 10 ms)
#                x 3 fitters (direct, GPU 1 ms, GPU 10 ms).
# Set CSI_TRUTH_PARAMETERS to a direct-likelihood JSON or a one-row legacy fit
# CSV to run another truth vector. Separate Slurm submissions should use
# separate CSI_RESULTS_ROOT values.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_fitting_readiness.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/fitting-readiness-${SLURM_ARRAY_JOB_ID:-mapping}}"
# The TigerAI virtual environment is shared with another checkout. Put this
# immutable run worktree first so imports match the recorded Git revision.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$DRIVER" ]]; then
    echo "Readiness driver not found: $DRIVER" >&2
    exit 1
fi
if [[ ! -f "$DATA_FILE" ]]; then
    echo "CSI template data not found: $DATA_FILE" >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > 18 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 18; got $TASK_ID" >&2
    exit 1
fi

GENERATORS=(continuous gpu-10ms)
FITTERS=(direct gpu-1ms gpu-10ms)
SIMULATION_SEEDS=(17 43 79)

OFFSET=$((TASK_ID - 1))
SEED_INDEX=$((OFFSET / 6))
WITHIN_SEED=$((OFFSET % 6))
GENERATOR_INDEX=$((WITHIN_SEED / 3))
FITTER_INDEX=$((WITHIN_SEED % 3))

GENERATOR="${GENERATORS[GENERATOR_INDEX]}"
FITTER="${FITTERS[FITTER_INDEX]}"
SIMULATION_SEED="${SIMULATION_SEEDS[SEED_INDEX]}"
OPTIMIZER_SEED=$((100 + SEED_INDEX))
LIKELIHOOD_SEED=$((100000 + SIMULATION_SEED))
SUBJECT="${CSI_TEMPLATE_SUBJECT:-1}"
RUN_LABEL="${GENERATOR}__${FITTER}__seed${SIMULATION_SEED}"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_LABEL"

truth_args=()
if [[ -n "${CSI_TRUTH_PARAMETERS:-}" ]]; then
    truth_args=(--truth-parameters "$CSI_TRUTH_PARAMETERS")
fi

command=(
    "$PYTHON" -u "$DRIVER" run-cell
    --generator "$GENERATOR"
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
    "${truth_args[@]}"
)

echo "Array task: $TASK_ID"
echo "Generator: $GENERATOR"
echo "Fitter: $FITTER"
echo "Simulation seed: $SIMULATION_SEED"
echo "Optimizer seed: $OPTIMIZER_SEED"
echo "Likelihood seed: $LIKELIHOOD_SEED"
echo "Template subject: $SUBJECT"
echo "Results: $OUTPUT_DIR"

# Mapping-only mode validates every array coordinate without compiling a model,
# allocating a GPU, or writing result files.
if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    "${command[@]}" --dry-run
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
