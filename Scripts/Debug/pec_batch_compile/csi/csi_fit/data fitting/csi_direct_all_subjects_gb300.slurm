#!/bin/bash
#SBATCH --job-name=csi-direct-subjects
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-97
#SBATCH --output=logs/csi-direct-subjects-%A_%a.out
#SBATCH --error=logs/csi-direct-subjects-%A_%a.err

set -euo pipefail

# Fit every empirical CSI participant with the continuous direct likelihood.
# The array index is the actual subject_nr value in data_to_fit_study3.csv.
# Each task uses the corresponding previous direct fit and legacy PNL fit as
# warm starts.  Solutions that touched the old gain ceiling also produce
# one-coordinate starts at gains 60 and 80.  The default and screened random
# starts fill any remaining slots in the eight-start budget.
# Results are written atomically and completed subjects are skipped by default,
# so the array can be safely resubmitted after interruption.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/Scripts/Debug/pec_batch_compile/csi/csi_fit/data fitting"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/../csi_direct_likelihood.py"
DATA_FILE="${CSI_DATA_FILE:-$SCRIPT_DIR/data_to_fit_study3.csv}"
PNL_FITS="${CSI_PNL_FITS:-$SCRIPT_DIR/fits}"
DIRECT_WARM_ROOT="${CSI_DIRECT_WARM_ROOT:-$SCRIPT_DIR/audit/direct-all-subjects-local-pilot}"
RESULTS_ROOT="${CSI_RESULTS_ROOT:-$SCRIPT_DIR/audit/direct-all-subjects-expanded-${SLURM_ARRAY_JOB_ID:-mapping}}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SUBJECT="${SLURM_ARRAY_TASK_ID:-1}"
if (( SUBJECT < 1 || SUBJECT > 97 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 1 and 97; got $SUBJECT" >&2
    exit 1
fi

STARTS="${CSI_DIRECT_STARTS:-8}"
MAX_ITERATIONS="${CSI_DIRECT_MAX_ITERATIONS:-200}"
RANDOM_START_CANDIDATES="${CSI_DIRECT_RANDOM_START_CANDIDATES:-32}"
POLISH_RESTARTS="${CSI_DIRECT_POLISH_RESTARTS:-2}"
POLISH_ITERATIONS="${CSI_DIRECT_POLISH_ITERATIONS:-150}"
OPTIMIZER_SEED=$((30000 + SUBJECT))

PNL_START="$PNL_FITS/expectation_3.2_real_sequences_single_csi_leak12_sub${SUBJECT}_triton.csv"
DIRECT_START="$DIRECT_WARM_ROOT/subject-$SUBJECT/fit.json"
OUTPUT_DIR="$RESULTS_ROOT/subject-$SUBJECT"
OUTPUT_FILE="$OUTPUT_DIR/fit.json"
TEMP_OUTPUT="$OUTPUT_DIR/fit.json.partial-${SLURM_JOB_ID:-local}"
SCORE_FILE="$OUTPUT_DIR/fresh-score.json"
TEMP_SCORE="$OUTPUT_DIR/fresh-score.json.partial-${SLURM_JOB_ID:-local}"
RESOURCE_LOG="$OUTPUT_DIR/resources.txt"

command=(
    "$PYTHON" -u "$DRIVER" fit
    --data "$DATA_FILE"
    --subject "$SUBJECT"
    --device cpu
    --ddm-time-step 0.001
    --ddm-spatial-points 65
    --lca-max-step 0.01
    --lca-integration-method rk4
    --ddm-bucket-size 256
    --native-lca-scan
    --native-ddm-forward
    --optimizer lbfgsb
    --initial-parameters "$DIRECT_START"
    --initial-parameters "$PNL_START"
    --gain-upper-bound 120
    --threshold-upper-bound 0.30
    --non-decision-time-upper-bound 0.50
    --gain-parameterization linear
    --gain-lift-value 60
    --gain-lift-value 80
    --gain-lift-source-bound 35
    --starts "$STARTS"
    --max-iterations "$MAX_ITERATIONS"
    --random-start-candidates "$RANDOM_START_CANDIDATES"
    --polish-restarts "$POLISH_RESTARTS"
    --polish-iterations "$POLISH_ITERATIONS"
    --coordinate-step 0.01
    --coordinate-levels 4
    --coordinate-rounds 12
    --coordinate-cycles 2
    --seed "$OPTIMIZER_SEED"
    --gradient-method autograd
    --output "$TEMP_OUTPUT"
)

echo "Subject: $SUBJECT"
echo "Optimizer seed: $OPTIMIZER_SEED"
echo "Direct warm start: $DIRECT_START"
echo "PNL warm start: $PNL_START"
echo "Output: $OUTPUT_FILE"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${CSI_MAPPING_ONLY:-0}" == "1" ]]; then
    exit 0
fi

for required in \
    "$PYTHON" \
    "$DRIVER" \
    "$DATA_FILE" \
    "$DIRECT_START" \
    "$PNL_START"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done

if [[ "${CSI_VALIDATE_ONLY:-0}" == "1" ]]; then
    echo "Validated all required paths for subject $SUBJECT."
    exit 0
fi

FIT_NEEDED=1
if [[ -s "$OUTPUT_FILE" && "${CSI_FORCE:-0}" != "1" ]]; then
    if "$PYTHON" -c 'import json, sys; json.load(open(sys.argv[1]))' "$OUTPUT_FILE"; then
        echo "Completed fit already exists; retaining subject $SUBJECT."
        FIT_NEEDED=0
    else
        echo "Existing result is not valid JSON; rerunning subject $SUBJECT." >&2
    fi
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$TEMP_OUTPUT" "$TEMP_SCORE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Git status:"
git -C "$REPO_ROOT" status --short
echo "Experiment source hashes:"
sha256sum \
    "$DRIVER" \
    "$SCRIPT_DIR/../direct_likelihood/fit.py" \
    "$SCRIPT_DIR/../direct_likelihood/model.py" \
    "$DIRECT_START" \
    "$PNL_START" \
    "$SCRIPT_DIR/csi_direct_all_subjects_gb300.slurm"

if (( FIT_NEEDED )); then
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "$RESOURCE_LOG" "${command[@]}"
    else
        "${command[@]}"
    fi
    mv "$TEMP_OUTPUT" "$OUTPUT_FILE"
fi

if [[ ! -s "$SCORE_FILE" || "${CSI_FORCE:-0}" == "1" ]]; then
    "$PYTHON" -u "$DRIVER" score \
        --data "$DATA_FILE" \
        --subject "$SUBJECT" \
        --device cpu \
        --ddm-time-step 0.001 \
        --ddm-spatial-points 65 \
        --lca-max-step 0.01 \
        --lca-integration-method rk4 \
        --ddm-bucket-size 256 \
        --native-lca-scan \
        --native-ddm-forward \
        --parameters "$OUTPUT_FILE" \
        --output "$TEMP_SCORE"
    "$PYTHON" -c '
import json, math, sys
fit = json.load(open(sys.argv[1]))
score = json.load(open(sys.argv[2]))
difference = abs(float(fit["log_likelihood"]) - float(score["log_likelihood"]))
diagnostics = score["diagnostics"]
if difference > 1e-8:
    raise SystemExit(f"fresh score differs from fit by {difference}")
if diagnostics["invalid_included_rows"]:
    raise SystemExit("fresh score contains invalid included rows")
if diagnostics["zero_probability_included_rows"]:
    raise SystemExit("fresh score contains zero-probability included rows")
if not math.isfinite(float(score["log_likelihood"])):
    raise SystemExit("fresh score is not finite")
' "$OUTPUT_FILE" "$TEMP_SCORE"
    mv "$TEMP_SCORE" "$SCORE_FILE"
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "Result: $OUTPUT_FILE"
echo "Fresh score: $SCORE_FILE"
echo "Resources: $RESOURCE_LOG"
