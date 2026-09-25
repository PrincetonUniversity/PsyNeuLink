#!/usr/bin/env bash
#SBATCH --job-name=dawa-gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=dawa-%j.log

# Della selects the partition from this resource request. Requesting gpu:a100
# explicitly selects a full A100 rather than a MIG slice.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch [Slurm options] dawa_gpu.slurm fit|recovery [Python driver options]

Required exported variables (absolute paths):
  DAWA_REPO_ROOT   PsyNeuLink checkout containing the DAWA drivers
  DAWA_RESULTS     Writable results directory, preferably on scratch

Optional exported variables:
  DAWA_PYTHON          Interpreter (default: $DAWA_REPO_ROOT/.venv/bin/python)
  DAWA_DATA            CSV (default: bundled model directory/flanker_data_part1.csv)
  DAWA_CUDA_MODULE     CUDA module to load, if your environment needs one
  DAWA_WORK_ROOT       Cache/temp storage (default: $DAWA_RESULTS/.work)
  DAWA_SOURCE_REVISION Base commit for a source snapshot without .git

Driver options such as --subject, --start, --estimates, and --output are passed
through. For arrays, the default subject is SLURM_ARRAY_TASK_ID; otherwise 1.
Each job gets a new default output directory named MODE-JOB_ID under DAWA_RESULTS.
EOF
}

case "${1:-}" in
    fit|recovery) dawa_mode=$1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${DAWA_REPO_ROOT:?Export DAWA_REPO_ROOT before submission}"
: "${DAWA_RESULTS:?Export DAWA_RESULTS before submission}"
export DAWA_PYTHON="${DAWA_PYTHON:-$DAWA_REPO_ROOT/.venv/bin/python}"
export DAWA_DATA="${DAWA_DATA:-$DAWA_REPO_ROOT/Scripts/Debug/pec_batch_compile/dawa/dawa_lca_model/flanker_data_part1.csv}"
export DAWA_WORK_ROOT="${DAWA_WORK_ROOT:-$DAWA_RESULTS/.work}"
for dawa_path in "$DAWA_REPO_ROOT" "$DAWA_RESULTS" "$DAWA_PYTHON" "$DAWA_DATA" "$DAWA_WORK_ROOT"; do
    [[ "$dawa_path" == /* ]] || { printf 'Use absolute paths: %s\n' "$dawa_path" >&2; exit 2; }
done
[[ -x "$DAWA_PYTHON" ]] || { printf 'Python is not executable: %s\n' "$DAWA_PYTHON" >&2; exit 2; }
[[ -f "$DAWA_DATA" ]] || { printf 'CSV not found: %s\n' "$DAWA_DATA" >&2; exit 2; }

if [[ -n "${DAWA_CUDA_MODULE:-}" ]]; then
    module load "$DAWA_CUDA_MODULE"
fi
export PYTHONPATH="$DAWA_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$DAWA_WORK_ROOT/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$DAWA_WORK_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$DAWA_RESULTS" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"

cd "$DAWA_REPO_ROOT"
"$DAWA_PYTHON" - <<'PY'
import os
from pathlib import Path
import socket
import psyneulink
import torch

assert Path(psyneulink.__file__).resolve().is_relative_to(Path(os.environ['DAWA_REPO_ROOT']).resolve()), psyneulink.__file__
assert torch.cuda.is_available(), 'This job requires a visible CUDA GPU'
print(f"Slurm job {os.environ['SLURM_JOB_ID']} on {socket.gethostname()}: {torch.cuda.get_device_name()}", flush=True)
print(f"PsyNeuLink source: {psyneulink.__file__}", flush=True)
PY

dawa_arguments=(
    --data "$DAWA_DATA"
    --subject "${SLURM_ARRAY_TASK_ID:-1}"
    --output "$DAWA_RESULTS/$dawa_mode-$SLURM_JOB_ID"
)
if [[ -n "${DAWA_SOURCE_REVISION:-}" ]]; then
    dawa_arguments+=(--source-revision "$DAWA_SOURCE_REVISION")
fi
exec "$DAWA_PYTHON" -u "Scripts/Debug/pec_batch_compile/dawa/dawa_pec_${dawa_mode}.py" "${dawa_arguments[@]}" "$@"
