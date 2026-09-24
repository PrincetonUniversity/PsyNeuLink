#!/usr/bin/env bash
# Source this file before setup, local runs, or sbatch. Override CSI_* first.
CSI_HANDOFF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CSI_REPO_ROOT="${CSI_REPO_ROOT:-$(cd "$CSI_HANDOFF_DIR/../../../../../.." && pwd)}"
export CSI_WORK_ROOT="${CSI_WORK_ROOT:-/scratch/gpfs/CSES/$USER/csi-handoff}"
export CSI_VENV="${CSI_VENV:-$CSI_WORK_ROOT/venv}"
export CSI_RESULTS_ROOT="${CSI_RESULTS_ROOT:-$CSI_WORK_ROOT/results}"
export CSI_DATA_FILE="${CSI_DATA_FILE:-$CSI_REPO_ROOT/Scripts/Debug/pec_batch_compile/csi/csi_fit/data fitting/data_to_fit_study3.csv}"

# Resolve symlinks too: a scratch-looking path must not lead back into home.
for csi_path in "$CSI_WORK_ROOT" "$CSI_VENV" "$CSI_RESULTS_ROOT"; do
    case "$(realpath -m -- "$csi_path")/" in
        "$(realpath -m -- "$HOME")/"*)
            echo "CSI storage must be outside home: $csi_path. Set CSI_WORK_ROOT to scratch." >&2
            return 1 ;;
    esac
    if [[ "$csi_path" != /* ]]; then
        echo "Use absolute CSI storage paths: $csi_path" >&2
        return 1
    fi
done
if [[ "$CSI_REPO_ROOT" != /* || "$CSI_DATA_FILE" != /* ]]; then
    echo "CSI_REPO_ROOT and CSI_DATA_FILE must be absolute paths." >&2
    return 1
fi

# Deliberately replace inherited cache paths, which often point into home.
export XDG_CACHE_HOME="$CSI_WORK_ROOT/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$CSI_WORK_ROOT/python"
export UV_PYTHON_BIN_DIR="$CSI_WORK_ROOT/bin"
export UV_TOOL_DIR="$CSI_WORK_ROOT/uv-tools"
export UV_TOOL_BIN_DIR="$CSI_WORK_ROOT/bin"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$CSI_WORK_ROOT/tmp"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$CSI_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$CSI_VENV/bin:$CSI_WORK_ROOT/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${CSI_CPUS:-4}}"
# PyTorch gives MKL_NUM_THREADS precedence over OMP_NUM_THREADS, including for
# the native OpenMP solver. Keep both aligned with the allocated CPU count.
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS="${CSI_BUILD_JOBS:-2}"
export CSI_PYTHON="${CSI_PYTHON:-$CSI_VENV/bin/python}"

csi_make_directories() {
    mkdir -p "$CSI_WORK_ROOT" "$CSI_RESULTS_ROOT" "$TMPDIR" \
        "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" \
        "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
        "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
}
