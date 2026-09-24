#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
mode="${1:-gpu}"
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
    echo "Usage: bash setup.sh [cpu|gpu] (gpu environment supports both runners)" >&2
    exit 2
fi
command -v uv >/dev/null || { echo "Install uv first; see ../README.md." >&2; exit 1; }
command -v "${CXX:-c++}" >/dev/null || {
    echo "A C++ compiler with OpenMP is required; load a GCC module first." >&2; exit 1;
}
if [[ -e "$CSI_VENV" ]]; then
    echo "Environment already exists: $CSI_VENV. Use a new CSI_VENV to avoid changing an active environment." >&2
    exit 1
fi
csi_make_directories
uv python install "${CSI_PYTHON_VERSION:-3.12}"
uv venv --python "${CSI_PYTHON_VERSION:-3.12}" "$CSI_VENV"
if [[ "$mode" == gpu ]]; then
    # Explicit selection also works on login nodes without a visible GPU.
    uv pip install --python "$CSI_VENV/bin/python" \
        --torch-backend "${CSI_TORCH_BACKEND:-cu128}" \
        -e "$CSI_REPO_ROOT[triton]" ninja
else
    uv pip install --python "$CSI_VENV/bin/python" --torch-backend cpu \
        -e "$CSI_REPO_ROOT" ninja
fi
uv pip check --python "$CSI_VENV/bin/python"
uv pip freeze --python "$CSI_VENV/bin/python" > "$CSI_VENV/requirements-resolved.txt"
"$CSI_VENV/bin/python" -c 'import sys, torch, psyneulink; print(sys.version); print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)'
echo "Environment ready: $CSI_VENV (no fit or Slurm submission performed)."
