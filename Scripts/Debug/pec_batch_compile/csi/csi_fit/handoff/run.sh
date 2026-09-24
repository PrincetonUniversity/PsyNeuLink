#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
if [[ ! -x "$CSI_PYTHON" ]]; then
    echo "Python not found: $CSI_PYTHON. Run handoff/setup.sh first or set CSI_PYTHON." >&2
    exit 1
fi
export PATH="$(dirname -- "$CSI_PYTHON"):$PATH"
exec "$CSI_PYTHON" "$CSI_HANDOFF_DIR/run_fit.py" "$@"
