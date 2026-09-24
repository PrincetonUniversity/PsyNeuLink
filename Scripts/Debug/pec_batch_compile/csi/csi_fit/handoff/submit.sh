#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
if [[ "${1:-}" != cpu && "${1:-}" != gpu ]]; then
    echo "Usage: bash submit.sh cpu|gpu [--dry-run] [sbatch options] -- [runner options]" >&2
    exit 2
fi
mode="$1"
shift
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
slurm_options=()
while (( $# )) && [[ "$1" != -- ]]; do
    slurm_options+=("$1")
    shift
done
if [[ "${1:-}" == -- ]]; then shift; fi
log_dir="$CSI_RESULTS_ROOT/logs"
command=(sbatch --export=ALL --chdir="$CSI_WORK_ROOT"
    --output="$log_dir/csi-$mode-%A_%a.out"
    --error="$log_dir/csi-$mode-%A_%a.err"
    "${slurm_options[@]}" "$CSI_HANDOFF_DIR/della_$mode.slurm" "$@")
printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
if (( dry_run )); then exit 0; fi
# Slurm opens logs before the job starts, so the submitter must create this.
csi_make_directories
mkdir -p "$log_dir"
exec "${command[@]}"
