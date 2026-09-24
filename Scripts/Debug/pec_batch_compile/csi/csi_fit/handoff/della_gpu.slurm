#!/usr/bin/env bash
#SBATCH --job-name=csi-gpu
#SBATCH --constraint=gpu40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=1
#SBATCH --chdir=/scratch/gpfs/CSES/dmturner/PsyNeuLink
#SBATCH --output=csi-gpu-%A_%a.out
#SBATCH --error=csi-gpu-%A_%a.err

set -euo pipefail
# Della selects the GPU partition from the resource request; its submission
# policy rejects an explicit --partition=gpu.
# submit.sh overrides the literal scratch chdir/log defaults above.
export CSI_REPO_ROOT="${CSI_REPO_ROOT:-/scratch/gpfs/CSES/dmturner/PsyNeuLink}"
exec bash "$CSI_REPO_ROOT/Scripts/Debug/pec_batch_compile/csi/csi_fit/handoff/run.sh" gpu "$@"
