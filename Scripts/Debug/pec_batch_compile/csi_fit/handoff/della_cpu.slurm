#!/usr/bin/env bash
#SBATCH --job-name=csi-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1
#SBATCH --chdir=/scratch/gpfs/CSES/dmturner/PsyNeuLink
#SBATCH --output=csi-cpu-%A_%a.out
#SBATCH --error=csi-cpu-%A_%a.err

set -euo pipefail
# Slurm runs a spooled copy: do not locate the checkout via BASH_SOURCE here.
# submit.sh overrides chdir/log paths above. For raw sbatch, override them on
# the command line when using another checkout; SBATCH cannot expand variables.
export CSI_REPO_ROOT="${CSI_REPO_ROOT:-/scratch/gpfs/CSES/dmturner/PsyNeuLink}"
exec bash "$CSI_REPO_ROOT/Scripts/Debug/pec_batch_compile/csi_fit/handoff/run.sh" cpu "$@"
