#!/bin/bash
#SBATCH --job-name=revaluation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-500

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/EGO
source .venv/bin/activate

echo "Running array task ${SLURM_ARRAY_TASK_ID}"

export PYTHONPATH=$PWD
python -u scripts/revaluation.py