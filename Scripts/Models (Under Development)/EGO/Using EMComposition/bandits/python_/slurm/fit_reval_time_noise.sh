#!/bin/bash
#SBATCH --job-name=reval_fit
#SBATCH --array=0-30
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/reval_fit_%A_%a.out

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/EGO
source .venv/bin/activate

export PYTHONPATH=$PWD
python scripts/fit_reval_time_noise.py ${SLURM_ARRAY_TASK_ID}