#!/bin/bash
#SBATCH --job-name=two_step
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/PsyNeuLink
source .venv/bin/activate

cd /scratch/gpfs/JDC/younes/projects/PsyNeuLink/Scripts/Models\ \(Under\ Development\)/EGO/Using\ EMComposition/bandits/python_

echo "Running array task ${SLURM_ARRAY_TASK_ID}"

export PYTHONPATH=$PWD
python -u scripts/two_step.py
