#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /scratch/gpfs/JDC/younes/projects/EGO
source .venv/bin/activate

export PYTHONPATH=$PWD
python -u scripts/revaluation_plot.py
python -u scripts/two_step_plot.py