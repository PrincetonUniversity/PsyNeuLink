#!/bin/bash
# Stab-flex neural-likelihood pipeline: simulate training configs (distributed), train the MDN,
# then run hierarchical EM with the neural likelihood and compare against the saved KDE-EM results.
#SBATCH --job-name=sf_nle
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/JDC/ap9344/stabflex_nle/slurm_%j.out

cd /scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/nle
PY=/scratch/gpfs/JDC/ap9344/PsyNeuLink/.venv/bin/python

$PY -u stabflex_nle.py gen --configs 4000 --workers 16 && \
$PY -u stabflex_nle.py train --epochs 30 && \
$PY -u stabflex_nle.py em
