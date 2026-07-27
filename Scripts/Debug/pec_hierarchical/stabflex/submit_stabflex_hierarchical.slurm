#!/bin/bash
# Overnight hierarchical Stability-Flexibility study: 4-parameter Laplace EM distributed over
# subjects on a single fat cpu node (one Dask worker per subject, a few LLVM threads each).
#SBATCH --job-name=sf_hier
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/JDC/ap9344/stabflex_hier/slurm_%j.out

cd /scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/stabflex

/scratch/gpfs/JDC/ap9344/PsyNeuLink/.venv/bin/python -u stabflex_recovery.py \
    --n_subjects 48 \
    --num_trials 250 \
    --num_estimates 10000 \
    --max_em_iterations 12 \
    --n_workers 48 \
    --worker_cores 4 \
    --seed 11 \
    --out_dir /scratch/gpfs/JDC/ap9344/stabflex_hier
