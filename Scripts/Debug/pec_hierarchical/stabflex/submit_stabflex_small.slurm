#!/bin/bash
# Smaller hierarchical Stability-Flexibility study sized to backfill and finish today, while the
# full run (submit_stabflex_hierarchical.slurm) waits for a whole fat node. Fewer trials/estimates
# but still 24 subjects -> meaningful group-level recovery + per-subject correlations.
#SBATCH --job-name=sf_small
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/JDC/ap9344/stabflex_hier_small/slurm_%j.out

cd /scratch/gpfs/JDC/ap9344/PsyNeuLink/Scripts/Debug/pec_hierarchical/stabflex

/scratch/gpfs/JDC/ap9344/PsyNeuLink/.venv/bin/python -u stabflex_recovery.py \
    --n_subjects 24 \
    --num_trials 120 \
    --num_estimates 1200 \
    --max_em_iterations 10 \
    --n_workers 16 \
    --worker_cores 2 \
    --seed 11 \
    --out_dir /scratch/gpfs/JDC/ap9344/stabflex_hier_small
