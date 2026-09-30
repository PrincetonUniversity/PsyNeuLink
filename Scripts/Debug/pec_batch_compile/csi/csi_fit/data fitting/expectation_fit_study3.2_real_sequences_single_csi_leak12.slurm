#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --array=1-97
#SBATCH --output=logs/%A_%a.out
#SBATCH -C amd,genoa

module load anaconda3/2024.2
conda activate psyneulink

python -u expectation_fit_study3.2_real_sequences_single_csi_leak12.py --subject_id $SLURM_ARRAY_TASK_ID --cpu_count $SLURM_CPUS_PER_TASK