#!/bin/bash
#SBATCH -t 144:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=53
#SBATCH --mem=16G
#SBATCH --output=logs/%A.out
#SBATCH -C amd,genoa

module load anaconda3/2024.2
conda activate psyneulink

python -u flanker_fit_lc_part3.py