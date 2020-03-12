#!/bin/bash
#SBATCH --job-name=pred_prey_search
#SBATCH --output=logs/%A.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

module load anaconda openmpi/gcc
conda activate psyneulink2

srun python run_cost_rate_search.py


