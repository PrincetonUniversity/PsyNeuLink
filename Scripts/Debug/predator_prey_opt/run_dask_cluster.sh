#!/bin/bash
#SBATCH --job-name=pred_prey_search
#SBATCH --output=logs/%A.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

module load anaconda intel-mpi/gcc
conda activate psyneulink2

WORKDIR=/scratch/gpfs/dmturner/${SLURM_JOB_ID}

# Create a directory on scratch for this job
mkdir $WORKDIR

rm scheduler.json

srun dask-mpi --no-nanny --scheduler-file scheduler.json --local-directory $WORKDIR --nthreads 1

# Cleanup the work directory and scheduler JSON
rm -rf $WORKDIR
rm scheduler.json

