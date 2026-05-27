#!/bin/bash
#SBATCH --job-name=pinocchio
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=pinocchio_%j.out
#SBATCH --error=pinocchio_%j.err

# Load modules (adattare al cluster)
# module load openmpi
# module load gsl
# module load fftw

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Bind MPI ranks to separate sockets/groups of cores
srun --mpi=pmix --cpu-bind=cores ./pinocchio.x parameter_file
