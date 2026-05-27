#!/bin/bash
#SBATCH --job-name=pinocchio
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=pinocchio_%j.out
#SBATCH --error=pinocchio_%j.err
#SBATCH --partition=regular

# Load modules (adjust to your cluster)
# module load openmpi/4.1
# module load gsl/2.7
# module load fftw/3.3

# OpenMP configuration
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Ensure MPI+OpenMP thread binding works correctly
export SLURM_CPU_BIND=none

# Run directory (adjust path)
RUNDIR=${SLURM_SUBMIT_DIR}
cd ${RUNDIR}

echo "=== Job ${SLURM_JOB_ID} ==="
echo "Nodes: ${SLURM_NNODES}, Tasks: ${SLURM_NTASKS}, CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "Start: $(date)"

# Run PINOCCHIO
srun --mpi=pmix ./pinocchio parameter_file

echo "End: $(date)"
