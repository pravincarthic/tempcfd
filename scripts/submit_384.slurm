#!/bin/bash
#SBATCH --job-name=pse-mack2
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=48          # 8 x 48 = 384 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=pse-%j.out

set -euo pipefail
module purge
module load gcc openmpi python                 # site-specific

# One BLAS thread per rank: the parallelism is over modes and nested
# threading only causes oversubscription.
export PSE_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

CFG=${1:-config/mach10_130kft.json}
OUT=${2:-results/$SLURM_JOB_ID}

srun --cpu-bind=cores python3 scripts/run_pse.py "$CFG" -o "$OUT"

# 384 frequencies / 384 ranks -> one mode per rank, dynamic scheduling.
# For fewer modes than ranks set parallel.group_size (e.g. 4) so each mode
# is solved by a 4-rank group that splits the wall-normal direction.
