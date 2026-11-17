#!/bin/bash
#SBATCH --job-name=pse-nemo-gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4              # A100 SXM (use 8 on DGX A100 nodes)
#SBATCH --ntasks-per-node=32           # 8 ranks per GPU under MPS
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=pse-gpu-%j.out

set -euo pipefail
module purge
module load gcc openmpi cuda python     # site-specific; needs cupy-cuda12x

export PSE_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PSE_BACKEND=gpu
# several ranks share each A100 through MPS (one daemon per node)
srun --ntasks-per-node=1 nvidia-cuda-mps-control -d || true

CFG=${1:-config/su2_nemo_air7.json}
OUT=${2:-results/$SLURM_JOB_ID}
srun --cpu-bind=cores python3 scripts/run_pse.py "$CFG" -o "$OUT"

srun --ntasks-per-node=1 bash -c 'echo quit | nvidia-cuda-mps-control' || true
