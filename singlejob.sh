#!/bin/bash
#BSUB -J "naviflow_test"
#BSUB -W 00:30
#BSUB -n 2
#BSUB -R "rusage[mem=20GB] span[hosts=1]"
#BSUB -q hpc
#BSUB -N
#BSUB -u s214960@dtu.dk
#BSUB -B

set -e

module purge
module load python3/3.13.2
module load latex/TexLive24
module load petsc/3.23.0-apr-2025-gcc-12.4.0-openblas-0.3.29-non-complex-slepc-elemental-superlu
source ~/venvs/cfd_env_Naviflow/bin/activate

#BASE_DIR="~/NaviFlow"
BASE_DIR="/zhome/86/f/169037/NaviFlow"

cd "$BASE_DIR"

bash run_experiment.sh --experiment lidDrivenCavity --reynolds 100 --mesh-type uniform --mesh-size coarse --postprocess