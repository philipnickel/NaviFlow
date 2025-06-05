#!/bin/bash
#BSUB -J staggered[1-6]              # One task per script
#BSUB -n 1                             # 1 core is likely enough
#BSUB -R "rusage[mem=25GB]"
#BSUB -W 05:00                         # Adjust as needed
#BSUB -o logs/staggered_%J_%I.out
#BSUB -e logs/staggered_%J_%I.err
#BSUB -q hpc
#BSUB -N
#BSUB -u s214960@dtu.dk
#BSUB -B


# Load modules or activate your environment
module load latex/TexLive24

source ~/miniconda3/etc/profile.d/conda.sh
conda activate petsc_gpu_env

# Change to project directory (assuming NaviFlow contains all scripts)
cd ~/NaviFlow

# Fetch the script path from the task file
SCRIPT=$(sed -n "${LSB_JOBINDEX}p" HPC\ jobscripts/staggered.txt)

echo "▶ Running: PYTHONPATH=$(pwd) python $SCRIPT"
PYTHONPATH=$(pwd) python "$SCRIPT"
