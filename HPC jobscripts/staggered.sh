#!/bin/bash
#BSUB -J staggered[1-6]              # One task per script
#BSUB -n 4                             # 1 core is likely enough
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 02:00                         # Adjust as needed
#BSUB -o logs/staggered_%J_%I.out
#BSUB -e logs/staggered_%J_%I.err

# Load modules or activate your environment
source ~/.bashrc
conda activate petsc_gpu_env

# Change to project directory (assuming NaviFlow contains all scripts)
cd ~/NaviFlow

# Fetch the script path from the task file
SCRIPT=$(sed -n "${LSB_JOBINDEX}p" staggered.txt)

echo "▶ Running: PYTHONPATH=$(pwd) python $SCRIPT"
PYTHONPATH=$(pwd) python "$SCRIPT"
