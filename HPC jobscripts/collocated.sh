#!/bin/bash
#BSUB -J cfd_array[1-100]               # 100 jobs in the array
#BSUB -n 4                              # number of cores per task
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==Intel_Platinum_8268]"  # Optional: request modern node
#BSUB -W 02:00                          # 2 hours per job
#BSUB -o logs/job_%J_%I.out
#BSUB -e logs/job_%J_%I.err

# Activate environment
source ~/.bashrc
conda activate petsc_gpu_env

# Read config path for this job
CONFIG=$(sed -n "${LSB_JOBINDEX}p" all_configs.txt)

# Run simulation
echo "Running config: $CONFIG"
python main.py --config "$CONFIG"
