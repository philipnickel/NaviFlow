#!/bin/bash
#BSUB -J collocated[1-55]              
#BSUB -n 4                            
#BSUB -R "rusage[mem=25GB]"
#BSUB -W 03:00                        
#BSUB -o logs/collocated_%J_%I.out
#BSUB -e logs/collocated_%J_%I.err
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


# Read config path for this job
CONFIG=$(sed -n "${LSB_JOBINDEX}p" HPC\ jobscripts/collocated.txt)

# Run simulation
echo "Running config: $CONFIG"
python main.py --config "$CONFIG"
