#!/bin/bash
#BSUB -J "naviflow[1-60]"  # Adjust number based on total combinations
#BSUB -o naviflow_%J_%I.out
#BSUB -e naviflow_%J_%I.err
#BSUB -W 03:00              # 48 hours
#BSUB -n 8                  # Request 8 cores
#BSUB -R "rusage[mem=20GB]" # 20GB
#BSUB -q normal             # Use normal queue
#BSUB -N
#BSUB -u s214960@dtu.dk

# Load required modules
module load python/3.10.4
module load gcc/12.1.0

# Base directory of the project
BASE_DIR="/dtu/3d-imaging-center/projects/2023_QIM_52_NaviFlow"
cd $BASE_DIR

# Define all experiment combinations
declare -a EXPERIMENTS=("lidDrivenCavity")
declare -a MESH_TYPES=("uniform" "unstructured")
declare -a MESH_SIZES=("coarse" "medium" "fine")
declare -a REYNOLDS=(100 400 

# Function to get experiment parameters from job array index
get_experiment_params() {
    local index=$1
    local exp_index=$(( (index - 1) / (${#MESH_TYPES[@]} * ${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local remaining=$(( (index - 1) % (${#MESH_TYPES[@]} * ${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local mesh_type_index=$(( remaining / (${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local mesh_size_index=$(( (remaining % (${#MESH_SIZES[@]} * ${#REYNOLDS[@]})) / ${#REYNOLDS[@]} ))
    local re_index=$(( remaining % ${#REYNOLDS[@]} ))
    
    echo "${EXPERIMENTS[$exp_index]} ${MESH_TYPES[$mesh_type_index]} ${MESH_SIZES[$mesh_size_index]} ${REYNOLDS[$re_index]}"
}

# Get parameters for this job array index
read exp mesh_type mesh_size re <<< $(get_experiment_params $LSB_JOBINDEX)



# Run the experiment
echo "Running: $exp, $mesh_type, $mesh_size, Re=$re"
bash run_experiment.sh \
    --experiment "$exp" \
    --reynolds "$re" \
    --mesh-type "$mesh_type" \
    --mesh-size "$mesh_size" \
    --postprocess 