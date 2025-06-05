#!/bin/bash
#BSUB -J "naviflow[1-12]"
#BSUB -o naviflow_%J_%I.out
#BSUB -e naviflow_%J_%I.err
#BSUB -W 03:00
#BSUB -n 8
#BSUB -R "rusage[mem=20GB]"
#BSUB -q normal
#BSUB -N
#BSUB -u s214960@dtu.dk

set -e

module purge
module load python3/3.13.2
module load petsc/3.23.0-apr-2025-gcc-12.4.0-openblas-0.3.29-non-complex-slepc-elemental-superlu
source ~/venvs/cfd_env_py313/bin/activate


BASE_DIR="/dtu/3d-imaging-center/projects/2023_QIM_52_NaviFlow"
cd "$BASE_DIR"

declare -a EXPERIMENTS=("channelFlow" "cylinderFlow")
declare -a MESH_TYPES=("unstructured")
declare -a MESH_SIZES=("coarse" "medium" "fine")
declare -a REYNOLDS=(5 20)

get_experiment_params() {
    local index=$1
    local exp_index=$(( (index - 1) / (${#MESH_TYPES[@]} * ${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local remaining=$(( (index - 1) % (${#MESH_TYPES[@]} * ${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local mesh_type_index=$(( remaining / (${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local mesh_size_index=$(( (remaining % (${#MESH_SIZES[@]} * ${#REYNOLDS[@]})) / ${#REYNOLDS[@]} ))
    local re_index=$(( remaining % ${#REYNOLDS[@]} ))
    
    echo "${EXPERIMENTS[$exp_index]} ${MESH_TYPES[$mesh_type_index]} ${MESH_SIZES[$mesh_size_index]} ${REYNOLDS[$re_index]}"
}

if (( LSB_JOBINDEX > 12 )); then
    echo "Invalid job index: $LSB_JOBINDEX"
    exit 1
fi

read exp mesh_type mesh_size re <<< $(get_experiment_params $LSB_JOBINDEX)

echo "Running: $exp, $mesh_type, $mesh_size, Re=$re"
bash run_experiment.sh \
    --experiment "$exp" \
    --reynolds "$re" \
    --mesh-type "$mesh_type" \
    --mesh-size "$mesh_size" \
    --postprocess
