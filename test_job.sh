#!/bin/bash
#BSUB -J "naviflow_test[1-1]"
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -q hpc
#BSUB -N
#BSUB -u s214960@dtu.dk
#BSUB -B

set -e

module purge
module load python3/3.13.2
module load petsc/3.23.0-apr-2025-gcc-12.4.0-openblas-0.3.29-non-complex-slepc-elemental-superlu
source ~/venvs/cfd_env_py313/bin/activate

#BASE_DIR="~/NaviFlow"
BASE_DIR="/zhome/86/f/169037/NaviFlow"

cd "$BASE_DIR"

declare -a EXPERIMENTS=("channelFlow")
declare -a MESH_TYPES=("unstructured")
declare -a MESH_SIZES=("medium")
declare -a REYNOLDS=(10)

get_experiment_params() {
    local index=$1
    local exp_index=$(( (index - 1) / (${#MESH_TYPES[@]} * ${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local remaining=$(( (index - 1) % (${#MESH_TYPES[@]} * ${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local mesh_type_index=$(( remaining / (${#MESH_SIZES[@]} * ${#REYNOLDS[@]}) ))
    local mesh_size_index=$(( (remaining % (${#MESH_SIZES[@]} * ${#REYNOLDS[@]})) / ${#REYNOLDS[@]} ))
    local re_index=$(( remaining % ${#REYNOLDS[@]} ))

    echo "${EXPERIMENTS[$exp_index]} ${MESH_TYPES[$mesh_type_index]} ${MESH_SIZES[$mesh_size_index]} ${REYNOLDS[$re_index]}"
}

read exp mesh_type mesh_size re <<< $(get_experiment_params $LSB_JOBINDEX)

# Set meaningful log file names
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
OUT_FILE="$LOG_DIR/${exp}_${mesh_type}_${mesh_size}_Re${re}.out"
ERR_FILE="$LOG_DIR/${exp}_${mesh_type}_${mesh_size}_Re${re}.err"

exec > "$OUT_FILE" 2> "$ERR_FILE"

echo "Running: $exp, $mesh_type, $mesh_size, Re=$re"
bash run_experiment.sh \
    --experiment "$exp" \
    --reynolds "$re" \
    --mesh-type "$mesh_type" \
    --mesh-size "$mesh_size" \
    --postprocess
