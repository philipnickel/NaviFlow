#!/bin/bash

# run_experiment.sh
# ----------------
# Run CFD experiments with specified parameters for various flow cases.
#
# This script provides a command-line interface to run CFD simulations with different
# parameters including experiment type, Reynolds number, mesh type, and mesh size.
# It automatically locates the appropriate configuration file and runs the simulation
# using main.py.
#
# The script supports two modes:
#   1. Normal mode - run with specific parameters (mesh type, size, and Reynolds number)
#   2. Debug mode - run using a config file from the experiment's debugging directory
#
# Available Parameters:
#   Experiments: lidDrivenCavity, channelFlow, cylinderFlow
#   Reynolds numbers: 100, 400, 1000, 3200, 5000
#   Mesh types: uniform, unstructured
#   Mesh sizes: coarse, medium, fine
#
# Note:
#   The script will automatically find the appropriate configuration file in the
#   experiments directory structure and run the simulation using main.py.
#   In debug mode, it uses a config file from the experiment's debugging directory.

# Function to display usage
usage() {
    echo "Usage:"
    echo "  Normal mode:"
    echo "    ./run_experiment.sh --experiment <name> --reynolds <number> --mesh-type <type> --mesh-size <size> [--postprocess]"
    echo "  Debug mode:"
    echo "    ./run_experiment.sh --experiment <name> --debug [--postprocess]"
    echo
    echo "Arguments:"
    echo "  --experiment    Name of the experiment (lidDrivenCavity, channelFlow, or cylinderFlow)"
    echo "  --reynolds      Reynolds number (100, 400, 1000, 3200, or 5000) [required in normal mode]"
    echo "  --mesh-type     Type of mesh (uniform or unstructured) [required in normal mode]"
    echo "  --mesh-size     Size of mesh (coarse, medium, or fine) [required in normal mode]"
    echo "  --debug         Run in debug mode using config from experiments/<experiment>/debugging/config.yaml"
    echo "  --postprocess   Run postprocessing after simulation completes"
    echo
    echo "Examples:"
    echo "  # Run a lid-driven cavity simulation with uniform coarse mesh at Re=100"
    echo "  ./run_experiment.sh --experiment lidDrivenCavity --reynolds 100 --mesh-type uniform --mesh-size coarse"
    echo
    echo "  # Run a channel flow simulation in debug mode with postprocessing"
    echo "  ./run_experiment.sh --experiment channelFlow --debug --postprocess"
    exit 1
}

# Parse command line arguments
EXPERIMENT=""
REYNOLDS=""
MESH_TYPE=""
MESH_SIZE=""
DEBUG=false
POSTPROCESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --reynolds)
            REYNOLDS="$2"
            shift 2
            ;;
        --mesh-type)
            MESH_TYPE="$2"
            shift 2
            ;;
        --mesh-size)
            MESH_SIZE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --postprocess)
            POSTPROCESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate arguments
if [ -z "$EXPERIMENT" ]; then
    echo "Error: --experiment is required"
    usage
fi

if [ "$DEBUG" = false ]; then
    if [ -z "$REYNOLDS" ] || [ -z "$MESH_TYPE" ] || [ -z "$MESH_SIZE" ]; then
        echo "Error: --reynolds, --mesh-type, and --mesh-size are required when not in debug mode"
        usage
    fi
fi

# Find config file
if [ "$DEBUG" = true ]; then
    CONFIG_PATH="experiments/$EXPERIMENT/debugging/config.yaml"
else
    CONFIG_PATH="experiments/$EXPERIMENT/ForReport/$MESH_TYPE/$MESH_SIZE/Re_$REYNOLDS/config.yaml"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Print run parameters
echo -e "\nRunning experiment with parameters:"
echo "  Experiment: $EXPERIMENT"
if [ "$DEBUG" = true ]; then
    echo "  Mode: Debug"
else
    echo "  Reynolds number: $REYNOLDS"
    echo "  Mesh type: $MESH_TYPE"
    echo "  Mesh size: $MESH_SIZE"
fi
echo "  Config file: $CONFIG_PATH"
if [ "$POSTPROCESS" = true ]; then
    echo "  Postprocessing: Enabled"
fi
echo

# Run the simulation
python main.py --config "$CONFIG_PATH"

# Run postprocessing if requested
if [ "$POSTPROCESS" = true ]; then
    echo -e "\nRunning postprocessing..."
    if [ "$DEBUG" = true ]; then
        python naviflow_collocated/utils/postprocess/postprocess.py --experiment "$EXPERIMENT/debugging" --all
    else
        python naviflow_collocated/utils/postprocess/postprocess.py --experiment "$EXPERIMENT/ForReport/$MESH_TYPE/$MESH_SIZE/Re_$REYNOLDS" --all
    fi
fi 