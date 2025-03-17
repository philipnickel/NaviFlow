#!/bin/bash

# run_quick_test.sh
# A simplified version for testing

# Default parameters
MESH_SIZE=127              # A larger mesh size that will take longer to run
REYNOLDS_NUMBER=100        # Just one Reynolds number
OUTPUT_DIR="./quick_results" # Separate results directory
LOG_DIR="./quick_logs"       # Separate logs directory

# Setup directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Function to display timestamp
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "$(timestamp) Starting quick test simulation with larger mesh (${MESH_SIZE}x${MESH_SIZE})"

JOB_NAME="mesh${MESH_SIZE}_Re${REYNOLDS_NUMBER}"
LOG_FILE="$LOG_DIR/${JOB_NAME}.log"

echo "$(timestamp) Running: $JOB_NAME"

# Run the simulation
python preconditioned_cg_cavity.py \
  --nx=${MESH_SIZE} \
  --ny=${MESH_SIZE} \
  --reynolds=${REYNOLDS_NUMBER} \
  --scheme=power_law \
  --max_iter=500 \
  --tolerance=1e-5 \
  --output="${JOB_NAME}" \
  > "$LOG_FILE" 2>&1

# Check result
if [ $? -eq 0 ]; then
  echo "$(timestamp) Completed: $JOB_NAME"
else
  echo "$(timestamp) Failed: $JOB_NAME"
fi

echo "$(timestamp) Test completed!"
