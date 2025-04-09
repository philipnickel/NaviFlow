#!/bin/bash

# run_m3_optimized.sh
# Advanced optimization script for Apple Silicon M3 Pro MacBook Pro
# This script intelligently distributes CFD simulations across performance and efficiency cores

# M3 Pro typically has 6 performance cores and 4 efficiency cores
# We'll use this knowledge to optimally distribute workloads

# Default parameters
P_CORES=6                   # Performance cores (adjust based on your specific M3 Pro model)
E_CORES=4                   # Efficiency cores (adjust based on your specific M3 Pro model)
MAX_P_JOBS=$P_CORES         # Max jobs on performance cores (resource-intensive jobs)
MAX_E_JOBS=$E_CORES         # Max jobs on efficiency cores (lighter jobs)
MESH_SIZES=(256 512)  # Mesh sizes to test
REYNOLDS_NUMBERS=(1000 3200 5000 7500 10000) # Reynolds numbers to test
OUTPUT_DIR="./results"      # Output directory
LOG_DIR="./logs"            # Log directory
MAX_ITER=100000               # Maximum iterations for each simulation
TOLERANCE=1e-5              # Convergence tolerance

# Command line arguments parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    --p-cores)
      P_CORES="$2"
      MAX_P_JOBS=$P_CORES
      shift 2
      ;;
    --e-cores)
      E_CORES="$2"
      MAX_E_JOBS=$E_CORES
      shift 2
      ;;
    --mesh-sizes)
      IFS=',' read -ra MESH_SIZES <<< "$2"
      shift 2
      ;;
    --reynolds)
      IFS=',' read -ra REYNOLDS_NUMBERS <<< "$2"
      shift 2
      ;;
    --max-iter)
      MAX_ITER="$2"
      shift 2
      ;;
    --tolerance)
      TOLERANCE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --p-cores N            Number of performance cores to use (default: 6)"
      echo "  --e-cores N            Number of efficiency cores to use (default: 4)"
      echo "  --mesh-sizes X,Y,Z     Comma-separated list of mesh sizes (default: 31,63,127,255)"
      echo "  --reynolds X,Y,Z       Comma-separated list of Reynolds numbers (default: 100,400,1000)"
      echo "  --max-iter N           Maximum iterations per simulation (default: 1000)"
      echo "  --tolerance X          Convergence tolerance (default: 1e-5)"
      echo "  --output-dir DIR       Output directory (default: ./results)"
      echo "  --help                 Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Setup directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Function to display timestamp
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# Function to get current memory usage
get_memory_usage() {
  # For macOS, use vm_stat and calculate used memory
  local pages_active=$(vm_stat | grep "Pages active" | awk '{print $3}' | tr -d '.')
  local pages_wired=$(vm_stat | grep "Pages wired down" | awk '{print $4}' | tr -d '.')
  local page_size=16384  # 16KB page size for macOS
  
  # Calculate memory usage in GB
  echo "scale=2; ($pages_active + $pages_wired) * $page_size / 1024 / 1024 / 1024" | bc
}

# Function to estimate memory requirement based on mesh size
estimate_memory_gb() {
  local nx=$1
  # Simple formula: memory ~ (nx*nx*8*factors)
  # This is an approximation - adjust the coefficient based on actual measurements
  echo "scale=2; ($nx * $nx * 8 * 10) / 1024 / 1024 / 1024" | bc
}

# Check if we have enough memory to start another job
can_start_job() {
  local nx=$1
  local mem_required=$(estimate_memory_gb $nx)
  local mem_used=$(get_memory_usage)
  local mem_total=$(sysctl -n hw.memsize | awk '{print $1 / 1024 / 1024 / 1024}')
  local mem_available=$(echo "$mem_total - $mem_used" | bc)
  
  # Check if we have enough memory (with a safety margin)
  if (( $(echo "$mem_available > $mem_required + 2" | bc -l) )); then
    return 0  # We can start the job
  else
    return 1  # Not enough memory
  fi
}

# Function to determine if a job should run on performance cores
should_use_performance_cores() {
  local mesh_size=$1
  local reynolds=$2
  
  # Criteria for using performance cores:
  # 1. Large mesh sizes (>= 127)
  # 2. High Reynolds numbers (>= 400)
  if [ "$mesh_size" -ge 127 ] || [ "$reynolds" -ge 400 ]; then
    return 0  # Use performance cores
  else
    return 1  # Use efficiency cores
  fi
}

# Function to run a job with CPU affinity for macOS
# Since macOS doesn't have taskset, we'll use a workaround with launchd priority
run_with_affinity() {
  local job_cmd="$1"
  local log_file="$2"
  local use_performance_cores=$3
  
  if [ "$use_performance_cores" -eq 1 ]; then
    # For performance cores, use normal priority
    # This ensures the job gets scheduled on performance cores when available
    eval "$job_cmd" > "$log_file" 2>&1 &
    echo $!
  else
    # For efficiency cores, use background priority via nice
    # This encourages macOS to schedule on efficiency cores
    nice -n 10 eval "$job_cmd" > "$log_file" 2>&1 &
    echo $!
  fi
}

# Arrays to track jobs on different core types
P_JOBS=()
E_JOBS=()

# Prepare the job list, splitting by core type
P_JOB_LIST=()
E_JOB_LIST=()

for MESH in "${MESH_SIZES[@]}"; do
  for RE in "${REYNOLDS_NUMBERS[@]}"; do
    JOB_NAME="mesh${MESH}_Re${RE}"
    JOB_CMD="python preconditioned_cg_cavity.py --nx=${MESH} --ny=${MESH} --reynolds=${RE} --scheme=power_law --max_iter=${MAX_ITER} --tolerance=${TOLERANCE} --quiet --output=${JOB_NAME}"
    
    # Determine if this job should use performance cores
    if should_use_performance_cores "$MESH" "$RE"; then
      P_JOB_LIST+=("$JOB_NAME:$JOB_CMD")
    else
      E_JOB_LIST+=("$JOB_NAME:$JOB_CMD")
    fi
  done
done

# Count jobs
P_TOTAL=${#P_JOB_LIST[@]}
E_TOTAL=${#E_JOB_LIST[@]}
TOTAL_JOBS=$((P_TOTAL + E_TOTAL))
P_COMPLETED=0
E_COMPLETED=0
COMPLETED_JOBS=0

echo "$(timestamp) Starting M3-optimized batch simulation with $TOTAL_JOBS total jobs"
echo "$(timestamp) Performance cores: $P_CORES (jobs: $P_TOTAL), Efficiency cores: $E_CORES (jobs: $E_TOTAL)"

# Function to display progress
show_progress() {
  local p_percent=$((P_COMPLETED * 100 / (P_TOTAL > 0 ? P_TOTAL : 1)))
  local e_percent=$((E_COMPLETED * 100 / (E_TOTAL > 0 ? E_TOTAL : 1)))
  local total_percent=$((COMPLETED_JOBS * 100 / TOTAL_JOBS))
  
  echo "$(timestamp) Progress: $COMPLETED_JOBS/$TOTAL_JOBS jobs completed ($total_percent%)"
  echo "$(timestamp) P-Cores: $P_COMPLETED/$P_TOTAL ($p_percent%), E-Cores: $E_COMPLETED/$E_TOTAL ($e_percent%)"
  echo "$(timestamp) Running: P-Cores: ${#P_JOBS[@]}/$MAX_P_JOBS, E-Cores: ${#E_JOBS[@]}/$MAX_E_JOBS"
}

# Function to check job completion status
check_job_completion() {
  # Check performance core jobs
  for i in "${!P_JOBS[@]}"; do
    if ! ps -p ${P_JOBS[$i]} > /dev/null; then
      echo "$(timestamp) P-Core job ${P_JOBS[$i]} completed"
      unset P_JOBS[$i]
      P_COMPLETED=$((P_COMPLETED + 1))
      COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
    fi
  done
  # Clean up array
  P_JOBS=("${P_JOBS[@]}")
  
  # Check efficiency core jobs
  for i in "${!E_JOBS[@]}"; do
    if ! ps -p ${E_JOBS[$i]} > /dev/null; then
      echo "$(timestamp) E-Core job ${E_JOBS[$i]} completed"
      unset E_JOBS[$i]
      E_COMPLETED=$((E_COMPLETED + 1))
      COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
    fi
  done
  # Clean up array
  E_JOBS=("${E_JOBS[@]}")
  
  show_progress
}

# Process performance core jobs
process_p_jobs() {
  local jobs_started=0
  
  while [ "${#P_JOBS[@]}" -lt "$MAX_P_JOBS" ] && [ "$P_COMPLETED" -lt "$P_TOTAL" ]; do
    local idx=$((P_COMPLETED + ${#P_JOBS[@]}))
    if [ "$idx" -lt "${#P_JOB_LIST[@]}" ]; then
      local job="${P_JOB_LIST[$idx]}"
      local job_name="${job%%:*}"
      local job_cmd="${job#*:}"
      local log_file="$LOG_DIR/${job_name}.log"
      
      # Extract mesh size from job name
      local mesh_size=$(echo "$job_name" | grep -o 'mesh[0-9]*' | sed 's/mesh//')
      
      # Check memory availability
      if can_start_job "$mesh_size"; then
        echo "$(timestamp) Starting P-Core job: $job_name"
        local pid=$(run_with_affinity "$job_cmd" "$log_file" 1)
        P_JOBS+=($pid)
        jobs_started=$((jobs_started + 1))
        
        # Brief pause to avoid overwhelming system
        sleep 0.5
      else
        # Not enough memory, try again later
        echo "$(timestamp) Waiting for memory to start P-Core job: $job_name"
        break
      fi
    else
      break
    fi
  done
  
  return $jobs_started
}

# Process efficiency core jobs
process_e_jobs() {
  local jobs_started=0
  
  while [ "${#E_JOBS[@]}" -lt "$MAX_E_JOBS" ] && [ "$E_COMPLETED" -lt "$E_TOTAL" ]; do
    local idx=$((E_COMPLETED + ${#E_JOBS[@]}))
    if [ "$idx" -lt "${#E_JOB_LIST[@]}" ]; then
      local job="${E_JOB_LIST[$idx]}"
      local job_name="${job%%:*}"
      local job_cmd="${job#*:}"
      local log_file="$LOG_DIR/${job_name}.log"
      
      # Extract mesh size from job name
      local mesh_size=$(echo "$job_name" | grep -o 'mesh[0-9]*' | sed 's/mesh//')
      
      # Check memory availability
      if can_start_job "$mesh_size"; then
        echo "$(timestamp) Starting E-Core job: $job_name"
        local pid=$(run_with_affinity "$job_cmd" "$log_file" 0)
        E_JOBS+=($pid)
        jobs_started=$((jobs_started + 1))
        
        # Brief pause to avoid overwhelming system
        sleep 0.5
      else
        # Not enough memory, try again later
        echo "$(timestamp) Waiting for memory to start E-Core job: $job_name"
        break
      fi
    else
      break
    fi
  done
  
  return $jobs_started
}

# Main execution loop
while [ "$COMPLETED_JOBS" -lt "$TOTAL_JOBS" ]; do
  # Process performance core jobs
  process_p_jobs
  p_started=$?
  
  # Process efficiency core jobs
  process_e_jobs
  e_started=$?
  
  # If no new jobs were started, check for completed jobs
  if [ "$p_started" -eq 0 ] && [ "$e_started" -eq 0 ]; then
    check_job_completion
    
    # If we're not at capacity and not starting new jobs, there's likely a memory constraint
    if [ "${#P_JOBS[@]}" -lt "$MAX_P_JOBS" ] || [ "${#E_JOBS[@]}" -lt "$MAX_E_JOBS" ]; then
      echo "$(timestamp) Waiting for resources to become available..."
      sleep 5
    else
      # If at capacity, just wait for jobs to complete
      sleep 2
    fi
  else
    # If we started new jobs, brief pause before continuing
    sleep 1
  fi
done

show_progress
echo "$(timestamp) All jobs completed successfully!"

# Generate summary report
echo "$(timestamp) Generating summary report..."
echo "# NaviFlow M3-Optimized Simulation Results" > "$OUTPUT_DIR/m3_optimized_report.md"
echo "Date: $(date)" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "## Performance Summary" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "| Mesh Size | Reynolds | Core Type | Iterations | Max Divergence | Runtime (s) |" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "|-----------|----------|-----------|------------|----------------|-------------|" >> "$OUTPUT_DIR/m3_optimized_report.md"

# Extract data from metadata files and determine core type
for METADATA in "$OUTPUT_DIR"/*_metadata.txt; do
  if [ -f "$METADATA" ]; then
    MESH=$(grep "Mesh:" "$METADATA" | awk -F: '{print $2}' | tr -d ' ')
    RE=$(grep "Reynolds:" "$METADATA" | awk -F: '{print $2}' | tr -d ' ')
    ITER=$(grep "Iterations:" "$METADATA" | awk -F: '{print $2}' | tr -d ' ')
    DIV=$(grep "Max Divergence:" "$METADATA" | awk -F: '{print $2}' | tr -d ' ')
    TIME=$(grep "Runtime:" "$METADATA" | awk -F: '{print $2}' | sed 's/ seconds//' | tr -d ' ')
    
    # Determine which core type was used
    if should_use_performance_cores "${MESH%%x*}" "$RE"; then
      CORE_TYPE="P-Core"
    else
      CORE_TYPE="E-Core"
    fi
    
    echo "| $MESH | $RE | $CORE_TYPE | $ITER | $DIV | $TIME |" >> "$OUTPUT_DIR/m3_optimized_report.md"
  fi
done

echo "" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "## System Information" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "* CPU: $(sysctl -n machdep.cpu.brand_string)" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "* Memory: $(sysctl -n hw.memsize | awk '{print $1 / 1024 / 1024 / 1024}') GB" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "* OS: $(sw_vers -productName) $(sw_vers -productVersion)" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "* Performance Cores: $P_CORES" >> "$OUTPUT_DIR/m3_optimized_report.md"
echo "* Efficiency Cores: $E_CORES" >> "$OUTPUT_DIR/m3_optimized_report.md"

echo "$(timestamp) M3-optimized report generated: $OUTPUT_DIR/m3_optimized_report.md"
echo "$(timestamp) All done!" 