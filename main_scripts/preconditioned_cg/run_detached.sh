#!/bin/bash

# run_detached.sh
# This script uses tmux to run simulations in a detachable session

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    echo "You can install it with: brew install tmux"
    exit 1
fi

# The script to run (default to m3_optimized.sh but allow overriding)
SCRIPT_TO_RUN="${1:-./run_m3_optimized.sh}"
SESSION_NAME="naviflow"

# Check if the script exists and is executable
if [ ! -x "$SCRIPT_TO_RUN" ]; then
    echo "Error: $SCRIPT_TO_RUN is not executable or doesn't exist."
    echo "Please make sure the script exists and is executable."
    exit 1
fi

# Check if a session with this name already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "A tmux session named '$SESSION_NAME' already exists."
    echo "You can attach to it with: tmux attach -t $SESSION_NAME"
    echo "Or you can kill it with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create a new detached tmux session and run the script
echo "Starting $SCRIPT_TO_RUN in a detached tmux session..."
tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_TO_RUN"

echo "Simulation started in the background."
echo "You can check on its progress with: tmux attach -t $SESSION_NAME"
echo "When done viewing progress, detach with Ctrl+B then D"
echo ""
echo "Other useful commands:"
echo "- Kill the session: tmux kill-session -t $SESSION_NAME"
echo "- List all sessions: tmux ls" 