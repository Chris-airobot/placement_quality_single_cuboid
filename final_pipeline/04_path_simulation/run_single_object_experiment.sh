#!/bin/bash

# Configuration
EXPERIMENT_RESULTS_FILE="/home/chris/Chris/placement_ws/src/data/box_simulation/v7/experiments/experiment_results_origin_box.jsonl"
# Companion resume file used by Experiment.py override logic
RESUME_FILE="$(dirname "$EXPERIMENT_RESULTS_FILE")/resume.txt"
SIMULATOR_FILE="/home/chris/Chris/placement_ws/src/placement_quality/path_simulation/model_testing/simulator.py"
ISAAC_SIM_PATH="/home/chris/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh"
SLEEP_TIME=10
START_INDEX=
# Function to get the latest index from experiment results and increment by 1
get_next_index() {
    if [ -f "$EXPERIMENT_RESULTS_FILE" ]; then
        # Use python for robust JSON parse of last non-empty line
        last_non_empty=$(tac "$EXPERIMENT_RESULTS_FILE" | grep -m1 . || true)
        if [ -n "$last_non_empty" ]; then
            python3 - <<'PY'
import sys, json
line = sys.stdin.read().strip()
try:
    idx = json.loads(line).get("index", -1)
    print(int(idx) + 1 if isinstance(idx, int) else 0)
except Exception:
    print(0)
PY
            <<< "$last_non_empty"
        else
            echo "0"
        fi
    else
        echo "0"
    fi
}

# Write resume index for Experiment.py to consume
write_resume_index() {
    local next_index=$1
    mkdir -p "$(dirname "$RESUME_FILE")"
    echo -n "$next_index" > "$RESUME_FILE"
}

# Function to run the experiment
run_experiment() {
    echo "Running experiment with Isaac Sim..."
    $ISAAC_SIM_PATH -m placement_quality.path_simulation.model_testing.sim_test --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
    return $?
}

# Main execution
echo "=== Infinite Experiment Runner Script ==="
echo "Results file: $EXPERIMENT_RESULTS_FILE"
echo "Simulator file: $SIMULATOR_FILE"

# Initialize failure counter and last read index
failure_count=0
last_read_index=-1

# Optional one-time start override
OVERRIDE_START_INDEX=${START_INDEX:-}
override_used=0

# Run experiment infinitely
while true; do
    echo ""
    echo "=== Starting new experiment cycle ==="
    
    # Always read the latest index from results file unless override (first cycle only)
    if [ -n "$OVERRIDE_START_INDEX" ] && [ $override_used -eq 0 ]; then
        echo "Using START_INDEX override: $OVERRIDE_START_INDEX"
        current_index=$OVERRIDE_START_INDEX
        # Align last_read_index so the 'updated' logic treats this as fresh
        current_latest_from_file=$(get_next_index)
        current_latest_actual=$((current_latest_from_file - 1))
        last_read_index=$((OVERRIDE_START_INDEX - 1))
        override_used=1
        # Persist override for Experiment.py via resume.txt
        write_resume_index "$OVERRIDE_START_INDEX"
    else
        current_latest_from_file=$(get_next_index)
        current_latest_actual=$((current_latest_from_file - 1))
    fi
    
    # Check if the results file has been updated since last read
    if [ $last_read_index -ne $current_latest_actual ]; then
        echo "Results file updated! Last read: $last_read_index, Current: $current_latest_actual"
        echo "Resetting failure count and using new index"
        failure_count=0
        current_index=$current_latest_from_file
        last_read_index=$current_latest_actual
        echo "Latest index from results: $current_latest_actual"
        echo "Next index to process: $current_index"
    else
        # Results file hasn't been updated, continue with current logic
        if [ $failure_count -eq 0 ]; then
            current_index=$current_latest_from_file
            echo "Latest index from results: $current_latest_actual"
            echo "Next index to process: $current_index"
        else
            echo "Continuing with failed index: $current_index"
        fi
    fi
    
    echo "Failure count: $failure_count"
    
    # Persist the start index for Experiment.py
    # write_resume_index $current_index
    
    echo "Running experiment with data_index: $current_index"
    
    # Run the experiment
    if run_experiment; then
        echo "✅ Experiment completed successfully!"
        echo "Waiting $SLEEP_TIME seconds before next experiment..."
        sleep $SLEEP_TIME
        
        # Reset failure counter on success
        failure_count=0
        
    else
        echo "❌ Experiment failed with exit code $?"
        failure_count=$((failure_count + 1))
        
        echo "Incrementing by $failure_count and retrying..."
        
        # Increment index by failure count for next attempt and persist
        current_index=$((current_index + failure_count))
        # write_resume_index $current_index
        
        echo "Waiting $SLEEP_TIME seconds before retry with index $current_index..."
        sleep $SLEEP_TIME
    fi
done 