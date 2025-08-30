#!/bin/bash

# List of id-net pairs (id net)
jobname_net_pairs=(
  "greedy_saintarnoult saint_arnoult"
  "greedy_provins provins"
  "greedy_ingolstadtcustom ingolstadt_custom"
)

# Limit number of parallel jobs
max_jobs=1

# Function to run a single command
run_urb_experiment() {
  jobname=$1
  netname=$2

  # Unique timestamp
  timestamp=$(date +%Y%m%d%H%M%S)
  jobname_timestamp="${jobname}_${timestamp}"

  echo "Running experiment with id=${jobname_timestamp} and net=$netname..."
  python scripts/greedy.py \
    --id "$jobname_timestamp" \
    --alg-conf config1 \
    --task-conf config1 \
    --env-conf config1_smooth_0 \
    --net "$netname" \
    --env-seed 42
}

# Loop through (id,net)
for idnet in "${jobname_net_pairs[@]}"; do
  set -- $idnet
  id=$1
  net=$2

  # Run in background
  run_urb_experiment "$id" "$net" &

  # Wait if too many jobs are running
  while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 1
  done
done

# Wait for all background jobs to finish
wait
echo "All jobs completed."
