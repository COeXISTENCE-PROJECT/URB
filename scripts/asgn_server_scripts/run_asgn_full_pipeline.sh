#!/bin/bash
# Example:
# NETWORK_NAME=ingolstadt_custom2 TASK_CONF=asgn_100k_grid RESULTS_BASE_DIR=/scratch/tmp/$USER/asgn run_asgn_full_pipeline.sh
# The variables are only set for that one command.

set -euo pipefail

PARTITION=${PARTITION:-rknodes}
QOS=${QOS:-big_bonk}
NETWORK_NAME=${NETWORK_NAME:?NETWORK_NAME must be set, e.g. NETWORK_NAME=saint_arnoult}
DEFAULT_EXPERIMENT_NAME="${NETWORK_NAME}_asgn"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}
if [[ "$EXPERIMENT_NAME" == "$DEFAULT_EXPERIMENT_NAME" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
PATH_PROGRAM="${PATH_PROGRAM:-/home/$USER/URB}"
RUN_ASGN_ARRAY="$PATH_PROGRAM/scripts/asgn_server_scripts/run_asgn_array.sh"
TASK_CONF=${TASK_CONF:?TASK_CONF must be set, e.g. TASK_CONF=asgn_100k_grid}
TASK_CONF_PATH="$PATH_PROGRAM/config/task_config/${TASK_CONF}.json"
if [[ ! -f "$TASK_CONF_PATH" ]]; then
  echo "Missing task config file: $TASK_CONF_PATH" >&2
  exit 1
fi

# Get the total array size by summing the total number of tasks for all sampling methods
ARRAY_SIZE=$(python3 - "$TASK_CONF_PATH" <<'PY'
import json
import sys

task_config = json.load(open(sys.argv[1]))
total_tasks = sum(int(value) for key, value in task_config.items() if key.startswith("num_tasks_"))
print(total_tasks)
PY
)

if [[ "$ARRAY_SIZE" -le 0 ]]; then
  echo "No enabled ASGN tasks found in $TASK_CONF_PATH" >&2
  exit 1
fi

echo "Submitting step 1"
jid1=$(sbatch --parsable \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",NETWORK_NAME="$NETWORK_NAME",SEED_BASE=100,TASK_CONF="$TASK_CONF" \
  --array=0-$((ARRAY_SIZE - 1)) \
  "$RUN_ASGN_ARRAY")

echo "Submitting step 2"
jid2=$(sbatch --parsable \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --dependency=afterok:${jid1} \
  --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",NETWORK_NAME="$NETWORK_NAME",PHASE=aggregate,TASK_CONF="$TASK_CONF" \
  --array=0-0 \
  "$RUN_ASGN_ARRAY")

echo "Submitting failure cleanup jobs"
sbatch --partition="$PARTITION" --qos="$QOS" --dependency=afternotok:${jid1} --wrap="scancel ${jid2}"

echo "Submitted:"
echo "  $jid1"
echo "  $jid2"
