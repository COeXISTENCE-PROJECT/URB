#!/bin/bash
#SBATCH --job-name=asgn_sim
#SBATCH --qos=big_bonk
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --partition=rknodes
#SBATCH --array=0-31

set -euo pipefail

PATH_PROGRAM="${PATH_PROGRAM:-/home/$USER/URB}"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="${PATH_SUMO_CONTAINER:-/shared/sets/singularity/sumo1_26_0patched.sif}"
ASGN_SERVER_DIR="$PATH_PROGRAM/scripts/asgn_server_scripts"
CMD_PATH="$ASGN_SERVER_DIR/run_asgn_single_internal.sh"
PRINTS_SAVE_PATH="$ASGN_SERVER_DIR/container_printouts/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-/scratch/tmp/$USER/asgn}"

mkdir -p "$ASGN_SERVER_DIR/container_printouts"
mkdir -p "$RESULTS_BASE_DIR"

# Required run settings
NETWORK_NAME=${NETWORK_NAME:?NETWORK_NAME must be set, e.g. NETWORK_NAME=saint_arnoult}

# Defaults can be overridden
DEFAULT_EXPERIMENT_NAME="${NETWORK_NAME}_asgn"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}
if [[ "$EXPERIMENT_NAME" == "$DEFAULT_EXPERIMENT_NAME" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
SEED_BASE=${SEED_BASE:-42}
PHASE=${PHASE:-sample}
if [[ "$PHASE" != "sample" && "$PHASE" != "aggregate" ]]; then
    echo "Unsupported ASGN phase: $PHASE. Use sample or aggregate." >&2
    exit 1
fi

ASGN_METHOD=
ASGN_NUM_SAMPLES=

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}" # e.g. 0-31
TASK_CONF=${TASK_CONF:?TASK_CONF must be set, e.g. TASK_CONF=asgn_100k_grid}
TASK_CONF_PATH="$PATH_PROGRAM/config/task_config/${TASK_CONF}.json"
if [[ ! -f "$TASK_CONF_PATH" ]]; then
    echo "Missing task config file: $TASK_CONF_PATH" >&2
    exit 1
fi

# Read sampling methods and corresponding numbers of samples
# Python script builds a task-id to method lookup, then returns the method and per-task sample count for TASK_INDEX.
if [[ "$PHASE" == "sample" ]]; then
    read -r ASGN_METHOD ASGN_NUM_SAMPLES ASGN_METHOD_TASK_INDEX < <(python3 - "$TASK_CONF_PATH" "$TASK_INDEX" <<'PY'
import json
import sys

task_config = json.load(open(sys.argv[1]))
task_index = int(sys.argv[2])

methods = ("greedy", "random", "balanced", "dirichlet", "grid", "hex")
task_cursor = 0
selected_method = None
selected_method_task_index = None
per_task_samples = None

for method in methods:
    num_tasks = int(task_config.get(f"num_tasks_{method}", 0))
    if num_tasks <= 0:
        continue

    total_samples = int(task_config.get(f"{method}_num_samples", 0))
    base, rem = divmod(total_samples, num_tasks)

    if task_cursor <= task_index < task_cursor + num_tasks:
        selected_method = method
        selected_method_task_index = task_index - task_cursor
        per_task_samples = base + (1 if selected_method_task_index < rem else 0)
        break

    task_cursor += num_tasks

if selected_method is None:
    raise SystemExit(f"Task index {task_index} is out of range for {sys.argv[1]}")

print(selected_method, per_task_samples, selected_method_task_index)

PY
    )
else
    ASGN_METHOD_TASK_INDEX=0
fi

# Use a deterministic but distinct seed per array task
ENV_SEED=$((SEED_BASE + TASK_INDEX))

# Build a per-task run id using only job id, method/phase, and task index.
if [[ -n "$ASGN_METHOD" ]]; then
    EXP_ID="${SLURM_ARRAY_JOB_ID}_${ASGN_METHOD}_${TASK_INDEX}"
else
    EXP_ID="${SLURM_ARRAY_JOB_ID}_${PHASE}_${TASK_INDEX}"
fi

# Run container by adding code by binding, run commands from run_asgn_single_internal.sh, save printouts to a file
singularity exec --cleanenv \
    --env TASK_ID="$TASK_INDEX" \
    --env EXP_ID="$EXP_ID" \
    --env EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    --env PHASE="$PHASE" \
    --env TASK_CONF="$TASK_CONF" \
    --env ASGN_METHOD="$ASGN_METHOD" \
    --env ASGN_NUM_SAMPLES="$ASGN_NUM_SAMPLES" \
    --env ASGN_METHOD_TASK_INDEX="$ASGN_METHOD_TASK_INDEX" \
    --env NETWORK_NAME="$NETWORK_NAME" \
    --env ENV_SEED="$ENV_SEED" \
    --env RESULTS_BASE_DIR="$RESULTS_BASE_DIR" \
    --bind "$RESULTS_BASE_DIR":"$RESULTS_BASE_DIR" \
    --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" \
    "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" > "$PRINTS_SAVE_PATH" 2>&1

# How to run: see run_asgn_full_pipeline.sh
