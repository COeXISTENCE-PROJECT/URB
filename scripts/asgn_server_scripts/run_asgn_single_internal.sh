#!/bin/bash
set -e # tells bash that it should exit the script if any statement returns a non-true return value

cd /app/

PHASE=${PHASE:-sample}
if [ "$PHASE" != "sample" ] && [ "$PHASE" != "aggregate" ]; then
    echo "Unsupported ASGN phase: $PHASE. Use sample or aggregate." >&2
    exit 1
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set by the launcher}
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:?RESULTS_BASE_DIR must be set by the launcher}
NETWORK_NAME=${NETWORK_NAME:?NETWORK_NAME must be set by the launcher}

if [ "$PHASE" != "aggregate" ]; then
    TASK_ID=${TASK_ID:?TASK_ID must be set by the launcher}
    EXP_ID=${EXP_ID:?EXP_ID must be set by the launcher}
    TASK_CONF=${TASK_CONF:?TASK_CONF must be set by the launcher}
    ENV_SEED=${ENV_SEED:?ENV_SEED must be set by the launcher}
fi

# Shared venv settings
VENV_DIR="/app/.venv"
REQ_FILE="/app/requirements.txt"
LOCK_DIR="${VENV_DIR}.lock"
READY_FILE="${VENV_DIR}/.ready"

# If venv is not fully ready, have exactly one task build it
if [ ! -f "$READY_FILE" ]; then
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "Creating shared venv at $VENV_DIR"

        cleanup() {
            rmdir "$LOCK_DIR" 2>/dev/null || true
        }
        trap cleanup EXIT # auto cleanup (remove lock) on termination/error - and auto error because of set -e. Won't handle kill -9 or node failure though

        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        source "${VENV_DIR}/bin/activate"

        python -m pip install --upgrade pip setuptools wheel
        python -m pip install --no-cache-dir -r "$REQ_FILE"

        touch "$READY_FILE"
        echo "Shared venv ready"
    else
        echo "Another task is preparing the shared venv; waiting..."
        while [ ! -f "$READY_FILE" ]; do
            if [ ! -d "$LOCK_DIR" ]; then
                echo "Shared venv setup appears to have failed: lock disappeared but $READY_FILE was not created." >&2
                exit 1
            fi
            sleep 2
        done
    fi
fi

# Activate shared venv for this task
source "${VENV_DIR}/bin/activate"

if [ "$PHASE" = "aggregate" ]; then
    echo "--- Running aggregation for $EXPERIMENT_NAME ---"
    python -u scripts/asgn_aggregate.py \
    --experiment-id "$EXPERIMENT_NAME" \
    --results-base-dir "$RESULTS_BASE_DIR"
    exit 0
fi

# Use /scratch (local bonk 2TB disk) for saving the results
EXPERIMENT_ROOT="$RESULTS_BASE_DIR/$EXPERIMENT_NAME"
BONK_SCRATCH_RESULTS="$EXPERIMENT_ROOT/$EXP_ID"
mkdir -p "$BONK_SCRATCH_RESULTS"
export RESULTS_BASE_DIR
echo "--- Results will be stored locally at: $BONK_SCRATCH_RESULTS ---"

TASK_CONF_PATH="/app/config/task_config/${TASK_CONF}.json"
if [ ! -f "$TASK_CONF_PATH" ]; then
    echo "Missing task config file: $TASK_CONF_PATH" >&2
    exit 1
fi

echo "--- Running: $EXP_ID | Env Seed: $ENV_SEED ---"
python -u scripts/asgn_simulations.py \
--id "$EXP_ID" \
--net "$NETWORK_NAME" \
--task-conf "$TASK_CONF" \
--mode sample \
--env-seed "$ENV_SEED"
