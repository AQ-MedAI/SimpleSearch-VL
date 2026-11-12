#!/usr/bin/env bash

: <<'USAGE'
============================================================

---- Training (2 nodes) ----
  Node 0 (head):   NODE_RANK=0 NNODES=2 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/qwen3vl8b_sft_dapo_2x_dynroll_4_6_150step.yaml
  Node 1 (worker): NODE_RANK=1 MASTER_ADDR="<head-node-ip>" NNODES=2 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/qwen3vl8b_sft_dapo_2x_dynroll_4_6_150step.yaml

---- Eval (2 nodes) ----
  Node 0 (head):   NODE_RANK=0 NNODES=2 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/eval_8B.yaml
  Node 1 (worker): NODE_RANK=1 MASTER_ADDR="<head-node-ip>" NNODES=2 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/eval_8B.yaml

---- Single-node (no need for NODE_RANK/MASTER_ADDR) ----
  NNODES=1 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/qwen3vl8b_sft_dapo_2x_dynroll_4_6_150step.yaml
  NNODES=1 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/eval_8B.yaml

---- Override params on the fly ----
  NODE_RANK=0 NNODES=2 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/qwen3vl8b_sft_dapo_2x_dynroll_4_6_150step.yaml trainer.total_epochs=2

============================================================
USAGE


# Do not print credentials or command arguments via shell tracing.
set +x
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RLLM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${1:-}" = "--check" ]; then
    exec bash "$SCRIPT_DIR/run.sh" "$@"
fi
CONFIG_FILE=${1:?"Usage: run_multi_node.sh <config_yaml> [extra hydra overrides...]"}
CONFIG_FILE=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
set -- "$CONFIG_FILE" "${@:2}"
export PYTHONPATH="${RLLM_DIR}/verl:${RLLM_DIR}:${PYTHONPATH:-}"

# Load credentials before Ray starts so its workers inherit the same environment.
ENV_FILE="${SIMPLESEARCH_ENV_FILE-${RLLM_DIR}/.env}"
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
elif [ -n "$ENV_FILE" ] && [ "${SIMPLESEARCH_ENV_FILE+x}" = x ]; then
    echo "ERROR: SIMPLESEARCH_ENV_FILE does not exist" >&2
    exit 1
fi
# run.sh will receive the already-loaded environment, without sourcing it again.
export SIMPLESEARCH_ENV_FILE=""
cd "$RLLM_DIR"
# Fail before stopping an existing Ray cluster or creating output directories.
bash "$SCRIPT_DIR/run.sh" --check "$@"
API_ENABLED=$(python3 - "$@" <<'PY_API'
import sys
from simple_agentic_search.config_validation import apply_overrides, load_config
cfg = apply_overrides(load_config(sys.argv[1]), sys.argv[2:])
print("true" if cfg.get("agent_model_api", {}).get("enable", False) else "false")
PY_API
)
if [ "$API_ENABLED" = true ]; then
    if [ "${NNODES:-1}" != 1 ] || [ "${NODE_RANK:-0}" != 0 ]; then
        echo "ERROR: API evaluation runs in one process; use run.sh instead" >&2
        exit 1
    fi
    exec bash "$SCRIPT_DIR/run.sh" "$@"
fi

# ==================== Node Configuration ====================
num_nodes=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}
if ! [[ "$num_nodes" =~ ^[1-9][0-9]*$ && "$NODE_RANK" =~ ^[0-9]+$ ]] || [ "$NODE_RANK" -ge "$num_nodes" ]; then
    echo "ERROR: require NNODES >= 1 and 0 <= NODE_RANK < NNODES" >&2
    exit 1
fi

# Auto-detect MASTER_ADDR on head node
if [ "$NODE_RANK" -eq 0 ] && [ -z "${MASTER_ADDR:-}" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi

if [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: MASTER_ADDR must be set for worker nodes"
    exit 1
fi

echo "=== Node Config: RANK=$NODE_RANK NNODES=$num_nodes MASTER=$MASTER_ADDR:$MASTER_PORT ==="

# ==================== Ray Setup ====================
for dependency in ray rsync; do
    command -v "$dependency" >/dev/null || { echo "ERROR: missing required command: $dependency" >&2; exit 1; }
done
ray stop --force 2>/dev/null || true
sleep 3

OBJECT_STORE_MEM=$(free -b | awk '/^Mem:/{printf "%.0f", $2 * 0.4}')

if [ "$NODE_RANK" -eq 0 ]; then
    # ==================== HEAD NODE ====================
    ray start --head --port=${MASTER_PORT} --dashboard-host="${RAY_DASHBOARD_HOST:-127.0.0.1}" --dashboard-port=8265 \
        --object-store-memory=$OBJECT_STORE_MEM --disable-usage-stats
    sleep 5

    # Wait for all nodes to join
    for ((i=1; i<=60; i++)); do
        CONNECTED=$(ray status 2>/dev/null | grep -c "node_" || true)
        echo "Connected nodes: $CONNECTED / $num_nodes"
        [ "$CONNECTED" -ge "$num_nodes" ] && break
        sleep 10
    done
    if [ "$CONNECTED" -lt "$num_nodes" ]; then
        echo "ERROR: timed out waiting for Ray worker nodes" >&2
        exit 1
    fi
    ray status

    # ==================== Compute scaled overrides ====================
    # Read base values from YAML, then scale by node count
    yq() {
        python3 - "$CONFIG_FILE" "$1" "${2:-}" <<'PY_VALUE'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    value = yaml.safe_load(stream)
for key in sys.argv[2].split("."):
    value = value.get(key) if isinstance(value, dict) else None
print(sys.argv[3] if value is None else value)
PY_VALUE
    }

    base_bsz=$(yq train_prompt_bsz 32)
    base_mini_bsz=$(yq train_prompt_mini_bsz 32)
    base_tasks=$(yq n_parallel_tasks 64)
    base_tools=$(yq n_parallel_tools 128)

    scaled_bsz=$((base_bsz * num_nodes))
    scaled_mini_bsz=$((base_mini_bsz * num_nodes))
    scaled_tasks=$((base_tasks * num_nodes))
    scaled_tools=$((base_tools * num_nodes))

    scaled_dapo_overrides=()
    dapo_enabled=$(yq dapo_sampling.enable False)
    if [ "$dapo_enabled" = "True" ] || [ "$dapo_enabled" = "true" ]; then
        base_dapo_target=$(yq dapo_sampling.target_batch_size "$base_bsz")
        base_dapo_target=${base_dapo_target:-$base_bsz}
        scaled_dapo_target=$((base_dapo_target * num_nodes))

        base_dapo_gen_batch=$(yq dapo_sampling.gen_batch_size "")
        if [ -n "$base_dapo_gen_batch" ]; then
            scaled_dapo_gen_batch=$((base_dapo_gen_batch * num_nodes))
        else
            dapo_multiplier=$(yq dapo_sampling.multiplier "")
            if [ -z "$dapo_multiplier" ]; then
                dapo_multiplier=$(yq dapo_sampling.over_sampling_multiplier 1.0)
            fi
            scaled_dapo_gen_batch=$(python3 -c "import math; print(int(math.ceil(${scaled_dapo_target} * float('${dapo_multiplier}'))))")
        fi
        if [ "$scaled_dapo_gen_batch" -lt "$scaled_dapo_target" ]; then
            scaled_dapo_gen_batch=$scaled_dapo_target
        fi
        scaled_dapo_overrides=(
            "++rllm.dapo_sampling.target_batch_size=${scaled_dapo_target}"
            "data.gen_batch_size=${scaled_dapo_gen_batch}"
        )
        echo "=== Scaled DAPO for $num_nodes nodes: target=$scaled_dapo_target gen_batch=$scaled_dapo_gen_batch ==="
    fi

    echo "=== Scaled for $num_nodes nodes: bsz=$scaled_bsz tasks=$scaled_tasks tools=$scaled_tools ==="

    # ==================== Launch training via run.sh ====================
    cd "$RLLM_DIR"
    exec bash simple_agentic_search/run/run.sh "$@" \
        "${scaled_dapo_overrides[@]}" \
        trainer.nnodes=${num_nodes} \
        data.train_batch_size=${scaled_bsz} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${scaled_mini_bsz} \
        rllm.workflow.n_parallel_tasks=${scaled_tasks} \
        rllm.workflow.n_parallel_tools=${scaled_tools}

else
    # ==================== WORKER NODE ====================
    sleep 30
    worker_connected=false
    for ((i=1; i<=10; i++)); do
        ray start --address="${MASTER_ADDR}:${MASTER_PORT}" \
            --object-store-memory=$OBJECT_STORE_MEM --disable-usage-stats && { worker_connected=true; break; }
        echo "Retrying ray start ($i/10)..."
        sleep 30
    done
    if [ "$worker_connected" != true ]; then
        echo "ERROR: worker could not join the Ray head after retries" >&2
        exit 1
    fi
    echo "Worker node $NODE_RANK connected to $MASTER_ADDR:$MASTER_PORT"
    sleep infinity
fi

# burning