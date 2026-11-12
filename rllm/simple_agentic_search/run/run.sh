#!/usr/bin/env bash
# ============================================================
# Universal launcher for simple_agentic_search experiments.
#
# Usage:
#   bash run/run.sh <config_yaml>  [extra hydra overrides...]
#
# Examples:
#   bash simple_agentic_search/run/run.sh --check simple_agentic_search/config/local/eval_api.yaml
#   bash simple_agentic_search/run/run.sh simple_agentic_search/config/local/eval_api.yaml
# ============================================================
# Never trace sourced credentials, even if invoked with bash -x.
set +x
set -euo pipefail

CHECK_ONLY=false
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=true
    shift
fi
CONFIG_FILE=${1:?"Usage: run.sh [--check] <config_yaml> [extra hydra overrides...]"}
shift
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found" >&2
    exit 1
fi
CONFIG_FILE=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
RLLM_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export SIMPLESEARCH_CONFIG_FILE="$CONFIG_FILE"
export PYTHONPATH="${RLLM_DIR}/verl:${RLLM_DIR}:${PYTHONPATH:-}"

# Source only a trusted local environment file, never include it in snapshots.
# An explicitly empty SIMPLESEARCH_ENV_FILE disables file loading (env-only use).
ENV_FILE="${SIMPLESEARCH_ENV_FILE-${RLLM_DIR}/.env}"
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
elif [ -n "$ENV_FILE" ] && [ "${SIMPLESEARCH_ENV_FILE+x}" = x ]; then
    echo "ERROR: SIMPLESEARCH_ENV_FILE does not exist" >&2
    exit 1
fi

# All model/data/cache/output paths in YAML are relative to rllm/, regardless
# of the caller's working directory. No GPU/Ray/API calls happen in this check.
cd "$RLLM_DIR"
python3 -m simple_agentic_search.config_validation "$CONFIG_FILE" "$@"
if [ "$CHECK_ONLY" = true ]; then
    exit 0
fi

# ========= helper: read a YAML key via python (flat keys only) =========
yq() {
    python3 - "$CONFIG_FILE" "$1" "${2:-}" <<'PY_VALUE'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    value = yaml.safe_load(stream)
for key in sys.argv[2].split("."):
    value = value.get(key) if isinstance(value, dict) else None
if value is None:
    print(sys.argv[3])
elif isinstance(value, bool):
    print("True" if value else "False")
else:
    print(value)
PY_VALUE
}

# ========= read experiment config =========
mode=$(yq mode train)
rollout_mode=$(yq rollout_mode async)
rollout_name=$(yq rollout_name sglang)
dtype=$(yq dtype bfloat16)
adv_estimator=$(yq adv_estimator rloo)

kl_coef=$(yq kl_coef 0.001)
use_kl_loss=$(yq use_kl_loss False)
kl_loss_coef=$(yq kl_loss_coef 0.001)
clip_ratio_high=$(yq clip_ratio_high 0.28)
loss_agg_mode=$(yq loss_agg_mode seq-mean-token-sum)

train_prompt_bsz=$(yq train_prompt_bsz 16)
val_batch_size=$(yq val_batch_size 512)
n_resp_per_prompt=$(yq n_resp_per_prompt 8)
train_prompt_mini_bsz=$(yq train_prompt_mini_bsz 16)
n_parallel_tasks=$(yq n_parallel_tasks 32)
n_parallel_tools=$(yq n_parallel_tools 64)

max_prompt_length=$(yq max_prompt_length 4096)
max_response_length=$(yq max_response_length 32000)
use_dynamic_bsz=$(yq use_dynamic_bsz True)
actor_ppo_max_token_len_per_gpu=$(yq actor_ppo_max_token_len_per_gpu 40000)
infer_ppo_max_token_len_per_gpu=$(yq infer_ppo_max_token_len_per_gpu 40000)

offload=$(yq offload False)
gen_tp=$(yq gen_tp 2)
train_tp=$(yq train_tp 2)
train_pp=$(yq train_pp 2)
train_ep=$(yq train_ep 1)
train_etp=$(yq train_etp 1)
train_cp=$(yq train_cp 1)

temperature=$(yq temperature 0.7)
top_p=$(yq top_p 1.0)
top_k=$(yq top_k -1)
presence_penalty=$(yq presence_penalty 1.5)
repetition_penalty=$(yq repetition_penalty 1.0)
val_temperature=$(yq val_temperature "${temperature}")
val_top_p=$(yq val_top_p 0.95)
val_top_k=$(yq val_top_k "${top_k}")
val_presence_penalty=$(yq val_presence_penalty 1.5)
val_repetition_penalty=$(yq val_repetition_penalty 1.0)

project_name=$(yq project_name mars-agentic-search)
exp_name=$(yq exp_name experiment)
MODEL_PATH=$(yq model_path "")

# Build hydra list overrides for train_data / test_data
mapfile -d '' -t DATA_OVERRIDES < <(python3 - "$CONFIG_FILE" <<'PY_DATA'
import json
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)
for field in ("train_data", "test_data"):
    if field in cfg:
        value = json.dumps(cfg[field], separators=(",", ":"))
        sys.stdout.write(f"+rllm.workflow.workflow_args.{field}={value}\0")
PY_DATA
)

total_epochs=$(yq total_epochs 1)
total_training_steps=$(yq total_training_steps null)
save_freq=$(yq save_freq 10)
test_freq=$(yq test_freq 100)
val_before_train=$(yq val_before_train False)
n_gpus_per_node=$(yq n_gpus_per_node 8)
NNODES=$(yq nnodes 1)
log_backend=$(yq logger '["console","wandb"]')

gpu_memory_utilization=$(yq gpu_memory_utilization 0.50)
enforce_eager=$(yq enforce_eager True)
lr=$(yq lr 1e-6)
lr_warmup_steps_ratio=$(yq lr_warmup_steps_ratio 0.0)
lr_decay_style=$(yq lr_decay_style constant)
min_lr=$(yq min_lr 0.0)
stepwise_advantage_enable=$(yq stepwise_advantage.enable False)
stepwise_advantage_mode=$(yq stepwise_advantage.mode broadcast)
stepwise_advantage_normalize_by_steps=$(yq stepwise_advantage.normalize_by_steps False)

# Optional train-time overrides, used by 30B-A3B MoE configs and inert when absent.
actor_optim_optimizer_offload_fraction=$(yq actor_optim_override.optimizer_offload_fraction "")
actor_optim_overlap_cpu_optimizer_d2h_h2d=$(yq actor_optim_override.overlap_cpu_optimizer_d2h_h2d "")
actor_optim_use_precision_aware_optimizer=$(yq actor_optim_override.use_precision_aware_optimizer "")
actor_optim_optimizer_cpu_offload=$(yq actor_optim_override.optimizer_cpu_offload "")
actor_megatron_moe_router_dtype=$(yq actor_megatron_override.moe_router_dtype "")
actor_megatron_gradient_accumulation_fusion=$(yq actor_megatron_override.gradient_accumulation_fusion False)
actor_megatron_moe_permute_fusion=$(yq actor_megatron_override.moe_permute_fusion "")
actor_megatron_moe_aux_loss_coeff=$(yq actor_megatron_override.moe_aux_loss_coeff "")
actor_megatron_moe_z_loss_coeff=$(yq actor_megatron_override.moe_z_loss_coeff "")

OPTIONAL_ACTOR_OVERRIDES=()
add_optional_override() {
    local key="$1"
    local value="$2"
    if [ -n "$value" ]; then
        OPTIONAL_ACTOR_OVERRIDES+=("+${key}=${value}")
    fi
}
add_optional_override actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction "$actor_optim_optimizer_offload_fraction"
add_optional_override actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d "$actor_optim_overlap_cpu_optimizer_d2h_h2d"
add_optional_override actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer "$actor_optim_use_precision_aware_optimizer"
add_optional_override actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload "$actor_optim_optimizer_cpu_offload"
add_optional_override actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype "$actor_megatron_moe_router_dtype"
add_optional_override actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion "$actor_megatron_moe_permute_fusion"
add_optional_override actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff "$actor_megatron_moe_aux_loss_coeff"
add_optional_override actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff "$actor_megatron_moe_z_loss_coeff"

if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
else
    return_raw_chat="False"
fi

default_local_dir=$(yq default_local_dir "")
if [ -n "$default_local_dir" ]; then
    CKPTS_DIR="$default_local_dir"
else
    CKPTS_DIR="checkpoints/${project_name}/${exp_name}"
fi
rollout_save_dir_cfg=$(yq rollout_save_dir "")
ROLLOUT_SAVE_OVERRIDE=""
if [ -n "$rollout_save_dir_cfg" ]; then
    ROLLOUT_SAVE_OVERRIDE="+rllm.workflow.workflow_args.rollout_save_dir=${rollout_save_dir_cfg}"
fi

# ========= wandb offline logging =========
# Keep syncable offline wandb runs next to checkpoints/rollouts.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${CKPTS_DIR}/wandb}"
command -v rsync >/dev/null || { echo "ERROR: rsync is required for code snapshots" >&2; exit 1; }
mkdir -p "${WANDB_DIR}"

# ========= snapshot core code to checkpoint dir =========
SNAPSHOT_DIR="${CKPTS_DIR}/code_snapshot"
mkdir -p "${SNAPSHOT_DIR}"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='cache/' \
    --exclude='config/local/' --exclude='*.local.yaml' --exclude='*.local.yml' \
    --exclude='.env' --exclude='.env.*' \
    simple_agentic_search/ "${SNAPSHOT_DIR}/simple_agentic_search/"
# Active config was checked to reject inline credentials; private endpoints may
# still appear here. Treat snapshots and rollout outputs as private artifacts.
cp "${CONFIG_FILE}" "${SNAPSHOT_DIR}/"
echo "=== Code snapshot saved to ${SNAPSHOT_DIR} ==="

# ========= read agent_config keys =========
agent_max_llm_calls=$(yq agent_config.max_llm_calls 10)
agent_max_repeat_turn=$(yq agent_config.max_repeat_turn 3)
agent_max_prompt_tokens=$(yq agent_config.max_prompt_tokens 32000)
agent_max_response_tokens=$(yq agent_config.max_response_tokens 4096)
agent_tool_call_timeout=$(yq agent_config.tool_call_timeout 120)
agent_run_timeout=$(yq agent_config.agent_run_timeout 360)
agent_force_final_answer_timeout=$(yq agent_config.force_final_answer_timeout 60)
agent_llm_call_timeout=$(yq agent_config.llm_call_timeout "")
agent_llm_max_retries=$(yq agent_config.llm_max_retries "")
agent_image_search_first=$(yq agent_config.image_search_first False)
workflow_timeout=$(yq workflow_timeout "${agent_run_timeout}")
agent_tool_max_retries=$(yq agent_config.tool_max_retries 3)

OPTIONAL_AGENT_OVERRIDES=()
add_optional_agent_override() {
    local key="$1"
    local value="$2"
    if [ -n "$value" ]; then
        OPTIONAL_AGENT_OVERRIDES+=("+rllm.workflow.workflow_args.agent_config.${key}=${value}")
    fi
}
add_optional_agent_override llm_call_timeout "$agent_llm_call_timeout"
add_optional_agent_override llm_max_retries "$agent_llm_max_retries"

# ========= read tools_config keys =========
tools_extract_max_tokens=$(yq tools_config.extract_max_tokens 8192)
tools_extract_mode=$(yq tools_config.extract_mode text)
tools_search_max_queries=$(yq tools_config.search_max_queries 3)
tools_search_max_urls=$(yq tools_config.search_max_urls 3)
tools_search_cache_max_urls=$(yq tools_config.search_cache_max_urls 20)
tools_visit_max_urls=$(yq tools_config.visit_max_urls 3)
tools_visit_max_content_chars=$(yq tools_config.visit_max_content_chars 60000)
tools_api_max_retries=$(yq tools_config.api_max_retries 3)
tools_cache_db_path=$(yq tools_config.cache_db_path "simple_agentic_search/cache/mars_cache.db")

# ========= read enabled_tools + image_search config =========
# enabled_tools is a YAML list; extract as comma-separated for Hydra list override
ENABLED_TOOLS_OVERRIDE=$(python3 -c "
import yaml
with open(__import__('os').environ['SIMPLESEARCH_CONFIG_FILE']) as f:
    cfg = yaml.safe_load(f)
tc = cfg.get('tools_config', {})
et = tc.get('enabled_tools')
if et:
    print('+rllm.workflow.workflow_args.tools_config.enabled_tools=[' + ','.join(et) + ']')
else:
    print('')
")
# image_search nested config
REGION_SEARCH_OVERRIDES=$(python3 -c "
import yaml
with open(__import__('os').environ['SIMPLESEARCH_CONFIG_FILE']) as f:
    cfg = yaml.safe_load(f)
tc = cfg.get('tools_config', {})
rs = tc.get('image_search', {})
parts = []
for k, v in rs.items():
    if isinstance(v, dict):
        # nested dict (e.g. cache_dataset_mapping) — pass as Hydra dict
        items = ','.join(f'{dk}:{dv}' for dk, dv in v.items())
        parts.append(f'+rllm.workflow.workflow_args.tools_config.image_search.{k}={{{items}}}')
    else:
        parts.append(f'+rllm.workflow.workflow_args.tools_config.image_search.{k}={v}')
print(' '.join(parts))
")

# zoom_in nested config
ZOOM_IN_OVERRIDES=$(python3 -c "
import yaml
with open(__import__('os').environ['SIMPLESEARCH_CONFIG_FILE']) as f:
    cfg = yaml.safe_load(f)
tc = cfg.get('tools_config', {})
zc = tc.get('zoom_in', {})
parts = []
def encode(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, list):
        return '[' + ','.join(str(x) for x in value) + ']'
    if value is None:
        return 'null'
    return str(value)
for k, v in zc.items():
    parts.append(f'+rllm.workflow.workflow_args.tools_config.zoom_in.{k}={encode(v)}')
print(' '.join(parts))
")

# rollout sampling + mastered_sample_filter → rllm.xxx Hydra overrides
RLLM_EXTRA_OVERRIDES=$(python3 -c "
import yaml
with open(__import__('os').environ['SIMPLESEARCH_CONFIG_FILE']) as f:
    cfg = yaml.safe_load(f)
parts = []
for section in ('dynamic_rollout', 'dapo_sampling', 'mastered_sample_filter'):
    sec = cfg.get(section)
    if sec is None:
        continue
    for k, v in sec.items():
        if v is None:
            parts.append(f'++rllm.{section}.{k}=null')
        else:
            parts.append(f'++rllm.{section}.{k}={v}')
print(' '.join(parts))
")

DAPO_DATA_OVERRIDES=$(python3 -c "
import math
import yaml
with open(__import__('os').environ['SIMPLESEARCH_CONFIG_FILE']) as f:
    cfg = yaml.safe_load(f)
dapo = cfg.get('dapo_sampling') or {}
if dapo.get('enable', False):
    target = int(dapo.get('target_batch_size') or cfg.get('train_prompt_bsz', 16))
    explicit = dapo.get('gen_batch_size')
    if explicit is not None:
        gen_batch_size = int(explicit)
    else:
        multiplier = float(dapo.get('multiplier', dapo.get('over_sampling_multiplier', 1.0)))
        gen_batch_size = int(math.ceil(target * multiplier))
    print(f'data.gen_batch_size={max(target, gen_batch_size)}')
else:
    print('')
")

export SIMPLESEARCH_CHECKPOINT_DIR="$CKPTS_DIR"

# reward config → rllm.workflow.workflow_args.reward Hydra overrides
# Supports nested reward config, e.g. reward.global_efficiency.enable=true.
REWARD_OVERRIDES=$(python3 -c "
import yaml

with open(__import__('os').environ['SIMPLESEARCH_CONFIG_FILE']) as f:
    cfg = yaml.safe_load(f)

reward = cfg.get('reward')
if reward is None:
    print('')
else:
    if isinstance(reward, dict):
        ge = reward.get('global_efficiency')
        if isinstance(ge, dict) and ge.get('enable', False) and not ge.get('state_path'):
            ge['state_path'] = __import__('os').path.join(__import__('os').environ['SIMPLESEARCH_CHECKPOINT_DIR'], 'global_efficiency_state.json')

    def encode(value):
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if value is None:
            return 'null'
        return str(value)

    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f'{prefix}.{key}' if prefix else str(key)
                yield from flatten(child_prefix, child)
        else:
            yield f'+rllm.workflow.workflow_args.reward.{prefix}={encode(value)}'

    print(' '.join(flatten('', reward)))
")

# agent model API config -> rllm.workflow.workflow_args.agent_model_api Hydra overrides
mapfile -d '' -t AGENT_MODEL_API_OVERRIDES < <(python3 - "$CONFIG_FILE" <<'PY_API_CFG'
import sys
import yaml
from simple_agentic_search.config_validation import encode_hydra_value
with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)
for key, value in (cfg.get("agent_model_api") or {}).items():
    if key == "api_key":
        continue  # Credentials are resolved from the named environment variable in Python.
    encoded = encode_hydra_value(value)
    sys.stdout.write(f"+rllm.workflow.workflow_args.agent_model_api.{key}={encoded}\0")
PY_API_CFG
)


# ========= environment variables =========
export SGLANG_DISABLE_TORCH_MEMORY_SAVER=1
export TORCH_MEMORY_SAVER_DISABLE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TRANSFORMER_ENGINE_DISABLED=1

export MAX_PIXELS=$(yq max_pixels 8294400)
export MIN_PIXELS=$(yq min_pixels 65536)
export CACHE_DB_PATH="${tools_cache_db_path}"

# ========= eval-only overrides =========
EVAL_OVERRIDES=""
if [ "$mode" = "eval" ]; then
    EVAL_OVERRIDES="+trainer.val_only=True"
fi

# ========= launch =========
echo "=== Launching: project=${project_name} exp=${exp_name} mode=${mode} ==="
python -m simple_agentic_search.train \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.train_batch_size=${train_prompt_bsz} \
    ${DAPO_DATA_OVERRIDES} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    data.seed=3407 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    +actor_rollout_ref.rollout.presence_penalty=${presence_penalty} \
    +actor_rollout_ref.rollout.repetition_penalty=${repetition_penalty} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    +actor_rollout_ref.rollout.val_kwargs.presence_penalty=${val_presence_penalty} \
    +actor_rollout_ref.rollout.val_kwargs.repetition_penalty=${val_repetition_penalty} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.megatron.dtype=${dtype} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.megatron.dtype=${dtype} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.optim.lr_decay_style=${lr_decay_style} \
    actor_rollout_ref.actor.optim.min_lr=${min_lr} \
    \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=${actor_megatron_gradient_accumulation_fusion} \
    "${OPTIONAL_ACTOR_OVERRIDES[@]}" \
    \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.workflow.workflow_args.timeout=${workflow_timeout} \
    "${DATA_OVERRIDES[@]}" \
    ${ROLLOUT_SAVE_OVERRIDE} \
    +rllm.workflow.workflow_args.agent_config.max_llm_calls=${agent_max_llm_calls} \
    +rllm.workflow.workflow_args.agent_config.max_repeat_turn=${agent_max_repeat_turn} \
    +rllm.workflow.workflow_args.agent_config.max_prompt_tokens=${agent_max_prompt_tokens} \
    +rllm.workflow.workflow_args.agent_config.max_response_tokens=${agent_max_response_tokens} \
    +rllm.workflow.workflow_args.agent_config.tool_call_timeout=${agent_tool_call_timeout} \
    +rllm.workflow.workflow_args.agent_config.agent_run_timeout=${agent_run_timeout} \
    +rllm.workflow.workflow_args.agent_config.force_final_answer_timeout=${agent_force_final_answer_timeout} \
    +rllm.workflow.workflow_args.agent_config.tool_max_retries=${agent_tool_max_retries} \
    "${OPTIONAL_AGENT_OVERRIDES[@]}" \
    +rllm.workflow.workflow_args.agent_config.image_search_first=${agent_image_search_first} \
    +rllm.workflow.workflow_args.tools_config.extract_max_tokens=${tools_extract_max_tokens} \
    +rllm.workflow.workflow_args.tools_config.extract_mode=${tools_extract_mode} \
    +rllm.workflow.workflow_args.tools_config.search_max_queries=${tools_search_max_queries} \
    +rllm.workflow.workflow_args.tools_config.search_max_urls=${tools_search_max_urls} \
    +rllm.workflow.workflow_args.tools_config.search_cache_max_urls=${tools_search_cache_max_urls} \
    +rllm.workflow.workflow_args.tools_config.visit_max_urls=${tools_visit_max_urls} \
    +rllm.workflow.workflow_args.tools_config.visit_max_content_chars=${tools_visit_max_content_chars} \
    +rllm.workflow.workflow_args.tools_config.api_max_retries=${tools_api_max_retries} \
    ${ENABLED_TOOLS_OVERRIDE} \
    ${REGION_SEARCH_OVERRIDES} \
    ${ZOOM_IN_OVERRIDES} \
    ${REWARD_OVERRIDES} \
    "${AGENT_MODEL_API_OVERRIDES[@]}" \
    ${RLLM_EXTRA_OVERRIDES} \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_unknown=True \
    rllm.compact_filtering.mask_error=True \
    rllm.rejection_sample.enable=$(yq rejection_sample.enable False) \
    rllm.rejection_sample.multiplier=$(yq rejection_sample.multiplier 1.0) \
    rllm.stepwise_advantage.enable=${stepwise_advantage_enable} \
    rllm.stepwise_advantage.mode=${stepwise_advantage_mode} \
    rllm.stepwise_advantage.normalize_by_steps=${stepwise_advantage_normalize_by_steps} \
    \
    trainer.critic_warmup=0 \
    trainer.logger="${log_backend}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    ${EVAL_OVERRIDES} \
    "$@"
