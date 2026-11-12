#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$PROJECT_ROOT/ms-swift"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=32 \
NNODES=${WORLD_SIZE:-1} \
NODE_RANK=${RANK:-0} \
NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
megatron sft \
    --model "$PROJECT_ROOT/checkpoints/Qwen3-VL-8B-Instruct" \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "$PROJECT_ROOT/data/SimpleSearch-VL/SimpleSearch-VL-SFT-5K/sft_ms_swift.jsonl" \
    --load_from_cache_file true \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 2e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --save checkpoints/SFT_8B_megatron_lr2e5_3ep_evidenceversion_mf_format_5k_final \
    --tensorboard_dir /tmp/openmmseeker_tensorboard/SFT_8B_megatron_lr2e5_3ep_evidenceversion_mf_format_5k_final \
    --eval_interval 50000000 \
    --save_interval 500 \
    --max_length 64000 \
    --packing false \
    --num_workers 8 \
    --dataset_num_proc 128 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --attention_backend flash
