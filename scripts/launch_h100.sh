#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_h100.sh – DeepSpeed multi-GPU launcher for MultiMini-1.2B on H100s
#
# Single-node (all local GPUs):
#   bash scripts/launch_h100.sh --manifest data/chat_reasoning.jsonl
#
# Multi-node (example: 2 nodes × 8 GPUs):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 bash scripts/launch_h100.sh \
#       --manifest data/chat_reasoning.jsonl
#
# All extra args are forwarded to train.py (--compile, --no_flash_attn, etc.)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NGPU_PER_NODE="${NGPU_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# H100 NVLink + hopper-specific env
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_MIN_NCHANNELS=4
export NCCL_SOCKET_NTHREADS=4
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# Enable NVLink-aware collective comms
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

deepspeed \
    --num_gpus    "${NGPU_PER_NODE}" \
    --num_nodes   "${NNODES}" \
    --node_rank   "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --config           configs/model_1p2b.yaml \
    --deepspeed_config configs/deepspeed_zero3_bf16.json \
    --tokenizer        HuggingFaceTB/SmolLM2-360M-Instruct \
    --compile \
    "$@"
