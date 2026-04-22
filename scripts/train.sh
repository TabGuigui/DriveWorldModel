#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/cogvideox/nuscenes_history_traj_train.yaml}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/zero2.json}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PWD}/diffusers/src:${PWD}:${PYTHONPATH:-}"

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  --config "${CONFIG_PATH}" \
  --mixed-precision "${MIXED_PRECISION}" \
  --deepspeed-config "${DEEPSPEED_CONFIG}"
