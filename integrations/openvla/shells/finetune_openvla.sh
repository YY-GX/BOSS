#!/usr/bin/env bash
# LoRA fine-tune OpenVLA on a BOSS dataset. Run from inside your OpenVLA checkout.
#
#   DATASET_NAME=libero44       -> paper Table I, Setup A (44 original skills)
#   DATASET_NAME=libero_bl3_all -> paper Table I, Setup B (RAMG-augmented)
#
# Override any variable from the environment, e.g.
#   GPUS=8 DATASET_NAME=libero_bl3_all bash finetune_openvla.sh
set -euo pipefail

GPUS="${GPUS:-4}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
DATASET_NAME="${DATASET_NAME:-libero44}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-./datasets}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./runs/${DATASET_NAME}/1.0.0}"
ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-./adapter-tmp/${DATASET_NAME}/1.0.0}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
IMAGE_AUG="${IMAGE_AUG:-True}"
SAVE_STEPS="${SAVE_STEPS:-5000}"

WANDB_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS=(--wandb_project "${WANDB_PROJECT:-openvla-boss}" --wandb_entity "${WANDB_ENTITY}")
fi

torchrun --standalone --nnodes 1 --nproc-per-node "${GPUS}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --adapter_tmp_dir "${ADAPTER_TMP_DIR}" \
  --lora_rank "${LORA_RANK}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --image_aug "${IMAGE_AUG}" \
  --save_steps "${SAVE_STEPS}" \
  "${WANDB_ARGS[@]}"
