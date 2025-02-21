#!/bin/zsh

# Define variables
CPUS_PER_TASK=32
GPUS=4
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla/logs"
LOG_FILE="train_libero44_v0_%j.out"
JOB_NAME="openvla_libero44"
SCRIPT_PATH="/mnt/arc/yygx/pkgs_baselines/openvla/vla-scripts/finetune.py"
VLA_PATH="openvla/openvla-7b"
DATA_ROOT_DIR="/mnt/arc/yygx/pkgs_baselines/openvla/datasets"
DATASET_NAME="libero44"
RUN_ROOT_DIR="/mnt/arc/yygx/pkgs_baselines/openvla/runs/1.0.0"
ADAPTER_TMP_DIR="/mnt/arc/yygx/pkgs_baselines/openvla/adapter-tmp/1.0.0"
LORA_RANK=32
BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-4
IMAGE_AUG=True
WANDB_PROJECT="openvla-libero"
WANDB_ENTITY="yygx"
SAVE_STEPS=5000

# Submit the job
sbatch \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS $SCRIPT_PATH \
    --vla_path $VLA_PATH \
    --data_root_dir $DATA_ROOT_DIR \
    --dataset_name $DATASET_NAME \
    --run_root_dir $RUN_ROOT_DIR \
    --adapter_tmp_dir $ADAPTER_TMP_DIR \
    --lora_rank $LORA_RANK \
    --batch_size $BATCH_SIZE \
    --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --image_aug $IMAGE_AUG \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --save_steps $SAVE_STEPS"
