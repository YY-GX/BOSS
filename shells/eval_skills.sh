#!/bin/zsh

# Change to the specified directory
cd /mnt/arc/yygx/pkgs_baselines/LIBERO/libero || { echo "Directory change failed"; exit 1; }

# Execute the sbatch command with the wrapped commands
sbatch --gpus 1 --cpus-per-task=8 \
       -o /mnt/arc/yygx/pkgs_baselines/LIBERO/libero/outs/debug_%j.out \
       -J eval \
       --wrap="export CUDA_VISIBLE_DEVICES=0 && \
               export MUJOCO_EGL_DEVICE_ID=0 && \
               python lifelong/evaluate.py --run_id 2 --task_id 0 --load_task 0 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 1 --load_task 1 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 2 --load_task 2 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 3 --load_task 3 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 4 --load_task 4 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 5 --load_task 5 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 6 --load_task 6 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 7 --load_task 7 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 8 --load_task 8 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos && \
               python lifelong/evaluate.py --run_id 2 --task_id 9 --load_task 9 --version 'train_base_90_v0' --seed 10000 --device_id 0 --benchmark 'libero_90' --policy 'bc_transformer_policy' --algo 'base' --save-videos"


#export CUDA_VISIBLE_DEVICES=1 && \
#export MUJOCO_EGL_DEVICE_ID=1 && \
#python lifelong/evaluate.py --run_id 3 --task_id 0 --load_task 0 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 1 --load_task 1 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 2 --load_task 2 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 3 --load_task 3 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 4 --load_task 4 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 5 --load_task 5 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 6 --load_task 6 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 7 --load_task 7 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 8 --load_task 8 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos && \
#python lifelong/evaluate.py --run_id 3 --task_id 9 --load_task 9 --version "train_base_90_v0" --seed 10000 --device_id 0 --benchmark "libero_90" --policy "bc_transformer_policy" --algo "base" --save-videos


