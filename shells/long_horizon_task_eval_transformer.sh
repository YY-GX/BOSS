#!/bin/zsh

cd /mnt/arc/yygx/pkgs_baselines/LIBERO/ || exit

for task_order_index in {7..16}; do
    echo "Starting task_order_index: $task_order_index"
    export CUDA_VISIBLE_DEVICES=6 && export MUJOCO_EGL_DEVICE_ID=6 && python libero/lifelong/eval_skill_chain_new.py --task_order_index $task_order_index --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/skill_policies_without_wrist_camera_view/Sequential/BCTransformerPolicy_seed10000/all"
    echo "Completed task_order_index: $task_order_index"
done
