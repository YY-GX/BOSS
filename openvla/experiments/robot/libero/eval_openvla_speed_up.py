"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
"""

import os
# current_working_directory = os.getcwd()
# print(os.getcwd())
# os.chdir("../../../../")
# print(os.getcwd())
from libero.libero import benchmark
from libero.libero.utils.video_utils import VideoWriter
# os.chdir(current_working_directory)
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import numpy as np
import tqdm

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../../")
# sys.path.append("../../../")
# sys.path.append("openvla/")
from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_dummy_action_parallel,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    get_libero_subproc_env,
    get_batch_action_given_batch_obs,
)
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

import argparse
from pathlib import Path
from typing import Optional, Union



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for LIBERO tasks")

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla/runs/libero44/1.0.0/openvla-7b+libero44+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug",
        # "runs/bl3_all/1.0.0/openvla-7b+libero_bl3_all+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug",  # data_aug
        help="Pretrained checkpoint path",
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load with 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load with 4-bit quantization")
    parser.add_argument("--center_crop", action="store_true", default=True, help="Center crop images (default: True)")

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="boss_44",
        help="Task suite. Options: boss_44, ch1, ch2_2_modifications, ch2_3_modifications, data_augmentation, libero_90",
    )
    parser.add_argument("--num_steps_wait", type=int, default=5, help="Number of steps to wait in sim")
    parser.add_argument("--num_trials_per_task", type=int, default=20, help="Number of rollouts per task")

    #################################################################################################################
    # Utils
    #################################################################################################################
    parser.add_argument("--run_id_note", type=str, default=None, help="Extra note to add in run ID for logging")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs/", help="Directory for eval logs")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for reproducibility")

    return parser.parse_args()



def eval_libero(cfg):
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    # cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # yy: TODO: I don't get what this part is about
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        # if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        #     cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        # cfg.unnorm_key = "libero44"
        cfg.unnorm_key = "libero_bl3_all"
        print(f"model.norm_stats: {model.norm_stats}")
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}-{cfg.seed}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(os.path.join(cfg.local_log_dir, run_id), exist_ok=True)
    local_log_save_dir = os.path.join(cfg.local_log_dir, run_id)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    tasks_success_list = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        # env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        # yy: parallel env
        env, task_description = get_libero_subproc_env(task, num_trials_per_task=cfg.num_trials_per_task,
                                                       resolution=256)

        # yy: add traj for debugging
        actions_traj, proprios_traj = [], []

        # yy: add my video recorder
        video_folder_pth = os.path.join(local_log_save_dir, f"{task_id}_videos_{task_description}")
        os.makedirs(video_folder_pth, exist_ok=True)
        video_writer_agentview = VideoWriter(os.path.join(video_folder_pth, "agentview"), save_video=True,
                                             single_video=False)
        video_writer_wristcameraview = VideoWriter(os.path.join(video_folder_pth, "wristcameraview"), save_video=True,
                                                   single_video=False)

        # Start episodes
        task_successes = 0
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        env.reset()

        # Set initial states
        # obs = env.set_init_state(initial_states[episode_idx % initial_states.shape[0]])
        # yy: need to use list of inits
        indices = np.arange(cfg.num_trials_per_task) % initial_states.shape[0]
        initial_states = initial_states[indices]
        env.set_init_state(initial_states)


        # Setup
        t = 0
        dones = [False] * cfg.num_trials_per_task  # yy: add dones list
        if cfg.task_suite_name == "boss_44":
            max_steps = 400  # longest training demo has 373 steps
        elif ((cfg.task_suite_name == "ch1") or
              (cfg.task_suite_name == "ch2_2_modifications") or
              (cfg.task_suite_name == "ch2_3_modifications")):
            max_steps = 400  # longest training demo has 373 steps

        while t < max_steps + cfg.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    # obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    # yy: action shall be of shape [env_num, ...]
                    env.step(get_libero_dummy_action_parallel(cfg.num_trials_per_task))
                    t += 1
                    continue

                # yy: assemble all actions -> batched actions
                actions = get_batch_action_given_batch_obs(observations, cfg, resize_size, model, task_description,
                                                           processor, get_action)

                actions_traj.append(np.array(actions))  # [20, 7]
                proprios_traj.append(np.array([np.concatenate(
                    (obs_["robot0_eef_pos"], quat2axisangle(obs_["robot0_eef_quat"]), obs_["robot0_gripper_qpos"])
                ) for obs_ in observations]))  # [20, 9 (?)]

                # Execute action in environment
                obs, reward, done, info = env.step(actions)

                # yy: my video recorder save obses
                video_writer_agentview.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )
                video_writer_wristcameraview.append_vector_obs(
                    obs, dones, camera_name="robot0_eye_in_hand_image"
                )

                # check whether succeed
                # yy: dones is also of shape [env_num, ...], need some modifications
                for k in range(cfg.num_trials_per_task):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    task_successes += np.sum(dones)
                    total_successes += np.sum(dones)
                    break

                t += 1

            except Exception as e:
                print(f"Caught exception: {e}")
                log_file.write(f"Caught exception: {e}\n")
                break

        video_writer_agentview.save(save_video_name="video_agentview")
        video_writer_wristcameraview.save(save_video_name="video_wristcameraview")
        # save actions_traj [[20, 7], [20, 7], ...] and proprio_traj [[20, 9], [20, 9], ...]
        trajs_folder_pth = os.path.join(local_log_save_dir, f"{task_id}_trajs_{task_description}")
        os.makedirs(trajs_folder_pth, exist_ok=True)
        actions_traj = np.concatenate(actions_traj, axis=1)
        proprios_traj = np.concatenate(proprios_traj, axis=1)
        np.save(os.path.join(trajs_folder_pth, "actions_traj.npy"), actions_traj)
        np.save(os.path.join(trajs_folder_pth, "proprios_traj.npy"), proprios_traj)

        total_episodes += cfg.num_trials_per_task

        tasks_success_list.append(float(task_successes) / float(cfg.num_trials_per_task))
        np.save(os.path.join(local_log_save_dir, f"tasks_success_list_{cfg.task_suite_name}_seed_{cfg.seed}.npy"),
                np.array(tasks_success_list))

        # Log current results
        print(f"Current task success rate: {float(task_successes) / float(cfg.num_trials_per_task)}")
        print(f"# episodes completed so far: {total_episodes}")
        print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        log_file.write(f"Current task success rate: {float(task_successes) / float(cfg.num_trials_per_task)}\n")
        log_file.write(f"# episodes completed so far: {total_episodes}\n")
        log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        log_file.flush()

    # Save local log file
    log_file.close()

    # Save rollouts / succ list
    np.save(os.path.join(local_log_save_dir, f"tasks_success_list_{cfg.task_suite_name}_seed_{cfg.seed}.npy"),
            np.array(tasks_success_list))


if __name__ == "__main__":
    cfg = parse_args()
    eval_libero(cfg)
