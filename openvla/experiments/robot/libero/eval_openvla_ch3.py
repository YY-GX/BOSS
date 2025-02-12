import argparse
import sys
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_working_directory = os.getcwd()
os.chdir(os.environ['PYTHONPATH'])
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, SequentialEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.metric import (
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    safe_device,
    torch_load_model,
)
from libero.lifelong.main import get_task_embs
import robomimic.utils.obs_utils as ObsUtils
from libero.lifelong.algos import get_algo_class
os.chdir(current_working_directory)


import numpy as np
import torch
import warnings
import pickle
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
import draccus
from types import SimpleNamespace

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA CH3 Script")
    parser.add_argument("--model_path_folder", type=str,
                        default="./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/",
                        help="only used to quickly access a cfg file. "
                             "Just use a path where cfg is saved, such as BOSS-CH1's checkpoint path. ")
    parser.add_argument("--openvla_ckpt", type=str,
                        default="runs/libero44/1.0.0/openvla-7b+libero44+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug")
    parser.add_argument("--seed", type=int, required=True, default=10000)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    return args




def initialize_robot_state(crr_state, robot_init_sim_state):
    # 0: timestep; 1-40: states; 41-76: vel_info;
    modified_state = crr_state.copy()
    # initial robot states
    modified_state[1:10] = robot_init_sim_state[1:10]
    # zeroize all velocity related states
    modified_state[41:] = robot_init_sim_state[41:]
    return modified_state


def reset_env_init_states(env, obs, info, init_states_ls, env_num, task_indexes):
    obs_ls = []
    for k in range(env_num):
        if info[k]['is_init']:
            # next task's initial state is extracted,
            #  and then passed to be modifed as I only wanna change robot related state
            init_state_ = initialize_robot_state(env.get_sim_state()[k], init_states_ls[task_indexes[k]][k, :])[
                None, ...]
            obs_ = env.set_init_state(init_state_, k)
            obs_ls.append(obs_[0])
        else:
            obs_ = obs[k]
            obs_ls.append(obs_)
    obs = np.stack(obs_ls)
    return obs


def openvla_select_action(obs, task_description, model, openvla_cfg, resize_size=224):
    """
    obs: single env obs
    """

    cfg = openvla_cfg
    img = get_libero_image(obs, resize_size)
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    action = get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
    )
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action).tolist()
    return np.array(action)


def main():
    args = parse_args()

    openvla_cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.openvla_ckpt,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        task_suite_name="boss_44",
        num_steps_wait=5,
        num_trials_per_task=20,
        run_id_note=None,
        local_log_dir="./experiments/logs",
        use_wandb=False,
        wandb_project="YOUR_WANDB_PROJECT",
        wandb_entity="YOUR_WANDB_ENTITY",
        seed=10000,
        unnorm_key="libero44"
    )

    openvla_model = get_model(openvla_cfg)

    # Loop 10 long horizon tasks
    for lht_idx in range(1, 11):
        lht_name = f"ch3_{lht_idx}"
        """
        Preparation for Evaluation
        """
        # Get the benchmarks
        benchmark = get_benchmark(lht_name)()
        n_tasks = benchmark.n_tasks
        task_id_ls = benchmark.task_indexes



        # Obtain language descriptions
        descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
        print("======= Tasks Language =======")
        print(f"{descriptions}")
        print("======= Tasks Language =======")

        save_dir = f"./experiments/logs/ch3/{lht_name}_seed{args.seed}"
        os.system(f"mkdir -p {save_dir}")

        # For collecting necessary list of items
        # For sequential env, need to obtain: cfg_ls, initial_states_ls
        cfg_ls, init_states_ls, task_ls = [], [], []
        task_embs = []
        for task_idx, task_id in enumerate(task_id_ls):  # task_id is the actual id of the task. task_idx is just the index.
            print(f">> Evaluate on original Task {task_id}")
            # Obtain useful info from saved model - checkpoints / cfg
            model_index = task_id
            model_path = args.model_path_folder
            model_path = os.path.join(model_path, f"task{model_index}_model.pth")
            if not os.path.exists(model_path):
                print(f">> {model_path} does NOT exist!")
                print(f">> Env_{task_id} evaluation fails.")
                exit(1)
            _, cfg, _ = torch_load_model(
                model_path, map_location=args.device_id
            )

            # Modify some attributes of cfg via args
            cfg.benchmark_name = "boss_44"
            cfg.folder = get_libero_path("datasets")
            cfg.bddl_folder = get_libero_path("bddl_files")
            cfg.init_states_folder = get_libero_path("init_states")
            cfg.device = args.device_id
            # cfg_ls here
            cfg_ls.append(cfg)

            # Obtain language embs & task
            task_embs += get_task_embs(cfg, descriptions)
            benchmark.set_task_embs(task_embs)
            task = benchmark.get_task(task_idx)
            # task_ls here
            task_ls.append(task)

            init_states_path = os.path.join(
                cfg.init_states_folder, task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            indices = np.arange(cfg['eval']['n_eval']) % init_states.shape[0]
            # init_states_ls here
            init_states_ls.append(init_states[indices])  # each element with shape [env_num, ...]



        """
        Start Evaluation
        """
        cfg = cfg_ls[0]
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

        save_stats_pth = os.path.join(
            save_dir,
            f"long_horizon_task_{lht_name}.stats",
        )

        video_folder = os.path.join(
            save_dir,
            f"long_horizon_task_{lht_name}_videos",
        )

        os.system(f"mkdir -p {video_folder}")

        with Timer() as t:
            # video recorder preparation
            video_writer_agentview = VideoWriter(os.path.join(video_folder, "agentview"), save_video=True,
                                                 single_video=False)
            video_writer_wristcameraview = VideoWriter(os.path.join(video_folder, "wristcameraview"), save_video=True,
                                                       single_video=False)

            # env preparation
            env_args = {
                "bddl_file_name": [
                    os.path.join(
                        cfg_.bddl_folder,
                        task_ls[i].problem_folder,
                        task_ls[i].bddl_file
                    )
                    for i, cfg_ in enumerate(cfg_ls)
                ],
                "camera_heights": [256 for _, cfg_ in enumerate(cfg_ls)],
                "camera_widths": [256 for _, cfg_ in enumerate(cfg_ls)],
            }
            env_num = cfg['eval']['n_eval']
            env = SubprocVectorEnv(
                [
                    lambda: SequentialEnv(n_tasks=len(cfg_ls), init_states_ls=init_states_ls, **env_args)
                    for _ in range(env_num)
                ]
            )
            env.reset()
            env.seed(cfg.seed)
            init_states_ = init_states_ls[0]
            obs = env.set_init_state(init_states_)
            dones = [False] * env_num
            task_indexes = [0 for _ in range(env_num)]
            steps = 0
            num_success = 0
            level_success_rate = {int(task_idx): 0 for task_idx in range(n_tasks)}
            dummy_action = np.zeros((env_num, 7))
            dummy_action[:, -1] = -1.0
            for _ in range(5):  # simulate the physics without any actions
                obs, reward, done, info = env.step(dummy_action)

            # formal start of the evaluation
            with torch.no_grad():
                while steps < (cfg.eval.max_steps * n_tasks):
                    # print("--------------------------------------------------------------------")
                    # print(steps)
                    steps += 1
                    if steps % (cfg.eval.max_steps // 30) == 0:
                        print(f"[INFO] Steps: {steps}; Task Indexes: {task_indexes}.", flush=True)
                        print(f"Evaluation takes {t.get_middle_past_time()} seconds", flush=True)

                    actions = np.zeros((1, 7))
                    for k in range(env_num):
                        task_description = task_ls[task_indexes[k]].language
                        action = openvla_select_action(obs=obs[k], task_description=task_description, openvla_cfg=openvla_cfg, model=openvla_model)
                        actions = np.vstack([actions, action])
                    actions = actions[1:, ...]
                    obs, reward, done, info = env.step(actions)
                    task_indexes = [kv['task_index'] for kv in info]

                    # reset robot arm if move to a new skill. Modify the obs as well.
                    if np.array([info[is_init_idx]['is_init'] for is_init_idx in range(env_num)]).any():
                        obs = reset_env_init_states(env, obs, info, init_states_ls, env_num, task_indexes)

                    video_writer_agentview.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )
                    video_writer_wristcameraview.append_vector_obs(
                        obs, dones, camera_name="robot0_eye_in_hand_image"
                    )

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    if all(dones):
                        break

                for k in range(env_num):
                    num_success += int(dones[k])

                """
                level_info
                """
                level_info = np.array([kv['complete_id'] for kv in info])
                for level, succ_ls in level_success_rate.items():
                    level_success_rate[level] = np.sum(level_info >= level) / env_num

            video_writer_agentview.save(save_video_name="video_agentview")
            video_writer_wristcameraview.save(save_video_name="video_wristcameraview")
            success_rate = num_success / env_num
            env.close()

            eval_stats = {
                "success_rate": success_rate,
                "level_success_rate": level_success_rate
            }

            torch.save(eval_stats, save_stats_pth)

        with open(os.path.join(save_dir, f"succ_rate_evaluation_on_ori_envs_{lht_name}.npy"), 'wb') as f:
            np.save(f, success_rate)
        with open(os.path.join(save_dir, f"level_succ_{lht_name}.pkl"), 'wb') as f:
            pickle.dump(level_success_rate, f)

        print(f"Results are saved at {save_stats_pth}")
        print(success_rate)


if __name__ == "__main__":
    main()


