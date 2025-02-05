import argparse
import sys
import os

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, SequentialEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time
import ast
import time
from contextlib import contextmanager
from PIL import Image

@contextmanager
def timeit():
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")




benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_90": "LIBERO_90",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def initialize_robot_state(crr_state, robot_init_sim_state):
    # yy: 0: timestep; 1-40: states; 41-76: vel_info;
    modified_state = crr_state.copy()
    # initial robot states
    modified_state[1:10] = robot_init_sim_state[1:10]
    # zeroize all velocity related states
    modified_state[41:] = robot_init_sim_state[41:]
    return modified_state


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"],
    )
    # parser.add_argument("--task_id", type=int, required=True)
    # yy: load task ids that you want to execute in sequence
    parser.add_argument("--task_id_ls", type=str, required=True)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--is_local_eval", type=int, required=True, default=1)
    parser.add_argument("--seed", type=int, required=True)
    # parser.add_argument("--load_task_ls", type=int, required=True)
    # parser.add_argument("--load_task", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        # assert args.load_task in list(
        #     range(10)
        # ), "[error] load_task should be in [0, ..., 9]"
        assert type(ast.literal_eval(args.task_id_ls)) == list
    return args


def main():
    args = parse_args()
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/
    # yy: make the str a list
    args.task_id_ls = ast.literal_eval(args.task_id_ls)

    experiment_dir = os.path.join(
        args.experiment_dir,
        f"{args.benchmark}/"
        + f"{algo_map[args.algo]}/"
        + f"{policy_map[args.policy]}_seed{args.seed}",
    )

    # find the checkpoint
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            # yy: obtain the newest one
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    if experiment_id == 0:
        print(f"[error] cannot find the checkpoint under {experiment_dir}")
        sys.exit(0)

    run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")


    # yy: load a list of policies, in order to execute these tasks in sequence
    checkpoints_ls = []
    cfg_ls = []
    for i, task_id in enumerate(args.task_id_ls):
        model_path = os.path.join(run_folder, f"task{task_id}_model.pth")
        sd, cfg, _ = torch_load_model(
            model_path, map_location=args.device_id
        )
        checkpoints_ls.append(sd)
        cfg_ls.append(cfg)
    cfg = cfg_ls[0]
    # try:
    #     if args.algo == "multitask":
    #         model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
    #         sd, cfg, previous_mask = torch_load_model(
    #             model_path, map_location=args.device_id
    #         )
    #     else:
    #         # yy: load a list of policies, in order to execute these tasks in sequence
    #         checkpoints_ls = []
    #         cfg_ls = []
    #         for i, task_id in enumerate(args.task_id_ls):
    #             model_path = os.path.join(run_folder, f"task{task_id}_model.pth")
    #             sd, cfg, _ = torch_load_model(
    #                 model_path, map_location=args.device_id
    #             )
    #             checkpoints_ls.append(sd)
    #             cfg_ls.append(cfg)
    #         cfg = cfg_ls[0]
    #
    # except:
    #     print(f"[error] cannot find the checkpoint at {str(model_path)}")
    #     sys.exit(0)

    # yy: these are folders, which shall be the same for all tasks
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.device = args.device_id

    algo_ls = []
    for i, task_id in enumerate(args.task_id_ls):
        algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
        # algo.policy.previous_mask = previous_mask
        algo.policy.load_state_dict(checkpoints_ls[i])
        algo.eval()
        algo.reset()
        algo_ls.append(algo)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]  # yy: why 10? cuz load_task_id is from 0 - 9.
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # yy: task is sth like:
    """
    Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )
    """
    task = benchmark.get_task(args.task_id_ls[0])

    ### ======================= start evaluation ============================

    # yy: just use get_dataset() to do some necessary initilization
    dataset, shape_meta = get_dataset(
        dataset_path=os.path.join(
            cfg.folder, benchmark.get_task_demonstration(args.task_id_ls[0])
        ),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    dataset = GroupedTaskDataset(
        [dataset], task_embs[args.task_id_ls[0]: args.task_id_ls[0] + 1]
    )


    test_loss = 0.0

    # evaluate success rate
    if args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
    else:
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_eval_skill_chain.stats",
        )

    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_eval_skill_chain_videos",
    )

    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        # yy: here is where the env is defined <= comes from task.bddl_file
        # yy: cfg.bddl_folder -> "bddl_files"; task.problem_folder -> "libero_90"; task.bddl_file -> sth like: "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"
        env_args = {
            "bddl_file_name": [os.path.join(
                cfg_.bddl_folder, benchmark.get_task(args.task_id_ls[i]).problem_folder,
                benchmark.get_task(args.task_id_ls[i]).bddl_file
            ) for i, cfg_ in enumerate(cfg_ls)],
            "camera_heights": [cfg_.data.img_h for i, cfg_ in enumerate(cfg_ls)],
            "camera_widths": [cfg_.data.img_w for i, cfg_ in enumerate(cfg_ls)],
        }

        env_num = 20
        # yy: set init_states_ls
        init_states_ls = []
        for i, cfg_ in enumerate(cfg_ls):
            task_ = benchmark.get_task(args.task_id_ls[i])
            if args.is_local_eval == 1:
                cfg_.init_states_folder = cfg_.init_states_folder.replace("/mnt/arc/yygx/pkgs_baselines/", "/home/yygx/UNC_Research/pkgs_simu/")

            init_states_path = os.path.join(
                cfg_.init_states_folder, task_.problem_folder, task_.init_states_file
            )
            # print(f"init_states_path: {init_states_path}")
            init_states = torch.load(init_states_path)
            # print(f"init_states: {init_states}, size: {init_states.shape}")
            indices = np.arange(env_num) % init_states.shape[0]
            # print(f"indices: {indices}")
            # print(f"init_states[indices]: {init_states[indices]}, size: {init_states[indices].shape}")
            # yy: init_states[indices],shape -> [20, 77]
            init_states_ls.append(init_states[indices])
        # print(f"len(init_states_ls): {len(init_states_ls)}")
        # print([is_.shape for is_ in init_states_ls])
        # yy: this is for the 1st task
        init_states_ = init_states_ls[0]

        env = SubprocVectorEnv(
            [lambda: SequentialEnv(n_tasks=len(cfg_ls), init_states_ls=init_states_ls, **env_args)
             for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        # task_emb = benchmark.get_task_emb(args.task_id)

        num_success = 0
        task_indexes = [0 for _ in range(env_num)]
        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                with timeit():
                    steps += 1

                    actions = np.zeros((1, 7))
                    for k in range(env_num):
                        task_emb = benchmark.get_task_emb(args.task_id_ls[task_indexes[k]])
                        cfg = cfg_ls[task_indexes[k]]
                        algo = algo_ls[task_indexes[k]]
                        data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                        """
                            agentview_rgb <class 'torch.Tensor'> torch.Size([20, 3, 128, 128])
                            eye_in_hand_rgb <class 'torch.Tensor'> torch.Size([20, 3, 128, 128])
                            gripper_states <class 'torch.Tensor'> torch.Size([20, 2])
                            joint_states <class 'torch.Tensor'> torch.Size([20, 7])
                        """
                        for key, v in data['obs'].items():
                            data['obs'][key] = v[k, ...][None, ...]
                        data['task_emb'] = data['task_emb'][k, ...][None, ...]
                        # yy: 20 * 768
                        # print(data['task_emb'].size())
                        actions = np.vstack([actions, algo.policy.get_action(data)])
                    actions = actions[1:, ...]
                    obs, reward, done, info = env.step(actions)
                    task_indexes = [kv['task_index'] for kv in info]
                    print(f"Step: {steps}")
                    print(task_indexes)

                    # yy: obs shape: (20,). In it, each element is an OrderedDict
                    obs_ls = []
                    for k in range(env_num):
                        if info[k]['is_init']:
                            # yy: next task's initial state is extracted,
                            #  and then passed to be modifed as I only wanna change robot related state
                            init_state_ = initialize_robot_state(env.get_sim_state()[k], init_states_ls[task_indexes[k]][k, :])[None, ...]
                            obs_ = env.set_init_state(init_state_, k)
                            obs_ls.append(obs_[0])
                        else:
                            obs_ = obs[k]
                            obs_ls.append(obs_)
                    obs = np.stack(obs_ls)

                    video_writer.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    print(dones)
                    if all(dones):
                        break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        os.system(f"mkdir -p {args.save_dir}")
        torch.save(eval_stats, save_folder)
    print(
        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)


if __name__ == "__main__":
    main()
