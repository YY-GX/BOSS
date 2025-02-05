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
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
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


# yy: map from modified benchmark to original path:
# 00-09 (00-10 for modified)
modified_mapping = {
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"],
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl",
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_with_popcorn_in_the_top_drawer.bddl"],
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_with_top_drawer_open.bddl"],
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"],
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate": [
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet": [
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_with_the_bottom_drawer_open.bddl"],
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate": [
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate": [
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate": [
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_with_popcorn_in_the_bowl.bddl"],
}

# 10-19 (00-11 for modified)
modified_mapping = {
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet": [
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_with_top_drawer_open.bddl",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_with_popcorn_on_top_of_the_cabinet.bddl",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle": [
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl": [
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove": [
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_with_popcorn_in_the_frying_pan.bddl"],
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove": [
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_with_stove_turned_on.bddl"],
    "KITCHEN_SCENE3_turn_on_the_stove": [
        "KITCHEN_SCENE3_turn_on_the_stove_with_moka_pot_on_the_stove.bddl"],
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_with_top_drawer_open.bddl"],
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_with_popcorn_in_the_bottom_drawer.bddl"],
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet": [
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet_with_popcorn_in_the_balck_bowl.bddl"],
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_with_popcorn_in_the_drawer.bddl"],
}


def create_index_mapping(dict_map):
    output_map = {}
    key_index = 0
    value_index = 0

    for key, values in dict_map.items():
        for _ in values:
            output_map[value_index] = key_index
            value_index += 1
        key_index += 1

    return output_map


benchmark_map = {
    "libero_90": "LIBERO_90",
    "libero_10": "LIBERO_10",
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

# TODO
"""
TODO check this:
Example command: 
python lifelong/evaluate_on_modified_task.py --seed 1 --benchmark "modified_libero" --policy "bc_transformer_policy" --algo "base" --task_id 0 --load_task 0 --version "modify_v0" --device_id 0 --save-videos --seed 10000 --model_path "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/train_base_90_v0/Sequential/BCTransformerPolicy_seed10000/00_09/task0_model.pth" 
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    # parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--model_path_folder", type=str, default="/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/train_base_90_v0/Sequential/BCTransformerPolicy_seed10000/00_09/task0_model.pth")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal", "yy_try", "modified_libero"],
    )
    parser.add_argument("--task_id", type=int, required=True)
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
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    # yy: TODO: I modify here - change back when not yy_try
    args.save_dir = f"experiments_saved/modified/{args.benchmark}/{args.version}/"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        # yy: modified by me - original is 10
        assert args.load_task in list(
            range(100)
        ), "[error] load_task should be in [0, ..., 99]"
    return args



def main():
    args = parse_args()
    # yy: I feel that i only need to modify this as the original one. For all the other ones, I could use the modified one (?)
    # experiment_dir = os.path.join(
    #     args.experiment_dir,
    #     f"{args.benchmark}/"
    #     + f"{args.version}/"
    #     + f"{algo_map[args.algo]}/"
    #     + f"{policy_map[args.policy]}_seed{args.seed}",
    # )
    # experiment_dir = args.experiment_dir

    # model_path = args.model_path
    # sd, cfg, previous_mask = torch_load_model(
    #     model_path, map_location=args.device_id
    # )
    # exit(0)

    # find the checkpoint
    index_mapping = create_index_mapping(modified_mapping)
    model_index = index_mapping[args.task_id]
    model_path = ""
    try:
        if args.algo == "multitask":
            # model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
            model_path = args.model_path_folder
            model_path = os.path.join(model_path, f"task{model_index}_model.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
        else:
            # model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
            model_path = args.model_path_folder
            model_path = os.path.join(model_path, f"task{model_index}_model.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    """
    What I need to modify in cfg:
    - benchmark_name
    - version
    """
    # yy: modify these attributes
    cfg.benchmark_name = args.benchmark
    cfg.version = args.version

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(args.task_id + 1)] = 0.0
                # we never train norm layers
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    # yy: TODO: I modify here - change back when need to eval a chain of skills
    # descriptions = [benchmark.get_task(i).language for i in range(10)]
    descriptions = [benchmark.get_task(i).language for i in range(1)]
    print("======= Task Language =======")
    print(f"{descriptions[0]}")
    print("======= Task Language =======")
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)

    ### ======================= start evaluation ============================
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    # yy: the commented code is just used for some necessary initialization, which I've done in the above line of code
    # # 1. evaluate dataset loss
    # try:
    #     dataset, shape_meta = get_dataset(
    #         dataset_path=os.path.join(
    #             cfg.folder, benchmark.get_task_demonstration(args.task_id)
    #         ),
    #         obs_modality=cfg.data.obs.modality,
    #         initialize_obs_utils=True,
    #         seq_len=cfg.data.seq_len,
    #     )
    #     dataset = GroupedTaskDataset(
    #         [dataset], task_embs[args.task_id : args.task_id + 1]
    #     )
    # except:
    #     print(
    #         f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
    #     )
    #     sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    if args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
    else:
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",
        )

    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
    )


    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = 20
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        # task_emb = benchmark.get_task_emb(args.task_id)
        # yy: I modified this - change back if need to eval a chain of skills
        task_emb = benchmark.get_task_emb(0)

        num_success = 0
        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)
                obs, reward, done, info = env.step(actions)
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
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
        # # yy: save video
        # video_writer.save()
    print(
        f"[info] finish for ckpt at {model_path} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)


if __name__ == "__main__":
    main()

