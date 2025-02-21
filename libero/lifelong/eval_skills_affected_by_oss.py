import argparse
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
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
from libero.lifelong.policy_starter import PolicyStarter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
import json


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model_path_folder", type=str,
                        default="./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/", required=True)
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["boss_44", "ch1", "ch2_2_modifications", "ch2_3_modifications", "factor_1", "factor_2", "libero_90",],
        default="ch1"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--is_debug", type=int, default=1)
    parser.add_argument("--is_wrist_camera_view", type=int, default=0)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    return args

def main():
    args = parse_args()

    # Get mapping
    mapping_pth = f"libero/mappings/{args.benchmark}.json"
    with open(mapping_pth, 'r') as json_file:
        mapping = json.load(json_file)
    
    # Get the benchmarks
    if args.is_debug:
        benchmark = get_benchmark(args.benchmark)(n_tasks=1)
    else:
        benchmark = get_benchmark(args.benchmark)()
    n_tasks = benchmark.n_tasks
    task_id_ls = benchmark.task_indexes

    # Obtain language descriptions
    descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
    print("======= Tasks Language =======")
    print(f"{descriptions}")
    print("==============================")

    succ_list = []
    eval_task_id = []
    for idx, task_id in enumerate(task_id_ls):  # task_id is the actual id of the task. idx is just the index.
        print(f">> Evaluate on modified Task {task_id}")
        # Obtain useful info from saved model - checkpoints / cfg
        index_mapping = create_index_mapping(mapping)
        model_index = index_mapping[task_id]  # model_index is the id for original model index
        print(f">> Load model checkpoint id: {model_index}")
        model_path = args.model_path_folder
        model_path = os.path.join(model_path, f"task{model_index}_model.pth")

        if not os.path.exists(model_path):
            print(f">> {model_path} does NOT exist!")
            print(f">> Modified env_{task_id} evaluation fails.")
            continue
        sd, cfg, previous_mask = torch_load_model(
            model_path, map_location=args.device_id
        )

        # Modify some attributes of cfg via args
        cfg.benchmark_name = args.benchmark
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")
        cfg.device = args.device_id

        save_dir = os.path.join(args.model_path_folder, f"eval_tasks_on_modified_envs_seed{args.seed}",
                                f"evaluation_task{task_id}_benchmark_{args.benchmark}on_modified_envs")
        print(f">> Create folder {save_dir}")
        os.system(f"mkdir -p {save_dir}")

        # Create algo
        algo = safe_device(PolicyStarter(n_tasks, cfg), cfg.device)
        algo.policy.load_state_dict(sd)

        # Obtain language embs
        task_embs = get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)
        task = benchmark.get_task(idx)
    
        """
        Start Evaluation
        """
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
        algo.eval()
        test_loss = 0.0

        save_stats_pth = os.path.join(
            save_dir,
            f"load_ori_{model_index}_on_modified_{task_id}.stats",
        )
    
        video_folder = os.path.join(
            save_dir,
            f"load_ori_{model_index}_on_modified_{task_id}_videos",
        )

        os.system(f"mkdir -p {video_folder}")

        with Timer() as t:
            video_writer_agentview = VideoWriter(os.path.join(video_folder, "agentview"), save_video=True,
                                                 single_video=False)
            if args.is_wrist_camera_view:
                video_writer_wristcameraview = VideoWriter(os.path.join(video_folder, "wristcameraview"), save_video=True,
                                                           single_video=False)


            env_args = {
                "bddl_file_name": os.path.join(
                    cfg.bddl_folder, task.problem_folder, task.bddl_file
                ),
                "camera_heights": cfg.data.img_h,
                "camera_widths": cfg.data.img_w,
            }
    
            env_num = cfg['eval']['n_eval']
            eng_ls = [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
            env = SubprocVectorEnv(
                eng_ls
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
            task_emb = benchmark.get_task_emb(idx)
    
            num_success = 0
            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros((env_num, 7)))
    
            with torch.no_grad():
                while steps < cfg.eval.max_steps:
                    steps += 1

                    video_writer_agentview.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )
                    if args.is_wrist_camera_view:
                        video_writer_wristcameraview.append_vector_obs(
                            obs, dones, camera_name="robot0_eye_in_hand_image"
                        )
                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)


                    actions = algo.policy.get_action(data)
                    obs, reward, done, info = env.step(actions)

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    if all(dones):
                        break
                for k in range(env_num):
                    num_success += int(dones[k])

            video_writer_agentview.save(save_video_name="video_agentview")
            if args.is_wrist_camera_view:
                video_writer_wristcameraview.save(save_video_name="video_wristcameraview")
            success_rate = num_success / env_num
            env.close()
    
            eval_stats = {
                "loss": test_loss,
                "success_rate": success_rate,
            }

            succ_list.append(success_rate)
            torch.save(eval_stats, save_stats_pth)

            with open(os.path.join(args.model_path_folder, f"eval_tasks_on_modified_envs_seed{args.seed}",
                                   f"succ_list_evaluation_on_modified_envs_benchmark_{args.benchmark}.npy"), 'wb') as f:
                np.save(f, np.array(succ_list))

        with open(os.path.join(args.model_path_folder, f"eval_tasks_on_modified_envs_seed{args.seed}",
                               f"succ_list_evaluation_on_modified_envs_benchmark_{args.benchmark}.npy"), 'wb') as f:
            np.save(f, np.array(succ_list))
        print(
            f"[info] finish for ckpt at {model_path} in {t.get_elapsed_time()} sec for rollouts"
        )
        print(f"Results are saved at {save_stats_pth}")
        print(test_loss, success_rate)
        eval_task_id.append(task_id)

    print(f"[INFO] Finish evaluating modified env list: {eval_task_id}")

if __name__ == "__main__":
    main()

