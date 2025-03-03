import random
import os
from pathlib import Path

import numpy as np

import libero.libero.envs.bddl_utils as BDDLUtils

from scale_up_bddl_generation import bddl_dict2file
from scale_up_bddl_modification import modify_environment, open_regions_for_each_scene
from robosuite.utils.errors import RandomizationError
from libero.libero.envs import OffScreenRenderEnv

# ================ Params ================
seed_ls = [10001] # [10000, 10001, 10002]
num_diff_combination = 20
# if u only want to generate 1 modified bddl file with ADDITIONAL_NUM modifications
combination_list = [1 for _ in range(44)]
# Define ur own combination_list, each number is the number of modified bddl files to be generated for each bddl file
combination_list = [50 for _ in range(44)]
bddl_folder_single_step = "libero/libero/bddl_files/ch1/"
bddl_folder_boss_44 = "libero/libero/bddl_files/boss_44/"
dst_bddl_folder = "libero/libero/bddl_files/"
ADDITIONAL_NUM = 2
# ================ Params ================


def bddl_env_test(task_bddl_file):
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
    try:
        env = OffScreenRenderEnv(**env_args)
        env.reset()
    except RandomizationError as e:
        print(e)
        print(f">> Error for {task_bddl_file}")
        return False
    return True

bddl_files = [
    l9 for l9 in os.listdir(bddl_folder_boss_44)
    if any(l9[:-5] in element for element in os.listdir(bddl_folder_single_step))
]


failed_seed_bddl_ls = {seed: [] for seed in seed_ls}
combination_dict = {}
for seed in seed_ls:
    random.seed(seed)
    print("==================================================================")
    print(f">> Random seed: {seed}")

    dst_bddl_folder_new = Path(dst_bddl_folder) / f"data_augmentation_multiple_num_{ADDITIONAL_NUM}"
    dst_bddl_folder_new.mkdir(parents=True, exist_ok=True)

    for i, bddl_file in enumerate(sorted(bddl_files)):
        combination_dict[bddl_file] = None
        num_diff_combination = combination_list[i]
        if '.bddl' not in bddl_file:
            continue
        print('----------------------------------------------------------------')
        print(f"Index: {i}; bddl file: {bddl_file}")
        if 'LIVING' in bddl_file:
            scene_key = "_".join(bddl_file.split("_")[:3])
        else:
            scene_key = "_".join(bddl_file.split("_")[:2])
        bddl_file_pth = os.path.join(bddl_folder_boss_44, bddl_file)
        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_pth)

        cnt = 0
        dead_loop_cnt = 0
        modified_problem_ls = []
        while cnt < num_diff_combination:
            if dead_loop_cnt > 10000:
                print(f"[ERROR] Dead loop for bddl file: {bddl_file}; Only {cnt}/{num_diff_combination} combinations obtained :(")
                failed_seed_bddl_ls[seed].append(bddl_file)
                break
            dead_loop_cnt += 1
            dst_bddl_file = dst_bddl_folder_new / f"{bddl_file[:-5]}_{cnt}.bddl"
            try:
                modified_problem, diversity_num, chosen_open_regions = modify_environment(parsed_problem, open_regions=open_regions_for_each_scene[scene_key], is_multiple=True)

                for _ in range(ADDITIONAL_NUM):
                    modified_problem, diversity_num, chosen_open_regions = modify_environment(modified_problem,
                                                                                              open_regions=chosen_open_regions,
                                                                                              is_multiple=True)

                same_flag = False
                for mp in modified_problem_ls:
                    if mp == modified_problem:
                        same_flag = True
                if not same_flag:
                    modified_problem_ls.append(modified_problem)
                    bddl_dict2file(modified_problem, new_bddl_filename=str(dst_bddl_file))
                    if bddl_env_test(str(dst_bddl_file)):
                        cnt += 1
                        combination_dict[bddl_file] = cnt
                    else:
                        os.remove(str(dst_bddl_file))

            except ValueError as e:
                continue

print(f"combination_dict: \n{combination_dict}")
print(f"combination_dict values: \n{combination_dict.values()}")