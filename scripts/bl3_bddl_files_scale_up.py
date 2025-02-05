import random
import os
from pathlib import Path

import numpy as np

import libero.libero.envs.bddl_utils as BDDLUtils

from scale_up_bddl_generation import bddl_dict2file
from scale_up_bddl_modification import modify_environment, open_regions_for_each_scene

# ================ Params ================
seed_ls = [10001] # [10000, 10001, 10002]
num_diff_combination = 20
combination_list = [40, 39, 38, 38, 37, 38, 52, 51, 51, 51, 52, 51, 51, 18, 18, 30, 51, 49, 50, 49, 38, 45, 44, 44, 45, 44, 50, 43, 40, 29, 11, 24, 54, 54, 54, 54, 69, 23, 23, 23, 24, 30, 43, 43]
bddl_folder_single_step = "libero/libero/bddl_files/single_step/"
bddl_folder_libero_90 = "libero/libero/bddl_files/libero_90/"
dst_bddl_folder = "libero/libero/bddl_files/"
# ================ Params ================


bddl_files = [
    l9 for l9 in os.listdir(bddl_folder_libero_90)
    if any(l9[:-5] in element for element in os.listdir(bddl_folder_single_step))
]


failed_seed_bddl_ls = {seed: [] for seed in seed_ls}
combination_dict = {}
for seed in seed_ls:
    random.seed(seed)
    print("==================================================================")
    print(f">> Random seed: {seed}")

    # dst_bddl_folder_new = Path(dst_bddl_folder) / f"bl3_seed{seed}"
    dst_bddl_folder_new = Path(dst_bddl_folder) / f"bl3_all"
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
        bddl_file_pth = os.path.join(bddl_folder_libero_90, bddl_file)
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
                modified_problem, diversity_num = modify_environment(parsed_problem, open_regions=open_regions_for_each_scene[scene_key])
                same_flag = False
                for mp in modified_problem_ls:
                    if mp == modified_problem:
                        same_flag = True
                if not same_flag:
                    modified_problem_ls.append(modified_problem)
                    cnt += 1
                    combination_dict[bddl_file] = cnt
                    bddl_dict2file(modified_problem, new_bddl_filename=str(dst_bddl_file))
            except ValueError as e:
                continue

# print(f"BDDL files that cannot generate {num_diff_combination} combinations of modified environments: {failed_seed_bddl_ls}")
print(f"combination_dict: \n{combination_dict}")
print(f"combination_dict values: \n{combination_dict.values()}")
np.save(f"/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/combination_dict.npy", combination_dict)