import torch
from libero.libero.envs import OffScreenRenderEnv
import os

import os
import os


# suite_name = "yy_try"
# suite_name = "modified_libero"
# # bddl_file_name = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/{suite_name}/SHELF_TABLE_SCENE_moving_potato_from_shelf_to_the_plate_on_the_table.bddl"
# # out_file = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/init_files/{suite_name}"
#
# bddl_file_name = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/{suite_name}/KITCHEN_SCENE_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl"
# bddl_file_name = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/yy_try/SHELF_TABLE_SCENE_moving_popcorn_from_table_to_topside_of_wooden_shelf.bddl"
# bddl_file_name = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/yy_try/SHELF_TABLE_SCENE_moving_chocolate_pudding_from_table_to_topside_of_wooden_shelf_demo.hdf5"
# bddl_file_name = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/modified_libero/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"



def create_init(bddl_file_name, suite_name):
    out_file = f"/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/init_files/{suite_name}"
    # out_file = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/init_files/{suite_name}"
    out_file = os.path.join(out_file, bddl_file_name.split(suite_name)[-1].replace(".bddl", ".pruned_init").lstrip('/'))

    env_args = {
                "bddl_file_name": bddl_file_name,
                "camera_heights": 128,
                "camera_widths": 128,
            }
    env = OffScreenRenderEnv(**env_args)
    dim = env.get_sim_state().shape[0]
    N = 3
    env.reset()
    init_states = torch.from_numpy(env.get_sim_state().reshape((1, dim)))
    for _ in range(N - 1):
        env.reset()
        init_states = torch.vstack([init_states, torch.from_numpy(env.get_sim_state().reshape((1, dim)))])
    torch.save(init_states, out_file)

# suite_name = "modified_libero"
# suite_name = "single_step"
# suite_name = "libero_90"
# suite_name = "ablation_1"
suite_name = "multi_step_2"
# suite_name = "multi_step_3"
# folder = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/{suite_name}"
folder = f"/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/{suite_name}"
ran_string = """

"""
for bddl_file_name in os.listdir(folder):
    if not (".bddl" in bddl_file_name):
        continue
    if bddl_file_name in ran_string:
        continue
    print(bddl_file_name)
    bddl_file_name_pth = os.path.join(folder, bddl_file_name)
    create_init(bddl_file_name_pth, suite_name)


