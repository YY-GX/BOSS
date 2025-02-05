from libero.libero.envs import OffScreenRenderEnv
import os
from robosuite.utils.errors import RandomizationError


bddl_folder = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/bl3_seed10000"
for i, f in enumerate(os.listdir(bddl_folder)):
    print(i, f)
    task_bddl_file = os.path.join(bddl_folder, f)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
    try:
        env = OffScreenRenderEnv(**env_args)
        env.reset()
    except RandomizationError as e:
        print(e)
        print(f">> Error for {task_bddl_file}")