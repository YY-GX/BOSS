import torch
from libero.libero.envs import OffScreenRenderEnv
import os

def create_init(bddl_file_name, suite_name):
    out_file = f"/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/init_files/{suite_name}"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    config_dict = {
        ".init": 50,
        ".pruned_init": 20,
    }
    for suffix, N in config_dict.items():
        out_file = os.path.join(out_file, bddl_file_name.split(suite_name)[-1].replace(".bddl", suffix).lstrip('/'))
        env_args = {
                    "bddl_file_name": bddl_file_name,
                    "camera_heights": 128,
                    "camera_widths": 128,
                }
        env = OffScreenRenderEnv(**env_args)
        dim = env.get_sim_state().shape[0]
        env.reset()
        init_states = torch.from_numpy(env.get_sim_state().reshape((1, dim)))
        for _ in range(N - 1):
            env.reset()
            init_states = torch.vstack([init_states, torch.from_numpy(env.get_sim_state().reshape((1, dim)))])
        torch.save(init_states, out_file)


suite_name = "gl_size"
folder = f"/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/bddl_files/{suite_name}"
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


