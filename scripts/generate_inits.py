import torch
from libero.libero.envs import OffScreenRenderEnv
import os
import random

def create_init(bddl_file_name, suite_name):
    N_init, N_pruned_init = 50, 20
    out_folder = f"/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/init_files/{suite_name}"
    os.makedirs(out_folder, exist_ok=True)

    # First, generate the full 50 samples
    base_file = os.path.join(out_folder, bddl_file_name.split(suite_name)[-1].replace(".bddl", ".init").lstrip('/'))
    env_args = {
        "bddl_file_name": bddl_file_name,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)
    dim = env.get_sim_state().shape[0]
    env.reset()
    init_states = torch.from_numpy(env.get_sim_state().reshape((1, dim)))
    for _ in range(N_init):
        env.reset()
        init_states = torch.vstack([init_states, torch.from_numpy(env.get_sim_state().reshape((1, dim)))])
    torch.save(init_states, base_file)

    # Sample 20 from the 50 to create .pruned_init
    pruned_file = base_file.replace(".init", ".pruned_init")
    indices = random.sample(range(N_init), N_pruned_init)
    pruned_states = init_states[indices]
    torch.save(pruned_states, pruned_file)

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


