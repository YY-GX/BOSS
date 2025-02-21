import os

from libero.libero.envs import OffScreenRenderEnv
import h5py
import numpy as np
from PIL import Image
from pathlib import Path

def hdf5_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):  # If the item is a group, recursively convert it
            result[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):  # If the item is a dataset, convert it to a numpy array
            result[key] = item[()]
        else:
            raise TypeError(f"Unsupported HDF5 item type: {type(item)}")
    return result

def load_hdf5_file_to_dict(file_path):
    with h5py.File(file_path, 'r') as f:
        return hdf5_to_dict(f)

is_wrist = False

# wrist example
bddl_path = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/modified_libero/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_with_popcorn_in_the_bowl.bddl"
demos_path = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_demo.hdf5"
img_saved_folder = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/sam/try_imgs/wrist_imgs"

# agent example
bddl_path = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/modified_libero/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"
# bddl_path = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl"
demos_path = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5"
img_saved_folder = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/sam/try_imgs/agent_imgs"

Path(img_saved_folder).mkdir(parents=True, exist_ok=True)

img_size = 512

env_args = {
    "bddl_file_name": bddl_path,
    "camera_heights": img_size,
    "camera_widths": img_size,
}
env = OffScreenRenderEnv(**env_args)
# crr_state = env.get_sim_state()
# ori_state = np.load("init_state.npy")
# ori_state[38:41] = crr_state[38:41]
# env.set_init_state(ori_state)

demo_pth = demos_path
data_dict = load_hdf5_file_to_dict(demo_pth)['data']
for demo_idx in list(data_dict.keys()):
    print(f">> demo_idx: {demo_idx}")
    demo = data_dict[demo_idx]
    # np.save("init_state.npy", demo['states'][0])
    actions = demo['actions']
    for i, action in enumerate(actions):
        print(f">> Steps: {i}")
        obs, _, _, _ = env.step(action)
        if is_wrist:
            image = Image.fromarray(obs['robot0_eye_in_hand_image'])
        else:
            image = obs['agentview_image']
            image = np.flip(image, axis=0)
            image = Image.fromarray(image)
        image.save(os.path.join(img_saved_folder, f"demo_{demo_idx}_idx{i}.png"))
    env.reset()
    exit(0)

env.close()
