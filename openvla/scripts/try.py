# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv

import torch
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# # Libero env
# bddl_file_pth = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet.bddl"
# env_args = {
#                 "bddl_file_name": bddl_file_pth,
#                 "camera_heights": 128,
#                 "camera_widths": 128,
#             }
# env = OffScreenRenderEnv(**env_args)
# obs = env.reset()
# for _ in range(5):  # simulate the physics without any actions
#     obs, _, _, _ = env.step(np.zeros(7))
# image = Image.fromarray(obs["agentview_image"][::-1])
# prompt = "In: What action should the robot take to close the top drawer of the cabinet?\nOut:"
# # Predict Action (7-DoF; un-normalize for BridgeData V2)
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
# print(">> type(action): ", type(action))
# print(">> action.shape: ", action.shape)

# Simpler env
env = simpler_env.make('google_robot_pick_coke_can')
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
image = get_image_from_maniskill2_obs_dict(env, obs)
image = Image.fromarray(image)
prompt = "In: What action should the robot take to pick coke can?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(">> type(action): ", type(action))
print(">> action.shape: ", action.shape)
