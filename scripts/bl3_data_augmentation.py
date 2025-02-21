import os
import time

from libero.libero.envs import OffScreenRenderEnv
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from robosuite.wrappers import VisualizationWrapper
from libero.libero.envs import *
from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
import shutil
import json
import subprocess
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
import libero.libero.utils.utils as libero_utils
from libero.libero.benchmark import get_benchmark, task_orders
from pathlib import Path

class CreateDemos:
    def __init__(
            self,
            benchmark,
            is_render=False,
            img_size=128
    ):
        self.benchmark = benchmark
        self.is_render = is_render
        self.img_size = img_size
        self.ori_demos_folder = "libero/datasets/libero_90/"
        self.ori_bddl_folder = f"libero/libero/bddl_files/libero_90/"
        self.modified_bddl_folder = f"libero/libero/bddl_files/{self.benchmark}/"
        self.task_order_index = 0

        benchmark = get_benchmark("libero_90")(task_order_index=self.task_order_index)
        self.ori_task_names = benchmark.get_task_names()
        # self.ori_task_names = [bddl_name.split('.')[0] for bddl_name in os.listdir(self.ori_bddl_folder)]
        self.demos_pths = sorted([os.path.join(self.ori_demos_folder, task_name + "_demo.hdf5") for task_name in
                           self.ori_task_names])

        # self.dataset_path = "libero/datasets/bl3"
        self.dataset_path = f"libero/datasets/{self.benchmark}"
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)

        self.initialize()

    def initialize(self, num_task_to_process=1000000):  # 1000000 means no limitation
        # # Copy ori demos to ori folder
        # self.copy_files(self.ori_demos_folder, os.path.join(self.dataset_path_tmp, "ori"))
        # Create new demos based on: 1. ori demo 2. modified bddl
        mapping_pth = f"libero/mappings/{self.benchmark}.json"
        with open(mapping_pth, 'r') as json_file:
            mapping = json.load(json_file)
        self.ori_task_names = sorted(self.ori_task_names)
        print(f"Original task names: {self.ori_task_names}")
        # For each libero_90 task, obtain the modified version of dataset from it.
        for i, task_name in enumerate(self.ori_task_names[:num_task_to_process]):
            # yy: I added this here
            if i < 5:
                continue

            print(f"===================================================================================================")
            print(f">> Index: {i}; Original Task Name: {task_name}")
            ori_demo_path = self.demos_pths[i]
            # # Jump demos that don't contain any demos
            # if os.path.exists(ori_demo_path):
            #     f = h5py.File(ori_demo_path, "r")
            #     demo_num = len(list(f["data"].keys()))
            #     if demo_num > 0:
            #         continue

            modified_bddl_ls = mapping[task_name]  # list of modified envs' bddl files
            for modified_idx, modified_bddl_name in enumerate(modified_bddl_ls):
                modified_bddl_path = os.path.join(self.modified_bddl_folder, modified_bddl_name)
                dst_demo_path = os.path.join(self.dataset_path, task_name + f"_{modified_idx}_demo.hdf5")
                print(f"Modified bddl path: {modified_bddl_path}")
                self.create_modified_demos(
                    ori_demo_path,
                    modified_bddl_path,
                    dst_demo_path
                )

    def hdf5_to_dict(self, group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If the item is a group, recursively convert it
                result[key] = self.hdf5_to_dict(item)
            elif isinstance(item, h5py.Dataset):  # If the item is a dataset, convert it to a numpy array
                result[key] = item[()]
            else:
                raise TypeError(f"Unsupported HDF5 item type: {type(item)}")
        return result

    def load_hdf5_file_to_dict(self, file_path):
        with h5py.File(file_path, 'r') as f:
            return self.hdf5_to_dict(f)


    def copy_files(self, source_folder, destination_folder):
        # Check if source folder exists
        if not os.path.exists(source_folder):
            print(f"Source folder '{source_folder}' does not exist.")
            return

        # Create destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        # Check if the destination folder already contains files or directories
        if os.listdir(destination_folder):
            print(f"Destination folder '{destination_folder}' is not empty. No files copied.")
            return

        # Iterate over all files in the source folder
        for filename in os.listdir(source_folder):
            # Construct full file path
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(source_file):
                # Copy the file
                shutil.copy2(source_file, destination_file)  # copy2 preserves metadata
                print(f"Copied: {source_file} to {destination_file}")
            else:
                print(f"Skipped: {source_file} (not a file)")

    def create_modified_demos(
            self,
            ori_demo_path,
            modified_bddl_path,
            dst_demo_path
    ):
        """
        Inputs: 1 ori demo + 1 modified bddl
        Returns: Save 1 modified demo.hdf5 and return None
        """

        cmd = [
            "python", "scripts/DemoProcessor.py",
            "--use-camera-obs",
            "--dataset_path",
            dst_demo_path,
            "--demo_file",
            ori_demo_path,
            "--bddl_path",
            modified_bddl_path
        ]

        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check the result
        if result.returncode == 0:
            print("Command executed successfully:")
            print(result.stdout)  # Output of the command
        else:
            print("Command failed:")
            print(result.stderr)  # Error message



    def replay_demos(
            self,
            bddl_path,
            demos_path
    ):
        """
        Inputs: 1 ori demo + 1 modified bddl
        Returns: Save 1 modified demo.hdf5 and return None
        """

        controller_config = load_controller_config(default_controller="OSC_POSE")
        config = {
            "robots": ["Panda"],
            "controller_configs": controller_config,
            "camera_heights": self.img_size,
            "camera_widths": self.img_size,
        }

        assert os.path.exists(bddl_path)
        problem_info = BDDLUtils.get_problem_info(bddl_path)
        problem_name = problem_info["problem_name"]
        config["env_configuration"] = "single-arm-opposed"
        print(problem_name)
        env = TASK_MAPPING[problem_name](
            bddl_file_name=bddl_path,
            **config,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
        )

        env = VisualizationWrapper(env)
        obs = env.reset()
        if self.is_render:
            env.render()

        demo_pth = demos_path
        demo_dict = self.load_hdf5_file_to_dict(demo_pth)['data']
        demo_keys = list(demo_dict.keys())
        cnt_succ = 0


        demos_bl3 = []
        for demo_idx in demo_keys:
            print(f">> demo_idx: {demo_idx}")
            demo = demo_dict[demo_idx]
            demo_bl3 = demo.copy()
            robot_states, states, dones = [], [], []
            actions = demo['actions']
            for i, action in enumerate(actions):
                # robot_states.append(env.get_robot_state_vector(obs)[None, ...])
                # states.append(env.sim.get_state().flatten()[None, ...])
                obs, _, done, info = env.step(action)
                dones.append(done)
                if self.is_render:
                    env.render()
                if done:
                    cnt_succ += 1
                    # # Only append demos whose traj is successful
                    # robot_states, states, dones = np.vstack(robot_states), np.vstack(states), np.vstack(dones)
                    # demo_bl3['robot_states'], demo_bl3['states'], demo_bl3['dones'] = robot_states, states, dones
                    break

            print(f"#succ_demo_{cnt_succ} / #total_demo_{len(demo_keys)}")
            env.reset()

        env.close()
        print(f"Final: #succ_demo_{cnt_succ} / #total_demo_{len(demo_keys)}")


if __name__ == '__main__':
    # This is for single_step demo creation
    # create_demos = CreateDemos(benchmark="single_step", is_render=False)

    # This is for scalable demo creation
    create_demos = CreateDemos(benchmark="bl3_all", is_render=False)

    # create_demos.replay_demos(
    #     bddl_path="/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate.bddl",
    #     demos_path="/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate_demo.hdf5"
    # )