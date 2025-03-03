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
        self.ori_demos_folder = "libero/datasets/boss_44/"
        self.ori_bddl_folder = f"libero/libero/bddl_files/boss_44/"
        self.modified_bddl_folder = f"libero/libero/bddl_files/{self.benchmark}/"
        self.task_order_index = 0

        benchmark = get_benchmark("boss_44")(task_order_index=self.task_order_index)
        self.ori_task_names = benchmark.get_task_names()
        # self.ori_task_names = [bddl_name.split('.')[0] for bddl_name in os.listdir(self.ori_bddl_folder)]
        self.demos_pths = sorted([os.path.join(self.ori_demos_folder, task_name + "_demo.hdf5") for task_name in
                           self.ori_task_names])

        self.dataset_path = f"libero/datasets/{self.benchmark}"
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)

        self.start_demos_generation()

    def start_demos_generation(self, num_task_to_process=1000000):  # 1000000 means no limitation
        # Create new demos based on: 1. ori demo 2. modified bddl
        mapping_pth = f"libero/mappings/{self.benchmark}.json"
        with open(mapping_pth, 'r') as json_file:
            mapping = json.load(json_file)
        self.ori_task_names = sorted(self.ori_task_names)
        print(f"Original task names: {self.ori_task_names}")
        # For each boss_44 task, obtain the modified version of dataset from it.
        for i, task_name in enumerate(self.ori_task_names[:num_task_to_process]):
            # I added this here
            if i < 5:
                continue

            print(f"===================================================================================================")
            print(f">> Index: {i}; Original Task Name: {task_name}")
            ori_demo_path = self.demos_pths[i]

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


if __name__ == '__main__':
    create_demos = CreateDemos(benchmark="data_augmentation", is_render=False)
