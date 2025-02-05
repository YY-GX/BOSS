import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action

import numpy as np
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
import h5py

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

def set_init_state(env, init_state):
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    env._check_success()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[]
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0

    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        # Run environment step

        env.step(action)
        env.render()
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    print(count)
    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    print(TASK_MAPPING[problem_name])
    print(args.bddl_file)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    # print(env.__class__.__bases__[0].__name__)
    # print(env.sim.set_state_from_flattened)c
    # exit(0)


    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)
    env.reset()
    env.render()

    demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/yy_try/KITCHEN_SCENE_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5"
    demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/yy_try/KITCHEN_SCENE_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5"
    # demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5"
    demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/yy_try/SHELF_TABLE_SCENE_moving_popcorn_from_table_to_topside_of_wooden_shelf_demo.hdf5"
    demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5"
    demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_demo.hdf5"
    demo_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5"

    data_dict = load_hdf5_file_to_dict(demo_pth)['data']
    action_to_remove_1 = np.array([0., 0., 0., 0., 0., -0., -1.])
    action_to_remove_2 = np.array([0., 0., 0., 0., 0., -0., 1.])
    print(data_dict.keys())

    for demo_idx in list(data_dict.keys()):
        # demo_idx = "demo_1"
        print(f">> demo_idx: {demo_idx}")
        demo = data_dict[demo_idx]
        actions = demo['actions']
        # set_init_state(env, demo['states'][0])
        for i, action in enumerate(actions):
            # if np.all(action == action_to_remove_1) or np.all(action == action_to_remove_2):
            #     print("======= JUMP =======")
            #     continue
            # print(f"> action idx: {i}")
            # print(action)

            obs, _, _, _ = env.step(action)
            # print(obs.keys())
            # print(obs['agentview_image'].shape)
            # from PIL import Image
            # import numpy as np
            # # Convert the NumPy array to an image
            # image = Image.fromarray(obs['agentview_image'])
            # # Save the image
            # image.save("image_128.png")
            # exit(0)

            env.render()
        env.reset()
        env.render()

    env.close()





