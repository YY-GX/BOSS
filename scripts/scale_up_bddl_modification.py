import random
import copy
from scale_up_bddl_generation import bddl_dict2file
import libero.libero.envs.bddl_utils as BDDLUtils
import os
import numpy as np
import re

# Provided parsed_problem dict
parsed_problem = {
    'problem_name': 'libero_kitchen_tabletop_manipulation',
    'fixtures': {'kitchen_table': ['kitchen_table'],
                 'wooden_cabinet': ['wooden_cabinet_1']},
    'regions': {'kitchen_table_wooden_cabinet_init_region': {'target': 'kitchen_table',
                                                             'ranges': [[-1e-10, -0.3, 0.0, -0.29999999999]],
                                                             'extra': [],
                                                             'yaw_rotation': [3.141592653589793, 3.141592653589793],
                                                             'rgba': [0, 0, 1, 0]},
                'kitchen_table_akita_black_bowl_init_region': {'target': 'kitchen_table',
                                                               'ranges': [[-0.025, -0.025, 0.025, 0.025]],
                                                               'extra': [],
                                                               'yaw_rotation': [0.0, 0.0],
                                                               'rgba': [0, 0, 1, 0]},
                'kitchen_table_plate_init_region': {'target': 'kitchen_table',
                                                    'ranges': [[-0.025, 0.225, 0.025, 0.275]],
                                                    'extra': [],
                                                    'yaw_rotation': [0.0, 0.0],
                                                    'rgba': [0, 0, 1, 0]},
                'wooden_cabinet_1_top_side': {'target': 'wooden_cabinet_1',
                                              'ranges': [],
                                              'extra': [],
                                              'yaw_rotation': [0, 0],
                                              'rgba': [0, 0, 1, 0]},
                'wooden_cabinet_1_top_region': {'target': 'wooden_cabinet_1',
                                                'ranges': [],
                                                'extra': [],
                                                'yaw_rotation': [0, 0],
                                                'rgba': [0, 0, 1, 0]},
                'wooden_cabinet_1_middle_region': {'target': 'wooden_cabinet_1',
                                                   'ranges': [],
                                                   'extra': [],
                                                   'yaw_rotation': [0, 0],
                                                   'rgba': [0, 0, 1, 0]},
                'wooden_cabinet_1_bottom_region': {'target': 'wooden_cabinet_1',
                                                   'ranges': [],
                                                   'extra': [],
                                                   'yaw_rotation': [0, 0],
                                                   'rgba': [0, 0, 1, 0]}},
    'objects': {'akita_black_bowl': ['akita_black_bowl_1'], 'plate': ['plate_1']},
    'scene_properties': {},
    'initial_state': [['on', 'akita_black_bowl_1', 'kitchen_table_akita_black_bowl_init_region'],
                      ['on', 'plate_1', 'kitchen_table_plate_init_region'],
                      ['on', 'wooden_cabinet_1', 'kitchen_table_wooden_cabinet_init_region']],
    'goal_state': [['open', 'wooden_cabinet_1_bottom_region']],
    'language_instruction': ['open', 'the', 'bottom', 'drawer', 'of', 'the', 'cabinet'],
    'obj_of_interest': ['wooden_cabinet_1']
}





# External objects provided
small_objects = ['onion', 'egg', 'chocolate_pudding', 'lemon', 'popcorn', 'potato']
large_objects = ['corn', 'white_yellow_mug', 'red_coffee_mug', 'porcelain_mug', 'chefmate_8_frypan']
containers = ['plate', 'akita_black_bowl', 'chefmate_8_frypan']
state_change_objects = {'cabinet': ['close', 'open'], 'microwave': ['close', 'open'], 'stove': ['turnoff', 'turnon']}
# TODO: Do this if time permit
open_regions_for_each_scene = {
    "KITCHEN_SCENE1": [[-0.3, -0.025, -0.12, 0.3], [0.12, -0.025, 0.3, 0.3]],
    "KITCHEN_SCENE2": [[-0.3, 0.025, -0.25, 0.3], [0.2, 0.025, 0.3, 0.3]],
    "KITCHEN_SCENE3": [[-0.3, -0.275, -0.175, 0.3], [0.25, -0.275, 0.3, 0.3]],
    "KITCHEN_SCENE4": [[0.15, -0.5, 0.275, 0.055]],
    "KITCHEN_SCENE5": [[-0.3, -0.275, -0.2, -0.025], [0.15, -0.275, 0.3, -0.025]],
    "KITCHEN_SCENE6": [[-0.3, -0.3, -0.2, 0.05], [0.1, -0.3, 0.3, 0.05]],
    "KITCHEN_SCENE7": [[-0.3, -0.1, -0.15, 0.15], [0.15, -0.1, 0.3, 0.15]],
    "KITCHEN_SCENE8": [[0.08, -0.18, 0.13, -0.13], [0.025, 0.225, 0.075, 0.275], [-0.35, -0.18, -0.25, 0.2]],
    "KITCHEN_SCENE9": [[-0.3, -0.2, -0.2, 0.2], [0.15, -0.2, 0.3, 0.2]],
    "KITCHEN_SCENE10": [[-0.3, -0.2, -0.2, 0.3], [0.15, -0.2, 0.3, 0.3]],
    "LIVING_ROOM_SCENE5": [[0.15, -0.3, 0.25, 0.3]],
    "LIVING_ROOM_SCENE6": [[0.15, -0.15, 0.25, -0.1], [0.15, 0.1, 0.25, 0.15]],
}

# For type-1
def regions_available_for_putting(parsed_problem):
    regions_non_available = []
    for goal_state in parsed_problem['goal_state']:
        region_name = goal_state[-1]
        if 'cabinet' in region_name:
            continue
        regions_non_available.append(region_name)
    opened_drawer_region = []
    for init_state in parsed_problem['initial_state']:
        region_name = init_state[-1]
        regions_non_available.append(region_name)
        if 'open' == init_state[0] and 'cabinet' in region_name:
            opened_drawer_region.append(region_name)
    available_open_regions = []
    for region_name in parsed_problem['regions'].keys():
        if ("cabinet" in region_name) and ("region" in region_name):
            continue
        if region_name in regions_non_available:
            continue
        available_open_regions.append(region_name)
    non_interest_container_regions = []
    for object_name in parsed_problem['objects'].keys():
        if object_name in containers:
            for object in parsed_problem['objects'][object_name]:
                if object not in parsed_problem['obj_of_interest']:
                    non_interest_container_regions.append(object)
    available_container_regions = []
    available_container_regions += opened_drawer_region
    available_container_regions += non_interest_container_regions

    return available_open_regions, available_container_regions




# Create a modified version of parsed_problem
def modify_environment(parsed_problem, open_regions=[], is_debug=False, is_multiple=False):
    """

    Usage: The output of this function will be either be ValueError or correct output (i.e., modified parsed_problem)

    Args:
        parsed_problem: dict version of bddl
        open_regions: one value (i.e., list) from open_regions_for_each_scene
        is_debug: turn on debug mode or not

    Returns:
        parsed_problem: modified dict version of bddl
        diversity_num: number of combinations of different modified env

    """
    chosen_open_regions = open_regions
    diversity_num = None
    parsed_problem = copy.deepcopy(parsed_problem)
    modification_type = random.choice([1, 2, 3])
    # modification_type = tmp
    if is_debug:
        print(f">> modification_type: {modification_type}")

    if modification_type == 1:
        # Type 1: Change position or add an external object
        existing_objects = [item
                            for sublist in parsed_problem['objects'].values()
                            for item in sublist
                            if item not in parsed_problem.get('obj_of_interest', [])]


        reg_open, reg_container = regions_available_for_putting(parsed_problem)
        diversity_num = 1 * (len(large_objects) + len(small_objects)) + len(reg_open) * (len(existing_objects) + len(large_objects) + len(small_objects)) + len(reg_container) * len(small_objects)
        if diversity_num == 0:
            raise ValueError(f"[ERROR] Diversity number is {diversity_num}!")
        if is_debug:
            print(f">> Diversity number is {diversity_num}")

        type_1_category = random.choice([1, 2, 3, 4])

        # Modify existing objects
        if type_1_category == 1:
            # if random.choice([True, False]):
            if len(reg_open) == 0:
                raise ValueError(f"[ERROR] No reg_open!")
            chosen_region_open = random.choice(reg_open)
            # Modify position of an existing object
            if len(existing_objects) == 0:
                raise ValueError(f"[ERROR] existing_objects have 0 objects!")
            chosen_object = random.choice(existing_objects)
            if is_debug:
                print(f"> Modify existing object {chosen_object} to region {chosen_region_open}")
            # Remove existing state
            parsed_problem['initial_state'] = [state for state in parsed_problem['initial_state'] if state[1] != chosen_object]
            # Add new state
            parsed_problem['initial_state'].append(['on', chosen_object, chosen_region_open])

        # elif type_1_category == 1.5:
        #     # Add existing objects to open_regions_for_each_scene
        #     if debug:
        #         print(f"len(open_regions): {len(open_regions)}")
        #     if len(open_regions) == 0:
        #         raise ValueError(f"[ERROR] open_regions empty!")
        #     if len(existing_objects) == 0:
        #         raise ValueError(f"[ERROR] existing_objects have 0 objects!")
        #     chosen_object = random.choice(existing_objects)
        #     if debug:
        #         print(f"chosen_object: {chosen_object}")
        #     # TODO
        #     parsed_problem['regions'][f'kitchen_table_{chosen_object[:-2]}_init_region']['ranges'] = open_regions

        elif type_1_category == 2:
            # Add external object
            if len(reg_open) == 0:
                raise ValueError(f"[ERROR] No reg_open!")
            chosen_region_open = random.choice(reg_open)
            external_object = random.choice(small_objects + large_objects)
            if is_debug:
                print(f"> Add external object {external_object} to open region {chosen_region_open}")
            if external_object not in parsed_problem['objects']:
                parsed_problem['objects'][external_object] = [f'{external_object}_1']
            else:
                raise ValueError(f"[ERROR] external object already exists!")
            parsed_problem['initial_state'].append(['on', f'{external_object}_1', chosen_region_open])

        elif type_1_category == 3:
            # Add external objects to open_regions_for_each_scene
            if len(open_regions) == 0:
                raise ValueError(f"[ERROR] open_regions empty!")
            chosen_open_region = [random.choice(open_regions)]
            chosen_open_regions = [region for region in open_regions if region != chosen_open_region]
            external_object = random.choice(small_objects + large_objects)
            if external_object not in parsed_problem['objects']:
                parsed_problem['objects'][external_object] = [f'{external_object}_1']
            else:
                raise ValueError(f"[ERROR] external object already exists!")
            table_name = [fixture for fixture in parsed_problem['fixtures'].keys() if 'table' in fixture][0]
            parsed_problem['regions'][f'{table_name}_{external_object}_init_region'] = {
                'target': 'kitchen_table',
                'ranges': chosen_open_region,
                'extra': [],
                'yaw_rotation': [0.0, 0.0],
                'rgba': [0, 0, 1, 0]
            }
            parsed_problem['objects'][external_object] = [f"{external_object}_1"]
            parsed_problem['initial_state'].append(['on', f"{external_object}_1", f'{table_name}_{external_object}_init_region'])
        elif type_1_category == 4:
            if len(reg_container) == 0:
                raise ValueError(f"[ERROR] No reg_container!")
            chosen_region_container = random.choice(reg_container)
            external_object = random.choice(small_objects)
            if is_debug:
                print(f"> Add external object {external_object} to container region {chosen_region_container}")
            if external_object not in parsed_problem['objects']:
                parsed_problem['objects'][external_object] = [f'{external_object}_1']
            else:
                raise ValueError(f"[ERROR] external object already exists!")
            parsed_problem['initial_state'].append(['on', f'{external_object}_1', chosen_region_container])


    elif modification_type == 2:
        # Type 2: Change state of an object (e.g., open a drawer)
        fixtures_state_change = [fixture for fixture in parsed_problem['fixtures'].keys() if any(
            object_state_change in fixture for object_state_change in state_change_objects.keys())]
        if len(fixtures_state_change) == 0:
            raise ValueError(f"[ERROR] No fixtures_state_change!")
        chosen_fixture = random.choice(fixtures_state_change)

        if is_debug:
            print(f">> Chosen fixture: {chosen_fixture}")


        if 'cabinet' in chosen_fixture:
            related_region_names = [region_name
                                    for region_name in parsed_problem['regions'].keys()
                                    if (chosen_fixture in parsed_problem['regions'][region_name]['target']) and
                                    ('region' in region_name) and
                                    (region_name not in [goal_state[-1] for goal_state in parsed_problem['goal_state']])]
        else:
            related_region_names = [parsed_problem['regions'][region_name]['target']
                                    for region_name in parsed_problem['regions'].keys()
                                    if (chosen_fixture in parsed_problem['regions'][region_name]['target']) and
                                    ('region' in region_name) and
                                    (parsed_problem['regions'][region_name]['target'] not in [goal_state[-1] for goal_state in
                                                         parsed_problem['goal_state']])]
        if len(related_region_names) == 0:
            raise ValueError(f"[ERROR] No related_region_names!")
        chosen_region_state_change = random.choice(related_region_names)

        if is_debug:
            print(f">> Chosen region_state_change to modify: {chosen_region_state_change}")

        for region_keyword in state_change_objects.keys():
            if region_keyword in chosen_region_state_change:
                parsed_problem['initial_state'].append([state_change_objects[region_keyword][1], chosen_region_state_change])

        diversity_num = len(fixtures_state_change) * len(related_region_names)

        if is_debug:
            print(f">> Diversity number is {diversity_num}")


    elif modification_type == 3:
        # Type 3: Put a small external object into a container
        containers_of_interests = [obj for obj in parsed_problem['obj_of_interest'] if any([container in obj for container in containers])]
        if len(containers_of_interests) == 0:
            raise ValueError(f"[ERROR] No containers_of_interests!")
        chosen_container = random.choice(containers_of_interests)
        chosen_small_object = random.choice(small_objects)
        if is_debug:
            print(f">> Chosen object {chosen_small_object} on chosen container {chosen_container}")
        if chosen_small_object not in parsed_problem['objects']:
            parsed_problem['objects'][chosen_small_object] = [f'{chosen_small_object}_1']
        else:
            raise ValueError(f"[ERROR] external object already exists!")
        parsed_problem['initial_state'].append(['on', f'{chosen_small_object}_1', chosen_container])
        diversity_num = len(containers_of_interests) * len(small_objects)

    if is_multiple:
        return parsed_problem, diversity_num, chosen_open_regions
    else:
        return parsed_problem, diversity_num



# yy: test type-1
# import pprint
# # Example usage
# seed = 1
# random.seed(seed)
# modified_problem, diversity_num = modify_environment(parsed_problem)
# bddl_dict2file(modified_problem)
# # pprint.pprint(modified_problem)

# # yy: test type-2
# seed = 1
# import libero.libero.envs.bddl_utils as BDDLUtils
# random.seed(seed)
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.bddl"
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE3_turn_on_the_stove.bddl"
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl"
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE7_open_the_microwave.bddl"
# parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_pth)
# modified_problem, diversity_num = modify_environment(parsed_problem)
# bddl_dict2file(modified_problem)

# # yy: test type-3
# seed = 0
# import libero.libero.envs.bddl_utils as BDDLUtils
# random.seed(seed)
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.bddl"
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE3_turn_on_the_stove.bddl"
# bddl_file_pth = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl"
# parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_pth)
# modified_problem, diversity_num = modify_environment(parsed_problem)
# bddl_dict2file(modified_problem)


# # yy: test for all - Note: add ", tmp=1" to modify_environment()'s params
# bddl_folder_single_step = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/single_step/"
# bddl_folder = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/"
# bddl_files = [
#     l9 for l9 in os.listdir(bddl_folder)
#     if any(l9[:-5] in element for element in os.listdir(bddl_folder_single_step))
# ]
#
# diversity_num_total_ls_ls = []
# seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# for seed in seeds:
#     random.seed(seed)
#     print("==================================================================")
#     print(f">> Random seed: {seed}")
#     available_region_number = []
#     diversity_num_total_ls = []
#     for bddl_file in sorted(bddl_files):
#         if '.bddl' not in bddl_file:
#             continue
#         print('----------------------------------------------------------------')
#         print(bddl_file)
#         if 'LIVING' in bddl_file:
#             scene_key = "_".join(bddl_file.split("_")[:3])
#         else:
#             scene_key = "_".join(bddl_file.split("_")[:2])
#
#         bddl_file_pth = os.path.join(bddl_folder, bddl_file)
#         parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_pth)
#         # yy: Test regions_available_for_putting()
#         if False:
#             reg_1, reg_2 = regions_available_for_putting(parsed_problem)
#             print(reg_1, reg_2)
#             available_region_number.append(len(reg_1) + len(reg_2))
#
#         diversity_num_total = 0
#
#         try:
#             modified_problem, diversity_num = modify_environment(parsed_problem, open_regions=open_regions_for_each_scene[scene_key], tmp=1)
#         except ValueError as e:
#             print(f"ValueError occurred: {e}")
#             diversity_num = 0
#         print(diversity_num)
#         diversity_num_total += diversity_num
#         # bddl_dict2file(modified_problem)
#
#         try:
#             modified_problem, diversity_num = modify_environment(parsed_problem, open_regions=open_regions_for_each_scene[scene_key], tmp=2)
#         except ValueError as e:
#             print(f"ValueError occurred: {e}")
#             diversity_num = 0
#         print(diversity_num)
#         diversity_num_total += diversity_num
#         # bddl_dict2file(modified_problem)
#
#         try:
#             modified_problem, diversity_num = modify_environment(parsed_problem, open_regions=open_regions_for_each_scene[scene_key], tmp=3)
#         except ValueError as e:
#             print(f"ValueError occurred: {e}")
#             diversity_num = 0
#         print(diversity_num)
#         diversity_num_total += diversity_num
#         # bddl_dict2file(modified_problem)
#
#         print(f"Total {diversity_num_total}")
#         diversity_num_total_ls.append(diversity_num_total)
#     print(f">> diversity_num_total_ls: {diversity_num_total_ls}")
#     diversity_num_total_ls_ls.append(diversity_num_total_ls)
#     # yy: Test regions_available_for_putting()
#     if False:
#         print(f">> available_region_number: {available_region_number}")
#         print(f"{sum(np.array(available_region_number) == 0)} / {len(available_region_number)}")
#
# result = [max(values) for values in zip(*diversity_num_total_ls_ls)]
# print(f"Total Combinations: {result}")