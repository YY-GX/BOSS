import os
import shutil

# Define paths
datasets_dir = "./datasets"
allowed_files = {
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5",
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_demo.hdf5",
    "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it_demo.hdf5",
    "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_demo.hdf5",
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo.hdf5",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5",
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_demo.hdf5",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_demo.hdf5",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_demo.hdf5",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_demo.hdf5",
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_demo.hdf5",
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_demo.hdf5",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5",
    "KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_demo.hdf5",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack_demo.hdf5",
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate_demo.hdf5",
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE6_close_the_microwave_demo.hdf5",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_demo.hdf5",
    "KITCHEN_SCENE7_open_the_microwave_demo.hdf5",
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate_demo.hdf5",
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate_demo.hdf5",
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5",
    "KITCHEN_SCENE8_turn_off_the_stove_demo.hdf5",
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf_demo.hdf5",
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf_demo.hdf5",
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet_demo.hdf5",
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5",
    "KITCHEN_SCENE9_turn_on_the_stove_demo.hdf5",
}

# Get the only folder in the datasets directory
subdirs = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
if len(subdirs) == 1:
    old_folder = os.path.join(datasets_dir, subdirs[0])
    new_folder = os.path.join(datasets_dir, "boss_44")
    os.rename(old_folder, new_folder)

    # Remove unwanted files
    for file in os.listdir(new_folder):
        file_path = os.path.join(new_folder, file)
        if os.path.isfile(file_path) and file not in allowed_files:
            os.remove(file_path)
    print("Folder renamed and unwanted files removed successfully.")
else:
    print("Error: More than one folder found in ./datasets or no folder exists.")
