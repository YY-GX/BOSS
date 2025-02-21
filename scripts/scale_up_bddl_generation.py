
def bddl_dict2file(parsed_problem, new_bddl_filename="./debug.bddl"):
    # Step 1: Reconstruct BDDL parts

    # Fixtures section
    fixtures_str = "\n    ".join([f"{' '.join(names)} - {key}" for key, names in parsed_problem['fixtures'].items()])

    # Objects section
    objects_str = "\n    ".join([f"{' '.join(names)} - {key}" for key, names in parsed_problem['objects'].items()])


    regions_str = ""
    for combined_region_name, region_attrs in parsed_problem['regions'].items():
        # Extract region_name (ignore target_name from the key, as it was just a unique identifier)
        target_name = region_attrs['target']
        region_name = combined_region_name[len(target_name) + 1:] if target_name else combined_region_name

        # Start building the region definition string
        region_str = f"({region_name}\n"
        if 'target' in region_attrs and region_attrs['target']:
            region_str += f"      (:target {region_attrs['target']})\n"
        if 'ranges' in region_attrs and region_attrs['ranges']:
            ranges_str = "\n        ".join([f"({x[0]} {x[1]} {x[2]} {x[3]})" for x in region_attrs['ranges']])
            region_str += f"      (:ranges (\n        {ranges_str}\n      ))\n"
        if 'yaw_rotation' in region_attrs and region_attrs['yaw_rotation'] != [0, 0]:
            yaw_rotation_str = " ".join([str(x) for x in region_attrs['yaw_rotation']])
            region_str += f"      (:yaw_rotation (\n        ({yaw_rotation_str})\n      ))\n"
        if 'rgba' in region_attrs and region_attrs['rgba'] != [0, 0, 1, 0]:
            rgba_str = " ".join([str(x) for x in region_attrs['rgba']])
            region_str += f"      (:rgba ({rgba_str}))\n"
        region_str += "    )"
        regions_str += f"    {region_str}\n"



    # Scene properties section (if available)
    scene_properties_str = ""
    for property_name, property_attrs in parsed_problem['scene_properties'].items():
        scene_property_str = f"({property_name}\n"
        if 'floor_style' in property_attrs:
            scene_property_str += f"      (:floor {property_attrs['floor_style']})\n"
        if 'wall_style' in property_attrs:
            scene_property_str += f"      (:wall {property_attrs['wall_style']})\n"
        scene_property_str += "    )"
        scene_properties_str += f"    {scene_property_str}\n"

    # Objects of Interest section
    obj_of_interest_str = "    " + "\n    ".join(parsed_problem['obj_of_interest'])

    # Initial state section
    init_state_str = "\n    ".join(
        ["({} {})".format(predicate[0], " ".join(predicate[1:])) for predicate in parsed_problem['initial_state']])

    # Goal state section
    goal_state_str = "\n    ".join(
        ["({} {})".format(predicate[0], " ".join(predicate[1:])) for predicate in parsed_problem['goal_state']])

    # Step 2: Construct Full BDDL String
    bddl_content = f"""(define (problem {parsed_problem['problem_name']})
      (:domain robosuite)
      (:language {" ".join(parsed_problem['language_instruction'])})
    
      (:regions
    {regions_str}  )
    
      (:fixtures
        {fixtures_str}
      )
    
      (:objects
        {objects_str}
      )
    
      (:obj_of_interest
        {obj_of_interest_str}
      )
    
      (:init
        {init_state_str}
      )
    
      (:goal
        (And
          {goal_state_str}
        )
      )
    """

    # Add scene properties if available
    if scene_properties_str:
        bddl_content += f"\n  (:scene_properties\n{scene_properties_str}  )\n"

    # End the define block
    bddl_content += "\n)"

    # Step 3: Save the new BDDL file
    with open(new_bddl_filename, "w") as f:
        f.write(bddl_content)

    # print(f"New BDDL file generated: {new_bddl_filename}")


if __name__ == '__main__':
    problem_filename = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl"
    new_bddl_filename = "./debug.bddl"