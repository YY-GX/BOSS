from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.envs.predicates import get_predicate_fn_dict, get_predicate_fn
import numpy as np
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info



# yy: Define the scene
# yy: Because I use the kitchentable problem set, so if object's initial region is not target at the table,
#  it will not be correctly randomly placed
@register_mu(scene_type="kitchen")
class ShelfTableScene(InitialSceneTemplates):
    def __init__(self):
        # yy: Define basic info for fixtures and objects
        self.workspace_name = "kitchen_table"
        self.fixture_objs = {
            "wooden_shelf": {
                "region_centroid_xy": [0.0, -0.3],
                "target_name": self.workspace_name,
                "region_half_len": 0.01,
                "yaw_rotation": (np.pi, np.pi),
                "f_num": 1
            }
        }
        self.objects_objs = {
            "popcorn": {
                "region_centroid_xy": [0, 0],
                "target_name": self.workspace_name,
                "region_half_len": 0.01,
                "o_num": 1
            },
            "chocolate_pudding": {
                "region_centroid_xy": [0, 0.2],
                "target_name": self.workspace_name,
                "region_half_len": 0.01,
                "o_num": 1
            },
            # yy: if over 1 num for the object, shall use list() to encapsulate attributes. Example:
            # "plate": {
            #     "region_centroid_xy": [[0.0, 0.10], [0.0, 0.20]],
            #     "target_name": [self.workspace_name, self.workspace_name],
            #     "region_half_len": [0.01, 0.01],
            #     "o_num": 2
            # },
        }

        fixture_num_info = {f_k: f_v['f_num'] for f_k, f_v in self.fixture_objs.items()}
        fixture_num_info[self.workspace_name] = 1
        object_num_info = {o_k: o_v['o_num'] for o_k, o_v in self.objects_objs.items()}

        super().__init__(
            workspace_name=self.workspace_name,
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info
        )

    @property
    def init_states(self):
        # format: [target_name + obj_name_init_region]
        states = []

        # default init: each thing on its corresponding init_region
        for f_k, f_v in self.fixture_objs.items():
            if f_v["f_num"] > 1:
                for idx in range(f_v["f_num"]):
                    states += [("On", f"{f_k}_{idx}", f"{f_v['target_name'][idx]}_{f_k}_{idx}_init_region")]
            else:
                states += [("On", f"{f_k}_1", f"{f_v['target_name']}_{f_k}_init_region")]
        for o_k, o_v in self.objects_objs.items():
            if o_v["o_num"] > 1:
                for idx in range(o_v["o_num"]):
                    states += [("On", f"{o_k}_{idx}", f"{o_v['target_name'][idx]}_{o_k}_{idx}_init_region")]
            else:
                states += [("On", f"{o_k}_1", f"{o_v['target_name']}_{o_k}_init_region")]

        # special init
        # yy: Define special initialization region here


        print(states)
        return states

    def define_regions(self):
        # For fixtures
        for f_k, f_v in self.fixture_objs.items():
            if f_v["f_num"] > 1:
                for idx in range(f_v["f_num"]):
                    self.regions.update(
                        self.get_region_dict(region_name=f"{f_k}_{idx}_init_region",
                                             region_centroid_xy=f_v["region_centroid_xy"][idx],
                                             target_name=f_v["target_name"][idx],
                                             region_half_len=f_v["region_half_len"][idx],
                                             yaw_rotation=f_v["yaw_rotation"][idx])
                    )
            else:
                self.regions.update(
                    self.get_region_dict(region_name=f"{f_k}_init_region",
                                         region_centroid_xy=f_v["region_centroid_xy"],
                                         target_name=f_v["target_name"],
                                         region_half_len=f_v["region_half_len"],
                                         yaw_rotation=f_v["yaw_rotation"])
                )


        # For objects
        for o_k, o_v in self.objects_objs.items():
            if o_v["o_num"] > 1:
                for idx in range(o_v["o_num"]):
                    self.regions.update(
                        self.get_region_dict(region_name=f"{o_k}_{idx}_init_region",
                                             region_centroid_xy=o_v["region_centroid_xy"][idx],
                                             target_name=o_v["target_name"][idx],
                                             region_half_len=o_v["region_half_len"][idx])
                    )
            else:
                self.regions.update(
                    self.get_region_dict(region_name=f"{o_k}_init_region",
                                         region_centroid_xy=o_v["region_centroid_xy"],
                                         target_name=o_v["target_name"],
                                         region_half_len=o_v["region_half_len"])
                )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)








# yy: functions
def create_task(
        scene_name,
        language,
        objects_of_interest,
        goal_states
):
    register_task_info(language,
                       scene_name=scene_name,
                       objects_of_interest=objects_of_interest,
                       goal_states=goal_states)
def save_bddl(path):
    bddl_file_names, failures = generate_bddl_from_task_info(folder=path)
    print(bddl_file_names)
    print("Encountered some failures: ", failures)
    return bddl_file_names, failures







if __name__ == '__main__':
    # yy: Define tasks here
    scene_name = "shelf_table_scene"
    language = "Moving popcorn from table to topside of wooden shelf"
    create_task(scene_name=scene_name,
                language=language,
                objects_of_interest=[f"popcorn_1"],
                goal_states=[("On", f"popcorn_1", f"wooden_shelf_1_top_side")]
    )

    scene_name = "shelf_table_scene"
    language = "Moving chocolate pudding from table to topside of wooden shelf"
    create_task(scene_name=scene_name,
                language=language,
                objects_of_interest=[f"chocolate_pudding_1"],
                goal_states=[("On", f"chocolate_pudding_1", f"wooden_shelf_1_top_side")]
    )

    # save file
    path = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/yy_try"
    save_bddl(path)

