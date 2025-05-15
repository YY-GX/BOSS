import os
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)


class TurbosquidObjects(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # yy: I add these
        if obj_name.startswith("smaller_"):
            base_name = obj_name[len("smaller_"):]
        elif obj_name.startswith("larger_"):
            base_name = obj_name[len("larger_"):]
        else:
            base_name = obj_name

        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/turbosquid_objects/{obj_name}/{base_name}.xml",
            ),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}


@register_object
class WoodenTray(TurbosquidObjects):
    def __init__(
        self,
        name="wooden_tray",
        obj_name="wooden_tray",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WhiteStorageBox(TurbosquidObjects):
    def __init__(
        self,
        name="white_storage_box",
        obj_name="white_storage_box",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (0, 0)
        self.rotation_axis = "y"


@register_object
class WoodenShelf(TurbosquidObjects):
    def __init__(
        self,
        name="wooden_shelf",
        obj_name="wooden_shelf",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WoodenTwoLayerShelf(TurbosquidObjects):
    def __init__(
        self,
        name="wooden_two_layer_shelf",
        obj_name="wooden_two_layer_shelf",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WineRack(TurbosquidObjects):
    def __init__(
        self,
        name="wine_rack",
        obj_name="wine_rack",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WineBottle(TurbosquidObjects):
    def __init__(
        self,
        name="wine_bottle",
        obj_name="wine_bottle",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class DiningSetGroup(TurbosquidObjects):
    """This dining set group is mostly for visualization"""

    def __init__(
        self,
        name="dining_set_group",
        obj_name="dining_set_group",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BowlDrainer(TurbosquidObjects):
    def __init__(
        self,
        name="bowl_drainer",
        obj_name="bowl_drainer",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class MokaPot(TurbosquidObjects):
    def __init__(
        self,
        name="moka_pot",
        obj_name="moka_pot",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BlackBook(TurbosquidObjects):
    def __init__(
        self,
        name="black_book",
        obj_name="black_book",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 4)


@register_object
class YellowBook(TurbosquidObjects):
    def __init__(
        self,
        name="yellow_book",
        obj_name="yellow_book",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 4)


@register_object
class RedCoffeeMug(TurbosquidObjects):
    def __init__(
        self,
        name="red_coffee_mug",
        obj_name="red_coffee_mug",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)


@register_object
class DeskCaddy(TurbosquidObjects):
    def __init__(
        self,
        name="desk_caddy",
        obj_name="desk_caddy",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class PorcelainMug(TurbosquidObjects):
    def __init__(
        self,
        name="porcelain_mug",
        obj_name="porcelain_mug",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)


@register_object
class WhiteYellowMug(TurbosquidObjects):
    def __init__(
        self,
        name="white_yellow_mug",
        obj_name="white_yellow_mug",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)


# yy: For Local Generalize
# scale: 1.5
@register_object
class LargerMokaPot(TurbosquidObjects):
    def __init__(
        self,
        name="larger_moka_pot",
        obj_name="larger_moka_pot",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)

# scale: 0.75
@register_object
class SmallerMokaPot(TurbosquidObjects):
    def __init__(
        self,
        name="smaller_moka_pot",
        obj_name="smaller_moka_pot",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)


# scale: 1.25
@register_object
class LargerRedCoffeeMug(TurbosquidObjects):
    def __init__(
        self,
        name="larger_red_coffee_mug",
        obj_name="larger_red_coffee_mug",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)


# scale: 0.6
@register_object
class SmallerRedCoffeeMug(TurbosquidObjects):
    def __init__(
        self,
        name="smaller_red_coffee_mug",
        obj_name="smaller_red_coffee_mug",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)



# scale: 0.75
@register_object
class SmallerPorcelainMug(TurbosquidObjects):
    def __init__(
        self,
        name="smaller_porcelain_mug",
        obj_name="smaller_porcelain_mug",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)
