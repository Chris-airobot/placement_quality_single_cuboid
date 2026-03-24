from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import omni
from omni.isaac.core.objects import FixedCuboid, VisualCuboid, DynamicCuboid, FixedCylinder
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units, add_reference_to_stage
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.objects import VisualCylinder

from pxr import UsdPhysics, Usd, UsdGeom, Gf

PEDESTAL_SIZE = np.array([0.27, 0.22, 0.10])   # X, Y, Z in meters (v7)

class RRTTask(BaseTask):
    def __init__(
        self,
        name: str,
        use_physics: bool=False,
        initial_position: Optional[np.ndarray] = None,
        initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self.use_physics = use_physics
        self._ycb_name = None
        self._ycb = None
        self._ycb_prim_path = None
        self._frame = None
        # self.box_dims = np.array([0.031, 0.096, 0.190])
        self.box_dims = np.array([0.143, 0.0915,  0.051])
        self._ycb_initial_position = initial_position
        self._ycb_initial_orientation = initial_orientation
        self._ycb_target_position = target_position
        self._ycb_target_orientation = target_orientation
        self._obstacle_walls = OrderedDict()
        self.ground_plane = None

        # Pedestal params
        self.pedestal_radius = 0.08  # meters
        self.pedestal_height = 0.10  # meters

        # Place the pedestal so its top is at z=0 (i.e., flush with ground)
        self.pedestal_center_z = self.pedestal_height / 2

        return

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        if self.use_physics:
            scene.add_default_ground_plane()

            self.ground_plane = DynamicCuboid(
                prim_path="/World/CollisionGround",
                name="collision_ground",
                position=np.array([0.0, 0.0, -0.0005]),  # Match your visual ground position
                scale=np.array([20.0, 20.0, 0.001]),     # Match size and thickness
                color=np.array([0.0, 0.0, 0.0])     # Make it invisible if you want (alpha=0)
            )
            scene.add(self.ground_plane)

    

        # Add the object to the scene
        self._ycb = self.set_ycb(name="ycb_object", 
                                 prim_path="/World/Ycb_object", 
                                 use_physics=True)
        # self._ycb = DynamicCuboid(
        #     prim_path="/World/Ycb_object",
        #     name="Ycb_object",
        #     position=np.array([0, 0, 0], dtype=float),
        #     orientation=np.array([0, 0, 0, 1], dtype=float),
        #     scale=self.box_dims.tolist(),
        #     color=np.array([0.8, 0.8, 0.8], dtype=float)
        # )
        
        
        scene.add(self._ycb)

        

        # Add the robot to the scene
        self._robot = self.set_robot()
        scene.add(self._robot)


        # Add the visual object:
        placement_pos = np.array([300000, 0.2, 0.1])         # [x, y, z] in meters
        placement_quat = np.array([0, 0, 0, 1])           # [x, y, z, w]

        self.preview_box = VisualCuboid(
                prim_path="/World/PreviewBox",
                name="preview_box",
                position=placement_pos,
                orientation=placement_quat,
                scale=self.box_dims,  # Match your YCB object size
                color=np.array([0.0, 1.0, 0.0])
        )
        scene.add(self.preview_box)


        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._frame = XFormPrim("/World/target", scale=[.04,.04,.04], name="target")
        self._frame.set_default_state(np.array([0.3, 0, 0.5]),
                                np.array([0, 0, 0, 1]))
        
        scene.add(self._frame)

        # Visual pedestals for pick/place (keep visible; collision handled in simulator)
        self.pick_pedestal = FixedCuboid(
            prim_path="/World/PickPedestal",
            name="PickPedestalVis",
            position=np.array([0.2, -0.3, 0.05]),
            scale=PEDESTAL_SIZE,
            color=np.array([0.6, 0.6, 0.6])
        )
        self.place_pedestal = FixedCuboid(
            prim_path="/World/PlacePedestal",
            name="PlacePedestalVis",
            position=np.array([0.3, 0.0, 0.05]),
            scale=PEDESTAL_SIZE,
            color=np.array([0.5, 0.5, 0.5])
        )
        scene.add(self.pick_pedestal)
        scene.add(self.place_pedestal)

        # Set the object parameters
        self.set_params(
            object_position=self._ycb_initial_position,
            object_orientation=self._ycb_initial_orientation,
        )

        # Add the robot and object to the task objects
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._ycb.name] = self._ycb
        # self._task_objects[self._target.name] = self._target

        # Move the task objects to their frame
        self._move_task_objects_to_their_frame()
        return



    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        
        self._franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name)


    def set_ycb(self, name: str, prim_path: str, use_physics: bool=False) -> XFormPrim:
        """Create a simple rigid box with the specified dimensions.
        
        Returns:
            XFormPrim: The created box object
        """
        # Create the initial object prim path and name
        self._ycb_prim_path = find_unique_string_name(
            initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._ycb_name = find_unique_string_name(
            initial_name=name, is_unique_fn=lambda x: not self.scene.object_exists(x)
        )

        if use_physics:
            import omni
            import omni.physx
            from pxr import UsdGeom, UsdShade, PhysxSchema
            stage = omni.usd.get_context().get_stage()

            # Create the main box prim
            object_prim = UsdGeom.Cube.Define(stage, prim_path)
            
            # Set RigidBody and Mass
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(object_prim.GetPrim())
            UsdPhysics.CollisionAPI.Apply(object_prim.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(object_prim.GetPrim())
            mass_api.CreateMassAttr(0.5)
            
            # Set the box dimensions using the box_dims from line 40
            object_prim.CreateSizeAttr(1.0)  # Base size of 1
            object_prim.AddScaleOp().Set(tuple(self.box_dims))  # Scale to actual dimensions
            object_prim.AddTranslateOp().Set((0, 0, 0))  # Centered
            
            # ---- PhysX Material ----
            material_path = prim_path + "/physx_material"

            for mat_type in ["PhysxSchema.PhysxMaterial", "PhysicsMaterial"]:
                if not stage.GetPrimAtPath(material_path).IsValid():
                    mat_prim = stage.DefinePrim(material_path, mat_type)
                else:
                    mat_prim = stage.GetPrimAtPath(material_path)
                # Set attributes if they exist
                try:
                    mat_prim.GetAttribute("physxStaticFriction").Set(1)
                    mat_prim.GetAttribute("physxDynamicFriction").Set(1)
                    mat_prim.GetAttribute("physxRestitution").Set(0.0)
                    break  # Succeed and exit loop
                except Exception:
                    continue

            # ---- Material Binding ----
            def bind_physx_material(prim, mat_prim):
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(UsdShade.Material(mat_prim))

            bind_physx_material(object_prim.GetPrim(), mat_prim)

        else:
            # For non-physics case, just create a visual cube
            import omni
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            object_prim = UsdGeom.Cube.Define(stage, prim_path)
            object_prim.CreateSizeAttr(1.0)
            object_prim.AddScaleOp().Set(tuple(self.box_dims))
            object_prim.AddTranslateOp().Set((0, 0, 0))

        return XFormPrim(prim_path=prim_path, name=name)



    def set_params(
        self,
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[np.ndarray] = None,
        preview_box_position: Optional[np.ndarray] = None,
        preview_box_orientation: Optional[np.ndarray] = None,
        pick_pedestal_position: Optional[np.ndarray] = None,
        place_pedestal_position: Optional[np.ndarray] = None,
        # object_target_position: Optional[np.ndarray] = None,
        # object_target_orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Set the object parameters."""
        if object_position is not None or object_orientation is not None:            
            self._ycb.set_world_pose(position=object_position, orientation=object_orientation)
        if preview_box_position is not None or preview_box_orientation is not None:
            self.preview_box.set_world_pose(position=preview_box_position, orientation=preview_box_orientation)
        if pick_pedestal_position is not None:
            self.pick_pedestal.set_world_pose(position=pick_pedestal_position)
        if place_pedestal_position is not None:
            self.place_pedestal.set_world_pose(position=place_pedestal_position)
        # if object_target_position is not None or object_target_orientation is not None:
        #     self._target.set_world_pose(position=object_target_position, orientation=object_target_orientation)
        return

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["object_initial_position"] = {"value": self._ycb_initial_position, "modifiable": True}
        params_representation["object_initial_orientation"] = {"value": self._ycb_initial_orientation, "modifiable": True}
        params_representation["object_target_position"] = {"value": self._ycb_target_position, "modifiable": True}
        params_representation["object_target_orientation"] = {"value": self._ycb_target_orientation, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        # ycb_position, ycb_orientation = self._ycb.get_local_pose()
        return {
            self._robot.name: {
                "joint_positions": np.array(joints_state.positions),
                "joint_velocities": np.array(joints_state.velocities),
            },
            # self._ycb.name: {"position": np.array(ycb_position), "orientation": np.array(ycb_orientation)},
        }

    def target_reached(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        ycb_position, _ = self._ycb.get_world_pose()
        if np.mean(np.abs(np.array(end_effector_position) - np.array(ycb_position))) < (0.035 / get_stage_units()):
            return True
        else:
            return False


    def add_obstacle(self, position: np.ndarray = None, orientation=None):
        """[summary]

        Args:
            position (np.ndarray, optional): [description]. Defaults to np.array([0.1, 0.1, 1.0]).
        """
        # TODO: move to task frame if there is one
        cube_prim_path = find_unique_string_name(
            initial_name="/World/WallObstacle", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="wall", is_unique_fn=lambda x: not self.scene.object_exists(x))
        if position is None:
            position = np.array([0.6, 0.1, 0.2]) / get_stage_units()
        if orientation is None:
            orientation = euler_angles_to_quat(np.array([0, 0, np.pi / 3]))
        cube = self.scene.add(
            VisualCuboid(
                name=cube_name,
                position=position + self._offset,
                orientation=orientation,
                prim_path=cube_prim_path,
                size=1.0,
                scale=np.array([0.1, 0.5, 0.6]) / get_stage_units(),
                color=np.array([0, 0, 1.0]),
            )
        )
        self._obstacle_walls[cube.name] = cube
        return cube

    def remove_obstacle(self, name: Optional[str] = None) -> None:
        """[summary]

        Args:
            name (Optional[str], optional): [description]. Defaults to None.
        """
        if name is not None:
            self.scene.remove_object(name)
            del self._obstacle_walls[name]
        else:
            obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]
        return

    def get_obstacles(self) -> List:
        return list(self._obstacle_walls.values())

    def get_obstacle_to_delete(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
        return self.scene.get_object(obstacle_to_delete)

    def obstacles_exist(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if len(self._obstacle_walls) > 0:
            return True
        else:
            return False

    def cleanup(self) -> None:
        """[summary]"""
        obstacles_to_delete = list(self._obstacle_walls.keys())
        for obstacle_to_delete in obstacles_to_delete:
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]
        return



    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return (1e15 * np.ones(9), 1e13 * np.ones(9))


    def set_prim_color(self, prim_path, color=[1,0,0], alpha=1.0):
        """
        Sets the display color and (optionally) alpha for a prim at prim_path.
        - prim_path: str, absolute USD path (e.g. "/World/VisualCube")
        - color: list/tuple of 3 floats [R,G,B], each in [0,1]
        - alpha: float in [0,1], only works if supported
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[set_prim_color] Prim {prim_path} not found!")
            return False

        try:
            geom = UsdGeom.Gprim(prim)
            # Set color
            geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            # Set alpha (if supported)
            if alpha < 1.0:
                try:
                    geom.CreateDisplayOpacityAttr().Set([alpha])
                except Exception:
                    print(f"[set_prim_color] Alpha not supported for {prim_path}.")
            return True
        except Exception as e:
            print(f"[set_prim_color] Failed to set color for {prim_path}: {e}")
            return False

    