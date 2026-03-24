# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from abc import ABC, abstractmethod
from typing import Optional

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from pyquaternion import Quaternion
from omni.isaac.franka import Franka
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import omni
from helper import *
import numpy as np

class MyPickPlace(ABC, BaseTask):
    """[summary]

    Args:
        name (str): [description]
        cube_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_target_orientation: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        set_camera: bool = True,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_cube = None
        self._cube = None
        self._cube_final = None
        self._camera = None
        self._set_camera = set_camera
        self._cube_initial_position = cube_initial_position
        self._cube_initial_orientation = cube_initial_orientation
        self._cube_target_position = target_position
        self._cube_target_orientation = cube_target_orientation
        self._cube_size = cube_size
        
        if self._cube_initial_position is None:
            # self._cube_initial_position, self._cube_target_position, self._cube_initial_orientation, self._cube_target_orientation = task_randomization()
            self.cube_init()
        if self._cube_size is None:
            self._cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
    

        return

    def cube_init(self, first_time=True):
        self._cube_initial_position, self._cube_initial_orientation = pose_init()
        self._cube_target_position, self._cube_target_orientation = pose_init()
        if not first_time:
            self.set_params(
                cube_position=self._cube_initial_position,
                cube_orientation=self._cube_initial_orientation,
                cube_target_position=self._cube_target_position,
                cube_target_orientation=self._cube_target_orientation,
            )
        return 

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        scene.add_default_ground_plane()


        initial_cube_prim_path = find_unique_string_name(
            initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))
        self._cube = scene.add(
            DynamicCuboid(
                name=cube_name,
                position=self._cube_initial_position,
                orientation=self._cube_initial_orientation,
                prim_path=initial_cube_prim_path,
                scale=self._cube_size,
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        )


        final_cube_prim_path = find_unique_string_name(
            initial_name="/World/Cube_final", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        final_cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))

        self._cube_final = scene.add(
            DynamicCuboid(
                name=final_cube_name,
                position=self._cube_target_position,
                orientation=self._cube_target_orientation,
                prim_path=final_cube_prim_path,
                scale=self._cube_size,
                size=1.0,
                color=np.array([1, 0, 0]),
            )
        )


        self.attach_face_markers(initial_cube_prim_path)
        self._task_objects[self._cube.name] = self._cube
        self._task_objects[self._cube_final.name] = self._cube_final

        self._robot: Franka = self.set_robot()
        self._robot.set_enabled_self_collisions(True)
        scene.add(self._robot)

        # set up the camera
        orientation_degrees = np.deg2rad([0, -90, 180]) # or [0, -90, 180]
        orientation = euler2quat(orientation_degrees[0], orientation_degrees[1], orientation_degrees[2])
        if self._set_camera:
            self._camera = Camera(
                prim_path="/World/Franka/panda_hand/geometry/realsense/realsense_camera",
                name="realsense_camera",
                translation=[0.05, 0.0, 0.05],
                orientation=orientation,
                resolution=[640, 480],
            )
                    # Set the focal length (in millimeters) to match something similar to a RealSense D435.
            self._camera.set_focal_length(0.193)

            # Set the clipping range. Typically, near and far clipping distances are in meters.
            self._camera.set_clipping_range(0.0001, 1000.0)
        # stage.GetRootLayer().Save()


        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()

        return

    @abstractmethod
    def set_robot(self) -> None:
        raise NotImplementedError
    

    def attach_face_markers(self, cube_prim_path, cube_size=1.0, offset=0.05):
        """
        Attaches a small sphere marker (with a label in its name) to each face of the cube.
        These markers are visual-only and will not be used for collisions.
        
        Args:
            cube_prim_path (str): The prim path of the cube.
            cube_size (float): The overall (uniform) size of the cube.
            offset (float): Extra offset to push the marker outwards from the face.
        """
        stage = omni.usd.get_context().get_stage()
        half = cube_size / 2.0

        # Define for each face: a translation and a label.
        markers = {
            "front":  (Gf.Vec3d(half + offset, 0, 0),        "1"),
            "back":   (Gf.Vec3d(-half - offset, 0, 0),       "2"),
            "left":   (Gf.Vec3d(0, half + offset, 0),        "3"),
            "right":  (Gf.Vec3d(0, -half - offset, 0),       "4"),
            "top":    (Gf.Vec3d(0, 0, half + offset),        "5"),
            "bottom": (Gf.Vec3d(0, 0, -half - offset),       "6")
        }
        
        # Get the cube prim.
        cube_prim = stage.GetPrimAtPath(cube_prim_path)
        if not cube_prim:
            print(f"Cube prim not found at {cube_prim_path}")
            return

        # Loop over each face.
        for face, (translation, label) in markers.items():
            # Create a marker Xform as a child of the cube.
            marker_path = cube_prim.GetPath().AppendChild(f"marker_{face}")
            marker = UsdGeom.Xform.Define(stage, marker_path)
            marker.AddTranslateOp().Set(translation)
            
            # Under the marker, create a small sphere.
            sphere_path = marker_path.AppendChild("sphere")
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.03)
            
            # (Optional) You could also set a custom attribute on the marker or sphere
            # to indicate the face number, or name the prim accordingly.
            # print(f"Attached marker for face '{face}' with label {label} at {translation}")

    def cube_pose_finalization(self) -> None:
        """ Both cubes finished setting up.
        """
        cube_position, cube_orientation = self._cube.get_world_pose()   
        cube_target_position, cube_target_orientation = self._cube_final.get_world_pose()
        # print(f"Cube initial position: {cube_position}")
        # print(f"Cube initial orientation: {cube_orientation}")
        # print(f"Cube target position: {cube_target_position}")
        # print(f"Cube target orientation: {cube_target_orientation}")
        self.set_params(
            cube_position=cube_position,
            cube_orientation=cube_orientation,
            cube_target_position=cube_target_position,
            cube_target_orientation=cube_target_orientation,
        )

        self._cube_final.set_world_pose(position=[1000,1000,0.5], orientation=cube_target_orientation)
        return


    def set_params(
        self,
        cube_position: Optional[np.ndarray] = None,
        cube_orientation: Optional[np.ndarray] = None,
        cube_target_position: Optional[np.ndarray] = None,
        cube_target_orientation: Optional[np.ndarray] = None,
    ) -> None:
        if cube_target_position is not None:
            self._cube_target_position = cube_target_position
        if cube_target_orientation is not None:
            self._cube_target_orientation = cube_target_orientation
        if cube_position is not None or cube_orientation is not None:
            self._cube.set_world_pose(position=cube_position, orientation=cube_orientation)
        return

    def get_params(self) -> dict:
        params_representation = dict()
        position, orientation = self._cube.get_world_pose()

        params_representation["cube_current_position"] = {"value": position, "modifiable": True}
        params_representation["cube_current_orientation"] = {"value": orientation, "modifiable": True}

        params_representation["cube_target_position"] = {"value": self._cube_target_position, "modifiable": True}
        params_representation["cube_target_orientation"] = {"value": self._cube_target_orientation, "modifiable": True}
        params_representation["cube_name"] = {"value": self._cube.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        cube_position, cube_orientation = self._cube.get_world_pose()
        end_effector_position, end_effector_orientation = get_current_end_effector_pose()

        observations = {
            self._cube.name: {
                "cube_current_position": cube_position,
                "cube_current_orientation": cube_orientation,
                "cube_target_position": self._cube_target_position,
                "cube_target_orientation": self._cube_target_orientation,
            },
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
                "end_effector_orientation": end_effector_orientation,
            }
        }
        return observations



        

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper

        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return

    def calculate_metrics(self) -> dict:
        """[summary]"""
        raise NotImplementedError

    def is_done(self) -> bool:
        """[summary]"""
        raise NotImplementedError
    

