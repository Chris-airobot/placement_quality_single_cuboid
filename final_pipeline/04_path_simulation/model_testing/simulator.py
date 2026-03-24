import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import numpy as np
import omni
from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from scipy.spatial.transform import Rotation as R
from collision_check import GroundCollisionDetector
from pxr import Gf
from pxr import Sdf, UsdLux, Tf
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import ContactSensor
from RRT_controller import RRTController
from RRT_task import RRTTask
from functools import partial
from typing import List

PEDESTAL_SIZE = np.array([0.27, 0.22, 0.10])   # X, Y, Z in meters

# ---- data sources (no argparse) ----
SIM_PATH = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.jsonl"  # or *_sample.jsonl/json
PEDESTAL_POSES_PATH = "/home/chris/Chris/placement_ws/src/pedestal_poses.json"


class Simulator:
    def __init__(self, use_physics=False):
        # Basic variables
        self.world = None
        self.object = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.gripper = None
        self.task = None
        self.stage = omni.usd.get_context().get_stage()
        self.use_physics = use_physics
        # Counters and timers
        self.grasps = []
        self.placements = []
        self.current_grasp = None
        self.current_placement = None

        # State tracking
        self.base_path = "/World/Franka"

        self.state = "SETUP"
        self.open = True
        self.collision_counter = 0
        self.step_counter = 0
        self.pedestal_collision_counter = 0
        self.contact_sensors = []
        self.force_threshold = 0.8
        self.contact_force = 0.0
        self.forced_completion = False

        self.test_data: list[dict] = None
        self.current_data = None
        self.data_index = 0
        self.results = []


    def start(self):
        # Simulation Environment Setup
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/600.0)
        self.task = RRTTask("RRT_task", use_physics=self.use_physics)
        self.world.add_task(self.task)
        self.world.reset()
        
        # Robot and planner setup
        self.robot: Franka = self.world.scene.get_object(self.task.get_params()["robot_name"]["value"])
        self.gripper: ParallelGripper = self.robot.gripper
        self.articulation_controller = self.robot.get_articulation_controller()
        self.controller = RRTController(
            name="RRT_controller",
            robot_articulation=self.robot,
            ground_plane=self.task.ground_plane,
        )
        print(f"RRT Controller created")

        self.current_data = self.test_data[self.data_index]

        self.data_index += 1

        self.setup_contact_sensors()  # New method using random_collection.py style
        self.setup_collision_detection()

        self.world.reset()

        self.setup_kinematics()
        self._add_light_to_stage()


        self.task.set_params(
            object_position=np.array(self.current_data["initial_object_pose"][:3]),
            object_orientation=np.array(self.current_data["initial_object_pose"][3:]),
            preview_box_position=np.array(self.current_data["final_object_pose"][:3]),
            preview_box_orientation=np.array(self.current_data["final_object_pose"][3:]),
        )

        # Initialize pedestals for the first case
        ped = self.current_data.get("pedestal", {})
        pick_pose = ped.get("pick", {"position": [0.2, -0.3, 0.05]})
        place_pose = ped.get("place", {"position": [0.3, 0.0, 0.05]})
        self.update_pedestals(pick_pose, place_pose)


    def setup_contact_sensors(self):
        """Set up contact sensors using the style from random_collection.py"""
        # Define all robot parts to monitor for contact
        panda_prim_names = [
            "Franka/panda_leftfinger",
            "Franka/panda_rightfinger",
            "Ycb_object"  # The object being manipulated
        ]

        self.contact_sensors = []
        for i, link_name in enumerate(panda_prim_names):
            sensor: ContactSensor = self.world.scene.add(
                ContactSensor(
                    prim_path=f"/World/{link_name}/contact_sensor",
                    name=f"contact_sensor_{i}",
                    min_threshold=0.0,
                    max_threshold=1e7,
                    radius=0.1,
                )
            )
            # Use raw contact data for detailed contact information
            sensor.add_raw_contact_data_to_frame()
            self.contact_sensors.append(sensor)

        # Add physics callback for contact monitoring (random_collection.py style)
        self.world.add_physics_callback("contact_sensor_callback", 
                                       partial(self.on_sensor_contact_report, sensors=self.contact_sensors))


    def on_sensor_contact_report(self, dt, sensors: List[ContactSensor]):
        """
        Physics-step callback: checks all sensors, sets contact state accordingly.
        Uses the same style as random_collection.py
        """
        for sensor in sensors:
            frame_data = sensor.get_current_frame()
            if "in_contact" in frame_data and frame_data["in_contact"]:
                # We have contact! Extract the bodies, force, etc.
                for c in frame_data["contacts"]:
                    body0 = c["body0"]
                    body1 = c["body1"]
                    # Optionally, check which bodies are in contact:
                    if "Ycb_object" in (body0 + body1):
                        self.contact_force = frame_data["force"]


    def reset(self):
        """Reset the simulation environment"""
        print("Resetting simulation environment, robot invisible")
        self.world.reset()
        self.controller.reset()
        self.robot.gripper.open()
        self.step_counter = 0
        self.collision_counter = 0
        self.forced_completion = False
        # Reset object preview to current case (sim-driven)
        self.task.set_params(
            object_position=np.array(self.current_data["initial_object_pose"][:3]),
            object_orientation=np.array(self.current_data["initial_object_pose"][3:]),
            preview_box_position=np.array(self.current_data["final_object_pose"][:3]),
            preview_box_orientation=np.array(self.current_data["final_object_pose"][3:]),
        )
        # Also reset pedestal positions to avoid duplicates/drift across attempts
        ped = self.current_data.get("pedestal", {})
        pick_pose = ped.get("pick", {"position": [0.2, -0.3, 0.05]})
        place_pose = ped.get("place", {"position": [0.3, 0.0, 0.05]})
        try:
            self.update_pedestals(pick_pose, place_pose)
        except Exception:
            pass


    def slow_close_gripper_panda(self, step_size=0.001, min_pos=0.0, max_pos=0.04, n_steps=60):
        """
        Slowly closes the Franka/Panda gripper (with 2 finger joints) by incrementing joint positions
        to avoid physics penetration, with stepping after each command.
        Arguments:
        gripper: your ParallelGripper instance
        world: your IsaacSim World instance (for .step())
        step_size: increment per step (meters)
        min_pos: fully closed position (usually 0)
        max_pos: fully open position (usually 0.04 for Panda)
        n_steps: maximum number of increments
        Returns: None
        """
        # Get current finger positions (assuming symmetric for Panda)
        positions = list(self.gripper.get_joint_positions())
        # Each finger: move toward closed (min_pos), clamp for safety
        new_pos = [
            max(positions[0] - step_size, min_pos),
            min(positions[1] + step_size, -min_pos)
        ]
        self.gripper.apply_action(ArticulationAction(joint_positions=new_pos))
        positions = list(self.gripper.get_joint_positions())
        # Stop early if fingers reached closed position (within tolerance)
        if abs(new_pos[0] - min_pos) < 1e-4 and abs(new_pos[1] + min_pos) < 1e-4:
            return

    def calculate_placement_pose(self, grasp_pose, object_initial_pose, object_final_pose):
        """
        Given a grasp pose defined in the object's local frame at the initial pose,
        compute the corresponding grasp pose in world for the final object pose.
        Object poses are [x, y, z, qw, qx, qy, qz].
        Returns [final_grasp_position, final_grasp_quat_wxyz].
        """
        grasp_position_initial         = np.array(grasp_pose[0])
        grasp_orientation_wxyz_initial = np.array(grasp_pose[1])

        # Initial object pose in world: [x, y, z, qw, qx, qy, qz]
        object_position_initial         = np.array(object_initial_pose[:3])
        object_orientation_wxyz_initial = np.array(object_initial_pose[3:])

        # Final object pose in world: [x, y, z, qw, qx, qy, qz]
        object_position_final           = np.array(object_final_pose[:3])
        object_orientation_wxyz_final   = np.array(object_final_pose[3:])
    


        # scipy Rotation.from_quat expects [x, y, z, w]
        rotation_grasp_initial = R.from_quat([
            grasp_orientation_wxyz_initial[1],
            grasp_orientation_wxyz_initial[2],
            grasp_orientation_wxyz_initial[3],
            grasp_orientation_wxyz_initial[0]
        ]).as_matrix()

        rotation_object_initial = R.from_quat([
            object_orientation_wxyz_initial[1],
            object_orientation_wxyz_initial[2],
            object_orientation_wxyz_initial[3],
            object_orientation_wxyz_initial[0]
        ]).as_matrix()

        rotation_object_final = R.from_quat([
            object_orientation_wxyz_final[1],
            object_orientation_wxyz_final[2],
            object_orientation_wxyz_final[3],
            object_orientation_wxyz_final[0]
        ]).as_matrix()

        # --- 3) Compute the fixed hand‐to‐object transform ---------------

        # R_grasp_in_object = R_object_initial^T * R_grasp_initial
        rotation_grasp_in_object_frame = (
            rotation_object_initial.T @ rotation_grasp_initial
        )

        # p_grasp_in_object = R_object_initial^T * (p_grasp_initial – p_object_initial)
        translation_grasp_in_object_frame = (
            rotation_object_initial.T @ (grasp_position_initial - object_position_initial)
        )

        # --- 4) Re‐apply that to the new object pose --------------------

        # p_grasp_final = R_object_final * p_grasp_in_object + p_object_final
        final_grasp_position = (
            rotation_object_final @ translation_grasp_in_object_frame
            + object_position_final
        )

        # R_grasp_final = R_object_final * R_grasp_in_object
        rotation_final_grasp = (
            rotation_object_final @ rotation_grasp_in_object_frame
        )

        # --- 5) Convert back to quaternion [w, x, y, z] -----------------

        # scipy gives [x, y, z, w]
        quaternion_final_grasp_xyzw = R.from_matrix(
            rotation_final_grasp
        ).as_quat()

        # reorder to [w, x, y, z]
        quaternion_final_grasp_wxyz = np.array([
            quaternion_final_grasp_xyzw[3],
            quaternion_final_grasp_xyzw[0],
            quaternion_final_grasp_xyzw[1],
            quaternion_final_grasp_xyzw[2]
        ])

        # --- 6) Stack into your final grasp pose ------------------------

        return [final_grasp_position, quaternion_final_grasp_wxyz]

    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])


    def setup_collision_detection(self):
        """Set up the ground collision detector"""
        stage = omni.usd.get_context().get_stage()
        self.collision_detector = GroundCollisionDetector(stage, non_colliding_part=f"{self.base_path}/panda_link0")

        # Create a virtual ground plane at z=-0.05 (lowered to avoid false positives)
        self.collision_detector.create_virtual_ground(
            size_x=20.0,
            size_y=20.0,
            position=Gf.Vec3f(0, 0, -0.001/2)  # Lower the ground to avoid false positives
        )

        # Create initial pedestals for collision detection
        self.collision_detector.create_virtual_pedestal(
            position=Gf.Vec3f(0.2, -0.3, 0.05),
            size_x=float(PEDESTAL_SIZE[0]),
            size_y=float(PEDESTAL_SIZE[1]),
            size_z=float(PEDESTAL_SIZE[2])
        )
        self.collision_detector.create_virtual_place_pedestal(
            position=Gf.Vec3f(0.3, 0.0, 0.05),
            size_x=float(PEDESTAL_SIZE[0]),
            size_y=float(PEDESTAL_SIZE[1]),
            size_z=float(PEDESTAL_SIZE[2])
        )

        # Define robot parts to check for collisions (explicitly excluding the base link0)
        self.robot_parts_to_check = [
            f"{self.base_path}/panda_link1",
            f"{self.base_path}/panda_link2",
            f"{self.base_path}/panda_link3",
            f"{self.base_path}/panda_link4",
            f"{self.base_path}/panda_link5",
            f"{self.base_path}/panda_link6",
            f"{self.base_path}/panda_link7",
            f"{self.base_path}/panda_hand",
            f"{self.base_path}/panda_leftfinger",
            f"{self.base_path}/panda_rightfinger"
        ]
        
    def check_for_collisions(self):
        """Check if any robot parts are colliding with ground, pedestal, or box."""
        ground_hit = any(
            self.collision_detector.is_colliding_with_ground(part_path)
            for part_path in self.robot_parts_to_check
        )
        pedestal_hit = any(
            self.collision_detector.is_colliding_with_pedestal(part_path)
            for part_path in self.robot_parts_to_check
        )

        place_pedestal_hit = any(
            self.collision_detector.is_colliding_with_place_pedestal(part_path)
            for part_path in self.robot_parts_to_check
        )



        # box_hit = any(
        #     self.collision_detector.is_colliding_with_box(part_path)
        #     for part_path in self.robot_parts_to_check
        # )
        self.collision_detected = ground_hit or pedestal_hit or place_pedestal_hit
        return self.collision_detected

    def check_object_pedestal_collision(self):
        """Check if the grasped object is colliding with the pedestal."""
        return self.collision_detector.is_object_colliding_with_pedestal()

    def check_object_place_pedestal_collision(self):
        """Check if the grasped object is colliding with the place pedestal."""
        return self.collision_detector.is_colliding_with_place_pedestal()

    def update_pedestals(self, pick_pose, place_pose):
        """Update both visual pedestals and collision detector pedestal positions"""
        # Update visual pedestals in the task
        pick_pos = pick_pose["position"]
        place_pos = place_pose["position"]

        self.task.set_params(
            pick_pedestal_position=pick_pos,
            place_pedestal_position=place_pos
        )

        # Update pick pedestal position
        self.collision_detector.update_pick_pedestal_position(pick_pos)
        # Update place pedestal position
        self.collision_detector.update_place_pedestal_position(place_pos)

    def setup_kinematics(self):
        """Set up the kinematics solver for the Franka robot"""
        # Load kinematics configuration for the Franka robot
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        # Print valid frame names for debugging
        # print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        # Configure the articulation kinematics solver with the end effector
        end_effector_name = "panda_hand"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.robot,
            self._kinematics_solver,
            end_effector_name
        )

    def update(self, grasp_pose):
        import carb
        """Update the robot's position based on the target's position"""
        # object_position, object_orientation = self._target.get_world_pose()
        
        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self.robot.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        # Compute inverse kinematics to find joint positions
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            np.array(grasp_pose[0:3]), 
            np.array(grasp_pose[3:])
        )
        
        # Apply the joint positions if IK was successful
        if success:
            self.robot.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken")
        
        return success
    

    def check_grasp_success(self):
        """Check if the grasp was successful"""
        # print(f"You are in the grasp success check")
        print(f"Gripper position: {self.robot._gripper.get_joint_positions()}")
        # Check if gripper is fully closed (no object grasped) with tolerance
        gripper_tolerance = 0.001  # 1mm tolerance
        if self.robot._gripper.get_joint_positions()[0] < gripper_tolerance or self.robot._gripper.get_joint_positions()[1] < gripper_tolerance:
            # print("Gripper is fully closed, didn't grasp object")
            return False
            
        return True
    
    def check_ik(self, position, orientation):
        """Check if the IK is successful"""
        return self.controller._make_new_plan(
            np.array(position), 
            np.array(orientation)
        )
