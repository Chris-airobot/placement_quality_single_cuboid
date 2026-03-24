import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to the Python path to access collision_check.py
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from isaacsim import SimulationApp

# Display options for the simulation viewport
DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

# Simulation configuration
CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

# Initialize the simulation application
simulation_app = SimulationApp(CONFIG)

import os
import rclpy
import datetime 
from omni.isaac.core.utils import extensions
import rclpy

from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni

import numpy as np
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import json
import tf2_ros
from rclpy.time import Time
import struct
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, quat2euler
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import random
import math
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

import carb
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux

# Import the collision detection functionality
from collision_check import GroundCollisionDetector
from pxr import UsdGeom, Gf, Usd

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")


class StandaloneIKWithCollision:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None
        self.world: World = None
        
        # Collision detection variables
        self.collision_detector = None
        self.base_path = "/World/panda"
        self.robot_parts_to_check = []
        self.collision_detected = False

    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()
        
        # Load robot and target
        self._articulation, self._target = self.load_assets()
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        
        # Add the box object to the scene for collision detection
        from omni.isaac.core.objects import VisualCuboid
        box_dims = np.array([0.143, 0.0915, 0.051])
        self._box = VisualCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array([0.2, -0.3, 0.125]),  # Adjusted z to sit properly on pedestal (0.05 + 0.10 + 0.051/2)
            scale=box_dims.tolist(),
            color=np.array([0.8, 0.8, 0.8])
        )
        self.world.scene.add(self._box)
        
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        # Set up the physics scene
        self.world.reset()
        
        # Set up the kinematics solver
        self.setup_kinematics()

    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def load_assets(self):
        """Load the Franka robot and target frame"""
        # Add the Franka robot to the stage
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        articulation = Articulation(robot_prim_path)
        
        # Add the target frame to the stage (for IK control)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        
        return articulation, target

    def setup_kinematics(self):
        """Set up the kinematics solver for the Franka robot"""
        # Load kinematics configuration for the Franka robot
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)
        
        # Print valid frame names for debugging
        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
        
        # Configure the articulation kinematics solver with the end effector
        end_effector_name = "panda_hand"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, 
            self._kinematics_solver, 
            end_effector_name
        )

    def setup_collision_detection(self):
        """Set up the ground collision detector"""
        stage = omni.usd.get_context().get_stage()
        self.collision_detector = GroundCollisionDetector(stage, non_colliding_part="/World/panda/panda_link0")
        
        # Create a virtual ground plane at z=0 (ground level)
        self.collision_detector.create_virtual_ground(
            size_x=20.0, 
            size_y=20.0, 
            position=Gf.Vec3f(0, 0, 0)  # Ground level
        )

        self.collision_detector.create_virtual_pedestal(
            position=Gf.Vec3f(0.2, -0.3, 0.05)
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
        # Note: panda_link0 is deliberately excluded since it's expected to touch the ground

    def change_box_color(self, is_colliding):
        """Change the box color based on collision status"""
        try:
            if is_colliding:
                self._box.color = np.array([1.0, 0.0, 0.0])  # Red
            else:
                self._box.color = np.array([0.8, 0.8, 0.8])  # Light gray
        except Exception as e:
            print(f"Error changing box color: {e}")

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
        box_hit = any(
            self.collision_detector.is_colliding_with_box(part_path)
            for part_path in self.robot_parts_to_check
        )
        self.collision_detected = ground_hit or pedestal_hit or box_hit
        
        # Change box color based on collision status
        self.change_box_color(box_hit)
        
        # Print detailed collision status for debugging
        if self.collision_detected:
            collision_types = []
            if ground_hit:
                collision_types.append("GROUND")
            if pedestal_hit:
                collision_types.append("PEDESTAL")
            if box_hit:
                collision_types.append("BOX")
            print(f"COLLISION DETECTED! Types: {', '.join(collision_types)}")
        else:
            print("No collision detected.")
        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
            
        return self.collision_detected

    def update(self, step):
        """Update the robot's position based on the target's position"""
        # Get the target position and orientation
        target_position, target_orientation = self._target.get_world_pose()
        
        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        
        # Compute inverse kinematics to find joint positions
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, 
            target_orientation
        )
        
        # Apply the joint positions if IK was successful
        if success:
            self._articulation.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken")
        
        # Check for collisions with the ground
        collision = self.check_for_collisions()

    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()
        
        print("=== IK with Collision Detection Tool ===")
        print("This tool allows you to test IK and collision detection.")
        print("You can drag the target frame to move the robot via IK.")
        print("The system will detect collisions with:")
        print("  - Ground (virtual ground plane)")
        print("  - Pedestal (cylinder)")
        print("  - Box (gripper finger contact)")
        print("Collision status will be printed to the console.")
        print("Press Ctrl+C to exit.")
        print("=====================================")
        
        # Main simulation loop
        while simulation_app.is_running():
            # Step the simulation
            self.world.step(render=True)
            
            # Update the robot's position
            self.update(step=1.0/60.0)

            # Now access the Jacobian safely!
            ee_index = self._articulation._articulation_view.get_link_index("panda_hand")
            physx_interface = self._articulation._articulation_view._physics_view
            jacobian = np.zeros((6, 7))
            jacobians =physx_interface.get_jacobians()
            # Get index of the end-effector link

            # Extract the Jacobian for your robot and EE link:
            # First index: env/robot (usually 0 if just one), second: link index
            jacobian = jacobians[0, ee_index, :, :]  # shape (6, 7)
            print(f"Jacobian is:\n{jacobian}")

        # Cleanup when simulation ends
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example with collision detection
    env = StandaloneIKWithCollision()
    env.run()
    
    # Close the simulation application
    simulation_app.close()