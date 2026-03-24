#!/usr/bin/env python3
"""
ROS2 node to visualize a point cloud and grasp poses.

This node:
  - Loads a PCD file (your gelatin box) using Open3D.
  - Publishes the point cloud as a sensor_msgs/PointCloud2 message.
  - Reads grasp poses (in the object's local frame) defined as a dictionary.
  - Transforms each grasp pose by the object's pose (which can later be randomized in simulation).
  - Creates a MarkerArray to show the coordinate axes (X/red, Y/green, Z/blue) for each grasp.
  - Publishes the MarkerArray so they are visible in RViz.
  
You can then verify that when the object's pose changes in the simulator, the transformed grasp axes (visualized via MarkerArray)
appear correctly around the object.
"""
import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add the placement workspace root to the Python path so we can import ycb_simulation module
placement_ws_root = "/home/chris/Chris/placement_ws/src/placement_quality/"
if placement_ws_root not in sys.path:
    sys.path.insert(0, placement_ws_root)

from isaacsim import SimulationApp
import json

DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON   = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height":1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
    "physics_dt": 1.0/60.0,  # Physics timestep (60Hz)
    "rendering_dt": 1.0/30.0,  # Rendering timestep (30Hz)
}

simulation_app = SimulationApp(CONFIG)

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from transforms3d.euler import euler2quat
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from ycb_simulation.utils.helper import draw_frame, transform_relative_pose, local_transform
from scipy.spatial.transform import Rotation as R
import pyquaternion
from placement_quality.path_simulation.model_testing.utils import get_flipped_object_pose
ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"

from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

class GraspVisualizer(Node):
    def __init__(self):
        super().__init__('grasp_visualizer')
        
        ################################
        ##### ROS parameters 
        ################################


        self.get_logger().info("Grasp Visualizer Node Initialized (ROS2)")

        # --- Load the PCD file ---
        # NOTE: Update the file path to your actual PCD file!
        grasps_file = "/home/chris/Chris/placement_ws/src/grasps.json"
        # object_poses_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/experiments/test_different_dimensions/object_poses_box.json"
        object_poses_file = "/home/chris/Chris/placement_ws/src/object_poses_box.json"
        
        # Load the original PCD data once
        # self.original_pcd = o3d.io.read_point_cloud(pcd_file)
        
        self.all_object_poses = json.load(open(object_poses_file))
        
        # --- Read Grasp Poses in Object Local Frame ---
        with open(grasps_file, 'r') as f:
            self.grasp_poses = json.load(f)


        
        # --- Object Pose in the Global Frame ---
        # This pose is normally provided by your simulator (or set randomly). For now, we use the identity (no translation/rotation).
        # When the simulator sets a random object pose, update this dictionary.
        self.object_pose = self.all_object_poses[1]

        self.transformed_grasp_pose = None




        ################################
        ##### Isaac Sim parameters 
        ################################

         # Basic variables
        self.world = None
        self.scene = None
        self.object = None  # Changed to list of objects
        self.object_name = "009_gelatin_box.usd"  # Fixed to one object
        self.grasp_offset_top = [0, 0, -0.075]
        self.grasp_offset_left = [0, 0, -0.15]
        self.grasp_offset_right = [0, 0, -0.075]
        self.box_dims = np.array([ 0.143, 0.0915,  0.051])
        # self.box_dims = np.array([ 0.0915, 0.051,  0.143])



        # Store the last used pose for comparison
        self.last_transform_pose = self.object_pose.copy()

    def get_pregrasp(self,grasp_pos, grasp_quat, offset=0.15):
        # grasp_quat: [w, x, y, z]
        # scipy uses [x, y, z, w]!
        grasp_quat_xyzw = [grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]
        rot = R.from_quat(grasp_quat_xyzw)
        # Get approach direction (z-axis of gripper in world frame)
        approach_dir = rot.apply([0, 0, 1])  # [0, 0, 1] is z-axis
        # Compute pregrasp position (move BACK along approach vector)
        pregrasp_pos = np.array(grasp_pos) - offset * approach_dir
        return pregrasp_pos, grasp_quat  # Same orientation

    def setup(self):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.scene = self.world.scene
        self.scene.add_default_ground_plane()
        
        # Add Franka robot and target frame BEFORE world.reset()
        self.add_franka_and_target()
        
        # Create object
        self.object_pose["position"][0] = 0.2
        self.object_pose["position"][1] = -0.3
        self.object_pose["position"][2] = 0.1715
        self.object_pose["orientation_quat"] = [0.5, 0.5, -0.5, 0.5] #wxzy
        # self.object_pose["orientation_quat"] = [0, 0, 0, -1] # wxzy for [ 0.0915, 0.051,  0.143]
        self.create_object(self.object_pose["position"], 
                           self.object_pose["orientation_quat"])
        print(f"The object pose before flipis: {self.object_pose}")
        object_pose = np.concatenate([self.object_pose["position"], self.object_pose["orientation_quat"]])

        ## VERY IMPORTANT: Visualization for the object after flip
        # final_pose = get_flipped_object_pose(object_pose, 180, axis='y')
        # self.object_pose["position"] = final_pose[:3]
        # self.object_pose["orientation_quat"] = final_pose[3:]
        # self.create_object(self.object_pose["position"], 
        #                    self.object_pose["orientation_quat"])
        # print(f"The object pose after flip is: {self.object_pose}")
        
        # Reset the world AFTER adding all objects
        self.world.reset()
        
        # Setup kinematics AFTER world.reset()
        self.setup_kinematics()
        
        # grasp pose draw frame
        self.local_grasp_pose = [self.grasp_poses["15"]["position"], 
                                   self.grasp_poses["15"]["orientation_wxyz"]]
        print(f"The local grasp pose is: {self.local_grasp_pose}")
        self.world_grasp_pose = transform_relative_pose(self.local_grasp_pose, 
                                                        self.object_pose["position"], 
                                                        self.object_pose["orientation_quat"])
        

        tmp_pose = self.world_grasp_pose[0]+self.world_grasp_pose[1]

        self.transformed_grasp_pose = local_transform(tmp_pose, [0,0,-0.065])
        draw_frame(self.transformed_grasp_pose[0], self.transformed_grasp_pose[1])

        pregrasp_pos, pregrasp_quat = self.get_pregrasp(self.transformed_grasp_pose[0], self.transformed_grasp_pose[1], offset=0.15)
        print(f"The pregrasp pose is: {pregrasp_pos}, {pregrasp_quat}")

        # Move the target to the grasp pose (not pregrasp, but actual grasp)
        self._target.set_world_pose(self.transformed_grasp_pose[0], self.transformed_grasp_pose[1])

        # DON'T call move_franka_to_target() here - it will be called in the simulation loop


    def create_object(self, pos, quat):
        """Create one YCB object as rigid bodies with physics enabled"""


        # )

        from omni.isaac.core.objects import VisualCuboid   
        object = VisualCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array(pos),
            orientation=quat,
            scale=self.box_dims.tolist(),  # [x, y, z]
            color=np.random.rand(3)  # Give each a random color if you want
        )

        self.scene.add(object)
        # from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
        # # Add the target frame to the stage (for IK control)
        # add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        # self.target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        # self.target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        # self.scene.add(self.target)

            





    def pose_to_matrix(self, position, orientation):
        """
        Converts a pose (position + quaternion) into a 4x4 homogeneous transform.
        
        The orientation is assumed to be in [w, x, y, z] order.
        """
        T = np.eye(4)
        q = pyquaternion.Quaternion(orientation)
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = np.array(position)
        return T

    def matrix_to_pose(self, T):
        """
        Converts a 4x4 homogeneous transformation matrix into (position, quaternion)
        where quaternion is [w, x, y, z].
        """
        pos = T[:3, 3].tolist()
        q = pyquaternion.Quaternion(matrix=T[:3, :3])
        quat = [q.elements[0], q.elements[1], q.elements[2], q.elements[3]]
        return pos, quat

    def add_franka_and_target(self):
        # Add Franka robot to the stage
        
        
        # Load robot and target (following collision_visualization.py pattern)
        self._articulation, self._target = self.load_assets()
        
        # Add assets to the world scene
        self.scene.add(self._articulation)
        self.scene.add(self._target)

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
        # Load kinematics configuration for the Franka robot
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)
        # Configure the articulation kinematics solver with the end effector
        end_effector_name = "panda_hand"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, self._kinematics_solver, end_effector_name
        )

    def move_franka_to_target(self):
        # Get the target position and orientation
        target_position, target_orientation = self._target.get_world_pose()
        # Set robot base pose (assume fixed at origin for now)
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        # Compute IK to move end effector to target
        grasp_position = np.array([ 0.165, -0.55,   0.225]) 
        grasp_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            grasp_position, grasp_orientation
        )
        if success:
            self._articulation.apply_action(action)
        else:
            print("IK did not converge to a solution. No action is being taken")



def main(args=None):
    rclpy.init(args=args)
    node = GraspVisualizer()
    node.setup()
    try:
        while simulation_app.is_running():
            node.world.step(render=True)
            # Move Franka to the grasp pose every step (to keep it at the pose)
            node.move_franka_to_target()
            rclpy.spin_once(node, timeout_sec=0)
            # print(f"The transformed grasp pose is: {node.transformed_grasp_pose}")
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
