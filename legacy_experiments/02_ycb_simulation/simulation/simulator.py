import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import rclpy
import time

from rclpy.executors import SingleThreadedExecutor

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.sensor import ContactSensor
import omni

from simulation.task import YcbTask
from simulation.planner import YcbPlanner

from utils.vision import *
from utils.helper import *


class YcbCollection:
    def __init__(self):
        # Basic variables
        self.world = None
        self.object = None
        self.planner = None
        self.controller = None
        self.robot = None
        self.cameras = None
        self.task = None
        self.contact_sensors = None
        self.stage = omni.usd.get_context().get_stage()

        # Task variables
        self.sim_subscriber = None
        self.contact_message = None
        self.object_target_orientation = None  
        self.ee_grasping_orientation = None
        self.ee_placement_orientation = None 
        self.grasp_poses = []
        self.current_grasp_pose = None

        # Counters and timers
        self.grasp_counter = 0
        self.placement_counter = 0
        self.pcd_counter = 0
        self.stabilize_counter = 30
        self.setup_start_time = None
        self.tf_wait_start_time = None
        
        # State tracking
        self.state = "INIT"  # States: INIT, SETUP, GRASP, PLACE, REPLAY, RESET
        
        # Object state tracking
        self.object_grasped = None
        self.object_collision = False
        
        # Data logging state
        self.start_logging = True
        self.data_recorded = False


    def start(self):
        # Initialize ROS2 node
        rclpy.init()
        self.sim_subscriber = SimSubscriber(visualization=True)

        # Create an executor
        executor = SingleThreadedExecutor()
        executor.add_node(self.sim_subscriber)

        # Simulation Environment Setup
        self.world: World = World(stage_units_in_meters=1.0)
        self.task = YcbTask(set_camera=True)
        self.world.add_task(self.task)
        self.world.reset()
        
        # Robot and planner setup
        self.robot: Franka = self.world.scene.get_object(self.task.get_params()["robot_name"]["value"])
        self.controller = self.robot.get_articulation_controller()
        self.planner = YcbPlanner(
            name="ycb_planner",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
        )

        prim_names = [
            "Franka/panda_link0",
            "Franka/panda_link1",
            "Franka/panda_link2",
            "Franka/panda_link3",
            "Franka/panda_link4",
            "Franka/panda_link5",
            "Franka/panda_link6",
            "Franka/panda_link7",
            "Franka/panda_link8",
            "Franka/panda_hand",
            "Franka/panda_leftfinger",
            "Franka/panda_rightfinger",
            "Ycb_object"
        ]

        self.contact_sensors = []
        for i, link_name in enumerate(prim_names):
            sensor: ContactSensor = self.world.scene.add(
                ContactSensor(
                    prim_path=f"/World/{link_name}/contact_sensor",
                    name=f"contact_sensor_{i}",
                    frequency=60,
                    min_threshold=0.0,
                    max_threshold=1e7,
                    radius=-1,
                )
            )
            # Use raw contact data if desired
            sensor.add_raw_contact_data_to_frame()
            self.contact_sensors.append(sensor)

        self.world.reset()

        # Contact report for links
        self.world.add_physics_callback("contact_sensor_callback", self.on_sensor_contact_report)

        # Initially hide the robot
        self.robot.prim.GetAttribute("visibility").Set("invisible")

        # Robot movement setup
        euler_angles = [random.uniform(0, 360) for _ in range(3)]
        self.ee_placement_orientation = R.from_euler('xyz', euler_angles, degrees=True).as_quat()

        # TF setup
        tf_graph_generation()
        
        # Update state to begin setup
        self.state = "SETUP"
        
        
    def reset(self):
        """Reset the simulation environment"""
        print("Resetting simulation environment, robot invisible")
        self.world.reset()
        self.planner.reset()
        self.task.object_init(False)
        
        # Reset state flags
        self.state = "SETUP"
        self.robot.prim.GetAttribute("visibility").Set("invisible")
        self.start_logging = True
        self.data_recorded = False
        self.object_grasped = None
        
    def soft_reset(self, state="REPLAY"):
        """Soft reset the simulation environment"""
        print("Soft resetting simulation environment")
        self.world.reset()
        self.planner.reset()
        variantSet = self.task._object_final.prim.GetVariantSets().GetVariantSet("mode")
        variantSet.SetVariantSelection("physics")
        self.task._object.set_world_pose(position=self.task._buffer[0], orientation=self.task._buffer[1])
        self.start_logging = True
        self.data_recorded = False
        self.object_grasped = None
        object_target_position, object_target_orientation = self.task.pose_init()
        self.task._object_final.set_world_pose(position=object_target_position, orientation=object_target_orientation)

        self.state = state
        print("You are going to the state: ", self.state)
        # self.task._object_final.set_world_pose(position=self.task._buffer[2], orientation=self.task._buffer[3])
        # self.stage.RemovePrim("/World/Ycb_final")

        # self.task.object_init(False)

        
        
    def setup_environment(self):
        """Handle environment setup phase"""
        # Initialize setup timer if needed
        if not self.setup_start_time:
            self.setup_start_time = time.time()
            return False
            
        # Wait a moment before proceeding with setup
        if time.time() - self.setup_start_time < 1:
            return False
            
        # Initialize cameras and object
        self.task.object_pose_finalization()
        self.cameras = self.task.set_camera(self.task.get_params()["object_current_position"]["value"])
        start_cameras(self.cameras)
        
        self.tf_wait_start_time = time.time()
        print("Cameras started, waiting for point clouds and TF data...")
        
        # Reset setup timer
        self.setup_start_time = None
        return True
        
    def wait_for_tf_buffer(self):
        """Wait for TF buffer to be populated"""
        if self.sim_subscriber.latest_tf is None:
            return False
        
        # Ensure we wait at least 2 seconds for TF buffer
        current_time = time.time()
        if current_time - self.tf_wait_start_time < 2.0:
            return False
            
        try:
            camera1_exists = self.sim_subscriber.check_transform_exists("world", "camera_side1")
            camera2_exists = self.sim_subscriber.check_transform_exists("world", "camera_side2")
            camera3_exists = self.sim_subscriber.check_transform_exists("world", "camera_top")
            
            if camera1_exists and camera2_exists and camera3_exists:
                return True
        except Exception:
            pass
            
        return False
        
    def wait_for_point_clouds(self, path):
        """Wait for and process point clouds from all cameras"""
        pcds = self.sim_subscriber.get_latest_pcds()
        if pcds["pcd1"] is None or pcds["pcd2"] is None or pcds["pcd3"] is None:
            return False
            
        print("All point clouds received!")

        # Create a msg of CloudIndexed.msg

        self.pcd_counter += 1
        pcd_path = path + f"Pcd_{self.pcd_counter}/pointcloud.pcd"
        tcp_msg = merge_and_save_pointclouds(pcds, self.sim_subscriber.buffer, pcd_path)
        
        if tcp_msg["cloud_sources"]["cloud"] is not None:
            # Update the camera view points
            for camera in self.cameras:
                camera_position, _ = camera.get_world_pose()
                tcp_msg["cloud_sources"]["view_points"].append(
                    {
                        "x": camera_position[0],
                        "y": camera_position[1],
                        "z": camera_position[2]
                    }
                )
            self.grasp_poses = obtain_grasps(tcp_msg, 12345)

            task_params = self.task.get_params()
            valid_grasp_poses = []
            distance_threshold = 0.1  # Threshold in meters - adjust as needed
            
            for grasp in self.grasp_poses:
                transformed_pose = transform_relative_pose(grasp, [0, 0, 0.062])
                pos_error = np.linalg.norm(transformed_pose[0] - task_params["object_current_position"]["value"])
                
                # Only keep grasp poses within the threshold
                if pos_error <= distance_threshold:
                    valid_grasp_poses.append(transformed_pose)
            
            self.grasp_poses = valid_grasp_poses
             # If no valid grasp poses found, return False to trigger restart
            if not self.grasp_poses:
                self.current_grasp_pose = -1
                print("No valid grasp poses found within threshold. Restarting simulation...")
                return False
            # Update grasp poses list with only valid poses
            
            self.current_grasp_pose = self.grasp_poses.pop(0)
            
           
            
            # Keep the original visualization for Isaac Sim
            self.grasp_counter = 1
        
        # Make the robot visible again after setup
        self.robot.prim.GetAttribute("visibility").Set("inherited")
        print("Setup finished, robot visible again")
        return True
        

    def on_sensor_contact_report(self, dt):
        """Physics-step callback: checks all sensors, sets self.contact_message accordingly."""
        contacts_list = []  # track if at least one sensor had contact

        for sensor in self.contact_sensors:
            frame_data = sensor.get_current_frame()
            if frame_data["in_contact"]:
                # We have contact! Extract the bodies, force, etc.
                for c in frame_data["contacts"]:
                    body0_short = c["body0"].split("/")[-1]
                    body1_short = c["body1"].split("/")[-1]
                    if "panda" in body0_short  and "panda" in body1_short:
                        continue
                    # Optionally, you can store additional details like force, time, etc.
                    contacts_list.append({
                        "body0": body0_short,
                        "body1": body1_short,
                        "time": frame_data.get("time"),
                        "physics_step": frame_data.get("physics_step"),
                        "current_event": self.planner.get_current_event()
                    })

        # If, after checking all sensors, none had contact, reset self.contact_message to None
        # print(f"This is the contact message: {contacts_list}")
        self.contact_message = contacts_list if contacts_list else None



    def check_grasp_success(self):
        """Check if the grasp was successful"""
        # print(f"You are in the grasp success check")
        POSITION_THRESHOLD = 0.1  # 1cm threshold for object movement
        
        # Check if gripper is fully closed (no object grasped)
        if np.floor(self.robot._gripper.get_joint_positions()[0] * 100) == 0:
            # print("Gripper is fully closed, didn't grasp object")
            self.object_grasped = "FAILED"
            return False
            
        self.object_grasped = "SUCCESS"
        print("Object grasped successfully")
        return True
    