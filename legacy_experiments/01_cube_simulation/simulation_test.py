# import carb
from isaacsim import SimulationApp
CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

# import carb
import numpy as np
import asyncio
import json
import os
import glob
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
import tf2_ros
import time
import re

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka, KinematicsSolver
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from utilies.camera_utility import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.sensor import ContactSensor
from functools import partial
from typing import List

from rclpy.executors import SingleThreadedExecutor

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()


class TFSubscriber(Node):
    def __init__(self):
        super().__init__("tf_subscriber")
        self.latest_tf = None  # Store the latest TFMessage here
        self.latest_pcd = None # Store the latest Pointcloud here

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        # Create the subscription
        self.tf_subscription = self.create_subscription(
            TFMessage,           # Message type
            "/tf",               # Topic name
            self.tf_callback,    # Callback function
            10                   # QoS
        )

        self.pcd_subscription = self.create_subscription(
            PointCloud2,         # Message type
            "/depth_pcl",        # Topic name
            self.pcd_callback,   # Callback function
            10                   # QoS
        )

        # This will hold the accumulated (merged) point cloud.
        self.accumulated_pcd = []

        # Flag to control whether to process new pointcloud messages.
        self.pcd_callback_enabled = True



    def pcd_callback(self, msg):
        if not self.pcd_callback_enabled:
            return

        self.latest_pcd = msg
        
    def tf_callback(self, msg):
        # This callback is triggered for every new TF message on /tf
        self.latest_tf = msg



class TestRobotMovement:
    def __init__(self):
        self.world = None
        self.cube = None
        self.controller = None
        self.kinematics_solver = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.task_params = None
        self.cube_target_orientation = None  # Gripper orientation when the cube is about to be placed
        self.ee_target_orientation = None    # End effector orientation when the cube is about to be placed
        # self.setup_start_time = None
        self.contact = None
        self.cube_grasped = None
        self.contact_sensors = None
        self.pcd_poses = None
        self.pcd_orientations = None
        self.model = None
        self.grasping_orientation = None
        self.placement_orientation = None  
        self.camera = None
        self.data_logger = None

        self.enable_pcd = True
        self.cube_contacted = False
        self.setup_finished = False

    def start(self):
        # Set up the world
        self.world: World = World(stage_units_in_meters=1.0)
        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default

        # Set up the task
        self.task = PickPlaceCamera()
        self.world.add_task(self.task)
        self.world.reset()
        self.task_params = self.task.get_params()

        # Set up the robot components
        self.robot: Franka =  self.world.scene.get_object(self.task_params["robot_name"]["value"])
        self.controller = DataCollectionController(
            name = "data_collection_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )
        self.articulation_controller = self.robot.get_articulation_controller() 
        self.kinematics_solver = KinematicsSolver(self.robot, end_effector_frame_name="panda_hand")

        # self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        self.grasping_orientation = np.array([0, np.pi, 0])
        self.placement_orientation = np.array([0, np.pi, 0])
        
        self.contact_activation()
        # tf_graph_generation()
        merge_graphs(self.task._camera)
        start_camera(self.task._camera, self.enable_pcd)

            

        self.cube_target_orientation = self.task_params["cube_target_orientation"]["value"].tolist()
        # prims.create_prim(prim_path="/World/VisualCube", prim_type="Cube", 
        #                 position=self.task_params["cube_target_position"]["value"].tolist(),
        #                 orientation=self.cube_target_orientation,
        #                 scale=[0.05, 0.05, 0.05],)

    
    def contact_activation(self):
        panda_prim_names = [
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
            "Cube"
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
            # Use raw contact data if desired
            sensor.add_raw_contact_data_to_frame()
            self.contact_sensors.append(sensor)
        self.world.reset()
        # Contact report for links
        self.world.add_physics_callback("contact_sensor_callback", partial(self.on_sensor_contact_report, sensors=self.contact_sensors))



    def on_sensor_contact_report(self, dt, sensors: List[ContactSensor]):
        """Physics-step callback: checks all sensors, sets self.contact accordingly."""
        any_contact = False  # track if at least one sensor had contact
        self.cube_contacted = False
        self.cube_grasped = None

        for sensor in sensors:
            frame_data = sensor.get_current_frame()
            if frame_data["in_contact"]:
                # We have contact! Extract the bodies, force, etc.
                for c in frame_data["contacts"]:
                    body0 = c["body0"]
                    body1 = c["body1"]
                    if "panda" in body0 + body1 and "Cube" in body0 + body1:
                        # print("Cube in the ground")
                        self.cube_grasped = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"

                    if "GroundPlane" in body0 + body1 and "Cube" in body0 + body1:
                        # print("Cube in the ground")
                        self.cube_contacted = True
                    elif ("GroundPlane" in body0) or ("GroundPlane" in body1):
                        print("Robot hits the ground, and it will be recorded")
                        any_contact = True
                        self.contact = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"
                        
        # If, after checking all sensors, none had contact, reset self.contact to None
        if not any_contact:
            self.contact = None

    def reset(self):
        self.world.reset()
        self.controller.reset()

        self.task.cube_init(False)
        self.setup_finished = False

        # cube_initial_position, target_position, cube_initial_orientation = task_randomization()

        # # Create the cube position with z fixed at 0
        # self.task.set_params(
        #     cube_position=cube_initial_position,
        #     cube_orientation=cube_initial_orientation,
        #     target_position=target_position
        # )
        # # print(f"cube_position is :{np.array([x, y, 0])}")
        # # self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)

def main():
    rclpy.init()
    tf_node = TFSubscriber()
    
    # Create an executor (SingleThreadedExecutor is the simplest choice)
    executor = SingleThreadedExecutor()
    executor.add_node(tf_node)


    env = TestRobotMovement()
    env.start()

    # One grasp corresponding to many placements
    reset_needed = False        # Used when grasp is done 
    pcd_finished = False

    current_target = None
    current_orientation = None
    # Set a time limit in seconds for each movement
    movement_time_limit = 2  # adjust this value as needed

    while simulation_app.is_running():
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            env.reset()
            continue
    
        # 2) Spin ROS for a short time so callbacks are processed
        executor.spin_once(timeout_sec=0.01)

        # Technically only pcd ready should be sufficient because it seems pcd takes longer to be prepared
        if tf_node.latest_tf is None or tf_node.latest_pcd is None:
            continue


        if env.world.is_stopped() and not reset_needed:
            reset_needed = True
        if env.world.is_playing():
            # The grasp is done, no more placement 
            if reset_needed:
                env.reset()
                reset_needed = False

            if not env.setup_finished:
                if not hasattr(env, "setup_start_time"):
                    env.setup_start_time = time.time()
                if time.time() - env.setup_start_time < 1:
                    # Skip sending robot commands, letting other parts of the simulation continue.
                    continue
                else:
                    env.task.cube_pose_finalization()
                    observations = env.world.get_observations()
                    print(f"So, the cube positions is: {observations[env.task_params['cube_name']['value']]['cube_current_position']}")
                    env.pcd_poses, env.pcd_orientations = pcd_movements(observations[env.task_params["cube_name"]["value"]]["cube_current_position"],
                                                                        observations[env.task_params["robot_name"]["value"]]["end_effector_position"],
                                                                        observations[env.task_params["robot_name"]["value"]]["end_effector_orientation"],
                                                                        )
                    print(f"And your movements are: {env.pcd_poses}")
                
                    env.setup_finished = True
                    if hasattr(env, "setup_start_time"):
                        del env.setup_start_time
                    print(f"set up finished")
            try:
                observations = env.world.get_observations()
            except:
                print("Something wrong with getting the observation")
                env.reset()
                continue
            
            # movement for robot to obtain the pcd of the scene
            if not pcd_finished:
                # If there's no current target (or the previous one was reached), pop the next one.
                if current_target is None:
                    if len(env.pcd_poses) > 0:
                        current_target = env.pcd_poses.pop(0)
                        current_orientation = env.pcd_orientations.pop(0)
                        movement_start_time = time.time()  # start timer for this movement
                    
                if env.controller.is_stage_task_done(current_target, 0.01) or \
                    (movement_start_time and (time.time() - movement_start_time > movement_time_limit)):
                     # Target reached or the movement has exceeded the time limit, reset the timer and get the next target
                    movement_start_time = None
                    tf_node.accumulated_pcd.append(tf_node.latest_pcd)
                    if len(env.pcd_poses) > 0:
                        current_target = env.pcd_poses.pop(0)
                        current_orientation = env.pcd_orientations.pop(0)
                        movement_start_time = time.time()  # start timer for new target
                    else:
                        process_collected_pointclouds(tf_node.accumulated_pcd)
                        pcd_finished = True
                        tf_node.pcd_callback_enabled = False
                

                print(f"Your current target is: {current_target}")
                print(f"Your current orientation is: {current_orientation}")
                actions = env.controller._cspace_controller.forward(
                    target_end_effector_position=current_target,
                    target_end_effector_orientation=current_orientation,
                )
                
                
            else:
                # Use random orientation only after the grasp part trajectory has been collected  
                actions = env.controller.forward(
                    picking_position=observations[env.task_params["cube_name"]["value"]]["cube_current_position"],
                    placing_position=observations[env.task_params["cube_name"]["value"]]["cube_target_position"],
                    current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=env.placement_orientation,  
                    grasping_orientation=env.grasping_orientation,    
                )


            env.articulation_controller.apply_action(actions)

        if env.controller.is_done():
            print("----------------- done picking and placing ----------------- \n\n")
            reset_needed = True
            env.reset()


            
    simulation_app.close()

if __name__ == "__main__":
    main()
    

    


