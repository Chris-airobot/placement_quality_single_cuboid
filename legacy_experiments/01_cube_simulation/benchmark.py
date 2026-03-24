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
import re

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from utilies.camera_utility import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.sensor import ContactSensor
from functools import partial
from typing import List
import datetime
from learning_models.model_use import load_model, predict_single_sample
from learning_models.model_train import StabilityNet, eval_model
import torch
from rclpy.executors import SingleThreadedExecutor

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/benchmark/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")

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

    def pcd_callback(self, msg):
        self.latest_pcd = msg

    def tf_callback(self, msg):
        # This callback is triggered for every new TF message on /tf
        self.latest_tf = msg

class StartSimulation:
    def __init__(self, model_path):
        self.current_model_path = model_path
        match = re.search(r"run_seed_(\d+)", model_path)
        self.seed_number = int(match.group(1))
        folder_name = f"model_{self.seed_number}/"
        # Ensure the directory exists
        self.folder_path = os.path.join(DIR_PATH, folder_name)
        os.makedirs(self.folder_path, exist_ok=True) 

        self.world = None
        self.cube = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.task_params = None
        self.cube_target_orientation = None  # Gripper orientation when the cube is about to be placed
        self.ee_target_orientation = None    # End effector orientation when the cube is about to be placed
        self.camera = None
        self.contact = None
        self.cube_grasped = None
        self.contact_sensors = None
        self.model = None
        self.data_logger = None
        

        self.current_grasp_pose = None # Should be in a format of [np.array[x,y,z], np.array[x,y,z]]
        self.grasp_poses = []
        self.sorted_grasp_poses = []
        
        self.grasp_counter = 0
        self.grasping_failure = False
        self.cube_contacted = False
        self.enable_pcd = False

        self.sample = {
                "inputs":{
                    "grasp_position": None,
                    "grasp_orientation": None,
                    "cube_target_position": None,
                    "cube_target_orientation": None,
                    "cube_initial_orientation": None,
                    "cube_initial_position": None,
                },
                "outputs": {
                    "grasp_unsuccessful": None,
                    "position_difference": None,
                    "orientation_difference": None,
                    "shift_position": None,
                    "shift_orientation": None,
                    "contacts": None
                }
            }


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start(self):

        self.model: StabilityNet = load_model(self.current_model_path)

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


        self.contact_activation()
        tf_graph_generation()
        if self.enable_pcd:
            camera_graph_generation(self.task._camera)

        # Benchmark the evaluation of the model
        self.sample["inputs"]["cube_target_position"] = self.task_params["target_position"]["value"].tolist()
        self.sample["inputs"]["cube_initial_position"] = self.task_params["cube_position"]["value"].tolist()
        self.sample["inputs"]["cube_initial_orientation"] = self.task_params["cube_orientation"]["value"].tolist()
        self.sample["inputs"]["cube_target_orientation"] = self.task_params["cube_target_orientation"]["value"].tolist()
        self.cube_target_orientation = self.task_params["cube_target_orientation"]["value"].tolist()
        prims.create_prim(prim_path="/World/VisualCube", prim_type="Cube", 
                          position=self.sample["inputs"]["cube_target_position"],
                          orientation=self.cube_target_orientation,
                          scale=[0.05, 0.05, 0.05],)
        

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



    def _on_logging_event(self, val, tf_node: TFSubscriber):
        print(f"----------------- Model {self.seed_number} Grasping {self.grasp_counter} Start -----------------")
        
        if not self.world.get_data_logger().is_started():
            self.task_params = self.task.get_params()
            cube_name = self.task_params["cube_name"]["value"]
            cube_target_position = self.task_params["cube_target_position"]["value"]
            if tf_node.latest_tf is not None:
                tf_data = process_tf_message(tf_node.latest_tf)
            else:
                tf_data = None
            # A data logging function is called at every time step index if the data logger is started already.
            # We define the function here. The tasks and scene are passed to this function when called.

            def frame_logging_func(tasks, scene: Scene):
                cube_position, cube_orientation =  scene.get_object(cube_name).get_world_pose()
                ee_position, ee_orientation = get_current_end_effector_pose()

                surface, _ = get_upward_facing_marker("/World/Cube")

                return {
                    "joint_positions": self.robot.get_joint_positions().tolist(),# save data as lists since its a json file.
                    "applied_joint_positions": self.robot.get_applied_action().joint_positions.tolist(),
                    "ee_position": ee_position.tolist(),
                    "ee_orientation": ee_orientation.tolist(),
                    "cube_target_position": cube_target_position.tolist(), # Cube target position
                    "cube_position": cube_position.tolist(),
                    "cube_orientation": cube_orientation.tolist(),
                    "stage": self.controller.get_current_event(),
                    "surface": surface,
                    "ee_target_orientation":self.ee_target_orientation.tolist(),
                    "tf": tf_data,
                    "contact": self.contact,
                    "cube_in_ground": self.cube_contacted,
                    "cube_grasped": self.cube_grasped
                }

            self.data_logger.add_data_frame_logging_func(frame_logging_func) # adds the function to be called at each physics time step.
        if val:
            self.data_logger.start() # starts the data logging
        else:
            self.data_logger.pause()
        return



    def _on_save_data_event(self, log_path):
        print("----------------- Saving Start -----------------\n")
        self.data_logger.save(log_path=log_path) # Saves the collected data to the json file specified.

        print(f"----------------- Successfully saved it to {log_path} -----------------\n")
        self.data_logger.reset() # Resets the DataLogger internal state so that another set of data can be collected and saved separately.
        return


    def reset(self):
        self.world.reset()
        self.task._camera.initialize()
        self.controller.reset()
        self.grasping_failure = False

        cube_initial_position, target_position, cube_initial_orientation = task_randomization()

        # Create the cube position with z fixed at 0
        self.task.set_params(
            cube_position=cube_initial_position,
            cube_orientation=cube_initial_orientation,
            target_position=target_position
        )
        
        

    ### Finish this function, think about do we need to execute it first, or calculate the score first
    def save_grasps(self):
        file_path = os.path.join(self.folder_path, "grasp_poses.json")  # Adjust filename as needed
        data = {}
        
        
        for i in range(len(self.grasp_poses)):
            grasp = self.grasp_poses[i]
            sample = self.sample.copy()
            sample["inputs"]["grasp_position"] = grasp[0]
            sample["inputs"]["grasp_orientation"] = grasp[1]
            pred_cls, pred_reg = predict_single_sample(self.model, sample)
            data[f"grasp_{i}"] = {
                "inputs": sample["inputs"],
                "scores": [pred_cls, pred_reg],
            }
            
        sorted_grasps = sorted(
            data.values(), 
            key=lambda x: x["scores"][1], 
            reverse=True  # Higher scores first
        )

        with open(file_path, 'w') as file:
            json.dump(sorted_grasps, file, indent=4)
            
        print(f"Saved {len(sorted_grasps)} grasps to {file_path}")

        return sorted_grasps
        
        

def log_grasping(start_logging, env: StartSimulation, tf_node: TFSubscriber):
    # Logging sections
    if start_logging:
        env._on_logging_event(True, tf_node)
        start_logging = False

    return start_logging


def record_grasping(recorded, env: StartSimulation):
    # Recording section
    if not recorded:
        file_path = env.folder_path + f"/trajectories/Grasping_{env.grasp_counter}_{env.grasping_failure}.json"

        # Ensure the parent directories exist
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        # Check if the file exists; if not, create it
        if not os.path.exists(file_path):
            # Open the file in write mode and create an empty JSON structure
            with open(file_path, 'w') as file:
                json.dump({}, file)
        env._on_save_data_event(file_path)
        recorded = True
    return recorded



def main(model_path):
    rclpy.init()
    tf_node = TFSubscriber()
    
    # Create an executor (SingleThreadedExecutor is the simplest choice)
    executor = SingleThreadedExecutor()
    executor.add_node(tf_node)

    env = StartSimulation(model_path)
    env.start()

    reset_needed = True            # Used when grasp is done 
    start_logging = True           # Used for start logging
    recorded = False               # Used to check if the data has been recorded 
    pipeline = True                # Used to do the pipeline testing

    while simulation_app.is_running():
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            env.reset()
            env.grasp_counter += 1
            if not pipeline:
                continue
            else:
                break
        # 2) Spin ROS for a short time so callbacks are processed
        executor.spin_once(timeout_sec=0.01)

        # Technically only pcd ready should be sufficient because it seems pcd takes longer to be prepared
        if tf_node.latest_tf is None or tf_node.latest_pcd is None:
            continue

        if env.world.is_playing():
            # Current grasp pose is done, forward to next grasp pose 
            if reset_needed:
                env.reset()
                reset_needed = False
                start_logging = True
                recorded = False

                if env.grasp_poses and not pipeline: # There are saved existing poses and I want to test all the poses rather than one only
                    env.current_grasp_pose = env.grasp_poses.pop(0)
                    env.grasp_counter += 1
                else:
                    transformed_pcd = transform_pointcloud_to_frame(tf_node.latest_pcd, tf_node.buffer, 'panda_link0')
                    saved_path = save_pointcloud(transformed_pcd, env.folder_path)
                    env.grasp_poses = obtain_grasps(saved_path, 12346)
                    data = env.save_grasps()
                    env.sorted_grasp_poses = [ [grasp["inputs"]["grasp_position"], 
                                                grasp["inputs"]["grasp_orientation"]] for grasp in data]
                     
                    env.current_grasp_pose = env.sorted_grasp_poses.pop(0)
                    env.grasp_counter = 1
                
                env.ee_target_orientation = cube_orientation_to_ee_orientation(env.sample["inputs"]["cube_initial_orientation"],
                                                                               env.current_grasp_pose[1],
                                                                               env.sample["inputs"]["cube_target_orientation"],
                                                                               )

            else:
                # Recording Session
                start_logging = log_grasping(start_logging, env, tf_node)
                try:
                    observations = env.world.get_observations()
                except:
                    print("Something wrong with hitting the floor in observation")
                    env.reset()
                    env.grasp_counter += 1
                    if not pipeline:
                        continue
                    else:
                        break
                
                actions = env.controller.forward(
                    picking_position=observations[env.task_params["cube_name"]["value"]]["cube_current_position"],
                    placing_position=observations[env.task_params["cube_name"]["value"]]["cube_target_position"],
                    current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=np.array([0, 0, 0]),
                    placement_orientation=env.ee_target_orientation,  
                    grasping_orientation=env.current_grasp_pose[1],    
                )
                
                # Gripper fully closed, did not grasp the object
                if env.controller.get_current_event() == 4:
                    if np.floor(env.robot._gripper.get_joint_positions()[0] * 100) == 0 : 
                        env.grasping_failure = True
                        recorded = record_grasping(recorded, env)
                        reset_needed = True

                env.articulation_controller.apply_action(actions)

            if env.controller.is_done():
                print("----------------- done picking and placing ----------------- \n\n")
                recorded = record_grasping(recorded, env)
                env.reset()

            
    simulation_app.close()

if __name__ == "__main__":
    model_path = "/home/chris/Chris/placement_ws/src/data/models/run_seed_93345942/stability_net.pth"
    main(model_path)
    

    



