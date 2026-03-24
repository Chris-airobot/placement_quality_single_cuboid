# import carb
from isaacsim import SimulationApp
CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

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

from omni.isaac.core import World
from omni.isaac.core.utils import extensions
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
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

DIR_PATH = "/home/chris/Chris/placement_ws/src/test_data/"


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
    def __init__(self):
        self.world = None
        self.cube = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.task_params = None
        self.placement_orientation = None  # Gripper orientation when the cube is about to be placed
        self.camera = None
        self.contact = None
        self.cube_grasped = None
        self.contact_sensors = None
        self.current_grasp_pose = None # Should be in a format of [np.array[x,y,z], np.array[x,y,z]]
        self.grasp_poses = []
        
        self.data_logger = None

        self.pcd_counter = 0
        self.grasp_counter = 0
        self.placement_counter = 0
        

        self.replay_finished = True
        self.grasping_failure = False
        self.cube_contacted = False



    def start(self):
        # Set up the world
        self.world: World = World(stage_units_in_meters=1.0)

        ranges = [(-0.3, -0.1), (0.1, 0.3)]
        range_choice = ranges[np.random.choice(len(ranges))]
        
        # Generate x and y as random values between -π and π
        x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
        p, q = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])


        self.task = PickPlaceCamera()
        

        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default

        self.world.add_task(self.task)

        self.world.reset()

        panda_prim_names = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
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

        self.task_params = self.task.get_params()
        self.robot: Franka =  self.world.scene.get_object(self.task_params["robot_name"]["value"])
        
        # Set up the robot
        self.controller = DataCollectionController(
            name = "data_collection_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )
        self.articulation_controller = self.robot.get_articulation_controller() 

        self.task.set_params(
            cube_position=np.array([x, y, 0]),
            target_position=np.array([p, q, 0.075])
        )

        self.placement_orientation = euler_angles_to_quat(np.random.uniform(low=-np.pi, high=np.pi, size=3))

        tf_graph_generation()
        start_camera(self.task._camera, True)
        

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
                        # self.contact = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"
        # print(f"cube is in the ground: {self.cube_contacted}, time is {self.world.current_time_step_index}, stage is {self.controller.get_current_event()}")

        # If, after checking all sensors, none had contact, reset self.contact to None
        if not any_contact:
            self.contact = None





    def _on_logging_event(self, val, tf_node: TFSubscriber):
        print(f"----------------- Pcd {self.pcd_counter} Grasping {self.grasp_counter} Placement {self.placement_counter} Start -----------------")
        print(f'Cube position is: {self.task.get_params()["cube_position"]["value"]}\n')

        if not self.world.get_data_logger().is_started():
            self.task_params = self.task.get_params()
            robot_name = self.task_params["robot_name"]["value"]
            cube_name = self.task_params["cube_name"]["value"]
            target_position = self.task_params["target_position"]["value"]
            camera_name = self.task_params["camera_name"]["value"]
            if tf_node.latest_tf is not None:
                tf_data = process_tf_message(tf_node.latest_tf)
            else:
                tf_data = None
            # A data logging function is called at every time step index if the data logger is started already.
            # We define the function here. The tasks and scene are passed to this function when called.

            def frame_logging_func(tasks, scene: Scene):
                cube_position, cube_orientation =  scene.get_object(cube_name).get_world_pose()
                ee_position, ee_orientation =  scene.get_object(robot_name).end_effector.get_world_pose()
                # surface = surface_detection(quat_to_euler_angles(cube_orientation))
                camera_position, camera_orientation =  scene.get_object(camera_name).get_world_pose()
                surface, _ = get_upward_facing_marker("/World/Cube")

                return {
                    "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),# save data as lists since its a json file.
                    "applied_joint_positions": scene.get_object(robot_name).get_applied_action().joint_positions.tolist(),
                    "ee_position": ee_position.tolist(),
                    "ee_orientation": ee_orientation.tolist(),
                    "target_position": target_position.tolist(), # Cube target position
                    "cube_position": cube_position.tolist(),
                    "cube_orientation": cube_orientation.tolist(),
                    "stage": self.controller.get_current_event(),
                    "surface": surface,
                    "ee_target_orientation":self.placement_orientation.tolist(),
                    "camera_position": camera_position.tolist(),
                    "camera_orientation": camera_orientation.tolist(),
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


    # This is for replying the whole scene
    async def _on_replay_scene_event_async(self, data_file):
            self.data_logger.load(log_path=data_file)

            await self.world.play_async()
            self.world.add_physics_callback("replay_scene", self._on_replay_scene_step)
            return 


    def _on_replay_scene_step(self, step_size):
        if self.world.current_time_step_index < self.data_logger.get_num_of_data_frames():
            cube_name = self.task_params["cube_name"]["value"]
            camera_name = self.task_params["camera_name"]["value"]
            data_frame = self.data_logger.get_data_frame(data_frame_index=self.world.current_time_step_index)
            self.articulation_controller.apply_action(
                ArticulationAction(joint_positions=data_frame.data["applied_joint_positions"])
            )
            # Sets the world position of the goal cube to the same recoded position
            self.world.scene.get_object(cube_name).set_world_pose(
                position=np.array(data_frame.data["cube_position"]),
                orientation=np.array(data_frame.data["cube_orientation"])
            )
            # Sets the world position of the goal camera to the same recoded position
            self.world.scene.get_object(camera_name).set_world_pose(
                position=np.array(data_frame.data["camera_position"]),
                orientation=np.array(data_frame.data["camera_orientation"])
            )


        elif self.world.current_time_step_index == self.data_logger.get_num_of_data_frames():
            print("----------------- Replay Finished, now moving to Placement Phase -----------------\n")
            self.replay_finished = True
            self.controller._event = 4
            self.world.remove_physics_callback("replay_scene")
        return

    def reset(self):
        self.world.reset()
        self.task._camera.initialize()
        self.controller.reset()
        self.grasping_failure = False

        ranges = [(-0.3, -0.1), (0.1, 0.3)]
        range_choice = ranges[np.random.choice(len(ranges))]
        
        # Generate x and y as random values between -π and π
        x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
        p, q = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])

        # Create the cube position with z fixed at 0
        self.task.set_params(
            cube_position=np.array([x, y, 0]),
            target_position=np.array([p, q, 0.075])
        )
        print(f"cube_position is :{np.array([x, y, 0])}")
        self.placement_orientation = euler_angles_to_quat(np.random.uniform(low=-np.pi, high=np.pi, size=3))
        

def log_grasping(start_logging, env: StartSimulation, tf_node: TFSubscriber):
    # Logging sections
    if start_logging:
        env._on_logging_event(True, tf_node)
        start_logging = False

    return start_logging


def record_grasping(recorded, env: StartSimulation):
    # Recording section
    if not recorded:
        file_path = DIR_PATH + f"Pcd_{env.pcd_counter}/Grasping_{env.grasp_counter}/Placement_{env.placement_counter}_{env.grasping_failure}.json"

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


def replay_grasping(env: StartSimulation):
    print(f"----------------- Replaying Pcd {env.pcd_counter} Grasping {env.grasp_counter} ----------------- \n")

    file_path = DIR_PATH + f"Pcd_{env.pcd_counter}/Grasping_{env.grasp_counter}/Grasping.json"

    # If the replay data does not exist, create one
    if not os.path.exists(file_path):
        file_pattern = os.path.join(DIR_PATH, f"Pcd_{env.pcd_counter}/Grasping_{env.grasp_counter}/Placement_*.json")
        file_list = glob.glob(file_pattern)


        extract_grasping(file_list[0])
    asyncio.ensure_future(env._on_replay_scene_event_async(file_path))
    return True


def main():
    rclpy.init()
    tf_node = TFSubscriber()
    
    # Create an executor (SingleThreadedExecutor is the simplest choice)
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(tf_node)

    env = StartSimulation()
    env.start()

    # One grasp corresponding to many placements
    reset_needed = True            # Used when grasp is done 
    start_logging = True           # Used for start logging
    recorded = False               # Used to check if the data has been recorded
    replay = False                 # Used for replay data     
    grasp_collected = False     # Use when the grasping is done


    while simulation_app.is_running():
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            env.reset()
            reset_needed = False
            start_logging = True
            recorded = False
            replay = False
        # 2) Spin ROS for a short time so callbacks are processed
        executor.spin_once(timeout_sec=0.01)

        # tf_started = not tf_node.latest_tf==None
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
                replay = False
                grasp_collected = False
                if env.grasp_poses: # There are saved existing poses 
                    env.current_grasp_pose = env.grasp_poses.pop(0)
                    env.grasp_counter += 1
                else:
                    transformed_pcd = transform_pointcloud_to_frame(tf_node.latest_pcd, tf_node.buffer, 'panda_link0')
                    env.pcd_counter += 1
                    saved_path = save_pointcloud(transformed_pcd, env.pcd_counter)
                    env.grasp_poses = obtain_grasps(saved_path)
                    env.current_grasp_pose = env.grasp_poses.pop(0)
                    env.grasp_counter = 1

                env.placement_counter = 0
            
            # Replaying Session
            if replay:
                # This function should only be played once
                env.replay_finished = False
                replay_grasping(env)
                env.placement_counter += 1
                replay = False
                start_logging = True
                recorded = False

            elif env.replay_finished:
            # elif env.replay_finished and tf_started:
                # Recording Session
                start_logging = log_grasping(start_logging, env, tf_node)
                try:
                    observations = env.world.get_observations()
                except:
                    print("Something wrong with hitting the floor in observation")
                    if env.placement_counter <= 1:
                        env.reset()
                        reset_needed = False
                        start_logging = True
                        recorded = False
                        replay = False
                        continue
                    elif env.placement_counter > 1 and env.placement_counter < 200:
                        env.reset()
                        replay = True
                        continue
                
                placement_orientation = env.placement_orientation if grasp_collected else env.current_grasp_pose[1]
                actions = env.controller.forward(
                    picking_position=observations[env.task_params["cube_name"]["value"]]["position"],
                    # picking_position=env.current_grasp_pose[0], # This is wrong, grasp position is off
                    placing_position=observations[env.task_params["cube_name"]["value"]]["target_position"],
                    current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=np.array([0, 0.005, 0]),
                    placement_orientation=env.placement_orientation,  
                    grasping_orientation=env.current_grasp_pose[1],    
                )
                
                # Gripper fully closed, did not grasp the object
                if env.controller.get_current_event() == 4 and env.placement_counter <= 1:
                    if np.floor(env.robot._gripper.get_joint_positions()[0] * 100) == 0 : 
                        env.grasping_failure = True
                        recorded = record_grasping(recorded, env)
                        reset_needed = True

                env.articulation_controller.apply_action(actions)

            if env.controller.is_done():
                print("----------------- done picking and placing ----------------- \n\n")
                recorded = record_grasping(recorded, env)

                if env.placement_counter >= 200:
                    reset_needed = True
                else: 
                    env.reset()
                    replay = True
            
    simulation_app.close()

if __name__ == "__main__":
    main()


    



