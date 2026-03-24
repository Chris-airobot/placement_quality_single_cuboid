import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Use the correct relative import path
from utils.helper import SimSubscriber, process_tf_message, extract_replay_data

from simulation.simulator import YcbCollection
import numpy as np
import glob
import asyncio
import json


class YcbLogger:
    def __init__(self, sim_env: YcbCollection, dir_path: str):
        self.env = sim_env
        self.world = self.env.world
        self.task = self.env.task
        self.data_logger = self.world.get_data_logger()
        self.planner = self.env.planner
        self.controller = self.env.controller
        self.contact_sensors = self.env.contact_sensors
        self.dir_path = dir_path

    def log_grasping(self):
        # Logging sections
        if self.env.start_logging:
            self._on_logging_event(True, self.env.sim_subscriber)
            self.env.start_logging = False

        

    def _on_logging_event(self, val, ros2_node: SimSubscriber):
        from omni.isaac.franka import Franka
        from omni.isaac.core.scenes import Scene
        
        print(f"----------------- Grasping {self.env.grasp_counter} Placement {self.env.placement_counter} Start -----------------")

        if not self.world.get_data_logger().is_started():
            self.task_params = self.task.get_params()
            robot_name = self.task_params["robot_name"]["value"]
            object_name = self.task_params["object_name"]["value"]
            target_position: np.ndarray = self.task_params["object_target_position"]["value"]
            target_orientation: np.ndarray = self.task_params["object_target_orientation"]["value"]
            if ros2_node.all_transforms is not None:
                tf_data = process_tf_message(ros2_node.all_transforms)
            else:
                tf_data = None
            # A data logging function is called at every time step index if the data logger is started already.
            # We define the function here. The tasks and scene are passed to this function when called.
            def frame_logging_func(tasks, scene: Scene):
                robot: Franka = scene.get_object(robot_name)
                object_position, object_orientation =  scene.get_object(object_name).get_world_pose()
                ee_position, ee_orientation =  robot.end_effector.get_world_pose()
                return {
                    "joint_positions": robot.get_joint_positions().tolist(),# save data as lists since its a json file.
                    "applied_joint_positions": robot.get_applied_action().joint_positions.tolist(),
                    "ee_position": ee_position.tolist(),
                    "ee_orientation": ee_orientation.tolist(),
                    "target_position": target_position.tolist(), # object target position
                    "target_orientation": target_orientation.tolist(), # object target position
                    "object_position": object_position.tolist(),
                    "object_orientation": object_orientation.tolist(),
                    "stage": self.planner.get_current_event(),
                    # "surface": surface,
                    "ee_target_orientation":self.env.ee_placement_orientation.tolist(),
                    "tf": tf_data,
                    "contact_info": self.env.contact_message,
                    "object_in_ground": self.env.object_collision,
                    "object_grasped": self.env.object_grasped
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
        from omni.isaac.core.utils.types import ArticulationAction
        print(f"Replaying step {self.world.current_time_step_index} of {self.data_logger.get_num_of_data_frames()}")
        if self.world.current_time_step_index < self.data_logger.get_num_of_data_frames():
            object_name = self.env.task.get_params()["object_name"]["value"]
            data_frame = self.data_logger.get_data_frame(data_frame_index=self.world.current_time_step_index)
            self.controller.apply_action(
                ArticulationAction(joint_positions=data_frame.data["applied_joint_positions"])
            )
            # Sets the world position of the goal object to the same recoded position
            self.world.scene.get_object(object_name).set_world_pose(
                position=np.array(data_frame.data["object_position"]),
                orientation=np.array(data_frame.data["object_orientation"])
            )

        elif self.world.current_time_step_index == self.data_logger.get_num_of_data_frames():
            print("----------------- Replay Finished, now moving to Placement Phase -----------------\n")
            self.replay_finished = True
            self.planner._event = 4
            self.world.remove_physics_callback("replay_scene")
        return
    
    def replay_grasping(self):

        print(f"----------------- Replaying Pcd {self.env.pcd_counter} Grasping {self.env.grasp_counter} ----------------- \n")

        file_path = self.dir_path + f"Pcd_{self.env.pcd_counter}/Grasping_{self.env.grasp_counter}/Grasping.json"

        # If the replay data does not exist, create one
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}, preparing to extract data from Placement files")
            file_pattern = os.path.join(self.dir_path, f"Pcd_{self.env.pcd_counter}/Grasping_{self.env.grasp_counter}/Placement_*.json")
            file_list = glob.glob(file_pattern)

            extract_replay_data(file_list[0])
        print(f"File found, starting to replay")
        asyncio.ensure_future(self._on_replay_scene_event_async(file_path))
        return True
    
    def record_grasping(self, message: str="Failed"):
            # Recording section
        if not self.env.data_recorded:
            file_path = self.dir_path + f"Pcd_{self.env.pcd_counter}/Grasping_{self.env.grasp_counter}/Placement_{self.env.placement_counter}_{message}.json"

            # Ensure the parent directories exist
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)

            # Check if the file exists; if not, create it
            if not os.path.exists(file_path):
                # Open the file in write mode and create an empty JSON structure
                with open(file_path, 'w') as file:
                    json.dump({}, file)
            self._on_save_data_event(file_path)
            self.env.data_recorded = True
        # return recorded