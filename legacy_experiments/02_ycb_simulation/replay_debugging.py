from isaacsim import SimulationApp

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
}

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
import time
import datetime 
import re
from functools import partial
from typing import List
from rclpy.executors import SingleThreadedExecutor

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.sensor import ContactSensor

from simulation.logger import YcbLogger
from simulation.simulator import YcbCollection
from utils.vision import *
from utils.helper import *


# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

# base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
# time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")
DIR_PATH = "/home/chris/Chris/placement_ws/src/data/YCB_data/run_20250325_213554/"



def main():
    env = YcbCollection()
    env.start()
    logger = YcbLogger(env, DIR_PATH)

    setup_phase = 0  # Track which setup step we're on
    env.state = "REPLAY"
    env.pcd_counter = 2
    env.grasp_counter = 1
    env.placement_counter = 0
    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            if env.placement_counter <= 1:
                env.reset()
                continue
            elif env.placement_counter > 1 and env.placement_counter < 200:
                env.reset()
                env.state = "REPLAY"
                continue
        
        # Process ROS callbacks
        rclpy.spin_once(env.sim_subscriber, timeout_sec=0)
        
        # Wait for initial TF data
        if env.sim_subscriber.latest_tf is None and env.state != "INIT":
            continue
            
        # Check if simulation is stopped unexpectedly
        if env.world.is_stopped():
            print("Simulation stopped unexpectedly, resetting...")
            env.reset()
            
            # Reset counters for new grasp attempt
            if env.state != "REPLAY" or env.placement_counter >= 200:
                env.grasp_counter += 1
                env.placement_counter = 0
            continue
            
        # Skip if simulation is paused
        if not env.world.is_playing():
            continue
            
        # State machine for simulation flow
        if env.state == "SETUP":
            # Handle the different phases of setup sequentially
            if setup_phase == 0:
                # Environment and camera setup
                if env.setup_environment():
                    setup_phase = 1
                    
            elif setup_phase == 1:
                # Wait for TF buffer
                if env.wait_for_tf_buffer():
                    setup_phase = 2
                    
            elif setup_phase == 2:
                # Wait for point clouds
                if env.wait_for_point_clouds(DIR_PATH):
                    setup_phase = 0  # Reset for next time
                    
                    # Move to grasp state if we're just starting
                    if env.placement_counter == 0:
                        env.state = "GRASP"
                    else:
                        env.state = "REPLAY"
                        
        elif env.state == "GRASP":
            # Start logging for the grasp attempt
            env.start_logging = logger.log_grasping()
                
            # Perform grasping
            try:
                observations = env.world.get_observations()
                task_params = env.task.get_params()

                draw_frame(env.current_grasp_pose[0], env.current_grasp_pose[1])
                # For first attempts, use same orientation for grasp and place
                orientation = env.current_grasp_pose[1]
                
                # picking_position = observations[task_params["object_name"]["value"]]["object_current_position"]
                picking_position = env.current_grasp_pose[0]
                # Generate actions for the robot
                actions = env.planner.forward(
                    picking_position=picking_position,
                    placing_position=observations[task_params["object_name"]["value"]]["object_target_position"],
                    current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=orientation,  
                    grasping_orientation=orientation,
                )

                # Check current planner event phase
                current_event = env.planner.get_current_event()
                
                # Check if grasp failed
                if current_event == 4 and not env.check_grasp_success():
                    print("Grasp failed")
                    # Still use soft reset, but actually with different poses
                    logger.record_grasping()
                    if env.grasp_poses:
                        print("Poses left: ", len(env.grasp_poses))
                        env.current_grasp_pose = env.grasp_poses.pop(0)
                        env.current_grasp_pose = transform_relative_pose(env.current_grasp_pose, [0, 0, 0.05])
                        env.soft_reset()
                    else:
                        print("No more poses left, resetting environment")
                        env.reset()
                    env.grasp_counter += 1  # Increment grasp counter for new attempt
                    continue

                # Apply the actions to the robot
                env.controller.apply_action(actions)
                
                
                if current_event > 4 and current_event < 9:
                    # Print progress through placement phases
                    print(f"Placement in grasping phase in progress - event {current_event}/9")
                    
                # Check if we've completed all phases of the planner
                if env.planner.is_done():
                    print("----------------- First grasp and placement complete -----------------")
                    # Record the successful trajectory
                    logger.record_grasping()
                    env.data_recorded = True
                    
                    # Increment the placement counter and move to replay state
                    env.placement_counter += 1
                    
                    # Reset for next placement
                    env.reset()
                    env.state = "REPLAY"
            
            except Exception as e:
                print(f"Error during grasping: {e}")
                env.reset()
                continue
                
        elif env.state == "REPLAY":
            # Start the replay of the grasping trajectory
            print(f"Replaying grasp {env.grasp_counter}, placement {env.placement_counter}")
            env.robot.prim.GetAttribute("visibility").Set("inherited")
            logger.replay_grasping()
            
            # Move to place state - the replay will continue in the background
            # and the planner will pick up after the replay is done
            env.state = "PLACE"
            env.start_logging = True
            
        elif env.state == "PLACE":
            # Event 4 means the object has been grasped and lifted
            if env.planner.get_current_event() < 4:
                continue
            # Start logging for the placement if not already logging
            env.start_logging = logger.log_grasping()
            
            # Wait for replay to finish if planner is not at event 4 (post-grasp)
            
                
            # Replay is finished, proceed with placement
            try:
                observations = env.world.get_observations()
                task_params = env.task.get_params()
                
                # Use orientation for placement
                # picking_position = observations[task_params["object_name"]["value"]]["object_current_position"]
                # picking_position[2] += 0.1
                picking_position = env.current_grasp_pose[0]
                orientation = env.current_grasp_pose[1]
                
                # Generate actions for placement
                actions = env.planner.forward(
                    picking_position=picking_position,
                    placing_position=observations[task_params["object_name"]["value"]]["object_target_position"],
                    current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=env.ee_placement_orientation,  
                    grasping_orientation=orientation,
                )
                
                # Apply the actions to the robot
                env.controller.apply_action(actions)
                
                # Track the current planner event
                current_event = env.planner.get_current_event()
                
                # Add progress tracking
                if current_event > 4 and current_event < 9:
                    # Print progress through placement phases
                    print(f"Placement in placement phase in progress - event {current_event}/9")
                
                # Check if the entire planning sequence is complete
                if env.planner.is_done():
                    print(f"----------------- Placement {env.placement_counter} complete ----------------- \n\n")
                    # Record the placement data
                    logger.record_grasping()
                    env.data_recorded = True
                    
                    # Reset object state for next iteration
                    env.object_grasped = False
                    
                    # Increment placement counter
                    env.placement_counter += 1
                    
                    # Check if we've reached the maximum placements
                    if env.placement_counter >= 200:
                        print(f"Maximum placements reached for grasp {env.grasp_counter}")
                        env.grasp_counter += 1
                        env.placement_counter = 0
                        env.reset()
                        env.state = "SETUP"
                    else:
                        # Reset for next placement with the same grasp
                        env.reset()
                        env.state = "REPLAY"
                
            except Exception as e:
                print(f"Error during placement: {e}")
                env.reset()
                env.state = "REPLAY"
                continue
                
    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    main()