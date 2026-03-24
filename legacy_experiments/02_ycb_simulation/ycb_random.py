import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

import os
import rclpy
import datetime 
from omni.isaac.core.utils import extensions
from ycb_simulation.simulation.logger import YcbLogger
from simulation.simulator import YcbCollection
from ycb_simulation.utils.vision import *
from ycb_simulation.utils.helper import *


# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")



def main():
    env = YcbCollection()
    env.start()
    logger = YcbLogger(env, DIR_PATH)


    while simulation_app.is_running():
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            if env.placement_counter <= 1:
                env.reset()
                continue
            elif env.placement_counter > 1 and env.placement_counter < 200:
                env.soft_reset()
                continue
        
        # Process ROS callbacks
        rclpy.spin_once(env.sim_subscriber, timeout_sec=0)
        
        # Wait for initial TF data
        if env.sim_subscriber.latest_tf is None and env.state != "INIT":
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
                    env.state = "GRASP"

                elif env.current_grasp_pose == -1:
                    setup_phase = 0
                    env.current_grasp_pose = None
                    env.reset()
                    continue

        # STABILIZE state between reset and grasp
        elif env.state == "STABILIZE":

            print("Now in STABILIZE state")
            variantSet = env.task._object_final.prim.GetVariantSets().GetVariantSet("mode")
            current_mode = variantSet.GetVariantSelection()
            print("Object current mode:", current_mode)
            # Let the simulation run for a few steps to allow objects to stabilize
            env.stabilize_counter = env.stabilize_counter - 1 if hasattr(env, 'stabilize_counter') else 50  # Default 30 steps
            
            # When stabilization is complete, transition to GRASP
            if env.stabilize_counter <= 0:
                print("Stabilization complete, moving to grasp")
                env.state = "GRASP"
                # Hide the final object
                # env.task._object_final.prim.GetAttribute("visibility").Set("invisible")
                
                # Reset counter for next time
                env.stabilize_counter = 50

        elif env.state == "GRASP":
            # Start logging for the grasp attempt
            logger.log_grasping()
                
            # Perform grasping
            try:
                observations = env.world.get_observations()
                task_params = env.task.get_params()

                variantSet = env.task._object_final.prim.GetVariantSets().GetVariantSet("mode")
                # Switch to the "visual" variant, which omits the physics properties.
                variantSet.SetVariantSelection("visual")

                draw_frame(env.current_grasp_pose[0], env.current_grasp_pose[1])
                env.current_grasp_pose[0][2] += 0.01 if env.current_grasp_pose[0][2] < 0.01 else 0
                
                # Generate actions for the robot
                actions = env.planner.forward(
                    picking_position=env.current_grasp_pose[0],
                    placing_position=observations[task_params["object_name"]["value"]]["object_target_position"],
                    current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=env.current_grasp_pose[1],  
                    picking_orientation=env.current_grasp_pose[1],
                )

                # Check if grasp failed
                if  env.planner.get_current_event() == 4 and \
                    not env.check_grasp_success():
                    
                    print("Grasp failed, recording data and restarting")
                    logger.record_grasping(message="FAILED")
                    try_next_grasp_pose()
                    continue
                if  env.planner.get_current_event() > 4 and \
                    env.planner.get_current_event() < 7 and \
                    not env.check_grasp_success():
                    
                    print("Grasp failed, recording data and restarting")
                    logger.record_grasping(message="SLIPPED")
                    try_next_grasp_pose()
                    continue

                # Apply the actions to the robot
                env.controller.apply_action(actions)
                    
                # Check if we've completed all phases of the planner
                if env.planner.is_done():
                    print("----------------- First grasp and placement complete -----------------")
                    # Record the successful trajectory
                    logger.record_grasping(message="SUCCESS")
                    
                    # Increment the placement counter and move to replay state
                    env.placement_counter += 1
                    
                    # Reset for next placement
                    env.soft_reset("REPLAY")
            
            except Exception as e:
                print("Something went wrong with the grasp, recording data and restarting")
                logger.record_grasping(message="ERROR")
                try_next_grasp_pose()
                continue
                
        elif env.state == "REPLAY":
            # Robot and final object visible
            env.robot.prim.GetAttribute("visibility").Set("inherited")
            env.task._object_final.prim.GetAttribute("visibility").Set("inherited")

            # Start the replay of the grasping trajectory
            logger.replay_grasping()        
            if logger.data_logger.get_num_of_data_frames() >= 600:
                env.reset()
                continue
            # Move to place state - the replay will continue in the background
            env.state = "PLACE"
            env.start_logging = True

            # Generate random placement orientation
            euler_angles = [random.uniform(0, 360) for _ in range(3)]
            env.ee_placement_orientation = R.from_euler('xyz', euler_angles, degrees=True).as_quat()

        elif env.state == "PLACE":
            # Wait for replay to finish if planner is not at event 4 (post-grasp)
            if env.planner.get_current_event() < 4:
                continue
            # Start logging for the placement if not already logging
            logger.log_grasping()

            # Replay is finished, proceed with placement
            try:
                # Hide the final object
                # env.task._object_final.prim.GetAttribute("visibility").Set("invisible")
                observations = env.world.get_observations()
                task_params = env.task.get_params()
                variantSet = env.task._object_final.prim.GetVariantSets().GetVariantSet("mode")
                variantSet.SetVariantSelection("visual")
                
                # Generate actions for placement
                actions = env.planner.forward(
                    picking_position=env.current_grasp_pose[0],
                    placing_position=task_params["object_target_position"]["value"],
                    current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=env.ee_placement_orientation,  
                    picking_orientation=env.current_grasp_pose[1],
                )
                
                # Apply the actions to the robot
                env.controller.apply_action(actions)
                                        
                # Check if the entire planning sequence is complete
                if env.planner.is_done():
                    print(f"----------------- Placement {env.placement_counter} complete ----------------- \n\n")
                    # Record the placement data
                    logger.record_grasping(message="SUCCESS")

                    #
                    
                    # Increment placement counter
                    env.placement_counter += 1
                    
                    # Check if we've reached the maximum placements
                    if env.placement_counter >= 200:
                        print(f"Maximum placements reached for grasp {env.grasp_counter}")
                        try_next_grasp_pose()
                    else:
                        # Reset for next placement with the same grasp
                        env.soft_reset()
                
            except Exception as e:
                print(f"Error during placement: {e}")
                # if "Found zero norm quaternions in `quat`" in str(e):
                #     env.reset()
                #     continue
                env.soft_reset()
                continue
                
        

    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    main()