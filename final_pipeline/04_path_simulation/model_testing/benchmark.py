import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to path 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Create an alias for model_training.pointnet2 as pointnet2
# This needs to happen before any imports that use pointnet2
import model_training.pointnet2
sys.modules['pointnet2'] = model_training.pointnet2

# Create an alias for model_training.dataset as dataset
import model_training.dataset
sys.modules['dataset'] = model_training.dataset

import model_training.model
sys.modules['model'] = model_training.model

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
import datetime 
from omni.isaac.core.utils import extensions
from simulator import Simulator
import numpy as np
import rclpy
import torch
import open3d as o3d
import json
from placement_quality.cube_simulation import helper
from rclpy.executors import SingleThreadedExecutor
import time
from model_training.dataset import KinematicFeasibilityDataset
from model_training.pointnet2 import *
from model_training.model import GraspObjectFeasibilityNet, PointNetEncoder
from model_training.train import load_pointcloud
# from omni.isaac.core import SimulationContext

# # Before running simulation
# sim = SimulationContext(physics_dt=1.0/240.0)  # 240 Hz
PEDESTAL_SIZE = np.array([0.09, 0.11, 0.1])   # X, Y, Z in meters

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/benchmark"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")
# Define color codes
GREEN = '\033[92m'  # Green text
RED = '\033[91m'    # Red text
RESET = '\033[0m'   # Reset to default color
GRIP_HOLD_STEPS = 40


def model_prediction(model, data, device):
    data["final_object_pose"][1] = 0.720
    data["final_object_pose"][2] = 0.694
    data["final_object_pose"][3] = -0.009
    data["final_object_pose"][4] = 0.005
    # Define constant values as in the dataset
    const_xy = torch.tensor([0.2, -0.3], dtype=torch.float32)
    
    # Preprocess the initial and final poses as done in the dataset
    initial_pose = torch.cat([const_xy, torch.tensor(data["initial_object_pose"], dtype=torch.float32)])
    final_pose = torch.cat([const_xy, torch.tensor(data["final_object_pose"], dtype=torch.float32)])
    
    # Now use the preprocessed tensors with batch dimension
    print(f"input 1: {torch.tensor(data['grasp_pose'], dtype=torch.float32).unsqueeze(0).to(device)}")
    print(f"input 2: {initial_pose.unsqueeze(0).to(device)}")
    print(f"input 3: {final_pose.unsqueeze(0).to(device)}")
    raw_success, raw_collision = model(None, 
                                     torch.tensor(data["grasp_pose"], dtype=torch.float32).unsqueeze(0).to(device), 
                                     initial_pose.unsqueeze(0).to(device), 
                                     final_pose.unsqueeze(0).to(device))
    
    # Apply sigmoid to convert logits to probabilities
    pred_success = torch.sigmoid(raw_success)
    pred_collision = torch.sigmoid(raw_collision)
    
    # Extract scalar values from tensors
    pred_success_val = pred_success.item()
    pred_collision_val = pred_collision.item()
    
    # Get binary predictions based on threshold of 0.5
    pred_success_binary = pred_success > 0.5
    pred_collision_binary = pred_collision > 0.5  # True means "collision predicted"
    
    return pred_success_val, 1-pred_collision_val
    return pred_success_binary.item(), pred_collision_binary.item()


def robot_action(env: Simulator, grasp_pose, current_state, next_state):
    grasp_position, grasp_orientation = grasp_pose[0], grasp_pose[1]
    env.task._frame.set_world_pose(np.array(grasp_position), 
                                    np.array(grasp_orientation))

    actions = env.controller.forward(
        target_end_effector_position=np.array(grasp_position),
        target_end_effector_orientation=np.array(grasp_orientation),
    )
    
    if env.controller.ik_check:
        kps, kds = env.task.get_custom_gains()
        env.articulation_controller.set_gains(kps, kds)
        env.articulation_controller.apply_action(actions)

        if env.check_for_collisions():
            env.collision_counter += 1
            print(f"⚠️ Collision during {current_state} (strike {env.collision_counter}/3)")

        if env.collision_counter >= 3:
            print(f"🛑 Too many collisions during {current_state} — trying next grasp")
            env.collision_counter = 0
            env.state = "FAIL"
            return False
        if env.controller.is_done():
            print(f"----------------- {next_state} Plan Complete -----------------")
            env.collision_counter = 0
            env.state = next_state
            env.controller.reset()
            return True
    else:
        print(f"----------------- RRT cannot find a path for {current_state}, going to next grasp -----------------")
        env.state = "FAIL"
        return False
    


def main(checkpoint, use_physics, test_mode=False):
    test_mode = True
    object_frame_path = "/World/Ycb_object"
    pcd_topic = "/cam0/depth_pcl"

    # Initialize ROS2 node
    rclpy.init()
    sim_subscriber = helper.TFSubscriber(pcd_topic)

    # Create an executor# Create an executor
    executor = SingleThreadedExecutor()
    executor.add_node(sim_subscriber)

    env = Simulator(use_physics=use_physics)
    env.start()
    

    # if not test_mode:   
    #     helper.tf_graph_generation(object_frame_path)
    #     helper.set_cameras(object_position)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Add a flag to check if we have computed the static object feature
    static_feature_computed = False
    grasp_poses = []

    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with during the step function")
            env.reset()
            continue
        
        # Process ROS callbacks
        rclpy.spin_once(sim_subscriber, timeout_sec=0)

        # Wait for initial TF data
        # if not test_mode and (sim_subscriber.latest_tf is None or sim_subscriber.latest_pcd is None):
        #     continue
            
        # Calculate the static object feature once we have a valid pointcloud
        if not static_feature_computed:
            env.gripper.open()
            print("Computing static point-cloud embedding from live pointcloud...")
            raw_grasps = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation/tests.json"
            with open(raw_grasps, "r") as f:
                raw_grasps = json.load(f)
            for key in sorted(raw_grasps.keys(), key=int):
                item = raw_grasps[key]
                position = item["position"]
                orientation = item["orientation_wxyz"]
                # Each grasp: [ [position], [orientation] ]
                grasp_poses.append([position, orientation])
            saved_path = "/home/chris/Chris/placement_ws/src/box_cube_0.031_0.096_0.190.pcd"
            
            # Convert pointcloud to tensor
            object_pcd_np = load_pointcloud(saved_path)
            object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
            print(f"Loaded point cloud with {object_pcd.shape[0]} points...")
            
            # Forward once through PointNetEncoder
            with torch.no_grad():
                pn = PointNetEncoder(global_feat_dim=256).to(device)
                static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
            print("Done.\n")

            # Load the checkpoint 
            checkpoint_data = torch.load(checkpoint, map_location=device)
            raw_state_dict = checkpoint_data.get("state_dict", checkpoint_data)
            
            # Strip any unwanted prefix
            prefix_to_strip = "_orig_mod."
            cleaned_state_dict = {}
            for key, tensor in raw_state_dict.items():
                if key.startswith(prefix_to_strip):
                    new_key = key[len(prefix_to_strip):]
                else:
                    new_key = key
                cleaned_state_dict[new_key] = tensor
            
            # Initialize model without static feature yet
            model = GraspObjectFeasibilityNet(use_static_obj=True).to(device)
            # Register this feature with the model
            model.register_buffer('static_obj_feat', static_obj_feat)
            # Load the state dict but don't compute point cloud embedding yet
            model.load_state_dict(cleaned_state_dict)
            model.eval()

            
            static_feature_computed = True
            print("Model loaded from checkpoint.\n")

        if env.state == "SETUP":
            # Step 6: Find all grasps with their scores and sort them
            grasp_scores = []  # List to store (grasp, score, success_val, collision_val)
            
            object_position = [0.2, -0.3, env.current_data["initial_object_pose"][0]]
            object_orientation = env.current_data["initial_object_pose"][1:]
            # env.task._ycb.set_world_pose(object_position, object_orientation)
            env.task.set_params(
                object_position=object_position,
                object_orientation=object_orientation,
            )
            for grasp_pose in grasp_poses:
                grasp_pose_local = [grasp_pose[0], grasp_pose[1]]
                grasp_pose_world = helper.transform_relative_pose(grasp_pose_local, 
                                                                  object_position,
                                                                  object_orientation)
                grasp_pose_center = helper.local_transform(grasp_pose_world, [0, 0, -0.065])
                # grasp_pose_center = grasp_pose_world
                env.current_data["grasp_pose"] = grasp_pose_center[0] + grasp_pose_center[1]
                pred_success_val, pred_collision_val = model_prediction(model, env.current_data, device)
                score = pred_success_val * pred_collision_val
                
                # Add this grasp and its score to our list
                grasp_scores.append((grasp_pose_center, score, pred_success_val, pred_collision_val))
            
            # Sort grasps by score in descending order
            grasp_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Print top grasps
            print(f"There are {len(grasp_scores)} grasps")
            print("\nTop 5 grasps:")
            for i, (grasp, score, success, no_collision) in enumerate(grasp_scores[:5]):
                print(f"#{i+1}: Score: {score:.4f} (success: {success:.4f}, no collision: {no_collision:.4f})")


            ############################################################
            #### TODO: Filter out invalid grasps 
            ############################################################
            
            # Use the best grasp for execution
            if grasp_scores:
                env.state = "PREGRASP"  
                current_grasp = grasp_scores.pop(0)
                # print(f"There are {len(grasp_scores)} grasps left")
                grasp_position, grasp_orientation = current_grasp[0][0], current_grasp[0][1]
                pregrasp_position = grasp_position + np.array([0, 0, 0.2])

        elif env.state == "FAIL":
            # Try next grasp if available
            if grasp_scores:    
                current_grasp = grasp_scores.pop(0)
                grasp_position, grasp_orientation = current_grasp[0][0], current_grasp[0][1]
                pregrasp_position = grasp_position + np.array([0, 0, 0.2])
                env.state = "PREGRASP"  # Restart with new grasp
            else:
                print("No more grasps to try")
                env.data_index += 1
                env.current_data = env.test_data[env.data_index]
                env.state = "SETUP"
            env.reset()
            continue

        elif env.state == "PREGRASP":
            # On first entry to PREGRASP, initialize collision counter
            env.gripper.open()
            
            pregrasp_pose = [pregrasp_position, grasp_orientation]
            robot_action(env, pregrasp_pose, "PREGRASP", "GRASP")
            
        elif env.state == "GRASP":
            env.gripper.open()
            grasp_pose = [current_grasp[0][0], current_grasp[0][1]]
            if robot_action(env, grasp_pose, "GRASP", "GRIPPER"):
                env.open = False


        elif env.state == "GRIPPER":
            if env.open:
                env.gripper.open()  
            else:
                env.gripper.close()
            env.step_counter += 1
            if env.step_counter < GRIP_HOLD_STEPS:
                continue   # stay here until we've done enough steps
            # After grasp
            if not env.open:
                if env.check_grasp_success():
                    print("Successfully grasped the object")
                    env.state = "PREPLACE"
                    env.controller.reset()
                    # This one is ee's
                    placement_position, placement_orientation = env.calculate_placement_pose(env.current_data["grasp_pose"], 
                                                                                         env.current_data["initial_object_pose"], 
                                                                                         env.current_data["final_object_pose"])
                    pre_placement_position = placement_position + np.array([0, 0, 0.2])

                    # This one is object's placement preview
                    final_object_position = np.array([0.2, -0.3, env.current_data["final_object_pose"][0] + PEDESTAL_SIZE[2]])
                    final_object_orientation = env.current_data["final_object_pose"][1:]
                    env.task._placement_preview.set_world_pose(final_object_position, final_object_orientation)



                else:
                    print("----------------- Grasp failed -----------------")
                    # reset state & counter for a retry
                    env.task.set_params(
                        object_position=np.array([0.2, -0.3, env.current_data["initial_object_pose"][0]]),
                        object_orientation=np.array(env.current_data["initial_object_pose"][1:]),
                    )
                    env.state = "FAIL"
                    env.reset()
                    env.gripper.open()

            # After placement
            else:
                print("----------------- Gripper Open, Going to next grasp -----------------")
                env.state = "PREGRASP"
                current_grasp = grasp_scores.pop(0)
                grasp_position, grasp_orientation = current_grasp[0][0], current_grasp[0][1]
                pregrasp_position = grasp_position + np.array([0, 0, 0.2])
                env.reset()



        elif env.state == "PREPLACE":
            preplace_pose = [pre_placement_position, placement_orientation]
            robot_action(env, preplace_pose, "PREPLACE", "PLACE")
            
        elif env.state == "PLACE":
            place_pose = [placement_position, placement_orientation]
            if robot_action(env, place_pose, "PLACE", "GRIPPER"):
                env.open = True    
                
    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    model_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/models/model_20250626_220307/best_model_0_1045_pth"
    use_physics = True
    main(model_path, use_physics)