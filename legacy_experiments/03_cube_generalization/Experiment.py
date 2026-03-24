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

import datetime 
from omni.isaac.core.utils import extensions
from simulator import Simulator
import numpy as np

import json
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from cube_generalization.utils import *
from ycb_simulation.utils.helper import local_transform

PEDESTAL_SIZE = np.array([0.09, 0.11, 0.1])   # X, Y, Z in meters
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"experiments")
# Sidecar resume pointer (crash-safe)
RESUME_FILE = os.path.join(DIR_PATH, "resume.txt")
# Define color codes
GREEN = '\033[92m'  # Green text
RED = '\033[91m'    # Red text
RESET = '\033[0m'   # Reset to default color
GRIP_HOLD_STEPS = 60
GRIPPER_STEP_SIZE = 0.002  # Step size for gradual gripper movement
GRIPPER_OPEN_POS = 0.04    # Fully open position
GRIPPER_CLOSE_POS = 0.0    # Fully closed position

def gradual_gripper_control(env: Simulator, target_open, max_steps=100):
    """
    Gradually control the gripper to open or close position with force detection
    Args:
        env: Simulator instance
        target_open: True to open gripper, False to close gripper
        max_steps: Maximum number of steps to prevent infinite loops
    Returns:
        bool: True if gripper has reached target position or object grasped, False otherwise
    """
    current_positions = env.gripper.get_joint_positions()
    
    # Add step counter to prevent infinite loops
    if not hasattr(env, 'gripper_step_counter'):
        env.gripper_step_counter = 0
    else:
        env.gripper_step_counter += 1
    
    # Check for timeout
    if env.gripper_step_counter >= max_steps:
        print(f"⚠️ Gripper control timeout after {max_steps} steps - forcing completion")
        env.gripper_step_counter = 0
        env.forced_completion = True
        return True
    
    if target_open:
        # Gradually open gripper (move toward 0.04)
        new_positions = [
            min(current_positions[0] + GRIPPER_STEP_SIZE, GRIPPER_OPEN_POS),
            min(current_positions[1] + GRIPPER_STEP_SIZE, GRIPPER_OPEN_POS)
        ]
        # Fix: Use AND instead of OR - both fingers must reach target
        target_reached = (abs(current_positions[0] - GRIPPER_OPEN_POS) < 1e-3 and 
                         abs(current_positions[1] - GRIPPER_OPEN_POS) < 1e-3)
        
        # Add progress tracking
        if env.gripper_step_counter % 100 == 0:  # Print every 100 steps
            print(f"Opening gripper: pos1={current_positions[0]:.4f}, pos2={current_positions[1]:.4f}, target={GRIPPER_OPEN_POS}")
            
    else:
        # Check if object contact force exceeds threshold BEFORE moving
        force_grasped = env.contact_force > env.force_threshold
        
        if force_grasped:
            print(f"Force threshold exceeded ({env.contact_force:.3f}N > {env.force_threshold}N), stopping gripper closing")
            env.gripper_step_counter = 0  # Reset counter
            return True
        
        # Gradually close gripper (move toward 0.0)
        new_positions = [
            max(current_positions[0] - GRIPPER_STEP_SIZE, GRIPPER_CLOSE_POS),
            max(current_positions[1] - GRIPPER_STEP_SIZE, GRIPPER_CLOSE_POS)
        ]
        
        # Fix: Use AND instead of OR - both fingers must reach target
        target_reached = (abs(current_positions[0] - GRIPPER_CLOSE_POS) < 1e-3 and 
                         abs(current_positions[1] - GRIPPER_CLOSE_POS) < 1e-3)
        
        # Print force information for debugging
        if env.contact_force > 0:
            print(f"Contact force: {env.contact_force:.3f}N (threshold: {env.force_threshold}N)")
        
        # Add progress tracking
        if env.gripper_step_counter % 100 == 0:  # Print every 100 steps
            print(f"Closing gripper: pos1={current_positions[0]:.4f}, pos2={current_positions[1]:.4f}, target={GRIPPER_CLOSE_POS}")
    
    # Apply the gradual movement
    env.gripper.apply_action(ArticulationAction(joint_positions=new_positions))
    
    # Reset counter when target is reached
    if target_reached:
        env.gripper_step_counter = 0
    
    return target_reached


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
            # Only count collisions after grasping (post-grasp stages)
            post_grasp_stages = ["LIFT_1", "LIFT_2", "PLACE", "END"]
            if current_state in post_grasp_stages:
                env.collision_counter += 1
                print(f"⚠️ Collision during {current_state} (strike {env.collision_counter}/3)")

                # NEW: Abort this attempt if too many collisions
                if env.collision_counter >= 3:
                    print(f"🛑 Too many collisions during {current_state} — recording failure and moving on")
                    # Record failure due to collisions
                    env.results.append({
                        "index":             env.data_index,
                        "outcome":           "collision_limit",
                        "had_collision":     True,
                        "collision_counter": int(env.collision_counter),
                        "collision_stage":   current_state,

                        "bucket":            env.cur_bucket,

                        "dims":              env.cur_dims,
                        "grasp_pose":        env.cur_grasp_pose_fixed,
                        "init_pose":         env.cur_init_pose,
                        "final_pose":        env.cur_final_pose,
                    })
                    # Reset counters/state and move on
                    env.collision_counter = 0
                    env.state = "FAIL"
                    env.controller.reset()
                    return False
            else:
                # For pre-grasp stages, just print the collision but don't count it
                print(f"⚠️ Collision detected during {current_state} (not counted - pre-grasp stage)")
        # if env.collision_counter >= 300000000000000:
        #     print(f"🛑 Too many collisions during {current_state} — trying next grasp")
        #     env.collision_counter = 0
        #     env.state = "FAIL"
        #     env.results.append({
        #         "index": env.data_index,
        #         "grasp": True,
        #         "collision_counter": env.collision_counter,
        #         "reason": pose_diff,
        #         "forced_completion": env.forced_completion
        #     })
        #     return False
        if env.controller.is_done():
            print(f"----------------- {current_state} Plan Complete -----------------")
            env.state = next_state
            env.controller.reset()
            return True
    else:
        print(f"----------------- RRT cannot find a path for {current_state}, going to next grasp -----------------")
        env.state = "FAIL"
        env.results.append({
            "index":             int(env.data_index),
            "outcome":           f"motion_ik_fail_{current_state.lower()}",
            "had_collision":     bool(env.collision_counter > 0),
            "collision_counter": int(env.collision_counter),

            "bucket":            env.cur_bucket,

            "dims":              list(map(float, env.cur_dims)),
            "grasp_pose":        list(map(float, env.cur_grasp_pose_fixed)),
            "init_pose":         list(map(float, env.cur_init_pose)),
            "final_pose":        list(map(float, env.cur_final_pose)),
        })

        return False
    


def write_results_to_file(results, file_path, mode='a'):
    """Append results to a JSONL file, ensuring the directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

def read_resume_index(path: str):
    """Read resume index from text file, return int or None if missing/invalid."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = f.read().strip()
                if data:
                    return int(data)
    except Exception:
        pass
    return None

def write_resume_index(path: str, next_index: int):
    """Atomically write next index to resume file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(str(int(next_index)))
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort; do not crash the experiment on resume write issues
        pass

def main(use_physics):
    env = Simulator(use_physics=use_physics)

    results_file = os.path.join(DIR_PATH, "experiment_results.jsonl")
    # STRICT: require resume.txt; do not fall back
    resume_idx = read_resume_index(RESUME_FILE)
    if resume_idx is None or resume_idx < 0:
        raise SystemExit(f"[Experiment] ERROR: Cannot read a valid index from {RESUME_FILE}. "
                         "Ensure resume.txt exists and contains a non-negative integer.")
    env.data_index = int(resume_idx)

    print(f"[Experiment] Starting run at index {env.data_index} (from {RESUME_FILE})")

    env.start()
        
    write_interval = 1
    last_written = 0


    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with during the step function")
            env.reset()
            continue
        
        if env.state == "SETUP":
            # Trace which sample is being processed
            print(f"[Experiment] SETUP for index {env.data_index}")
            # Crash-safe: as soon as we begin processing index i, set resume to i+1
            write_resume_index(RESUME_FILE, int(env.data_index) + 1)
            pose_key = tuple(env.current_data["initial_object_pose"].tolist()) if isinstance(env.current_data["initial_object_pose"], np.ndarray) else tuple(env.current_data["initial_object_pose"])
            object_position = env.current_data["initial_object_pose"][:3]
            object_orientation = env.current_data["initial_object_pose"][3:]
            
            grasp_position = env.current_data["grasp_pose"][:3]
            grasp_orientation = env.current_data["grasp_pose"][3:]
            grasp_pose = grasp_position + grasp_orientation
            # Update the grasp pose to be in the tool_center frame of the object
            fixed_grasp_pose = local_transform(grasp_pose, [0, 0, 0])
            grasp_position = fixed_grasp_pose[:3]
            grasp_orientation = fixed_grasp_pose[3:]

            pregrasp_position, pregrasp_orientation = get_reachable_prepose(
                grasp_position, grasp_orientation, env, max_offset=0.10
            )
            if pregrasp_position is None:
                # Record a reachable-pregrasp IK failure and move to next case (no 'skip' wording)
                env.results.append({
                    "index":             int(env.data_index),
                    "outcome":           "pregrasp_ik_fail",
                    "had_collision":     False,
                    "collision_counter": int(env.collision_counter),
                    "bucket":            classify_bucket(env.current_data["initial_object_pose"][3:], 
                                                        env.current_data["final_object_pose"][3:]),
                    "dims":              list(map(float, env.current_data["object_dimensions"])),
                    "grasp_pose":        list(map(float, fixed_grasp_pose)),
                    "init_pose":         list(map(float, env.current_data["initial_object_pose"])),
                    "final_pose":        list(map(float, env.current_data["final_object_pose"])),
                })
                env.data_index += 1
                env.current_data = env.test_data[env.data_index]
                env.state = "SETUP"
                env.reset()
                continue

            placement_position, placement_orientation = env.calculate_placement_pose(
                fixed_grasp_pose, env.current_data["initial_object_pose"], env.current_data["final_object_pose"]
            )
            
            # === NEW: compute once and STASH on env so every state can log consistently ===
            init_q  = env.current_data["initial_object_pose"][3:]
            final_q = env.current_data["final_object_pose"][3:]
            env.cur_bucket            = classify_bucket(init_q, final_q)           # uses helper below
            env.cur_dims              = list(map(float, env.current_data["object_dimensions"]))
            env.cur_grasp_pose_fixed  = list(map(float, fixed_grasp_pose))         # [tx,ty,tz,qw,qx,qy,qz]
            env.cur_init_pose         = list(map(float, env.current_data["initial_object_pose"]))
            env.cur_final_pose        = list(map(float, env.current_data["final_object_pose"]))

            
            # Define overhead waypoints for two-stage aerial travel
            lift1_height = 0.2
            env.lift1_position = [
                float(grasp_position[0]),
                float(grasp_position[1]),
                float(grasp_position[2] + lift1_height),
            ]
            env.lift1_orientation = grasp_orientation

            air_clearance = 0.2
            env.lift2_position = [
                float(placement_position[0]),
                float(placement_position[1]),
                float(placement_position[2] + air_clearance),
            ]
            env.lift2_orientation = placement_orientation

            # This one is object's placement preview
            final_object_position = env.current_data["final_object_pose"][:3]
            final_object_orientation = env.current_data["final_object_pose"][3:]


            env.task.set_params(
                object_position=object_position,
                object_orientation=object_orientation,
                preview_box_position=final_object_position,
                preview_box_orientation=final_object_orientation,
                object_scale=np.array(env.current_data["object_dimensions"]),
            )

            # Reset controller to clear any state from IK checks during setup
            env.controller.reset()
            
            env.state = "PREGRASP"
            env.contact_force = 0.0

        elif env.state == "FAIL":
            # Don't count movement failures - only grasp failures are counted elsewhere
            env.data_index += 1
            env.current_data = env.test_data[env.data_index]
            
            env.state = "SETUP"
            env.reset()
            continue

        elif env.state == "PREGRASP":
            # On first entry to PREGRASP, initialize collision counter
            env.gripper.open()
            pregrasp_pose = [pregrasp_position, pregrasp_orientation]
            robot_action(env, pregrasp_pose, "PREGRASP", "GRASP")
            
        elif env.state == "GRASP":
            env.gripper.open()
            if robot_action(env, [grasp_position, grasp_orientation], "GRASP", "GRIPPER"):
                env.open = False

        elif env.state == "LIFT_1":
            lift1_pose = [env.lift1_position, env.lift1_orientation]
            robot_action(env, lift1_pose, "LIFT_1", "LIFT_2")

        elif env.state == "LIFT_2":
            lift2_pose = [env.lift2_position, env.lift2_orientation]
            robot_action(env, lift2_pose, "LIFT_2", "PLACE")
            
        elif env.state == "PLACE":
            place_pose = [placement_position, placement_orientation]
            if robot_action(env, place_pose, "PLACE", "GRIPPER"):
                env.open = True 

        elif env.state == "END":
            _, gripper_current_orientation = env.gripper.get_world_pose()
   
            env.task._frame.set_world_pose(np.array(env.lift2_position), np.array(gripper_current_orientation))
            actions = env.controller.forward(
                target_end_effector_position=np.array(env.lift2_position),
                target_end_effector_orientation=np.array(gripper_current_orientation),
            )
            
            if env.controller.ik_check:
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                env.articulation_controller.apply_action(actions)

                if env.check_for_collisions():
                    env.collision_counter += 1
                    print(f"⚠️ Collision during END (strike {env.collision_counter}/3)")

                    # NEW: Too many collisions while returning – treat as failure
                    if env.collision_counter >= 3:
                        print("🛑 Too many collisions during END — recording failure and moving on")
                        env.results.append({
                            "index":             int(env.data_index),
                            "outcome":           "collision_limit",
                            "had_collision":     bool(env.collision_counter > 0),
                            "collision_counter": int(env.collision_counter),
                            "collision_stage":   "END",
                            "bucket":            env.cur_bucket,
                            "dims":              list(map(float, env.cur_dims)),
                            "grasp_pose":        list(map(float, env.cur_grasp_pose_fixed)),
                            "init_pose":         list(map(float, env.cur_init_pose)),
                            "final_pose":        list(map(float, env.cur_final_pose)),
                        })
                        env.collision_counter = 0
                        env.state = "FAIL"
                        continue

                if env.controller.is_done():
                    print(f"----------------- END Plan Complete -----------------")
                    object_current_position, object_current_orientation = env.task._ycb.get_world_pose()
                    env.results.append({
                        "index":             int(env.data_index),
                        "outcome":           "success",
                        "had_collision":     bool(env.collision_counter > 0),
                        "collision_counter": int(env.collision_counter),

                        "bucket":            env.cur_bucket,

                        "dims":              list(map(float, env.cur_dims)),
                        "grasp_pose":        list(map(float, env.cur_grasp_pose_fixed)),
                        "init_pose":         list(map(float, env.cur_init_pose)),
                        "final_pose":        list(map(float, env.cur_final_pose)),

                        "actual_object_pos": list(map(float, object_current_position)),
                    })
                    # reset state & counter for a retry, this is the success case
                    env.collision_counter = 0
                    env.data_index += 1
                    env.current_data = env.test_data[env.data_index]
                    env.state = "SETUP"
                    env.reset()
                    continue
            else:
                print(f"----------------- RRT cannot find a path for END, going to next grasp -----------------")
                env.state = "FAIL"
                env.results.append({
                    "index":             int(env.data_index),
                    "outcome":           "motion_ik_fail_end",
                    "had_collision":     bool(env.collision_counter > 0),
                    "collision_counter": int(env.collision_counter),
                    "bucket":            env.cur_bucket,
                    "dims":              list(map(float, env.cur_dims)),
                    "grasp_pose":        list(map(float, env.cur_grasp_pose_fixed)),
                    "init_pose":         list(map(float, env.cur_init_pose)),
                    "final_pose":        list(map(float, env.cur_final_pose)),
                })


        elif env.state == "GRIPPER":
            # Use gradual gripper control instead of instant open/close
            if env.open:
                # Open gripper
                while not gradual_gripper_control(env, target_open=True, max_steps=100):
                    if env.forced_completion:
                        break
                    env.world.step(render=True)
                print("Gripper opening completed")
            else:
                # Close gripper
                while not gradual_gripper_control(env, target_open=False, max_steps=100):
                    if env.forced_completion:
                        break
                    env.world.step(render=True)
            
            env.step_counter += 1
            
            # Only proceed to next state when gripper has reached target position
            # and we've held it for the required number of steps
            if env.step_counter >= GRIP_HOLD_STEPS:
                # After grasp
                if not env.open:
                    if env.check_grasp_success() and not env.forced_completion:
                        print("Successfully grasped the object")
                        # Reset collision counter for post-grasp stages
                        env.collision_counter = 0
                        env.state = "LIFT_1"
                        env.controller.reset()
                        env.task.preview_box.set_world_pose(final_object_position, final_object_orientation)
                    else:
                        print("----------------- Grasp failed -----------------")
                        
                        # Count this as a grasp failure for this pose
                        pose_key = tuple(env.current_data["initial_object_pose"].tolist()) if isinstance(env.current_data["initial_object_pose"], np.ndarray) else tuple(env.current_data["initial_object_pose"])
                        
                        # reset state & counter for a retry, this is the failure case
                        env.results.append({
                            "index":             int(env.data_index),
                            "outcome":           "grasp_fail",
                            "had_collision":     bool(env.collision_counter > 0),
                            "collision_counter": int(env.collision_counter),

                            "bucket":            env.cur_bucket,

                            "dims":              list(map(float, env.cur_dims)),
                            "grasp_pose":        list(map(float, env.cur_grasp_pose_fixed)),
                            "init_pose":         list(map(float, env.cur_init_pose)),
                            "final_pose":        list(map(float, env.cur_final_pose)),
                        })
                        
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        object_next_orientation = env.current_data["initial_object_pose"][1:]
                        # while object_next_orientation == object_orientation:

                        #     env.data_index += 1
                        #     env.current_data = env.test_data[env.data_index]
                        #     object_next_orientation = env.current_data["initial_object_pose"][1:]

                        env.current_data = env.test_data[env.data_index]
                        env.state = "SETUP"
                        remaining = write_interval - ((len(env.results) - last_written) % write_interval)
                        if remaining == write_interval:
                            remaining = write_interval
                        print(f"[RESULT] index: {env.data_index}, grasp: failure, collision_counter: {env.collision_counter}")
                        print(f"    {remaining} results until next data write.")
                        env.reset()
                        continue

                # After placement
                else:
                    # # Wait for 2 seconds (120 steps at 60 FPS) for object to stabilize
                    stabilization_steps = 60 # 2 seconds at 60 FPS
                    for _ in range(stabilization_steps):
                        env.world.step(render=True)
                    
                    print("----------------- Object stabilization complete, Going to END -----------------")
                    env.state = "END"


            else:
                continue   # stay here until gripper reaches target and we've done enough steps


        # After appending a result to env.results (success or failure), check if we should write:
        if len(env.results) - last_written >= write_interval:
            write_results_to_file(env.results[last_written:], results_file, mode='a')
            last_written = len(env.results)

    # Cleanup when simulation ends
    simulation_app.close()

    # Write any remaining results not yet saved
    if len(env.results) > last_written:
        write_results_to_file(env.results[last_written:], results_file, mode='a')

if __name__ == "__main__":
    # model_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/training/models/model_20250804_175834/best_model_roc_20250804_175834.pth"
    use_physics = True
    main(use_physics)