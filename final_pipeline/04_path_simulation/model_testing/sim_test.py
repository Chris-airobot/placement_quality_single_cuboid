import os, sys
from collections import OrderedDict
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""
This Experiment runs cases directly from a sim deck file (JSON or JSONL),
one object with varied positions across pedestals. No GPD/point cloud/model paths.
"""

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
from simulator import Simulator
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.types import ArticulationAction
from placement_quality.path_simulation.model_testing.utils import pose_difference
# from omni.isaac.core import SimulationContext
import copy
# # Before running simulation
# sim = SimulationContext(physics_dt=1.0/240.0)  # 240 Hz
PEDESTAL_SIZE = np.array([0.27, 0.22, 0.10])   # X, Y, Z in meters

# ---- data sources (no argparse) ----
SIM_PATH = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.waypoints.jsonl"



base_dir = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"experiments")
RESUME_FILE = os.path.join(DIR_PATH, "resume.txt")

# Define color codes
GREEN = '\033[92m'  # Green text
RED = '\033[91m'    # Red text
RESET = '\033[0m'   # Reset to default color
GRIP_HOLD_STEPS = 60
GRIPPER_STEP_SIZE = 0.001  # Step size for gradual gripper movement
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
        # if env.contact_force > 0:
        #     print(f"Contact force: {env.contact_force:.3f}N (threshold: {env.force_threshold}N)")
        
        # Add progress tracking
        if env.gripper_step_counter % 100 == 0:  # Print every 100 steps
            print(f"Closing gripper: pos1={current_positions[0]:.4f}, pos2={current_positions[1]:.4f}, target={GRIPPER_CLOSE_POS}")
    
    # Apply the gradual movement
    env.gripper.apply_action(ArticulationAction(joint_positions=new_positions))
    
    # Reset counter when target is reached
    if target_reached:
        env.gripper_step_counter = 0
    
    return target_reached


# Removed model prediction: Experiment runs from sim file only


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

        if getattr(env, "count_collisions", False) and env.check_for_collisions():
            env.collision_counter += 1
            print(f"⚠️ Collision during {current_state} (strike {env.collision_counter}/3)")


        if env.controller.is_done():
            print(f"----------------- {current_state} Plan Complete -----------------")
            env.state = next_state
            env.controller.reset()
            return True
    else:
        print(f"----------------- RRT cannot find a path for {current_state}, going to next grasp -----------------")
        env.state = "FAIL"
        env.results.append({
            "index": env.data_index,
            "grasp": True,
            "collision_counter": env.collision_counter,
            "reason": "IK",
            "forced_completion": env.forced_completion,
            "labels": env.current_data.get("labels", {}),         # << NEW
            "initial_object_pose": env.current_data["initial_object_pose"],
            "final_object_pose": env.current_data["final_object_pose"],
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


def _read_json_or_jsonl(path):
    items = []
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    return items


def _build_test_data(cases):
    out = []
    for c in cases:
        # Waypoints deck is required
        wpd = {}
        for wp in c["waypoints"]:
            name = wp["name"]
            pose = wp["pose"]
            wpd[name] = [pose["position"], pose["orientation_quat"]]

        # Initial/final object world pose from builder extras
        pick_obj  = c["pick_object_world"]
        place_obj = c["place_object_world"]
        init_pose  = pick_obj["position"]  + pick_obj["orientation_quat"]
        final_pose = place_obj["position"] + place_obj["orientation_quat"]

        # Carry pedestal block through so we never look elsewhere
        ped = c.get("pedestal", {})

        rec = {
            "initial_object_pose": init_pose,
            "final_object_pose":   final_pose,
            "grasp_pose":          wpd["C1"][0] + wpd["C1"][1],  # for logs/visuals
            "waypoints":           wpd,                         # {"P1": [[...],[...]], ...}
            "object_dimensions":   c.get("object_dimensions", [0.143, 0.0915, 0.051]),
            "pedestal":            ped,
            "labels":              c.get("labels", {}),          # << NEW
        }
        out.append(rec)
    return out

def _maybe_update_pedestal(env, ped_pose):
    try:
        pos = ped_pose.get("position", None)
        if pos is None:
            return
        if hasattr(env, "collision_detector") and hasattr(env.collision_detector, "create_virtual_pedestal"):
            from pxr import Gf
            env.collision_detector.create_virtual_pedestal(
                position=Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
                size_x=float(PEDESTAL_SIZE[0]),
                size_y=float(PEDESTAL_SIZE[1]),
                size_z=float(PEDESTAL_SIZE[2]),
            )
    except Exception:
        pass

def main(use_physics):
    env = Simulator(use_physics=use_physics)
    results_file = os.path.join(DIR_PATH, "experiment_results_origin_box.jsonl")

    # STRICT: require a valid resume index (mirrors Experiment.py discipline)
    resume_idx = read_resume_index(RESUME_FILE)
    if resume_idx is None or resume_idx < 0:
        raise SystemExit(f"[Experiment] ERROR: Cannot read a valid index from {RESUME_FILE}. "
                         "Ensure resume.txt exists and contains a non-negative integer.")

    # Load sim cases and pedestal poses
    cases = _read_json_or_jsonl(SIM_PATH)
    env.test_data = _build_test_data(cases)
    if not env.test_data:
        raise SystemExit("[Experiment] No test cases loaded from sim file.")

    env.data_index   = int(resume_idx)
    env.current_data = env.test_data[env.data_index]


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

            # Step 6: Find all grasps with their scores and sort them
            init_pose = env.current_data["initial_object_pose"]
            print(f"env.current_data: {env.current_data}")
            object_position = init_pose[:3]
            object_orientation = init_pose[3:]
            
            # Use explicit world-frame waypoints from deck
            wpd = env.current_data["waypoints"]  # {"P1":[pos,quat], ..., "C2":[pos,quat]}
            # P1 (pre-grasp) & C1 (contact)
            pregrasp_position, pregrasp_orientation = wpd["P1"][0], wpd["P1"][1]
            grasp_position,    grasp_orientation    = wpd["C1"][0], wpd["C1"][1]
            # L1/L2 (lifts) and P2/C2 (pre-place/contact)
            env.lift1_position, env.lift1_orientation = wpd["L1"][0], wpd["L1"][1]
            env.lift2_position, env.lift2_orientation = wpd["L2"][0], wpd["L2"][1]
            pre_placement_position, placement_orientation = wpd["P2"][0], wpd["P2"][1]
            placement_position,     place_contact_quat    = wpd["C2"][0], wpd["C2"][1]

            # Final object pose for preview box
            final_pose = env.current_data["final_object_pose"]
            final_object_position    = np.array(final_pose[:3])
            final_object_orientation = final_pose[3:]

            # Update BOTH pedestals in simulator (positions come from the record)
            ped = env.current_data.get("pedestal", {})
            pick_pose  = ped.get("pick",  {"position": [0.2, -0.3, 0.05]})
            place_pose = ped.get("place", {"position": [0.3,  0.0, 0.05]})
            try:
                env.update_pedestals(pick_pose, place_pose)
            except Exception:
                pass

            # Cache dims/poses for uniform logging
            env.cur_dims       = list(map(float, env.current_data.get("object_dimensions", [0.143, 0.0915, 0.051])))
            env.cur_grasp_pose = list(map(float, wpd["C1"][0] + wpd["C1"][1]))  # C1 world
            env.cur_init_pose  = list(map(float, env.current_data["initial_object_pose"]))
            env.cur_final_pose = list(map(float, env.current_data["final_object_pose"]))

            env.task.set_params(
                object_position=object_position,
                object_orientation=object_orientation,
                preview_box_position=final_object_position,
                preview_box_orientation=final_object_orientation,
                pick_pedestal_position=np.array(pick_pose["position"]).astype(float),
                place_pedestal_position=np.array(place_pose["position"]).astype(float),
            )
                
            
             # ---- NEW: build P0 (P1 with +0.20m Z, same orientation) ----
            env.p0_position     = np.array(pregrasp_position, dtype=float).copy()
            env.p0_position[2] += 0.20
            env.p0_orientation  = np.array(pregrasp_orientation, dtype=float).copy()

            # ---- Collision counting is disabled until after C1 ----
            env.collision_counter = 0
            env.count_collisions  = False

            env.state = "PREGRASP0"   # start at P0 (then go P1 -> C1 ...)
            env.contact_force = 0.0

        elif env.state == "FAIL":
            env.data_index += 1
            if env.data_index >= len(env.test_data):
                print("[Experiment] Completed all cases.")
                break
            env.current_data = env.test_data[env.data_index]
            
            env.state = "SETUP"
            env.reset()
            continue

        elif env.state == "PREGRASP0":
            env.gripper.open()
            p0_pose = [env.p0_position, env.p0_orientation]
            robot_action(env, p0_pose, "PREGRASP0", "PREGRASP")

        elif env.state == "PREGRASP":
            # On first entry to PREGRASP, initialize collision counter
            env.gripper.open()
            pregrasp_pose = [pregrasp_position, pregrasp_orientation]
            robot_action(env, pregrasp_pose, "PREGRASP", "GRASP")
            
        elif env.state == "GRASP":
            env.gripper.open()
            if robot_action(env, [grasp_position, grasp_orientation], "GRASP", "GRIPPER"):
                env.open = False

        elif env.state == "LIFT":
            # Move up from contact to aerial clearance above grasp
            lift1_pose = [env.lift1_position, env.lift1_orientation]
            robot_action(env, lift1_pose, "LIFT", "PREPLACE_ONE")

        elif env.state == "PREPLACE_ONE":
            # Aerial move above placement target (lifted)
            preplace_one_pose = [env.lift2_position, env.lift2_orientation]
            robot_action(env, preplace_one_pose, "PREPLACE_ONE", "PREPLACE_TWO")
        
        elif env.state == "PREPLACE_TWO":   
            preplace_two_pose = [pre_placement_position, placement_orientation]  # P2
            robot_action(env, preplace_two_pose, "PREPLACE_TWO", "PLACE")
            
        elif env.state == "PLACE":
            place_pose = [placement_position, placement_orientation] # C2
            if robot_action(env, place_pose, "PLACE", "GRIPPER"):
                env.open = True 

        elif env.state == "END":
            _, gripper_current_orientation = env.gripper.get_world_pose()
   
            env.task._frame.set_world_pose(np.array(pre_placement_position), np.array(gripper_current_orientation))
            actions = env.controller.forward(
                target_end_effector_position=np.array(pre_placement_position),
                target_end_effector_orientation=np.array(gripper_current_orientation),
            )
            
            if env.controller.ik_check:
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                env.articulation_controller.apply_action(actions)

                if env.check_for_collisions():
                    env.collision_counter += 1
                    print(f"⚠️ Collision during END (strike {env.collision_counter}/3)")


                if env.controller.is_done():
                    print(f"----------------- END Plan Complete -----------------")
                    object_current_position, object_current_orientation = env.task._ycb.get_world_pose()
                    object_target_position, object_target_orientation = env.task.preview_box.get_world_pose()
                    initial_pose_v = np.concatenate([object_current_position, object_current_orientation])
                    final_pose_v   = np.concatenate([object_target_position, object_target_orientation])
                    pose_diff = pose_difference(initial_pose_v, final_pose_v)
                    env.results.append({
                        "index": env.data_index,
                        "grasp": True,
                        "collision_counter": env.collision_counter,
                        "reason": pose_diff,
                        "forced_completion": env.forced_completion,
                        "labels": env.current_data.get("labels", {}),         # << NEW
                        "initial_object_pose": env.current_data["initial_object_pose"],
                        "final_object_pose": env.current_data["final_object_pose"],
                    })
                    remaining = write_interval - ((len(env.results) - last_written) % write_interval)
                    if remaining == write_interval:
                        remaining = write_interval
                    print(f"[RESULT] index: {env.data_index}, grasp: success, pose_difference: {pose_diff}")
                    print(f"    {remaining} results until next data write.")
                    # reset state & counter for a retry, this is the success case
                    env.data_index += 1
                    if env.data_index >= len(env.test_data):
                        print("[Experiment] Completed all cases.")
                        break
                    env.current_data = env.test_data[env.data_index]
                    env.state = "SETUP"
                    env.reset()
                    continue
            else:
                print(f"----------------- RRT cannot find a path for END, going to next grasp -----------------")
                env.state = "FAIL"
                env.results.append({
                    "index": env.data_index,
                    "grasp": True,
                    "collision_counter": env.collision_counter,
                    "reason": "IK",
                    "forced_completion": env.forced_completion,
                    "labels": env.current_data.get("labels", {}),         # << NEW
                    "initial_object_pose": env.current_data["initial_object_pose"],
                    "final_object_pose": env.current_data["final_object_pose"],
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
                        # Start counting collisions only AFTER C1:
                        env.collision_counter = 0
                        env.count_collisions  = True
                        env.state = "LIFT"
                        env.controller.reset()
                        env.task.preview_box.set_world_pose(final_object_position, final_object_orientation)

                    else:
                        print("----------------- Grasp failed -----------------")
                        # reset state & counter for a retry, this is the failure case
                        env.results.append({
                            "index": env.data_index,
                            "grasp": False,
                            "collision_counter": env.collision_counter,
                            "reason": "gripper",
                            "forced_completion": env.forced_completion,
                            "labels": env.current_data.get("labels", {}),         # << NEW
                            "initial_object_pose": env.current_data["initial_object_pose"],
                            "final_object_pose": env.current_data["final_object_pose"],
                        })
                        
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        object_next_orientation = env.current_data["initial_object_pose"][1:]

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
    use_physics = True
    main(use_physics)