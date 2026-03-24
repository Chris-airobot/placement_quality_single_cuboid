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
GRASP_PATH = "/home/chris/Chris/placement_ws/src/placement_quality/path_simulation/model_testing/baseline_grasps.json"


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
GRASP_OFFSET = [0.0, 0.0, -0.065]  # Local offset from gripper frame to tool center


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


# -------- Predefined grasps support --------
def _read_grasps_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    grasps = []
    if isinstance(data, dict):
        for gid, g in data.items():
            gg = dict(g)
            gg.setdefault("id", gid)
            grasps.append(gg)
    elif isinstance(data, list):
        grasps = data
    face_to = {}
    for g in grasps:
        f = g.get("face")
        if not f:
            continue
        face_to.setdefault(str(f), []).append(g)
    return face_to

def _q_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]

def _q_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]

def _q_rot_vec(v, q):
    qv = [0.0, v[0], v[1], v[2]]
    qc = _q_conj(q)
    r = _q_mul(_q_mul(q, qv), qc)
    return [r[1], r[2], r[3]]

def _world_from_obj(T, pos_o, quat_o):
    pw = _q_rot_vec(pos_o, T["orientation_quat"])
    pw = [T["position"][0]+pw[0], T["position"][1]+pw[1], T["position"][2]+pw[2]]
    qw = _q_mul(T["orientation_quat"], quat_o)
    return pw, qw

def _apply_local_offset(position_w, orientation_w, offset_local):
    """Apply a local-frame offset to a world-frame pose.
    new_position = position_w + R(orientation_w) * offset_local; orientation unchanged.
    """
    off_w = _q_rot_vec(offset_local, orientation_w)
    return [position_w[0] + off_w[0], position_w[1] + off_w[1], position_w[2] + off_w[2]]

def _down_face_at_Ti(Ti):
    faces = {"+X":[1,0,0], "-X":[-1,0,0], "+Y":[0,1,0], "-Y":[0,-1,0], "+Z":[0,0,1], "-Z":[0,0,-1]}
    q = Ti["orientation_quat"]
    best, name = -1.0, None
    for nm, n_obj in faces.items():
        n_w = _q_rot_vec(n_obj, q)
        score = -n_w[2]  # dot with world-down
        if score > best:
            best, name = score, nm
    return name

def _synthesize_waypoints_from_grasp(Ti, Tt, grasp):
    # Local grasp in object frame
    gpos_o = grasp.get("position", [0.0,0.0,0.0])
    gq_o   = grasp.get("orientation_wxyz", [1.0,0.0,0.0,0.0])
    # Contact poses (gripper) in world
    c1_p_raw, c1_q = _world_from_obj(Ti, gpos_o, gq_o)
    c2_p_raw, c2_q = _world_from_obj(Tt, gpos_o, gq_o)
    # Apply tool-center offset in local gripper frame
    c1_p = _apply_local_offset(c1_p_raw, c1_q, GRASP_OFFSET)
    c2_p = _apply_local_offset(c2_p_raw, c2_q, GRASP_OFFSET)
    # P1: back off 0.10m along tool -Z in world (orientation same as C1)
    backoff_tool = [0.0, 0.0, -0.10]
    back_w = _q_rot_vec(backoff_tool, c1_q)
    p1_p = [c1_p[0]+back_w[0], c1_p[1]+back_w[1], c1_p[2]+back_w[2]]
    # L1/L2: lifts in world Z from the tool-center C1
    l1_p = [c1_p[0], c1_p[1], c1_p[2]+0.20]
    l2_p = [l1_p[0], l1_p[1], l1_p[2]]
    # P2: above place by 0.10 using tool-center C2
    p2_p = [c2_p[0], c2_p[1], c2_p[2]+0.10]
    return {
        "P1": [p1_p, c1_q],
        "C1": [c1_p, c1_q],
        "L1": [l1_p, c1_q],
        "L2": [l2_p, c1_q],
        "P2": [p2_p, c2_q],
        "C2": [c2_p, c2_q],
    }

def _centeredness(g):
    u = float(g.get("u_frac", 0.5)); v = float(g.get("v_frac", 0.5))
    return abs(u-0.5) + abs(v-0.5)

def _expand_with_predefined_grasps(deck_cases, face_to_grasps, seed=2025, model_top_k=3):
    """Build per-trial records containing up to 3 baseline attempts and up to 3 model attempts.

    Filtering uses ONLY the initial pose (exclude the face touching the ground).
    Baseline: 1 random grasp per remaining face, capped at 3.
    Model: top-3 centered grasps overall, capped at 3.
    """
    rng = np.random.default_rng(seed)
    trials = []
    faces = ["+X","-X","+Y","-Y"]
    for c in deck_cases:
        Ti = c.get("pick_object_world") or c.get("objW_pick") or {}
        Tt = c.get("place_object_world") or c.get("objW_place") or {}
        if not Ti or not Tt:
            continue
        down = _down_face_at_Ti(Ti)
        # candidates: exclude only the initial bottom face
        candidates = []
        for f in faces:
            if f == down:
                continue
            candidates.extend(face_to_grasps.get(f, []))
        # Baseline: exactly one random per remaining face
        baseline = []
        for f in faces:
            if f == down:
                continue
            bucket = face_to_grasps.get(f, [])
            if not bucket:
                continue
            j = int(rng.integers(0, len(bucket)))
            baseline.append(bucket[j])
        # Model: top-3 most centered overall
        model_sel = sorted(candidates, key=lambda gg: (_centeredness(gg), gg.get("id","")))[:model_top_k]
        # Enforce at most 3 per method
        if len(baseline) > 3:
            baseline = baseline[:3]
        if len(model_sel) > 3:
            model_sel = model_sel[:3]

        # Build attempts with synthesized waypoints (offset applied inside)
        attempts = []
        for idx, g in enumerate(baseline):
            wpd = _synthesize_waypoints_from_grasp(Ti, Tt, g)
            attempts.append({
                "policy": "B",
                "attempt_index": idx,
                "grasp_id": g.get("id"),
                "face": g.get("face"),
                "waypoints": wpd,
                "dims": g.get("dims_xyz", [0.143,0.0915,0.051]),
            })
        for idx, g in enumerate(model_sel):
            wpd = _synthesize_waypoints_from_grasp(Ti, Tt, g)
            attempts.append({
                "policy": "M",
                "attempt_index": idx,
                "grasp_id": g.get("id"),
                "face": g.get("face"),
                "waypoints": wpd,
                "dims": g.get("dims_xyz", [0.143,0.0915,0.051]),
            })

        init_pose  = Ti["position"]  + Ti["orientation_quat"]
        final_pose = Tt["position"] + Tt["orientation_quat"]
        ped_pick   = [float(Ti["position"][0]), float(Ti["position"][1]), 0.05]
        ped_place  = [float(Tt["position"][0]), float(Tt["position"][1]), 0.05]
        case_index = len(trials)
        trial_rec = {
            "trial_id": c.get("id", str(case_index)),
            "case_index": case_index,
            "initial_object_pose": init_pose,
            "final_object_pose":   final_pose,
            "pedestal":            {"pick": {"position": ped_pick}, "place": {"position": ped_place}},
            "pick_object_world":   Ti,
            "place_object_world":  Tt,
            "attempts":            attempts,
        }
        trials.append(trial_rec)
    return trials

def _build_attempt_record(trial_rec, attempt):
    """Combine trial-level data with a specific attempt to form an executable record."""
    return {
        "initial_object_pose": trial_rec["initial_object_pose"],
        "final_object_pose":   trial_rec["final_object_pose"],
        "waypoints":           attempt["waypoints"],
        "object_dimensions":   attempt.get("dims", [0.143,0.0915,0.051]),
        "pedestal":            trial_rec["pedestal"],
        "labels":              {
            "trial_id": trial_rec.get("trial_id"),
            "case_index": trial_rec.get("case_index"),
            "policy": attempt.get("policy"),
            "grasp_id": attempt.get("grasp_id"),
            "face": attempt.get("face"),
            "attempt_index": attempt.get("attempt_index"),
        },
        "pick_object_world":   trial_rec["pick_object_world"],
        "place_object_world":  trial_rec["place_object_world"],
    }

def _should_skip_attempt(progress, labels, env):
    """Return True if this attempt should be skipped due to trial-phase rules.

    Rules:
    - Execute baseline (policy 'B') until success OR 3 attempts; then stop running more 'B'.
    - After baseline finished (success or 3 attempts), execute model (policy 'M') until success OR 3 attempts.
    - Skip 'M' until baseline finished.
    """
    trial_id = labels.get("trial_id")
    policy   = labels.get("policy")
    tp = progress.setdefault(trial_id, {"B": {"attempts": 0, "success": False},
                                        "M": {"attempts": 0, "success": False}})
    # Determine caps for this trial
    case_index = labels.get("case_index")
    cap_B = 3
    cap_M = 3
    try:
        if case_index is not None:
            cap_B = env.trial_caps_B[int(case_index)]
            cap_M = env.trial_caps_M[int(case_index)]
    except Exception:
        pass
    if policy == "B":
        return tp["B"]["success"] or tp["B"]["attempts"] >= cap_B
    # policy == 'M'
    baseline_done = tp["B"]["success"] or tp["B"]["attempts"] >= cap_B or cap_B == 0
    if not baseline_done:
        return True
    return tp["M"]["success"] or tp["M"]["attempts"] >= cap_M

def _mark_attempt_result(progress, labels, success):
    """Update per-trial progress after an attempt completes."""
    trial_id = labels.get("trial_id")
    policy   = labels.get("policy")
    tp = progress.setdefault(trial_id, {"B": {"attempts": 0, "success": False},
                                        "M": {"attempts": 0, "success": False}})
    tp[policy]["attempts"] += 1
    if success:
        tp[policy]["success"] = True
    return tp[policy]["attempts"]


    

def main(use_physics):
    env = Simulator(use_physics=use_physics)
    results_file = os.path.join(DIR_PATH, "comparison_results.jsonl")

    # STRICT: require a valid resume index (mirrors Experiment.py discipline)
    resume_idx = read_resume_index(RESUME_FILE)
    if resume_idx is None or resume_idx < 0:
        raise SystemExit(f"[Experiment] ERROR: Cannot read a valid index from {RESUME_FILE}. "
                         "Ensure resume.txt exists and contains a non-negative integer.")

    # Load SIM deck and predefined grasps; expand per-attempt cases
    cases = _read_json_or_jsonl(SIM_PATH)
    face_to_grasps = _read_grasps_json(GRASP_PATH)
    trials = _expand_with_predefined_grasps(cases, face_to_grasps, seed=2025, model_top_k=3)
    if not trials:
        raise SystemExit("[Experiment] No test cases loaded from sim file.")
    # Build an execution schedule following your rule:
    # 1) Execute baseline attempts (up to 3) until first success or all 3 tried
    # 2) Then execute model attempts (up to 3) until first success or all 3 tried
    exec_records = []
    record_to_trial_index = []
    trial_first_indices = []
    trial_caps_B = []
    trial_caps_M = []
    for t_idx, t in enumerate(trials):
        # Mark the starting index of this trial
        trial_first_indices.append(len(exec_records))
        # Count available per policy and store caps
        b_list = [a for a in t["attempts"] if a["policy"] == "B"]
        m_list = [a for a in t["attempts"] if a["policy"] == "M"]
        trial_caps_B.append(min(3, len(b_list)))
        trial_caps_M.append(min(3, len(m_list)))
        # Baseline first
        for a in b_list:
            exec_records.append(_build_attempt_record(t, a))
            record_to_trial_index.append(t_idx)
        # Model next
        for a in m_list:
            exec_records.append(_build_attempt_record(t, a))
            record_to_trial_index.append(t_idx)

    # Save scheduling metadata on env
    env.test_data = exec_records
    env.record_to_trial_index = record_to_trial_index
    env.trial_first_indices = trial_first_indices
    env.trial_caps_B = trial_caps_B
    env.trial_caps_M = trial_caps_M

    # Interpret resume index as TRIAL index; start at that trial's first attempt
    if resume_idx >= len(trials):
        raise SystemExit(f"[Experiment] Resume trial index {resume_idx} out of range ({len(trials)} trials).")
    start_data_index = env.trial_first_indices[resume_idx]
    env.data_index   = int(start_data_index)
    env.current_data = env.test_data[env.data_index]


    env.start()
    # Track per-trial progress for baseline->model gating
    trial_progress = {}
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
            # Crash-safe: set resume to NEXT TRIAL only when we start the first attempt of a trial
            # Find the trial index for current record
            cur_trial_idx = env.record_to_trial_index[env.data_index]
            # If this record is the first attempt of its trial, set resume to next trial
            first_idx_for_trial = env.trial_first_indices[cur_trial_idx]
            if env.data_index == first_idx_for_trial:
                write_resume_index(RESUME_FILE, int(cur_trial_idx) + 1)

            # Skip attempts based on baseline->model rules
            if _should_skip_attempt(trial_progress, env.current_data.get("labels", {}), env):
                env.data_index += 1
                if env.data_index >= len(env.test_data):
                    print("[Experiment] Completed all cases.")
                    break
                env.current_data = env.test_data[env.data_index]
                env.state = "SETUP"
                env.reset()
                continue

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
                        "case_id": env.current_data.get("labels", {}).get("trial_id"),
                        "grasp": True,
                        "collision_counter": env.collision_counter,
                        "reason": pose_diff,
                        "forced_completion": env.forced_completion,
                        "labels": env.current_data.get("labels", {}),
                        "execution_number": _mark_attempt_result(trial_progress, env.current_data.get("labels", {}), success=True) or 0,
                        "collisions": bool(env.collision_counter > 0),
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
                    "case_id": env.current_data.get("labels", {}).get("trial_id"),
                    "grasp": True,
                    "collision_counter": env.collision_counter,
                    "reason": "IK",
                    "forced_completion": env.forced_completion,
                    "labels": env.current_data.get("labels", {}),
                    "execution_number": _mark_attempt_result(trial_progress, env.current_data.get("labels", {}), success=False) or 0,
                    "collisions": bool(env.collision_counter > 0),
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
                            "case_id": env.current_data.get("labels", {}).get("trial_id"),
                            "grasp": False,
                            "collision_counter": env.collision_counter,
                            "reason": "gripper",
                            "forced_completion": env.forced_completion,
                            "labels": env.current_data.get("labels", {}),
                            "execution_number": _mark_attempt_result(trial_progress, env.current_data.get("labels", {}), success=False) or 0,
                            "collisions": bool(env.collision_counter > 0),
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