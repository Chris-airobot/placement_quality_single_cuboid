#!/usr/bin/env python3
"""
Minimal visualization for the exact data-collection case:
- Loads pedestal poses, object poses (z + orientation), and one local-frame grasp
- Places the object on top of the chosen pedestal (x,y from pedestal; z = ped_top + object_z)
- Transforms the chosen grasp to world (including tool-center offset)
- Computes pregrasp P using the current retreat rule (P = C - d_a * z_tool)
- Draws frames for C and P, and prints diagnostics (P.z vs pedestal top)
- Adds the Franka robot and performs P -> C -> L with IK (same setup)

Style intentionally mirrors grasp_pose_visualization.py (SimulationApp, World, VisualCuboid, helpers).
"""
import os
import sys
import time
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
}

simulation_app = SimulationApp(CONFIG)

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
import omni
from pxr import Gf
from placement_quality.path_simulation.collision_check import GroundCollisionDetector

from placement_quality.ycb_simulation.utils.helper import draw_frame, transform_relative_pose, local_transform

# Selection: fixed to one recorded case (edit these)
P_IDX, G_ID, O_IDX = 2, 1000, 100
RAW_ROOT = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
RAW_CASE_PATH = os.path.join(RAW_ROOT, f"p{P_IDX}", f"data_{G_ID}.json")
RAW_CASE_KEY = str(O_IDX)

# Box and pedestal geometry
BOX_DIMS = np.array([0.143, 0.0915, 0.051], dtype=float)
PEDESTAL_DIMS = np.array([0.27, 0.22, 0.10], dtype=float)

# (selection defined above)

# Offsets and distances
GRASP_OFFSET = [0.0, 0.0, -0.065]  # tool-center offset along local -Z
D_A = 0.04                         # pregrasp retreat distance (m)
H_LIFT = 0.3


def build_scene(world: World, ped_center, obj_center, obj_quat_wxyz):
    world.scene.add_default_ground_plane()

    pedestal = VisualCuboid(
        prim_path="/World/Pedestal_view",
        name="Pedestal_view",
        position=np.array(ped_center, dtype=float),
        scale=PEDESTAL_DIMS.tolist(),
        color=np.array([0.6, 0.6, 0.6], dtype=float),
    )
    world.scene.add(pedestal)

    cube = VisualCuboid(
        prim_path="/World/Ycb_object",
        name="Ycb_object",
        position=np.array(obj_center, dtype=float),
        orientation=np.array(obj_quat_wxyz, dtype=float),
        scale=BOX_DIMS.tolist(),
        color=np.array([0.8, 0.8, 0.8], dtype=float),
    )
    world.scene.add(cube)

    return pedestal, cube


def compute_pregrasp(contact_pos_w, contact_quat_wxyz, d_a: float):
    q_xyzw = np.array([contact_quat_wxyz[1], contact_quat_wxyz[2], contact_quat_wxyz[3], contact_quat_wxyz[0]], dtype=float)
    Rcw = R.from_quat(q_xyzw).as_matrix()
    z_tool_w = Rcw[:, 2]
    pregrasp_pos = np.array(contact_pos_w, dtype=float) - float(d_a) * z_tool_w
    pregrasp_quat = np.array(contact_quat_wxyz, dtype=float)
    return pregrasp_pos, pregrasp_quat


def add_franka(world: World) -> Articulation:
    assets = get_assets_root_path()
    add_reference_to_stage(assets + "/Isaac/Robots/Franka/franka.usd", "/World/panda")
    robot = Articulation("/World/panda")
    world.scene.add(robot)
    return robot


def setup_kinematics(articulation: Articulation) -> ArticulationKinematicsSolver:
    cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula = LulaKinematicsSolver(**cfg)
    kin = ArticulationKinematicsSolver(articulation, lula, "panda_hand")
    return kin


def try_move(articulation: Articulation, kin: ArticulationKinematicsSolver, pos_w, quat_wxyz) -> bool:
    base_p, base_q = articulation.get_world_pose()
    kin._kinematics_solver.set_robot_base_pose(base_p, base_q)
    action, ok = kin.compute_inverse_kinematics(np.array(pos_w, dtype=float), np.array(quat_wxyz, dtype=float))
    if ok:
        articulation.apply_action(action)
    return bool(ok)


def _merge_full_joints(articulation: Articulation, arm_vec: np.ndarray, full_ref: np.ndarray) -> np.ndarray:
    arm_len = min(len(arm_vec), len(full_ref))
    full = np.array(full_ref, dtype=float)
    full[:arm_len] = arm_vec[:arm_len]
    return full


def check_robot_collision(detector, parts):
    ground_hit   = any(detector.is_colliding_with_ground(p)   for p in parts)
    pedestal_hit = any(detector.is_colliding_with_pedestal(p) for p in parts)
    return (ground_hit or pedestal_hit), ground_hit, pedestal_hit

def _get_target_arm_joints(world: World, articulation: Articulation, kin: ArticulationKinematicsSolver, pos_w, quat_wxyz) -> tuple:
    action, ok = kin.compute_inverse_kinematics(np.array(pos_w, dtype=float), np.array(quat_wxyz, dtype=float))
    if not ok:
        return None, False
    # Try to read joint positions from the action object if available; otherwise apply and revert
    arm_vec = getattr(action, 'joint_positions', None)
    if arm_vec is None:
        q_start_full = np.array(articulation.get_joint_positions(), dtype=float)
        articulation.apply_action(action)
        world.step(render=True)
        arm_vec = np.array(articulation.get_joint_positions(), dtype=float)[:len(q_start_full)]
        articulation.set_joint_positions(q_start_full)
        world.step(render=True)
    else:
        arm_vec = np.array(arm_vec, dtype=float)
    return arm_vec, True


def sweep_to(world, articulation, kin, pos_w, quat_wxyz, steps=60,
             detector=None, parts=None, q_start=None, leave_at_end=False):
    # save the call-site pose so we can restore it
    q_save = np.array(articulation.get_joint_positions(), dtype=float)

    # force a canonical start if provided
    if q_start is not None:
        articulation.set_joint_positions(np.asarray(q_start, dtype=float))
        world.step(render=True)

    # now capture the (possibly forced) start
    q0 = np.array(articulation.get_joint_positions(), dtype=float)

    # Update base pose for solver and compute IK joints robustly
    base_p, base_q = articulation.get_world_pose()
    kin._kinematics_solver.set_robot_base_pose(base_p, base_q)

    arm_vec, ok = _get_target_arm_joints(world, articulation, kin, pos_w, quat_wxyz)
    if not ok or arm_vec is None or (np.asarray(arm_vec).size == 0):
        if not leave_at_end:
            articulation.set_joint_positions(q_save); world.step(render=True)
        return False

    # build end joints
    q1 = np.array(q0, dtype=float)
    arm_vec = np.asarray(arm_vec, dtype=float)
    q1[:len(arm_vec)] = arm_vec[:len(q1)]

    # basic safety asserts
    assert detector is not None, "Detector is None (collision check disabled)."
    assert parts and len(parts) > 0, "Empty parts list (no links to check)."

    # interpolate in JOINT space; check every step
    n = max(1, int(steps))
    hit = False
    hit_g = hit_p = False
    hit_part = None

    for s in range(1, n + 1):
        a = s / float(n)
        q_s = (1.0 - a) * q0 + a * q1
        articulation.set_joint_positions(q_s)
        world.step(render=True)  # flush transforms before queries
        time.sleep(0.01)


        # query detector
        h_g = any(detector.is_colliding_with_ground(p)   for p in parts)
        h_p = any(detector.is_colliding_with_pedestal(p) for p in parts)
        if h_g or h_p:
            hit = True; hit_g = h_g; hit_p = h_p
            # (optional) find first offending link
            for p in parts:
                if detector.is_colliding_with_pedestal(p) or detector.is_colliding_with_ground(p):
                    hit_part = p; break
            print(f"[COLLISION @ step {s}/{n}] ground={hit_g} pedestal={hit_p}")
            if hit_part: print("hitted part:", hit_part)
            break

    # leave or restore
    if not leave_at_end:
        articulation.set_joint_positions(q_save)
        world.step(render=True)

    return not hit

def _half_extent_along_world_up(quat_wxyz, dims_xyz):
    q = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
    Rw = R.from_quat(q).as_matrix()
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    proj = np.abs(Rw.T @ world_up)
    return 0.5 * float(np.sum(proj * np.asarray(dims_xyz, dtype=float)))


def main():
    # Load the single recorded case
    raw = json.load(open(RAW_CASE_PATH))
    rec = raw[RAW_CASE_KEY]

    # Object world pose from the record
    obj_center = np.array(rec["object_pose_world"]["position"], dtype=float)
    obj_quat_wxyz = np.array(rec["object_pose_world"]["orientation_quat"], dtype=float)

    # Derive pedestal center so its top touches the object's bottom face
    half_up = _half_extent_along_world_up(obj_quat_wxyz, BOX_DIMS)
    ped_top_z = float(obj_center[2] - half_up)
    ped_center = np.array([obj_center[0], obj_center[1], ped_top_z - 0.5 * PEDESTAL_DIMS[2]], dtype=float)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
    pedestal, cube = build_scene(world, ped_center, obj_center, obj_quat_wxyz)
    # Add robot BEFORE reset to ensure proper initialization
    robot = add_franka(world)
    world.reset()
    world.step(render=True)
    # Setup kinematics AFTER reset
    # Capture a canonical "home" joint pose for deterministic sweeps
    HOME_Q = np.array(robot.get_joint_positions(), dtype=float)

    # Setup kinematics AFTER reset
    kin = setup_kinematics(robot)

    # Sanity: ensure starting pose is collision-free (warn if not)
    robot.set_joint_positions(HOME_Q); world.step(render=True)
    if check_robot_collision(
        GroundCollisionDetector(omni.usd.get_context().get_stage(), non_colliding_part="/World/panda/panda_link0"),
        [
            "/World/panda/panda_link1","/World/panda/panda_link2","/World/panda/panda_link3",
            "/World/panda/panda_link4","/World/panda/panda_link5","/World/panda/panda_link6",
            "/World/panda/panda_link7","/World/panda/panda_hand",
            "/World/panda/panda_leftfinger","/World/panda/panda_rightfinger",
        ]
    )[0]:
        print("[WARN] HOME_Q collides at start; results may be pessimistic.")

    # --- Collision detector setup (mirrors sim_test.py style) ---
    stage = omni.usd.get_context().get_stage()
    collision_detector = GroundCollisionDetector(stage, non_colliding_part="/World/panda/panda_link0")
    # Ground plane at z=0
    MARGIN = 0.0015
    collision_detector.create_virtual_ground(
        size_x=20.0,
        size_y=20.0,
        position=Gf.Vec3f(0, 0, 0)
    )
    # Pick pedestal collider
    collision_detector.create_virtual_pedestal(
        position=Gf.Vec3f(float(ped_center[0]), float(ped_center[1]), float(ped_center[2])),
        size_x=float(PEDESTAL_DIMS[0]),
        size_y=float(PEDESTAL_DIMS[1]),
        size_z=float(PEDESTAL_DIMS[2])
    )
    # Robot parts to check
    robot_parts_to_check = [
        "/World/panda/panda_link1",
        "/World/panda/panda_link2",
        "/World/panda/panda_link3",
        "/World/panda/panda_link4",
        "/World/panda/panda_link5",
        "/World/panda/panda_link6",
        "/World/panda/panda_link7",
        "/World/panda/panda_hand",
        "/World/panda/panda_leftfinger",
        "/World/panda/panda_rightfinger",
    ]

    # Use recorded contact, pregrasp, and lift directly
    contact_world_pos = np.array(rec["grasp_pose_contact_world"]["position"], dtype=float)
    contact_world_quat = np.array(rec["grasp_pose_contact_world"]["orientation_quat"], dtype=float)
    pregrasp_pos = np.array(rec["pregrasp_world"]["position"], dtype=float)
    pregrasp_quat = np.array(rec["pregrasp_world"]["orientation_quat"], dtype=float)
    lift_pos = np.array(rec["lift_world"]["position"], dtype=float)
    lift_quat = np.array(rec["lift_world"]["orientation_quat"], dtype=float)

    # Frames
    draw_frame(contact_world_pos, contact_world_quat)
    draw_frame(pregrasp_pos, pregrasp_quat)

    # Diagnostics
    print("=== Diagnostics ===")
    print(f"Pedestal top z = {ped_top_z:.6f}")
    print(f"Contact C.z    = {float(contact_world_pos[2]):.6f}")
    print(f"Pregrasp P.z   = {float(pregrasp_pos[2]):.6f}")
    print(f"P.z - ped_top  = {float(pregrasp_pos[2]) - ped_top_z:.6f}")

    # Execute looping P -> C -> L using joint-space sweep (verifies sweep logic)
    import copy
    tmp = copy.deepcopy(pregrasp_pos)
    tmp[2] += 0.2
    phases = [
        ("Lift#1",   tmp,  pregrasp_quat),  # from HOME to recorded lift
        ("P", pregrasp_pos, pregrasp_quat),
        ("C", contact_world_pos, contact_world_quat),
        ("Lift#2",   lift_pos,  lift_quat),  # from HOME to recorded lift
    ]
    phase_idx = 0
    try:
        while simulation_app.is_running():
            # Start every cycle from REST
            robot.set_joint_positions(HOME_Q)
            world.step(render=True)

            # Run the 4 legs as a single chained route
            for i, (label, tgt_pos, tgt_quat) in enumerate(phases):
                ok = sweep_to(
                    world, robot, kin, tgt_pos, tgt_quat,
                    steps=120,
                    detector=collision_detector,
                    parts=robot_parts_to_check,
                    q_start=(HOME_Q if i == 0 else None),  # only the FIRST leg starts from REST
                    leave_at_end=True                        # keep end pose to chain to the next leg
                )
                print(f"[SWEEP] {label}: {ok}")
                if not ok:
                    print(f"[ROUTE FAILED] at segment: {label}")
                    break

            # brief pause at the end of the route (optional)
            for _ in range(15):
                world.step(render=True)

            print(f"object_pose_world: {{'position': {obj_center.tolist()}, 'orientation_quat': {obj_quat_wxyz.tolist()}}}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
