"""
Visualize pairs from a processed pairs file (built by build_rich_pairs).

- Input: a processed JSON with keys:
    grasp_poses, initial_object_poses, final_object_poses,
    success_labels, collision_labels
- For the first K pairs, spawn two robots per pair:
    - Left robot targets the pick (initial) grasp pose with the initial object pose
    - Right robot targets the transferred grasp pose on the final object pose
      computed as T_world_final = T_world_obj_final @ (T_obj_init^-1 @ T_world_hand_pick)
"""

import os
import re
import json
import math
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R

from isaacsim import SimulationApp

CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 1 << 0 | 1 << 3 | 1 << 10 | 1 << 13 | 1 << 14,
    "physics_dt": 1.0 / 60.0,
    "rendering_dt": 1.0 / 30.0,
}
simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader


# Layout: pairs far apart; two robots within a pair are close together
PAIR_COLS = 3
PAIR_ROWS = 2
PAIR_SPACING_X = 4.0
PAIR_SPACING_Y = 3.0
INTRA_PAIR_DX = 0.6
FRAME_SCALE = 0.06
ROBOT_BASE_OFFSET = np.array([-0.35, 0.0, 0.0], dtype=float)


def parse_dims_from_filename(path: str) -> List[float]:
    m = re.search(r"reordered_object_([0-9.]+)_([0-9.]+)_([0-9.]+)\.json$", os.path.basename(path))
    if not m:
        m = re.search(r"object_([0-9.]+)_([0-9.]+)_([0-9.]+)\.json$", os.path.basename(path))
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    return [0.1, 0.1, 0.1]


def add_frame_prim(path: str, scale: float = FRAME_SCALE) -> XFormPrim:
    assets = get_assets_root_path()
    add_reference_to_stage(assets + "/Isaac/Props/UIElements/frame_prim.usd", path)
    return XFormPrim(path, scale=[float(scale), float(scale), float(scale)])


def add_franka(world: World, prim_path: str, name: str) -> Articulation:
    assets = get_assets_root_path()
    add_reference_to_stage(assets + "/Isaac/Robots/Franka/franka.usd", prim_path)
    robot = Articulation(prim_path, name=name)
    world.scene.add(robot)
    return robot

def setup_lighting():
    import omni
    from pxr import UsdLux, UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()

    def mk_distant(name, intensity, rot_xyz):
        path = f"/World/Lights/{name}"
        light = UsdLux.DistantLight.Define(stage, path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(*[float(v) for v in rot_xyz]))
        return light

    mk_distant("Key",   2000.0, (45.0,   0.0,  0.0))
    mk_distant("Fill1", 1800.0, (-35.0, 35.0, 0.0))
    mk_distant("Fill2", 1500.0, (35.0, -35.0, 0.0))
    mk_distant("SideR", 1200.0, (0.0,   90.0, 0.0))
    mk_distant("SideL", 1200.0, (0.0,  -90.0, 0.0))


def setup_kinematics(articulation: Articulation) -> ArticulationKinematicsSolver:
    cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula = LulaKinematicsSolver(**cfg)
    return ArticulationKinematicsSolver(articulation, lula, "panda_hand")


def load_processed_pairs(path: str, max_pairs: int, start_index: int = 0) -> List[dict]:
    with open(path, "r") as f:
        d = json.load(f)
    total = len(d.get("grasp_poses", []))
    start = max(0, min(int(start_index), total))
    end = max(start, min(start + int(max_pairs), total))
    out = []
    for i in range(start, end):
        out.append({
            "grasp": d["grasp_poses"][i],
            "init_obj": d["initial_object_poses"][i],
            "final_obj": d["final_object_poses"][i],
            "success": float(d["success_labels"][i]),
            "collision": float(d["collision_labels"][i]),
        })
    return out


def pose_wxyz_to_T(pose_wxyz: List[float]) -> np.ndarray:
    t = np.array(pose_wxyz[:3], dtype=float)
    qw, qx, qy, qz = map(float, pose_wxyz[3:7])
    Rw = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rw
    T[:3, 3] = t
    return T


def T_to_pose_wxyz(T: np.ndarray) -> List[float]:
    Rm = T[:3, :3]
    t = T[:3, 3]
    qx, qy, qz, qw = R.from_matrix(Rm).as_quat()
    return [float(t[0]), float(t[1]), float(t[2]), float(qw), float(qx), float(qy), float(qz)]


def main():
    # User config: path to processed pairs file and how many pairs to visualize
    processed_path = \
        "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/processed_data/processed_reordered_object_0.1_0.18_0.055.json"
    max_pairs = 5
    start_index = 130000  # set this to skip earlier pairs (e.g., 100)

    dims_xyz = np.array(parse_dims_from_filename(processed_path), dtype=float)
    # Load a larger candidate window, then downselect 1 per bucket
    candidates = load_processed_pairs(processed_path, max_pairs * 200, start_index)

    # Bucket helpers
    def face_id(qw,qx,qy,qz):
        rot = R.from_quat([qx,qy,qz,qw]); up = np.array([0,0,1])
        faces = {1:[0,0,1],2:[1,0,0],3:[0,0,-1],4:[-1,0,0],5:[0,-1,0],6:[0,1,0]}
        return max(faces, key=lambda k: np.dot(rot.apply(faces[k]), up))

    ADJ = {1:{2,4,5,6}, 2:{1,3,5,6}, 3:{2,4,5,6}, 4:{1,3,5,6}, 5:{1,2,3,4}, 6:{1,2,3,4}}

    def bucket_of(pair):
        qi = pair["init_obj"][3:7]; qf = pair["final_obj"][3:7]
        fi = face_id(*qi); fj = face_id(*qf)
        if fi == fj:
            dq = R.from_quat(qf[1:]+qf[:1]) * R.from_quat(qi[1:]+qi[:1]).inv()
            ang = dq.magnitude() * 180.0 / math.pi
            if ang < 1:   return "SAME"
            if 1 <= ang <= 30: return "SMALL"
            if 30 < ang <= 90: return "MEDIUM"
            return "MEDIUM"
        return "ADJACENT" if fj in ADJ[fi] else "OPPOSITE"

    buckets = {b: [] for b in ("SAME","SMALL","MEDIUM","ADJACENT","OPPOSITE")}
    for p in candidates:
        b = bucket_of(p)
        if b in buckets:
            buckets[b].append(p)

    order = ["SAME","SMALL","MEDIUM","ADJACENT","OPPOSITE"]
    pairs = []
    for b in order:
        if buckets[b]:
            pairs.append(buckets[b][0])
    # If not enough buckets available, fill from remaining candidates
    if len(pairs) < max_pairs:
        leftovers = []
        for b in order:
            leftovers.extend(buckets[b][1:])
        need = max_pairs - len(pairs)
        pairs.extend(leftovers[:need])

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    set_camera_view(eye=[14.0, 10.0, 7.0], target=[0.0, 0.0, 0.4])
    setup_lighting()
    world.reset()

    robots: List[Articulation] = []
    ik_solvers: List[ArticulationKinematicsSolver] = []
    targets: List[XFormPrim] = []

    for k, s in enumerate(pairs):
        # Compute transferred final grasp pose from initial hand/object
        T_wh_pick = pose_wxyz_to_T(s["grasp"])           # hand@world (pick)
        T_wo_init = pose_wxyz_to_T(s["init_obj"])        # obj_init@world
        T_oh = np.linalg.inv(T_wo_init) @ T_wh_pick        # hand@obj_init
        T_wo_final = pose_wxyz_to_T(s["final_obj"])      # obj_final@world
        T_wh_final = T_wo_final @ T_oh                     # transferred hand@world

        # Base position for this pair (grid over pairs)
        pair_row = k // PAIR_COLS
        pair_col = k % PAIR_COLS
        pair_base = np.array([pair_col * PAIR_SPACING_X, -pair_row * PAIR_SPACING_Y, 0.0], dtype=float)

        # Two robots within a pair: left and right, placed close to each other
        for m, (obj_pose, hand_pose) in enumerate(((s["init_obj"], T_to_pose_wxyz(T_wh_pick)),
                                                   (s["final_obj"], T_to_pose_wxyz(T_wh_final)))):
            idx = 2 * k + m
            intra = np.array([(-INTRA_PAIR_DX if m == 0 else INTRA_PAIR_DX), 0.0, 0.0], dtype=float)
            cell_xy = pair_base + intra

            # Object
            obj_pos = np.array(obj_pose[:3], dtype=float) + cell_xy
            obj_quat = np.array(obj_pose[3:7], dtype=float)  # wxyz
            cube = VisualCuboid(
                prim_path=f"/World/Object_{idx:02d}",
                name=f"object_{idx:02d}",
                position=np.zeros(3),
                scale=dims_xyz,
                color=np.array([0.8, 0.8, 0.8], dtype=float),
            )
            world.scene.add(cube)
            cube.set_world_pose(obj_pos, obj_quat)

            # Target frame (grasp)
            g_pos = np.array(hand_pose[:3], dtype=float) + cell_xy
            g_quat = np.array(hand_pose[3:7], dtype=float)  # wxyz
            tgt = add_frame_prim(f"/World/target_{idx:02d}")
            tgt.set_world_pose(g_pos, g_quat)

            # Robot
            robot = add_franka(world, f"/World/panda_{idx:02d}", name=f"franka_robot_{idx:02d}")
            base_pose = np.array([obj_pos[0], obj_pos[1], 0.0], dtype=float) + ROBOT_BASE_OFFSET
            robot.set_world_pose(base_pose, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
            kin = setup_kinematics(robot)

            robots.append(robot)
            ik_solvers.append(kin)
            targets.append(tgt)

            # Defaults
            cube.set_default_state(obj_pos, obj_quat)
            tgt.set_default_state(g_pos, g_quat)
            XFormPrim(robot.prim_path).set_default_state(base_pose, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    world.reset()
    try:
        while simulation_app.is_running():
            for robot, kin, tgt in zip(robots, ik_solvers, targets):
                tgt_pos, tgt_quat = tgt.get_world_pose()
                kin._kinematics_solver.set_robot_base_pose(*robot.get_world_pose())
                action, ok = kin.compute_inverse_kinematics(np.array(tgt_pos, dtype=float), np.array(tgt_quat, dtype=float))
                if ok:
                    robot.apply_action(action)
            world.step(render=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
