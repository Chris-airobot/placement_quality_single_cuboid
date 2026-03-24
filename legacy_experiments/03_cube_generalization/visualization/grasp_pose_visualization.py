"""
Grasp pose visualizer (positions from generate_contact_metadata + surface-frame orientations)

- Positions: generate_contact_metadata(dims) -> p_local; per instance: p_world = R_init @ p_local + t_obj
- Orientation per contact (no yaw/pitch tilts):
    Z_tool = -n_f (approach into object)
    X_tool = shorter in-plane edge (auto-pick: u_f or v_f by dims)
    Y_tool = Z × X
  Orthonormalize and export as quaternion (wxyz).

- One robot + one object per contact (up to 15, bottom face excluded).
- Robot goes to contact pose (with a small local -Z offset via local_transform).
"""

import numpy as np
import os
import json
from typing import List, Tuple, Iterable, Optional, Dict

from isaacsim import SimulationApp
import math
import argparse

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

# Isaac/Omni imports (after SimulationApp)
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid, VisualSphere, VisualCylinder
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from placement_quality.ycb_simulation.utils.helper import local_transform

# Pose/contact utilities from your grasp code

from placement_quality.cube_generalization.grasp_pose_generator import (
    generate_grasp_poses_including_bottom_updated,
    six_face_up_orientations,
    pose_from_R_t
)


# -----------------------------
# Tunables
# -----------------------------
PEDESTAL_RADIUS = 0.08
PEDESTAL_HEIGHT = 0.10
PEDESTAL_CENTER_Z = 0.05
PEDESTAL_TOP_Z = PEDESTAL_CENTER_Z + 0.5 * PEDESTAL_HEIGHT

INIT_SPIN_DEG = 0
EDGE_MARGIN = 0.005  # kept for aesthetic spacing only (meta already uses internal fractions)

GRID_COLS = 5
GRID_ROWS = 3
CELL_SPACING_X = 1.6
CELL_SPACING_Y = 1.4

ROBOT_BASE_OFFSET = np.array([-0.35, 0.0, 0.0])  # ~35 cm left of its object
FRAME_SCALE = 0.06
SPHERE_RADIUS = 0.010

# -----------------------------
# Clearance and gripper constraints (match experiment_generation.py)
# -----------------------------
WORLD_UP = np.array([0.0, 0.0, 1.0])
MIN_CLEARANCE = 0.02
PALM_DEPTH = 0.038
# Effective finger half-thickness near the palm contact (tuned to Panda)
FINGER_THICK = 0.03
GRIPPER_OPEN_MAX = 0.08
MIN_PALM_CLEAR = 0.01
# Match the runtime offset we command in the control loop
APPROACH_BACKOFF = 0.09

# Toggle to quickly enable/disable clearance filters
ENABLE_CLEARANCE_FILTERS = False

# Tilt controls (no yaw). 90° means no change relative to surface frame
ENABLE_TILT = True
TILT_DEG = 75.0

# Yaw controls (about local Z/approach axis)
ENABLE_YAW = True
YAW_DEG = 15.0

# Roll controls (about local Y/binormal axis)
ENABLE_ROLL = True
ROLL_DEG = 15.0

# Hand-to-TCP offset and optional extra insert along local -Z
HAND_TO_TCP_Z = 0.1034
EXTRA_INSERT = -0.0334

# -----------------------------
# 1) Contact position sampler (your function)
# -----------------------------
def generate_contact_metadata(
    dims_xyz: np.ndarray,
    approach_offset: float = 0.01,
    u_fracs_long: Tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80),
    v_fracs_short: Tuple[float, ...] = (0.35, 0.50, 0.65),
    include_faces: Optional[Tuple[str, ...]] = None,
) -> List[Dict]:
    """
    Generate a 2-D lattice of local contact points per face of a cuboid.

    - For each face, define in-plane axes ej (index j) and ek (index k).
    - 'u' moves along the *longer* in-plane side; 'v' moves along the *shorter* side.
    - Closing axis 'axis' is the unit vector along the shorter in-plane side.
    - Approach is -normal (into the object), so Z_tool aligns with approach later.

    Returns a list of dicts with keys:
      'face', 'u_frac', 'v_frac', 'fraction' (alias of the long-side frac for backward-compat),
      'p_local' (3,), 'approach' (3,), 'binormal' (3,), 'normal' (3,), 'axis' (3,)
    """
    metadata: List[Dict] = []

    # Face index mapping: (i, j, k) => normal axis index, two in-plane axis indices
    face_axes = {
        '+X': (0, 1, 2), '-X': (0, 1, 2),
        '+Y': (1, 0, 2), '-Y': (1, 0, 2),
        '+Z': (2, 0, 1), '-Z': (2, 0, 1),
    }

    dims_xyz = np.asarray(dims_xyz, dtype=float)
    half = 0.5 * dims_xyz
    I = np.eye(3)

    for face, (i, j, k) in face_axes.items():
        if include_faces is not None and face not in include_faces:
            continue

        sign = 1.0 if face[0] == '+' else -1.0
        normal = sign * I[i]                  # outward face normal
        approach = -normal                    # tool approaches into the face

        ej, ek = I[j], I[k]                   # in-plane unit axes (object frame)
        dj, dk = float(dims_xyz[j]), float(dims_xyz[k])

        # Decide which in-plane side is long vs short
        long_axis, long_len, short_axis, short_len = (ej, dj, ek, dk) if dj >= dk else (ek, dk, ej, dj)

        # Closing axis is along the *shorter* in-plane side
        axis = short_axis / (np.linalg.norm(short_axis) + 1e-12)

        # Binormal completes the orthogonal set in the face plane
        binormal = np.cross(approach, axis)
        binormal /= (np.linalg.norm(binormal) + 1e-12)

        # Base point: slightly outside the surface along the face normal
        base = normal * (half[i] + float(approach_offset))

        # 2-D lattice over the face (u along long side, v along short side)
        for u in u_fracs_long:
            for v in v_fracs_short:
                # Convert center-biased fractions to offsets in meters
                offset_u = (float(u) - 0.5) * long_len
                offset_v = (float(v) - 0.5) * short_len

                # Build contact point in the object frame
                p_local = base + long_axis * offset_u + short_axis * offset_v

                # Back-compat 'fraction' = frac along the long side (old code used 1-D along long)
                frac_alias = float(u)

                metadata.append({
                    'face': face,
                    'u_frac': float(u),
                    'v_frac': float(v),
                    'fraction': frac_alias,          # backward compatibility
                    'p_local': p_local.astype(float),
                    'approach': approach.astype(float),
                    'binormal': binormal.astype(float),
                    'normal': normal.astype(float),
                    'axis': axis.astype(float),      # closing axis (short side)
                })

    return metadata


# -----------------------------
# 2) Surface-frame orientation (no tilts), then map to world
# -----------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n)

def _quat_wxyz_from_R(Rm: np.ndarray) -> np.ndarray:
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

_FACE_BASIS = {
    '+X': (np.array([0.,1.,0.]), np.array([0.,0.,1.]), np.array([1.,0.,0.])),
    '-X': (np.array([0.,0.,1.]), np.array([0.,1.,0.]), -np.array([1.,0.,0.])),
    '+Y': (np.array([0.,0.,1.]), np.array([1.,0.,0.]), np.array([0.,1.,0.])),
    '-Y': (np.array([1.,0.,0.]), np.array([0.,0.,1.]), -np.array([0.,1.,0.])),
    '+Z': (np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])),
    '-Z': (np.array([0.,1.,0.]), np.array([1.,0.,0.]), -np.array([0.,0.,1.])),
}

def face_surface_frame(face: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        u_f, v_f, n_f = _FACE_BASIS[face]
    except KeyError:
        raise ValueError(f"Unknown face label: {face}")
    return u_f.copy(), v_f.copy(), n_f.copy()

def build_tool_orientation_from_meta(
    meta: dict,
    R_init_obj_to_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the exact axis chosen during contact generation:
      - Z_tool (approach)  = -n_f  (into the object)
      - X_tool (closing)   = meta['axis'] (shorter in-plane edge, already decided)
      - Y_tool (binormal)  = Z × X
    Then map to world with R_init.
    """
    # 1) Pull needed vectors (OBJECT frame)
    face        = meta["face"]
    n_f_obj     = face_surface_frame(face)[2]       # outward normal in object frame
    z_loc       = -n_f_obj                           # approach into object
    x_raw       = np.array(meta["axis"], dtype=float)

    # 2) Ensure X is in the face plane (it should be; project just in case)
    x_loc = x_raw - np.dot(x_raw, z_loc) * z_loc     # remove any normal component
    nrm = np.linalg.norm(x_loc)
    if nrm < 1e-12:
        # Fallback: pick something in-plane (rare unless meta['axis'] was wrong)
        u_f, v_f, _ = face_surface_frame(face)
        x_loc = u_f if np.linalg.norm(u_f) > np.linalg.norm(v_f) else v_f
    x_loc = x_loc / np.linalg.norm(x_loc)

    # Panda closes along local +Y. Make Y = short edge (meta['axis']),
    # Z = -n_f, and X = Y × Z (right-handed).
    y_raw = np.array(meta["axis"], dtype=float)
    y_loc = y_raw - np.dot(y_raw, z_loc) * z_loc
    nrm = np.linalg.norm(y_loc)
    if nrm < 1e-12:
        u_f, v_f, _ = face_surface_frame(face)
        y_loc = u_f if np.linalg.norm(u_f) < np.linalg.norm(v_f) else v_f
    y_loc = y_loc / np.linalg.norm(y_loc)

    # 2) Complete a right-handed frame
    x_loc = np.cross(y_loc, z_loc); x_loc /= np.linalg.norm(x_loc)
    y_loc = np.cross(z_loc, x_loc); y_loc /= np.linalg.norm(y_loc)

    # Make sure X matches the stored axis direction (robust, no ad-hoc flips)
    x_meta = np.array(meta["axis"], dtype=float)
    if np.dot(x_loc, x_meta) < 0.0:
        x_loc *= -1.0
        y_loc *= -1.0     # keep right-handedness

    print("[BUILD]", meta["face"],
      "dot(X,metaX)=", np.round(np.dot(x_loc, np.array(meta["axis"])), 3),
      "dot(Y,metaY)=", np.round(np.dot(y_loc, np.array(meta["binormal"])), 3),
      "handed=", np.dot(np.cross(x_loc, y_loc), z_loc))
    # 4) Map to WORLD
    x_w = R_init_obj_to_world @ x_loc
    y_w = R_init_obj_to_world @ y_loc
    z_w = R_init_obj_to_world @ z_loc

    # 5) Pack rotation (columns = axes) and enforce det=+1
    R_tool = np.column_stack([x_w, y_w, z_w])
    if np.dot(np.cross(x_w, y_w), z_w) < 0.0:
        y_w *= -1.0
        R_tool = np.column_stack([x_w, y_w, z_w])


    print("[BUILD]", face,
      " X", np.round(x_loc,3),
      " Y", np.round(y_loc,3),
      " Z", np.round(z_loc,3),
      " handed=", np.dot(np.cross(x_loc, y_loc), z_loc))



    

    # 6) Quaternion [w,x,y,z]
    q_xyzw = R.from_matrix(R_tool).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    return R_tool, q_wxyz

# -----------------------------
# Clearance helpers
# -----------------------------
def compute_jaw_span_for_axis(axis_obj: np.ndarray, dims_xyz: np.ndarray) -> float:
    half_extents = 0.5 * np.asarray(dims_xyz, dtype=float)
    return float(2.0 * np.sum(np.abs(axis_obj) * half_extents))

def passes_gripper_width(meta: dict, dims_xyz: np.ndarray, max_open: float = GRIPPER_OPEN_MAX) -> bool:
    axis_obj = np.asarray(meta["axis"], dtype=float)
    jaw_span = compute_jaw_span_for_axis(axis_obj, dims_xyz)
    return jaw_span <= float(max_open)

def passes_clearance(p_world: np.ndarray,
                     ped_top_z: float,
                     R_tool: np.ndarray,
                     min_contact_clear: float = MIN_CLEARANCE,
                     min_palm_clear: float = MIN_PALM_CLEAR,
                     palm_depth: float = PALM_DEPTH,
                     approach_backoff: float = APPROACH_BACKOFF,
                     finger_thick: float = FINGER_THICK) -> bool:
    # Contact itself above pedestal
    if (float(p_world[2]) - float(ped_top_z)) < float(min_contact_clear):
        return False
    # Palm-bottom clearance at backed-off target pose
    z_w = R_tool[:, 2]
    y_w = R_tool[:, 1]
    effective_palm_depth = max(float(palm_depth) - float(approach_backoff), 0.0)
    palm_center_z = float(p_world[2]) - effective_palm_depth * float(z_w[2])
    palm_bottom_z = palm_center_z - float(finger_thick) * abs(float(y_w[2]))
    return (palm_bottom_z - float(ped_top_z)) >= float(min_palm_clear)

# -----------------------------
# 3) Small viz/robot helpers
# -----------------------------
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

def ensure_sphere(world: World, prim_path: str, radius: float, color_rgb=(1.0, 0.2, 0.2)) -> VisualSphere:
    obj = world.scene.get_object(prim_path)
    if obj is not None:
        return obj
    obj = VisualSphere(
        prim_path=prim_path,
        name=prim_path.split('/')[-1],
        position=np.zeros(3),
        radius=float(radius),
        color=np.array(color_rgb, dtype=float),
    )
    world.scene.add(obj)
    return obj

def ensure_cuboid(world: World, prim_path: str, size_xyz: np.ndarray, color_rgb=(0.8, 0.8, 0.8)) -> VisualCuboid:
    obj = world.scene.get_object(prim_path)
    if obj is not None:
        return obj
    obj = VisualCuboid(
        prim_path=prim_path,
        name=prim_path.split('/')[-1],
        position=np.ones(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
        scale=np.array(size_xyz, dtype=float),
        color=np.array(color_rgb, dtype=float),
    )
    world.scene.add(obj)
    return obj

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

def setup_kinematics(articulation: Articulation) -> ArticulationKinematicsSolver:
    cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula = LulaKinematicsSolver(**cfg)
    print(lula.get_all_frame_names())
    return ArticulationKinematicsSolver(articulation, lula, "panda_hand")

# -----------------------------
# 4) Main
# -----------------------------
def main(args):
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    setup_lighting()
    set_camera_view(eye=[16.0, 13.0, 8.5], target=[0.0, 0.0, 0.4])
    world.reset()

    # Object dims and orientation
    # dims_xyz = np.array(sample_dims(1, seed=77)[0], dtype=float) * 1.5
    dims_xyz = np.array([0.143, 0.0915, 0.051], dtype=float)
    R_init = six_face_up_orientations(spin_degs=(INIT_SPIN_DEG,))["+Z"][0]

    # Base pedestal top (reference height for first instance)
    t_base = np.array([0.0, 0.0, PEDESTAL_TOP_Z], dtype=float)

    # Export all grasps in OBJECT frame (call with identity pose)
    local_R = np.eye(3, dtype=float)
    local_t = np.zeros(3, dtype=float)
    grasps_object_frame = generate_grasp_poses_including_bottom_updated(
        dims_xyz=dims_xyz,
        R_obj_to_world=local_R,
        t_obj_world=local_t,
        include_faces=('+X','-X','+Y','-Y'),
        # Straight grasps only (no yaw/tilt/roll)
        yaw_set=(0.0,),
        tilt_set=(0.0,),
        roll_set=(0.0,),
        # 10 contacts per face: 5 along long side x 2 along short side
        # (keep default long-side samples; reduce short-side to 2 center-biased samples)
        u_fracs_long=(0.20, 0.35, 0.50, 0.65, 0.80),
        v_fracs_short=(0.45, 0.55),
    )
    # === NEW: save in a format compatible with your loader (numeric string keys) ===
    grasp_lib = {}
    for idx, g in enumerate(grasps_object_frame):
        # Because R_obj_to_world is identity here, these are OBJECT-frame values
        pos_obj = (g.get('p_local', g['contact_position_world']))
        quat_obj_wxyz = g['tool_quaternion_wxyz']

        grasp_lib[str(idx)] = {
            # Back-compat fields your code already uses:
            "position": np.asarray(pos_obj, dtype=float).tolist(),
            "orientation_wxyz": np.asarray(quat_obj_wxyz, dtype=float).tolist(),

            # Extra metadata for smarter pairing / analysis (safe to ignore in runtime):
            "face": g.get("face"),
            "u_frac": g.get("u_frac"),
            "v_frac": g.get("v_frac"),
            "axis_obj": np.asarray(g.get("axis_obj"), dtype=float).tolist() if g.get("axis_obj") is not None else None,
            "angles_deg": g.get("angles_deg"),
            "dims_xyz": np.asarray(dims_xyz, dtype=float).tolist(),  # duplicated per entry to avoid non-numeric top keys
            "version": 2
        }

    out_path = "/home/chris/Chris/placement_ws/src/40_grasps_meta_data.json"  # <-- set this to your desired file path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(grasp_lib, f, indent=2)
    print(f"Saved {len(grasp_lib)} grasps to {out_path}")



    export_data = {}
    for idx, g in enumerate(grasps_object_frame, start=1):
        p = list(map(float, np.asarray(g['contact_position_world'], dtype=float).tolist()))
        q = list(map(float, np.asarray(g['tool_quaternion_wxyz'], dtype=float).tolist()))
        export_data[str(idx)] = {
            "position": p,
            "orientation_wxyz": q,
        }
    out_path = "/home/chris/Chris/placement_ws/src/40_grasps.json"
    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"Saved {len(export_data)} object-frame grasps to {out_path}")

        # === Visualization setup: single object + single robot ===

    # Put one pedestal at the origin in XY
    cell_xy = np.array([0.0, 0.0], dtype=float)
    ped_center = np.array([cell_xy[0], cell_xy[1], PEDESTAL_CENTER_Z], dtype=float)
    ped_top_z  = float(ped_center[2] + 0.5 * PEDESTAL_HEIGHT)

    # Object pose: resting on the pedestal
    half_extents = 0.5 * dims_xyz
    h_z = float(np.sum(np.abs(R_init[2, :]) * half_extents))  # projected half-height along world Z
    t_obj = np.array([cell_xy[0], cell_xy[1], ped_top_z + h_z], dtype=float)
    pos_obj, quat_obj = pose_from_R_t(R_init, t_obj)

    # === Build contacts from generate_contact_metadata (multiple positions per face) ===
    # Bottom face excluded, side faces only
    meta_all = generate_contact_metadata(
        dims_xyz=dims_xyz,
        approach_offset=0.01,
        include_faces=("+X", "-X", "+Y", "-Y"),
        # u_fracs_long / v_fracs_short defaults give 5 x 3 = 15 positions per face
    )

    grasps_all: List[Dict] = []
    for meta in meta_all:
        # Local contact position (object frame)
        p_local = np.asarray(meta["p_local"], dtype=float)  # (3,)

        # World contact position: p_world = R_init @ p_local + t_obj
        p_world = R_init @ p_local + t_obj

        # Tool orientation from surface frame + short edge axis
        R_tool, q_wxyz = build_tool_orientation_from_meta(
            meta,
            R_init_obj_to_world=R_init,
        )

        # Face normal in world (for the cos(theta) correction later)
        normal_obj = np.asarray(meta["normal"], dtype=float)
        face_normal_world = R_init @ normal_obj

        grasps_all.append(
            {
                "contact_position_world": p_world,
                "tool_quaternion_wxyz": q_wxyz,
                "tool_rotation": R_tool,
                "face_normal_world": face_normal_world,
                "face": meta["face"],
                "u_frac": meta["u_frac"],
                "v_frac": meta["v_frac"],
                "p_local": p_local,
            }
        )

    total = len(grasps_all)
    num_to_show = total if int(args.num_grasps) < 0 else min(int(args.num_grasps), total)
    grasps_all = grasps_all[:num_to_show]
    print(f"Contacts available: {total}; will cycle through: {len(grasps_all)}")

    if len(grasps_all) == 0:
        print("WARNING: grasps_all is empty – nothing to visualize.")
        # Keep going so you at least see the object and robot idle.

    # Pre-cache face normals for IK offset (same length/order as grasps_all)
    face_normals_world: List[np.ndarray] = [
        np.asarray(g["face_normal_world"], dtype=float) for g in grasps_all
    ]




    # --- Create scene prims (one of each) ---

    ped = VisualCylinder(
        prim_path="/World/Pedestal",
        name="pedestal",
        position=ped_center,
        radius=float(PEDESTAL_RADIUS),
        height=float(PEDESTAL_HEIGHT),
        color=np.array([0.6, 0.6, 0.6], dtype=float),
    )
    world.scene.add(ped)

    cube = ensure_cuboid(world, "/World/Object", dims_xyz, color_rgb=(0.8, 0.8, 0.8))
    cube.set_world_pose(np.array(pos_obj, dtype=float), np.array(quat_obj, dtype=float))

    contact_sphere = ensure_sphere(
        world,
        prim_path="/World/contacts/sphere",
        radius=SPHERE_RADIUS,
        color_rgb=(1.0, 0.2, 0.2),
    )

    contact_frame = add_frame_prim("/World/contacts/contact_frame")
    target = add_frame_prim("/World/target")

    robot = add_franka(world, "/World/panda", name="franka_robot")
    base_pose = np.array([t_obj[0], t_obj[1], 0.0], dtype=float) + ROBOT_BASE_OFFSET
    robot.set_world_pose(base_pose, np.array([1.0, 0.0, 0.0, 0.0]))
    kin = setup_kinematics(robot)

    # --- Small helper to apply a given grasp index to the visuals ---

    def apply_grasp(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        g = grasps_all[idx]
        p_world = np.array(g["contact_position_world"], dtype=float)
        q_wxyz = np.array(g["tool_quaternion_wxyz"], dtype=float)

        contact_sphere.set_world_pose(p_world, np.array([1.0, 0.0, 0.0, 0.0]))
        contact_frame.set_world_pose(p_world, q_wxyz)
        target.set_world_pose(p_world, q_wxyz)

        # For debugging
        face_lbl = g.get("face", "?")
        u_frac = g.get("u_frac", None)
        v_frac = g.get("v_frac", None)
        p_local = np.array(g.get("p_local", [np.nan, np.nan, np.nan]), dtype=float)
        print(
            f"[Grasp {idx:03d}] face={face_lbl} "
            f"p_local={np.round(p_local, 4).tolist()} "
            f"u={u_frac} v={v_frac}"
        )
        return p_world, q_wxyz

    # Initialize default states
    ped.set_default_state(ped_center, np.array([1.0, 0.0, 0.0, 0.0]))
    cube.set_default_state(pos_obj, quat_obj)

    # If there are grasps, set the first one as the default visual state
    if len(grasps_all) > 0:
        p0, q0 = apply_grasp(0)
    else:
        p0 = pos_obj
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    contact_sphere.set_default_state(p0, np.array([1.0, 0.0, 0.0, 0.0]))
    contact_frame.set_default_state(p0, q0)
    target.set_default_state(p0, q0)
    XFormPrim(robot.prim_path).set_default_state(
        base_pose, np.array([1.0, 0.0, 0.0, 0.0])
    )

    print(f"Valid contacts to cycle through: {len(grasps_all)}")
    world.reset()
    print("Starting simulation loop...")

        # --- Simulation loop: cycle through grasps one-by-one ---
    steps_per_grasp = 18  # ~3 seconds per grasp at 60Hz
    current_idx = 0        # which grasp is currently active
    frames_left = steps_per_grasp

    while simulation_app.is_running():
        if len(grasps_all) > 0:
            # When frames for this grasp are exhausted, switch to the next one
            if frames_left <= 0:
                current_idx = (current_idx + 1) % len(grasps_all)
                p_world, q_wxyz = apply_grasp(current_idx)
                frames_left = steps_per_grasp

            # IK to the current target
                       # IK to the current target
            tgt_pos, tgt_quat = target.get_world_pose()
            target_pose = tgt_pos.tolist() + tgt_quat.tolist()

            # Orientation-aware Z offset from panda_hand to TCP
            R_tgt = R.from_quat(
                [tgt_quat[1], tgt_quat[2], tgt_quat[3], tgt_quat[0]]
            ).as_matrix()
            z_tool_w = R_tgt[:, 2]
            n_face_w = face_normals_world[current_idx]

            cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_w)))
            z_local = -(float(HAND_TO_TCP_Z) + float(EXTRA_INSERT)) / cos_theta

            fixed_grasp_pose = local_transform(target_pose, [0.0, 0.0, float(z_local)])

            # --- Robustly handle local_transform output shape ---
            # Case 1: (pos, quat)
            if isinstance(fixed_grasp_pose, (tuple, list)) and len(fixed_grasp_pose) == 2:
                pos_part, quat_part = fixed_grasp_pose
                grasp_position = np.asarray(pos_part, dtype=float).reshape(3)
                grasp_orientation = np.asarray(quat_part, dtype=float).reshape(4)
            else:
                # Case 2: flat [x, y, z, w, x, y, z]
                fixed_arr = np.asarray(fixed_grasp_pose, dtype=float).ravel()
                if fixed_arr.shape[0] != 7:
                    raise RuntimeError(
                        f"local_transform returned shape {fixed_arr.shape}, expected 7 "
                        f"(either (pos, quat) or flat 7D pose)."
                    )
                grasp_position = fixed_arr[:3]
                grasp_orientation = fixed_arr[3:]

            base_p, base_q = robot.get_world_pose()
            kin._kinematics_solver.set_robot_base_pose(base_p, base_q)

            action, ok = kin.compute_inverse_kinematics(grasp_position, grasp_orientation)
            if ok:
                robot.apply_action(action)


            frames_left -= 1

        # Even if there are no grasps, still step the world so the window stays alive
        world.step(render=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp pose visualization")
    parser.add_argument("--num-grasps", type=int, default=36,
                        help="Number of grasps to visualize (-1 for all)")
    parser.add_argument("--grid-cols", type=int, default=6,
                        help="Number of columns in the grid layout")
    args = parser.parse_args()
    main(args)
