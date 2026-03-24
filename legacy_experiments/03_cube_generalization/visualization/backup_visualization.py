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
from typing import List, Tuple, Iterable

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
from placement_quality.cube_generalization.grasp_generator import (
    sample_dims,
    six_face_up_orientations,
    pose_from_R_t,
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
ENABLE_CLEARANCE_FILTERS = True

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
def generate_contact_metadata(dims: np.ndarray, approach_offset: float = 0.01, G_max: float = 0.08):
    """
    Computes up to 18 local contact points on a cuboid's 6 faces (3 per face),
    with metadata for approach, binormal, psi_max, and outward normal.
    """
    metadata = []
    face_axes = {
        '+X': (0,1,2), '-X': (0,1,2),
        '+Y': (1,0,2), '-Y': (1,0,2),
        '+Z': (2,0,1), '-Z': (2,0,1),
    }
    fractions = [0.25, 0.50, 0.75]
    half = dims * 0.5

    for face, (i, j, k) in face_axes.items():
        sign = 1 if face[0] == '+' else -1
        normal   = sign * np.eye(3)[i]      # outward face normal
        approach = -normal                  # gripper approach direction

        ej = np.eye(3)[j]
        ek = np.eye(3)[k]
        du, dv = dims[j], dims[k]
        long_vec, long_len = (ej, du) if du >= dv else (ek, dv)
        axis_vec = ek if du >= dv else ej  # shorter edge as closing axis
        axis = axis_vec / np.linalg.norm(axis_vec)
        binormal = np.cross(approach, axis)    # Z × X  (right-handed)
        binormal /= (np.linalg.norm(binormal) + 1e-12)
        # DEBUG: handedness check for the face basis you’re about to store
        triple = np.dot(np.cross(axis, binormal), approach)  # should be +1 after the fix
        print(f"[META] face={face:>2s} axis={axis} approach={approach} binormal={binormal}  (XxY)·Z={triple:+.0f}")


        base = normal * (half[i] + approach_offset)
        for frac in fractions:
            offset = (frac - 0.5) * long_len
            p_local = base + long_vec * offset
            metadata.append({
                'face':     face,
                'fraction': frac,
                'p_local':  p_local,
                'approach': approach,
                'binormal': binormal,
                'normal':   normal,
                'axis':     axis
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
        position=np.zeros(3),
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
def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    setup_lighting()
    set_camera_view(eye=[16.0, 13.0, 8.5], target=[0.0, 0.0, 0.4])
    world.reset()

    # Object dims and orientation
    # dims_xyz = np.array(sample_dims(1, seed=77)[0], dtype=float) * 1.5
    dims_xyz = np.array([0.111, 0.05, 0.149], dtype=float)
    R_init = six_face_up_orientations(spin_degs=(INIT_SPIN_DEG,))["+Z"][0]

    # Base pedestal top (reference height for first instance)
    t_base = np.array([0.0, 0.0, PEDESTAL_TOP_Z], dtype=float)

    # ---- positions from YOUR metadata (local) ----
    metadata = generate_contact_metadata(dims_xyz, approach_offset=0.01, G_max=0.08)
    # Exclude bottom face so we keep 5 faces × 3 = 15 contacts
    metadata = [m for m in metadata if m["face"] != "-Z"]

    # Filter: gripper opening limit using closing axis (shorter in-plane edge)
    if ENABLE_CLEARANCE_FILTERS:
        filtered = [m for m in metadata if passes_gripper_width(m, dims_xyz, GRIPPER_OPEN_MAX)]
        metadata = filtered
    metadata = metadata[: GRID_COLS * GRID_ROWS]

    print(f"Contacts (metadata) kept for visualization: {len(metadata)}")

    robots: List[Articulation] = []
    ik_solvers: List[ArticulationKinematicsSolver] = []
    targets: List[XFormPrim] = []

    colors = [
        (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8),
        (0.8, 0.8, 0.2), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8),
        (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.0, 1.0, 0.5),
        (1.0, 0.0, 0.5), (0.5, 1.0, 0.0), (0.0, 0.5, 1.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
    ]

    valid_contacts = 0
    face_normals_world: List[np.ndarray] = []
    for i, m in enumerate(metadata):
        row = i // GRID_COLS
        col = i % GRID_COLS
        cell_xy = np.array([col * CELL_SPACING_X, -row * CELL_SPACING_Y], dtype=float)

        # Per-cell pedestal geometry (compute top height for clearance checks)
        ped_center = np.array([cell_xy[0], cell_xy[1], PEDESTAL_CENTER_Z], dtype=float)
        ped_top_z  = float(ped_center[2] + 0.5 * PEDESTAL_HEIGHT)

        # Object pose at this cell (same rotation R_init)
        half_extents = 0.5 * dims_xyz
        h_z = float(np.sum(np.abs(R_init[2, :]) * half_extents))  # projected half-height onto world Z
        t_obj = np.array([cell_xy[0], cell_xy[1], ped_top_z + h_z], dtype=float)
        pos_obj, quat_obj = pose_from_R_t(R_init, t_obj)
        # (Optional) debug compare stored meta binormal with built frame
        x_chk = (R_init @ np.array(m["axis"], float))
        z_chk = (R_init @ (-face_surface_frame(m["face"])[2]))
        y_from_build = (R_init @ np.cross(z_chk/np.linalg.norm(z_chk), x_chk/np.linalg.norm(x_chk)))
        print("[CHECK]", m["face"], "meta_bin=", np.round(m["binormal"],3),
            "built_y=", np.round(y_from_build,3))


        # Contact position in WORLD for this instance
        p_local = np.array(m["p_local"], dtype=float)
        p_world = R_init @ p_local + t_obj

        # Orientation in WORLD from the face's surface frame, with optional tilt and yaw
        R_tool, q_wxyz = build_tool_orientation_from_meta(m, R_init)
        if ENABLE_TILT:
            # Tilt about local X (closing axis) by (TILT_DEG - 90°); 90° => identity
            tilt_rad = np.deg2rad(TILT_DEG - 90.0)
            if abs(tilt_rad) > 1e-9:
                R_tilt = R.from_rotvec(tilt_rad * R_tool[:, 0]).as_matrix()
                R_tool = R_tilt @ R_tool
        if ENABLE_ROLL:
            roll_rad = np.deg2rad(ROLL_DEG)
            if abs(roll_rad) > 1e-9:
                R_roll = R.from_rotvec(roll_rad * R_tool[:, 1]).as_matrix()
                R_tool = R_roll @ R_tool
        if ENABLE_YAW:
            yaw_rad = np.deg2rad(YAW_DEG)
            if abs(yaw_rad) > 1e-9:
                R_yaw = R.from_rotvec(yaw_rad * R_tool[:, 2]).as_matrix()
                R_tool = R_yaw @ R_tool
        # Refresh quaternion after any rotations
        q_xyzw = R.from_matrix(R_tool).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

        # Outward face normal in world frame
        n_face_world = R_init @ face_surface_frame(m["face"])[2]

        # Clearance checks (toggleable)
        if ENABLE_CLEARANCE_FILTERS:
            if not passes_clearance(p_world, ped_top_z, R_tool,
                                    MIN_CLEARANCE, MIN_PALM_CLEAR,
                                    PALM_DEPTH, APPROACH_BACKOFF, FINGER_THICK):
                continue

        # Now create visuals only for valid contacts
        ped = VisualCylinder(
            prim_path=f"/World/Pedestal_{i:02d}",
            name=f"pedestal_{i:02d}",
            position=ped_center,
            radius=float(PEDESTAL_RADIUS),
            height=float(PEDESTAL_HEIGHT),
            color=np.array([0.6, 0.6, 0.6], dtype=float),
        )
        world.scene.add(ped)

        cube = ensure_cuboid(world, f"/World/Object_{i:02d}", dims_xyz, color_rgb=colors[i % len(colors)])
        cube.set_world_pose(np.array(pos_obj, dtype=float), np.array(quat_obj, dtype=float))

        # Visual markers
        contact_sphere = ensure_sphere(world, f"/World/contacts/sphere_{i:02d}", SPHERE_RADIUS, (1.0, 0.2, 0.2))
        contact_sphere.set_world_pose(p_world, np.array([1.0, 0.0, 0.0, 0.0]))
        frame = add_frame_prim(f"/World/contacts/contact_{i:02d}")
        frame.set_world_pose(p_world, q_wxyz)

        # Robot + IK target
        robot = add_franka(world, f"/World/panda_{i:02d}", name=f"franka_robot_{i:02d}")
        base_pose = np.array([t_obj[0], t_obj[1], 0.0], dtype=float) + ROBOT_BASE_OFFSET
        robot.set_world_pose(base_pose, np.array([1.0, 0.0, 0.0, 0.0]))
        kin = setup_kinematics(robot)

        tgt = add_frame_prim(f"/World/target_{i:02d}")
        tgt.set_world_pose(p_world, q_wxyz)

        robots.append(robot)
        ik_solvers.append(kin)
        targets.append(tgt)
        face_normals_world.append(n_face_world)
        valid_contacts += 1

        # Default states (for clean reload)
        ped.set_default_state(ped_center, np.array([1.0, 0.0, 0.0, 0.0]))
        cube.set_default_state(pos_obj, quat_obj)
        frame.set_default_state(p_world, q_wxyz)
        tgt.set_default_state(p_world, q_wxyz)
        XFormPrim(robot.prim_path).set_default_state(base_pose, np.array([1.0, 0.0, 0.0, 0.0]))
        contact_sphere.set_default_state(p_world, np.array([1.0, 0.0, 0.0, 0.0]))

    print(f"Valid contacts visualized: {valid_contacts}")
    world.reset()
    print("Starting simulation loop...")

    try:
        while simulation_app.is_running():
            for i, (robot, kin, tgt) in enumerate(zip(robots, ik_solvers, targets)):
                tgt_pos, tgt_quat = tgt.get_world_pose()
                target_pose = tgt_pos.tolist() + tgt_quat.tolist()
                # Orientation-aware offset along local -Z (panda_hand -> TCP), corrected by cos(theta)
                R_tgt = R.from_quat([tgt_quat[1], tgt_quat[2], tgt_quat[3], tgt_quat[0]]).as_matrix()
                z_tool_w = R_tgt[:, 2]
                n_face_w = face_normals_world[i]
                cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_w)))
                z_local = -(float(HAND_TO_TCP_Z) + float(EXTRA_INSERT)) / cos_theta
                fixed_grasp_pose = local_transform(target_pose, [0.0, 0.0, float(z_local)])
                grasp_position = np.array(fixed_grasp_pose[:3], dtype=float)
                grasp_orientation = np.array(fixed_grasp_pose[3:], dtype=float)

                base_p, base_q = robot.get_world_pose()
                kin._kinematics_solver.set_robot_base_pose(base_p, base_q)
                action, ok = kin.compute_inverse_kinematics(grasp_position, grasp_orientation)
                if ok:
                    robot.apply_action(action)
            world.step(render=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
