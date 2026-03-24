import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence

from scipy.spatial.transform import Rotation as R
from placement_quality.cube_generalization.utils import six_face_up_orientations
import json

INIT_SPIN_DEG = 0
PEDESTAL_HEIGHT = 0.10
PEDESTAL_CENTER_Z = 0.05
PEDESTAL_TOP_Z = PEDESTAL_CENTER_Z + 0.5 * PEDESTAL_HEIGHT # 0.10
WORLD_UP = np.array([0.0, 0.0, 1.0])
# Environment and clearance parameters (align with visualization)
PEDESTAL_HEIGHT = 0.10
PEDESTAL_CENTER_Z = 0.05
PEDESTAL_TOP_Z = PEDESTAL_CENTER_Z + 0.5 * PEDESTAL_HEIGHT

MIN_CLEARANCE = 0.02
PALM_DEPTH = 0.038
FINGER_THICK = 0.03
GRIPPER_OPEN_MAX = 0.08
MIN_PALM_CLEAR = 0.01
APPROACH_BACKOFF = 0.09

# Grasp transform parameters
ENABLE_TILT = True
TILT_DEG = 75.0
ENABLE_YAW = True
YAW_DEG = 15.0
ENABLE_ROLL = True
ROLL_DEG = 15.0

# Panda hand offset (hand->TCP along local +Z) and extra insert along -Z
HAND_TO_TCP_Z = 0.1034
EXTRA_INSERT = -0.0334

# Initial and final orientation sampling
INIT_SPINS_DEG: Tuple[int, ...] = (0, 90)
FINAL_SPIN_CANDIDATES_DEG: Tuple[int, ...] = (0, 90, 180, 270)
N_FINAL_SPINS_PER_FACE: int = 1
FACE_LIST: Tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")



# Local surface frames for each cuboid face in the OBJECT frame
_FACE_BASIS = {
    '+X': (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
    '-X': (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), -np.array([1.0, 0.0, 0.0])),
    '+Y': (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    '-Y': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), -np.array([0.0, 1.0, 0.0])),
    '+Z': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    '-Z': (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), -np.array([0.0, 0.0, 1.0])),
}


def face_surface_frame(face: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        u_f, v_f, n_f = _FACE_BASIS[face]
    except KeyError as exc:
        raise ValueError(f"Unknown face label: {face}") from exc
    return u_f.copy(), v_f.copy(), n_f.copy()


def quat_wxyz_to_xyzw(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    assert q.shape[-1] == 4
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    assert q.shape[-1] == 4
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)



def pose_from_R_t(R_wo: np.ndarray, t_wo: np.ndarray) -> Tuple[List[float], List[float]]:
    q_xyzw = R.from_matrix(R_wo).as_quat()
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    return [float(t_wo[0]), float(t_wo[1]), float(t_wo[2])], [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])]


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
def sample_dims(n: int, min_s: float = 0.05, max_s: float = 0.20, seed: int = 0) -> List[Tuple[float, float, float]]:
    """
    Generate exactly n box dimension triplets (x, y, z) in metres with simple shape bias
    while enforcing constraints:
      - All edges lie within [min_s, max_s]
      - Each triplet has at least one edge exactly equal to min_s
    """
    rng = np.random.default_rng(seed)

    min_s = float(min_s)
    max_s = float(max_s)
    assert max_s >= min_s, "max_s must be >= min_s"

    def enforce_constraints(dims: np.ndarray) -> Tuple[float, float, float]:
        # Clip to bounds
        dims = np.clip(dims.astype(float), min_s, max_s)
        # Ensure at least one dimension equals min_s
        if not np.any(np.isclose(dims, min_s)):
            idx = int(np.argmin(dims))
            dims[idx] = min_s
        return float(dims[0]), float(dims[1]), float(dims[2])

    # Shape types to encourage some variety without breaking bounds
    shape_types = ("thin", "long", "tall", "cubey", "random")
    results: List[Tuple[float, float, float]] = []

    # Helper to sample a biased triplet within range
    span = max_s - min_s

    def sample_one(kind: str) -> Tuple[float, float, float]:
        if kind == "thin":
            # Two arbitrary, one near the lower end
            a = rng.uniform(min_s, max_s)
            b = rng.uniform(min_s, max_s)
            c = rng.uniform(min_s, min_s + 0.3 * span)
            dims = np.array([a, b, c], dtype=float)
        elif kind == "long":
            # One near upper end, others arbitrary
            hi = rng.uniform(min_s + 0.7 * span, max_s)
            lo1 = rng.uniform(min_s, max_s)
            lo2 = rng.uniform(min_s, max_s)
            # Randomize which axis is the long one
            idx = rng.integers(0, 3)
            vals = [lo1, lo2, rng.uniform(min_s, max_s)]
            vals[idx] = hi
            dims = np.array(vals, dtype=float)
        elif kind == "tall":
            # Similar to long: emphasize z
            a = rng.uniform(min_s, max_s)
            b = rng.uniform(min_s, max_s)
            c = rng.uniform(min_s + 0.7 * span, max_s)
            dims = np.array([a, b, c], dtype=float)
        elif kind == "cubey":
            s = rng.uniform(min_s, max_s)
            dims = np.array([s * rng.uniform(0.9, 1.1),
                             s * rng.uniform(0.9, 1.1),
                             s * rng.uniform(0.9, 1.1)], dtype=float)
        else:  # random
            dims = rng.uniform(min_s, max_s, size=3).astype(float)
        return enforce_constraints(dims)

    for i in range(n):
        kind = shape_types[int(rng.integers(0, len(shape_types)))]
        results.append(sample_one(kind))

    return results


def build_tool_orientation_from_meta(
    meta: Dict,
    R_obj_to_world: np.ndarray,
) -> np.ndarray:
    """
    Build a right-handed tool frame from contact metadata.

    Returns a 3x3 rotation matrix whose columns are [X_tool, Y_tool, Z_tool] in WORLD.
    """
    face = meta['face']
    n_f_obj = face_surface_frame(face)[2]
    z_loc = -n_f_obj

    # Start with the stored axis (shorter face edge)
    x_raw = np.asarray(meta['axis'], dtype=float)
    x_loc = x_raw - np.dot(x_raw, z_loc) * z_loc
    norm_x = np.linalg.norm(x_loc)
    if norm_x < 1e-12:
        u_f, v_f, _ = face_surface_frame(face)
        x_loc = u_f if np.linalg.norm(u_f) > np.linalg.norm(v_f) else v_f
    x_loc = x_loc / np.linalg.norm(x_loc)

    # Make Y = stored axis projected into the plane; complete the frame
    y_raw = np.asarray(meta['axis'], dtype=float)
    y_loc = y_raw - np.dot(y_raw, z_loc) * z_loc
    norm_y = np.linalg.norm(y_loc)
    if norm_y < 1e-12:
        u_f, v_f, _ = face_surface_frame(face)
        y_loc = u_f if np.linalg.norm(u_f) < np.linalg.norm(v_f) else v_f
    y_loc = y_loc / np.linalg.norm(y_loc)

    x_loc = np.cross(y_loc, z_loc); x_loc /= np.linalg.norm(x_loc)
    y_loc = np.cross(z_loc, x_loc); y_loc /= np.linalg.norm(y_loc)

    # Align X with stored meta axis direction
    x_meta = np.asarray(meta['axis'], dtype=float)
    if float(np.dot(x_loc, x_meta)) < 0.0:
        x_loc *= -1.0
        y_loc *= -1.0

    x_w = R_obj_to_world @ x_loc
    y_w = R_obj_to_world @ y_loc
    z_w = R_obj_to_world @ z_loc

    R_tool = np.column_stack([x_w, y_w, z_w])
    if float(np.dot(np.cross(x_w, y_w), z_w)) < 0.0:
        y_w *= -1.0
        R_tool = np.column_stack([x_w, y_w, z_w])
    return R_tool


def _quat_wxyz_from_R(Rm: np.ndarray) -> np.ndarray:
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _apply_local_rotations(
    R_tool: np.ndarray,
    enable_tilt: bool,
    tilt_deg: float,
    enable_yaw: bool,
    yaw_deg: float,
    enable_roll: bool,
    roll_deg: float,
) -> np.ndarray:
    R_out = R_tool
    if enable_tilt:
        tilt_rad = float(np.deg2rad(tilt_deg - 90.0))
        if abs(tilt_rad) > 1e-9:
            R_tilt = R.from_rotvec(tilt_rad * R_out[:, 0]).as_matrix()
            R_out = R_tilt @ R_out
    if enable_roll:
        roll_rad = float(np.deg2rad(roll_deg))
        if abs(roll_rad) > 1e-9:
            R_roll = R.from_rotvec(roll_rad * R_out[:, 1]).as_matrix()
            R_out = R_roll @ R_out
    if enable_yaw:
        yaw_rad = float(np.deg2rad(yaw_deg))
        if abs(yaw_rad) > 1e-9:
            R_yaw = R.from_rotvec(yaw_rad * R_out[:, 2]).as_matrix()
            R_out = R_yaw @ R_out
    return R_out


def generate_grasp_poses(
    dims_xyz: np.ndarray,
    R_obj_to_world: np.ndarray,
    t_obj_world: np.ndarray,
    enable_tilt: bool = False,
    tilt_deg: float = 90.0,
    enable_yaw: bool = False,
    yaw_deg: float = 0.0,
    enable_roll: bool = False,
    roll_deg: float = 0.0,
    filter_by_gripper_open: bool = True,
    gripper_open_max: float = 0.08,
    apply_hand_to_tcp: bool = False,
    hand_to_tcp_z: float = 0.1034,
    extra_insert: float = 0.0,
) -> List[Dict]:
    """
    Generate grasp pose candidates (position + orientation) for a cuboid object.

    Inputs:
      - dims_xyz: object dimensions [dx, dy, dz]
      - R_obj_to_world: 3x3 rotation of the object in world
      - t_obj_world: 3-vector translation of the object in world
      - Bottom face is excluded automatically (based on WORLD_UP)
      - enable_tilt/yaw/roll + angles: apply individually or in combination
      - apply_hand_to_tcp: if True, also return panda_hand pose with an orientation-aware
        offset along local -Z scaled by 1/cos(theta)

    Returns a list of dicts containing:
      - 'face', 'fraction', 'contact_position_world'
      - 'tool_quaternion_wxyz', 'tool_rotation', 'face_normal_world'
      - Optionally 'hand_position_world', 'hand_quaternion_wxyz' when apply_hand_to_tcp is True
    """
    dims_xyz = np.asarray(dims_xyz, dtype=float)
    R_obj_to_world = np.asarray(R_obj_to_world, dtype=float)
    t_obj_world = np.asarray(t_obj_world, dtype=float)

    contacts = generate_contact_metadata(dims_xyz, approach_offset=0.01)

    # Auto-detect and exclude the bottom face (most opposite to WORLD_UP)
    face_to_normal_obj = {
        '+X': np.array([ 1.0, 0.0, 0.0]),
        '-X': np.array([-1.0, 0.0, 0.0]),
        '+Y': np.array([ 0.0, 1.0, 0.0]),
        '-Y': np.array([ 0.0,-1.0, 0.0]),
        '+Z': np.array([ 0.0, 0.0, 1.0]),
        '-Z': np.array([ 0.0, 0.0,-1.0]),
    }
    dot_to_face: Dict[float, str] = {}
    for face, n_obj in face_to_normal_obj.items():
        n_world = R_obj_to_world @ n_obj
        dot_to_face[float(np.dot(n_world, WORLD_UP))] = face
    # Small numerical drift safety: pick min dot value (most downward)
    min_dot = min(dot_to_face.keys())
    bottom_face = dot_to_face[min_dot]
    contacts = [m for m in contacts if m['face'] != bottom_face]

    results: List[Dict] = []
    for meta in contacts:
        # Filter by gripper max opening using the stored closing axis (object frame)
        if filter_by_gripper_open:
            half_extents = 0.5 * dims_xyz
            axis_obj = np.asarray(meta['axis'], dtype=float)
            jaw_span = float(2.0 * np.sum(np.abs(axis_obj) * half_extents))
            if jaw_span > float(gripper_open_max):
                continue

        # Contact position in world
        p_local = np.asarray(meta['p_local'], dtype=float)
        p_world = R_obj_to_world @ p_local + t_obj_world

        # Base orientation from surface frame, then apply local rotations
        R_tool = build_tool_orientation_from_meta(meta, R_obj_to_world)
        R_tool = _apply_local_rotations(
            R_tool,
            enable_tilt, tilt_deg,
            enable_yaw, yaw_deg,
            enable_roll, roll_deg,
        )
        q_wxyz = _quat_wxyz_from_R(R_tool)

        # Face normal in world (outward)
        n_face_world = R_obj_to_world @ face_surface_frame(meta['face'])[2]

        result: Dict = {
            'face': meta['face'],
            'fraction': float(meta['fraction']),
            'contact_position_world': p_world.astype(float),
            'tool_quaternion_wxyz': q_wxyz.astype(float),
            'tool_rotation': R_tool.astype(float),
            'face_normal_world': n_face_world.astype(float),
        }

        if apply_hand_to_tcp:
            # Orientation-aware offset to convert TCP to panda_hand
            z_tool_w = R_tool[:, 2]
            cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_world)))
            z_local = -(float(hand_to_tcp_z) + float(extra_insert)) / cos_theta
            hand_position_world = p_world + R_tool @ np.array([0.0, 0.0, z_local], dtype=float)
            result['hand_position_world'] = hand_position_world.astype(float)
            result['hand_quaternion_wxyz'] = q_wxyz.astype(float)

        results.append(result)

    return results



def generate_grasp_poses_including_bottom(
    dims_xyz: np.ndarray,
    R_obj_to_world: np.ndarray,
    t_obj_world: np.ndarray,
    enable_tilt: bool = False,
    tilt_deg: float = 90.0,
    enable_yaw: bool = False,
    yaw_deg: float = 0.0,
    enable_roll: bool = False,
    roll_deg: float = 0.0,
    filter_by_gripper_open: bool = False,   # do NOT gate for training
    gripper_open_max: float = 0.08,
    apply_hand_to_tcp: bool = True,
    hand_to_tcp_z: float = 0.1034,
    extra_insert: float = -0.0334,
) -> List[Dict]:
    dims_xyz = np.asarray(dims_xyz, dtype=float)
    R_obj_to_world = np.asarray(R_obj_to_world, dtype=float)
    t_obj_world = np.asarray(t_obj_world, dtype=float)

    # Keep ALL contacts, including bottom face
    contacts = generate_contact_metadata(dims_xyz, approach_offset=0.01)

    results: List[Dict] = []
    for meta in contacts:
        if filter_by_gripper_open:
            half_extents = 0.5 * dims_xyz
            axis_obj = np.asarray(meta['axis'], dtype=float)
            jaw_span = float(2.0 * np.sum(np.abs(axis_obj) * half_extents))
            if jaw_span > float(gripper_open_max):
                continue

        p_local = np.asarray(meta['p_local'], dtype=float)
        p_world = R_obj_to_world @ p_local + t_obj_world

        R_tool = build_tool_orientation_from_meta(meta, R_obj_to_world)
        R_tool = _apply_local_rotations(
            R_tool,
            enable_tilt, tilt_deg,
            enable_yaw, yaw_deg,
            enable_roll, roll_deg,
        )
        q_wxyz = _quat_wxyz_from_R(R_tool)

        n_face_world = R_obj_to_world @ face_surface_frame(meta['face'])[2]
        result: Dict = {
            'face': meta['face'],
            'fraction': float(meta['fraction']),
            'contact_position_world': p_world.astype(float),
            'tool_quaternion_wxyz': q_wxyz.astype(float),
            'tool_rotation': R_tool.astype(float),
            'face_normal_world': n_face_world.astype(float),
        }

        if apply_hand_to_tcp:
            z_tool_w = R_tool[:, 2]
            cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_world)))
            z_local = -(float(hand_to_tcp_z) + float(extra_insert)) / cos_theta
            hand_position_world = p_world + R_tool @ np.array([0.0, 0.0, z_local], dtype=float)
            result['hand_position_world'] = hand_position_world.astype(float)
            result['hand_quaternion_wxyz'] = q_wxyz.astype(float)
            result['axis_obj'] = np.asarray(meta['axis'], dtype=float)  # NEW
            result['p_local']  = p_local.astype(float) 

        results.append(result)
    return results


from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

# assumes you already have these helpers defined above:
# - generate_contact_metadata(...)
# - build_tool_orientation_from_meta(...)
# - _apply_local_rotations(...)
# - _quat_wxyz_from_R(...)
# - face_surface_frame(...)

def generate_grasp_poses_including_bottom_updated(
    dims_xyz: np.ndarray,
    R_obj_to_world: np.ndarray,
    t_obj_world: np.ndarray,
    *,
    # --- orientation policy knobs (safe defaults) ---
    # yaw about Z_tool (approach): face-aware & center/edge-aware
    yaw_set: Tuple[float, ...]  = (-15.0, 0.0, 15.0),         # about Z_tool (approach)
    tilt_set: Tuple[float, ...] = (-10.0, 0.0, 10.0),         # about X_tool (closing)
    roll_set: Tuple[float, ...] = (-5.0, 0.0, 5.0),           # about Y_tool (binormal)

    # --- contact lattice knobs pass-through (optional) ---
    approach_offset: float = 0.01,
    include_faces: Optional[Tuple[str, ...]] = None,          # e.g., ('+X','-X','+Y','-Y')
    u_fracs_long: Optional[Tuple[float, ...]] = None,         # override long-side fractions
    v_fracs_short: Optional[Tuple[float, ...]] = None,        # override short-side fractions

    # --- filters/offsets (unchanged semantics) ---
    filter_by_gripper_open: bool = False,   # keep False during training; simple span check underestimates span when yaw≠0
    gripper_open_max: float = 0.08,
    apply_hand_to_tcp: bool = True,
    hand_to_tcp_z: float = 0.1034,
    extra_insert: float = -0.0334,
) -> List[Dict]:

    dims_xyz = np.asarray(dims_xyz, dtype=float)
    R_obj_to_world = np.asarray(R_obj_to_world, dtype=float)
    t_obj_world = np.asarray(t_obj_world, dtype=float)

    # Build kwargs for your (possibly updated) contact lattice function
    contact_kwargs = {'approach_offset': float(approach_offset)}
    if include_faces is not None:
        contact_kwargs['include_faces'] = include_faces
    if u_fracs_long is not None:
        contact_kwargs['u_fracs_long'] = u_fracs_long
    if v_fracs_short is not None:
        contact_kwargs['v_fracs_short'] = v_fracs_short


    contacts = generate_contact_metadata(dims_xyz, include_faces=('+X', '-X', '+Y', '-Y'),)

    results: List[Dict] = []
    for meta in contacts:
        face_label: str = str(meta['face'])  # e.g., '+X'

        # Uniform orientation sets for all faces/rows
        yaw_candidates  = yaw_set
        tilt_candidates = tilt_set

        # Optional check (kept from your code; conservative and ignores yaw effect)
        if filter_by_gripper_open:
            half_extents = 0.5 * dims_xyz
            axis_obj = np.asarray(meta['axis'], dtype=float)  # short in-plane axis in *object* frame
            jaw_span = float(2.0 * np.sum(np.abs(axis_obj) * half_extents))
            if jaw_span > float(gripper_open_max):
                continue

        # World contact position
        p_local = np.asarray(meta['p_local'], dtype=float)
        p_world = R_obj_to_world @ p_local + t_obj_world

        # Base tool frame at contact (no local rotations yet)
        R_tool = build_tool_orientation_from_meta(meta, R_obj_to_world)

        # Face normal in world (for hand offset calc)
        n_face_world = R_obj_to_world @ face_surface_frame(face_label)[2]

        # Enumerate orientation variants for this contact
        for yaw_deg in yaw_candidates:
            for tilt in tilt_candidates:
                for roll_deg in roll_set:
                    # Uniform sets; same conversion for tilt (90° = no-tilt baseline)
                    tilt_deg = 90.0 + float(tilt)

                    R_tool_var = _apply_local_rotations(
                        R_tool,
                        enable_tilt=True, tilt_deg=tilt_deg,
                        enable_yaw=True,  yaw_deg=float(yaw_deg),
                        enable_roll=True, roll_deg=float(roll_deg),
                    )
                    q_wxyz = _quat_wxyz_from_R(R_tool_var)

                    out: Dict = {
                        'face': face_label,
                        # propagate lattice ids if present
                        'u_frac': float(meta.get('u_frac', np.nan)),
                        'v_frac': float(meta.get('v_frac', np.nan)),
                        'contact_position_world': p_world.astype(float),
                        'tool_quaternion_wxyz': q_wxyz.astype(float),
                        'tool_rotation': R_tool_var.astype(float),
                        'face_normal_world': n_face_world.astype(float),

                        'axis_obj': np.asarray(meta.get('axis', [0.0, 0.0, 0.0]), dtype=float),
                        'p_local': p_local.astype(float),

                        # record the applied local angles for traceability
                        'angles_deg': {
                            'yaw_about_Ztool': float(yaw_deg),
                            'tilt_about_Xtool': float(tilt),
                            'roll_about_Ytool': float(roll_deg),
                            'tilt_param_passed': float(tilt_deg),  # the value actually passed to your helper
                        },
                    }

                    if apply_hand_to_tcp:
                        # Offset wrist along tool Z, accounting for approach angle vs face normal
                        z_tool_w = R_tool_var[:, 2]
                        cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_world)))
                        z_local = -(float(hand_to_tcp_z) + float(extra_insert)) / cos_theta
                        hand_position_world = p_world + R_tool_var @ np.array([0.0, 0.0, z_local], dtype=float)

                        out['hand_position_world'] = hand_position_world.astype(float)
                        out['hand_quaternion_wxyz'] = q_wxyz.astype(float)
                        out['axis_obj'] = np.asarray(meta.get('axis', [0.0, 0.0, 0.0]), dtype=float)
                        out['p_local']  = p_local.astype(float)

                    results.append(out)

    return results



def passes_clearance(
    p_world: np.ndarray,
    ped_top_z: float,
    R_tool: np.ndarray,
    min_contact_clear: float = MIN_CLEARANCE,
    min_palm_clear: float = MIN_PALM_CLEAR,
    palm_depth: float = PALM_DEPTH,
    approach_backoff: float = APPROACH_BACKOFF,
    finger_thick: float = FINGER_THICK,
) -> bool:
    """Match visualization clearance test (contact and palm-bottom clearance)."""
    if (float(p_world[2]) - float(ped_top_z)) < float(min_contact_clear):
        return False
    z_w = R_tool[:, 2]
    y_w = R_tool[:, 1]
    effective_palm_depth = max(float(palm_depth) - float(approach_backoff), 0.0)
    palm_center_z = float(p_world[2]) - effective_palm_depth * float(z_w[2])
    palm_bottom_z = palm_center_z - float(finger_thick) * abs(float(y_w[2]))
    return (palm_bottom_z - float(ped_top_z)) >= float(min_palm_clear)


def generate_experiments(
    num_objects: int = 4,
    dims_min: float = 0.05,
    dims_max: float = 0.20,
    seed: int = 980579,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    test_data: List[Dict] = []

    # Sample diverse object dimensions
    box_dim_lists = sample_dims(n=num_objects, min_s=dims_min, max_s=dims_max, seed=seed)
    print(box_dim_lists)

    for obj_idx, dims in enumerate(box_dim_lists):
        dims_xyz = np.array(dims, dtype=float)

        # Initial orientation candidates: multiple faces and spins
        R_init_map = six_face_up_orientations(spin_degs=INIT_SPINS_DEG)
        for init_face, R_list in R_init_map.items():
            for R_init in R_list:
                R_init = np.asarray(R_init, dtype=float)

                # Place object at pedestal with correct Z (bottom on pedestal top)
                half_extents = 0.5 * dims_xyz
                h_z_init = float(np.sum(np.abs(R_init[2, :]) * half_extents))
                t_init = np.array([0.2, -0.3, PEDESTAL_TOP_Z + h_z_init], dtype=float)
                pos_i, quat_i = pose_from_R_t(R_init, t_init)

                # Build grasp candidates for all 8 transform variants
                transform_variants = [
                    {"tilt": False, "yaw": False, "roll": False},
                    {"tilt": True,  "yaw": False, "roll": False},
                    {"tilt": False, "yaw": True,  "roll": False},
                    {"tilt": False, "yaw": False, "roll": True },
                    # {"tilt": True,  "yaw": True,  "roll": False},
                    # {"tilt": True,  "yaw": False, "roll": True },
                    # {"tilt": False, "yaw": True,  "roll": True },
                    {"tilt": True,  "yaw": True,  "roll": True },
                ]

                grasps_by_variant: List[Tuple[Dict, List[Dict]]] = []
                for tv in transform_variants:
                    grasps_tv = generate_grasp_poses(
                        dims_xyz=dims_xyz,
                        R_obj_to_world=R_init,
                        t_obj_world=t_init,
                        enable_tilt=tv["tilt"], tilt_deg=TILT_DEG,
                        enable_yaw=tv["yaw"],  yaw_deg=YAW_DEG,
                        enable_roll=tv["roll"], roll_deg=ROLL_DEG,
                        filter_by_gripper_open=True, gripper_open_max=GRIPPER_OPEN_MAX,
                        apply_hand_to_tcp=True,
                        hand_to_tcp_z=HAND_TO_TCP_Z,
                        extra_insert=EXTRA_INSERT,
                    )

                    valid_grasps_tv: List[Dict] = []
                    for g in grasps_tv:
                        p_world = np.asarray(g['contact_position_world'], dtype=float)
                        R_tool = np.asarray(g['tool_rotation'], dtype=float)
                        if passes_clearance(p_world, PEDESTAL_TOP_Z, R_tool):
                            valid_grasps_tv.append(g)
                    # Keep empty lists too; we still want coverage info per variant
                    grasps_by_variant.append((tv, valid_grasps_tv))

                # Final orientation candidates: select N spins per face
                spins_this_block = tuple(sorted(rng.choice(FINAL_SPIN_CANDIDATES_DEG, size=N_FINAL_SPINS_PER_FACE, replace=False)))
                R_final_map = six_face_up_orientations(spin_degs=spins_this_block)

                for fin_face in FACE_LIST:
                    for R_fin in R_final_map[fin_face]:
                        R_fin = np.asarray(R_fin, dtype=float)
                        # Place final object bottom on pedestal top as well
                        h_z_fin = float(np.sum(np.abs(R_fin[2, :]) * half_extents))
                        t_fin = np.array([0.2, -0.3, PEDESTAL_TOP_Z + h_z_fin], dtype=float)
                        pos_f, quat_f = pose_from_R_t(R_fin, t_fin)

                        # One block of trials per transform variant, sharing the same init/final
                        for tv, valid_grasps_tv in grasps_by_variant:
                            for g in valid_grasps_tv:
                                trial = {
                                    "object_dimensions": list(map(float, dims_xyz.tolist())),
                                    "initial_object_pose": list(map(float, pos_i + quat_i)),
                                    "final_object_pose": list(map(float, pos_f + quat_f)),
                                    # Save panda_hand pose for robot execution
                                    "grasp_pose": list(map(float, (g['hand_position_world'].tolist() + g['hand_quaternion_wxyz'].tolist()))),
                                    # Debug: which transform toggles generated this grasp
                                    "debug_info": {
                                        "face": g['face'],
                                        "fraction": float(g['fraction']),
                                        "face_normal_world": np.asarray(g['face_normal_world'], dtype=float).tolist(),
                                        "tilt_enabled": bool(tv["tilt"]),
                                        "yaw_enabled": bool(tv["yaw"]),
                                        "roll_enabled": bool(tv["roll"]),
                                        "tilt_deg": float(TILT_DEG),
                                        "yaw_deg": float(YAW_DEG),
                                        "roll_deg": float(ROLL_DEG),
                                    },
                                }
                                test_data.append(trial)

    return test_data


def main():
    data = generate_experiments(
        num_objects=4,
        dims_min=0.05,
        dims_max=0.20,
        seed=980579,
    )

    out_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiments.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} trials → {out_path}")





if __name__ == "__main__":
    main()