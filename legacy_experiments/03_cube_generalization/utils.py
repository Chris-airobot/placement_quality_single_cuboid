import random
from typing import Tuple, Optional
import numpy as np
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
import json
from collections import defaultdict
from typing import Optional, List

def sample_dims(n=400, min_s=0.05, max_s=0.20, seed=None):
    """Return `n` (default=400) [dx, dy, dz] triples:
       75 thin flat, 75 long, 75 tall, 75 cube-like, 100 fully random."""
    if seed is not None:
        random.seed(seed)

    out = []
    r = random.uniform            # alias

    # ---- compact helpers for four “meaningful” shapes -------------------
    def thin():   return sorted([r(min_s,0.06)]+[r(0.10,max_s) for _ in range(2)])
    def long():   return sorted([r(0.15,max_s)]+[r(min_s,0.10) for _ in range(2)])
    def tall():   return [r(min_s,0.10), r(min_s,0.10), r(0.15,max_s)]
    def cube():   s=r(0.08,0.12); return [s*r(0.9,1.1) for _ in range(3)]
    shapes = [thin,long,tall,cube]

    # ---- 300 meaningful (75 each) ---------------------------------------
    for f in shapes:
        out += [f() for _ in range(75)]

    # ---- 100 fully random -----------------------------------------------
    out += [[r(min_s,max_s) for _ in range(3)] for _ in range(100)]

    # ensure “≤ 0.079 m” rule
    for dims in out:
        if all(d >= 0.06 for d in dims):
            dims[random.randrange(3)] = r(min_s, 0.059)

    random.shuffle(out)

    return [[round(d,3) for d in random.sample(dims,3)] for dims in out]





def sample_object_poses(num_samples_per_surface, object_dims):
    """
    Returns a list of quaternions in the format of [w, x, y, z]
    sampled at intervals determined by orientation_intervals(num_samples) around the "free" axis.
    """
    
    # baseline Euler angles (deg) for each surface, and which index to vary:
    # roll (0), pitch (1), yaw (2)
    configs = {
        "top_surface":    (np.array([   0.0,   0.0,   0.0]), 2), # 0.01475
        "bottom_surface": (np.array([-180.0,   0.0,   0.0]), 2), # 0.01415
        "left_surface":   (np.array([  90.0,   0.0,   0.0]), 1), # 0.03586
        "right_surface":  (np.array([ -90.0,   0.0,   0.0]), 1), # 0.03644
        "front_surface":  (np.array([  90.0,   0.0,  90.0]), 1), # 0.04427
        "back_surface":   (np.array([  90.0,   0.0, -90.0]), 1), # 0.04447
    }

    pos = []
    bin_edges = np.linspace(0, 360, num_samples_per_surface + 1, endpoint=True)
    angles = [int(np.random.uniform(bin_edges[i], bin_edges[i+1])) for i in range(num_samples_per_surface)]
    for _, (base_euler, var_idx) in configs.items():
        for a in angles:
            e = base_euler.copy()
            e[var_idx] = a
            # convert to radians and then quaternion
            q = euler2quat(*np.deg2rad(e), axes='rxyz', scalar_first=True).tolist()
            surface = surface_detection(q)
            if surface == 1 or surface == 3:
                z = object_dims[2]/2
            if surface == 2 or surface == 4:
                z = object_dims[0]/2
            if surface == 5 or surface == 6:
                z = object_dims[1]/2
            pos.append([0.2, -0.3, z] + q)

    return pos



import math
from typing import Dict, Iterable

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
FACE_ORDER = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
# For each face (object-local), define (n_loc, u_loc, v_loc) with right-handed: u x v = n
FACE_LOCAL_BASIS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "+X": (np.array([+1, 0, 0], float), np.array([0, +1, 0], float), np.array([0, 0, +1], float)),
    "-X": (np.array([-1, 0, 0], float), np.array([0, 0, +1], float), np.array([0, +1, 0], float)),
    "+Y": (np.array([0, +1, 0], float), np.array([0, 0, +1], float), np.array([+1, 0, 0], float)),
    "-Y": (np.array([0, -1, 0], float), np.array([+1, 0, 0], float), np.array([0, 0, +1], float)),
    "+Z": (np.array([0, 0, +1], float), np.array([+1, 0, 0], float), np.array([0, +1, 0], float)),
    "-Z": (np.array([0, 0, -1], float), np.array([0, +1, 0], float), np.array([+1, 0, 0], float)),
}


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def rotation_for_face_up(face: str, spin_rad: float = 0.0) -> np.ndarray:
    """Return R_wo with the chosen face's outward normal mapped to +Z, and spun around +Z by spin_rad."""
    n_loc, u_loc, v_loc = FACE_LOCAL_BASIS[face]

    # World target basis: n_w = +Z; u_w = +X spun about +Z by spin; v_w = n_w x u_w
    c, s = math.cos(spin_rad), math.sin(spin_rad)
    n_w = WORLD_UP.copy()
    u_w = np.array([c, s, 0.0], float)
    v_w = unit(np.cross(n_w, u_w))

    L = np.column_stack([u_loc, v_loc, n_loc])
    W = np.column_stack([u_w,   v_w,   n_w])
    return W @ L.T  # maps local face basis to world target basis


def six_face_up_orientations(spin_degs: Iterable[float] = (0.0,)) -> Dict[str, List[np.ndarray]]:
    R_map: Dict[str, List[np.ndarray]] = {f: [] for f in FACE_ORDER}
    for f in FACE_ORDER:
        for deg in spin_degs:
            R_map[f].append(rotation_for_face_up(f, math.radians(float(deg))))
    return R_map










def surface_detection(quat):
    local_normals = {
        "1": np.array([0, 0, 1]),   # +z going up (0, 0, 0)
        "2": np.array([1, 0, 0]),   # +x going up (-90, -90, -90)
        "3": np.array([0, 0, -1]),  # -z going up (180, 0, -180)
        "4": np.array([-1, 0, 0]),  # -x going up (90, 90, -90)
        "5": np.array([0, -1, 0]),  # -y going up (-90, 0, 0)
        "6": np.array([0, 1, 0]),   # +y going up (90, 0, 0)
        }
    
    global_up = np.array([0, 0, 1]) 

    # Replace with your actual quaternion w,x,y,z
    rotation = R.from_quat(quat, scalar_first=True)

    # Transform normals to the world frame
    world_normals = {face: rotation.apply(local_normal) for face, local_normal in local_normals.items()}

    # Find the face with the highest dot product with the global up direction
    upward_face = max(world_normals, key=lambda face: np.dot(world_normals[face], global_up))
    
    return int(upward_face)

def pose_difference(initial_pose, final_pose):
    # Positions
    pos_init = np.array(initial_pose[:3])
    pos_final = np.array(final_pose[:3])
    pos_diff = np.linalg.norm(pos_final - pos_init)

    # Orientations (quaternions, assumed [x, y, z, w] or [w, x, y, z] - check your format!)
    quat_init = np.array(initial_pose[3:])
    quat_final = np.array(final_pose[3:])
    # If your quaternions are [w, x, y, z], convert to [x, y, z, w] for scipy
    quat_init_xyzw = np.roll(quat_init, -1)
    quat_final_xyzw = np.roll(quat_final, -1)
    r_init = R.from_quat(quat_init_xyzw)
    r_final = R.from_quat(quat_final_xyzw)
    # Relative rotation
    r_rel = r_final * r_init.inv()
    angle_diff = r_rel.magnitude()  # in radians
    angle_diff_deg = np.degrees(angle_diff)

    return {
        "position_error": pos_diff.tolist(),
        "orientation_error_deg": angle_diff_deg.tolist()
    }



# ----------------------------------------------------
# Variant: six poses with incremental rotations 60°→360°
# ----------------------------------------------------


def sample_fixed_surface_poses_increments(object_dims: list[float],
                                         base_angle_deg: float = 60.0,
                                         base_position: tuple[float, float, float] = (0.2, -0.3, 0.0)) -> list[list[float]]:
    """Return six poses – one per face – where the rotation about the face’s
    outward normal increases in *base_angle_deg* steps (60°, 120°, …, 360°).

    If *base_angle_deg* is not 60 you still get the 1×,2×,3×… multiples of that
    angle.
    """

    configs = {
        "top_surface":    (np.array([0.0,   0.0,   0.0]), 2),
        "bottom_surface": (np.array([-180.0, 0.0,   0.0]), 2),
        "left_surface":   (np.array([90.0,  0.0,   0.0]), 1),
        "right_surface":  (np.array([-90.0, 0.0,   0.0]), 1),
        "front_surface":  (np.array([90.0,  0.0,  90.0]), 1),
        "back_surface":   (np.array([90.0,  0.0, -90.0]), 1),
    }

    poses = []
    for idx, (key, (base_euler, axis_idx)) in enumerate(configs.items()):
        rot_angle = base_angle_deg * (idx + 1)  # 60,120,…,360
        e = base_euler.copy()
        e[axis_idx] += rot_angle

        q = euler2quat(*np.deg2rad(e), axes='rxyz').tolist()

        surface = surface_detection(q)
        if surface in (1, 3):
            z = object_dims[2] / 2
        elif surface in (2, 4):
            z = object_dims[0] / 2
        else:
            z = object_dims[1] / 2

        poses.append([base_position[0], base_position[1], z + base_position[2]] + q)

    return poses


# ----------------------------------------------------
# Flip & rotate a single pose
# ----------------------------------------------------


def flip_and_rotate_pose(initial_pose: list[float],
                         rotation_deg: float = 90.0,
                         flip_deg: float = 90.0) -> list[float]:
    """Flip a cuboid depending on which face is currently upward, then spin.

    The function:
      1. Determines which local axis (+X, +Y, +Z) (or their negatives) is most
         aligned with the world +Z (up) direction.
      2. Chooses another local axis (cyclic order X→Y→Z→X) to act as the flip
         axis and applies a 180° rotation about it **in the cuboid’s local
         frame**.
      3. Applies an additional *rotation_deg* spin about the local +Z axis.

    Parameters
    ----------
    initial_pose : list[float]
        Pose in the format ``[x, y, z, w, qx, qy, qz]`` (scalar-first quaternion).
    rotation_deg : float, default 60
        Amount to spin around the face’s outward normal **after** flipping.
    viewer_direction_world : 3-tuple, default (0, –1, 0)
        World-frame vector that points *towards* the viewer (i.e. from the object
        to you). The flip is a 180° rotation about this axis.

    Returns
    -------
    list[float]
        Transformed pose in the same scalar-first format.
    """

    # --- parse input ---
    pos = np.asarray(initial_pose[:3], dtype=float)
    q_wxyz = np.asarray(initial_pose[3:], dtype=float)

    # scipy uses [x, y, z, w]
    q_xyzw = np.roll(q_wxyz, -1)
    r_initial = R.from_quat(q_xyzw)

    # -----------------------------------------
    # 1. Detect upward axis in world frame
    # -----------------------------------------
    # Local basis vectors expressed in world frame
    local_axes_world = r_initial.apply(np.eye(3))  # columns: x,y,z in world
    up_vec = np.array([0.0, 0.0, 1.0])
    dots = local_axes_world @ up_vec  # dot products (3,)
    up_idx = int(np.argmax(np.abs(dots)))  # which axis points most upward

    # -----------------------------------------
    # 2. Pick flip axis (next axis cyclically)
    # -----------------------------------------
    flip_idx = (up_idx + 1) % 3  # X->Y, Y->Z, Z->X
    flip_axis_local = np.eye(3)[flip_idx]

    # 3. Local flip (default 90°) and spin
    r_flip_local = R.from_rotvec(np.deg2rad(flip_deg) * flip_axis_local)
    r_spin_local = R.from_rotvec(np.deg2rad(rotation_deg) * np.array([0, 0, 1]))

    # Compose: first flip, then spin (both local)
    r_local = r_spin_local * r_flip_local

    # Map back to world frame
    r_target = r_initial * r_local

    q_target_xyzw = r_target.as_quat()
    q_target_wxyz = np.roll(q_target_xyzw, 1)

    return pos.tolist() + q_target_wxyz.tolist()


# Canonical baselines  (Euler angles in degrees, rxyz order)

# Your six canonical Euler baselines (rxyz, degrees)
BASELINES = {
    "top"   : ([   0,   0,   0], 2),
    "bottom": ([-180,  0,   0], 2),
    "left"  : ([  90,  0,   0], 1),
    "right" : ([ -90,  0,   0], 1),
    "front" : ([  90,  0,  90], 1),
    "back"  : ([  90,  0, -90], 1),
}
FLIP_TO = {
    "top"   : "back",
    "bottom": "front",
    "left"  : "bottom",
    "right" : "top",
    "front" : "left",
    "back"  : "right",
}

def quat_wxyz_to_xyzw(q_wxyz):
    return np.roll(q_wxyz, -1)

def quat_xyzw_to_wxyz(q_xyzw):
    return np.roll(q_xyzw,  1)

def detect_upward_face(R_world_from_body):
    """
    Return (face_key, free_axis_idx) for whichever local ±X/±Y/±Z
    is most aligned with +Z world.
    """
    # world-frame directions of body X,Y,Z
    axes_world = R_world_from_body.apply(np.eye(3))      # (3,3)
    z_world    = np.array([0, 0, 1])
    dots       = axes_world @ z_world                    # dot products
    idx        = int(np.argmax(np.abs(dots)))            # 0,1,2
    axis_sign  = np.sign(dots[idx])                      # +1 or –1

    face_keys = {
        ( 2, +1): "top",
        ( 2, -1): "bottom",
        ( 0, +1): "right",
        ( 0, -1): "left",
        ( 1, +1): "back",
        ( 1, -1): "front",
    }
    face = face_keys[(idx, int(axis_sign))]
    free_axis_idx = BASELINES[face][1]
    return face, free_axis_idx



def flip_pose_keep_delta(pose_wxyz, object_dims, num_turns: int = 1):
    """
    1) Rotate the cube about a local axis (≠ current up) by 90°*num_turns
    2) Detect which face is now up and set z = half‑thickness for that face
       using object_dims = [dim_x, dim_y, dim_z]
    """
    assert num_turns in (0,1,2,3), "num_turns must be 1, 2 or 3"

    # split pos + quat
    pos     = np.array(pose_wxyz[:3], dtype=float)
    q_wxyz  = np.asarray(pose_wxyz[3:], dtype=float)
    R_cur   = R.from_quat(quat_wxyz_to_xyzw(q_wxyz))

    # 1) find which local axis is up (0=X,1=Y,2=Z)
    axes_world = R_cur.apply(np.eye(3))
    up_idx     = int(np.argmax(np.abs(axes_world @ [0,0,1])))

    # pick a different local axis to spin around
    spin_idx   = (up_idx + 1) % 3
    axis_body  = np.eye(3)[spin_idx]

    # apply local spin
    angle      = np.deg2rad(90 * num_turns)
    R_spin     = R.from_rotvec(angle * axis_body)
    R_final    = R_cur * R_spin

    # get final quaternion
    q_final   = quat_xyzw_to_wxyz(R_final.as_quat())

    # 2) detect which face is now up to choose z‑offset
    face, _   = detect_upward_face(R_final)
    if face in ("top","bottom"):
        pos[2] = object_dims[2]/2
    elif face in ("left","right"):
        pos[2] = object_dims[0]/2
    else:  # front/back
        pos[2] = object_dims[1]/2

    return pos.tolist() + q_final.tolist()


def get_prepose(grasp_pos, grasp_quat, offset=0.15):
    # grasp_quat: [w, x, y, z]
    # scipy uses [x, y, z, w]!
    grasp_quat_xyzw = [grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]
    rot = R.from_quat(grasp_quat_xyzw)
    # Get approach direction (z-axis of gripper in world frame)
    approach_dir = rot.apply([0, 0, 1])  # [0, 0, 1] is z-axis
    # Compute pregrasp position (move BACK along approach vector)
    pregrasp_pos = np.array(grasp_pos) - offset * approach_dir
    return pregrasp_pos, grasp_quat  # Same orientatio


def get_reachable_prepose(grasp_pos, grasp_quat,
                          env,                       # gives access to IK checker
                          max_offset=0.15,
                          min_offset=0.03,
                          step=0.02,
                          vertical_fallback=True):
    """
    Returns (pre_pos, pre_quat) for the first retreat distance that is
    IK-reachable and collision-free.  Falls back to a vertical lift if
    nothing along the approach works.

    grasp_quat = [w, x, y, z] (same as your code)
    """
    # Build world-frame approach unit vector
    rot = R.from_quat([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]])
    approach_dir = rot.apply([0, 0, 1])          # tool Z

    # Test a shrinking ladder of offsets: 0.15, 0.13, …, 0.03
    offset_list = np.arange(max_offset, min_offset - 1e-6, -step)
    for off in offset_list:
        candidate = np.array(grasp_pos) - off * approach_dir
        if env.check_ik(candidate, grasp_quat):
            return candidate, grasp_quat

    # Optional fallback: straight vertical lift above grasp
    if vertical_fallback:
        up_dir = np.array([0, 0, 1])
        for lift in offset_list:                  # reuse same distances
            candidate = np.array(grasp_pos) + lift * up_dir
            if env.check_ik(candidate, grasp_quat):
                return candidate, grasp_quat

    # Nothing worked – return None so caller can handle gracefully
    return None, None


def quat_wxyz_to_R(q):
    w, x, y, z = map(float, q)
    n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
    w, x, y, z = w/n, x/n, y/n, z/n
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], float)

def corners_local(dx, dy, dz):
    hx, hy, hz = dx/2.0, dy/2.0, dz/2.0
    return np.array([[sx*hx, sy*hy, sz*hz]
                     for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], float)

def face_id_from_R(R_obj):
    # Map world +Z alignment of object axes to IDs: 1: +Z(top), 2:+X, 3:-Z(bottom), 4:-X, 5:-Y, 6:+Y
    axes = {
        1: np.array([0,0, 1.0]),  # +Z (top)
        2: np.array([1.0,0,0 ]),  # +X
        3: np.array([0,0,-1.0]),  # -Z (bottom)
        4: np.array([-1.0,0,0]),  # -X
        5: np.array([0,-1.0,0]),  # -Y
        6: np.array([0, 1.0,0])   # +Y
    }
    # Take object Z/X/-Z/-X/-Y/+Y normals in world, pick which aligns most with +Z_world
    z_world = np.array([0,0,1.0])
    cand = {
        1: R_obj[:,2],   # +Z_obj in world
        2: R_obj[:,0],   # +X_obj
        3: -R_obj[:,2],  # -Z_obj
        4: -R_obj[:,0],  # -X_obj
        5: -R_obj[:,1],  # -Y_obj
        6: R_obj[:,1],   # +Y_obj
    }
    best_k = max(cand, key=lambda k: float(np.dot(cand[k], z_world)))
    return int(best_k)

def hand_object_local_features(hand_pose_wxyz, obj_pose_wxyz):
    t_h = np.array(hand_pose_wxyz[:3], float)
    qh  = hand_pose_wxyz[3:7]
    t_o = np.array(obj_pose_wxyz[:3], float)
    qo  = obj_pose_wxyz[3:7]
    R_h = quat_wxyz_to_R(qh)
    R_o = quat_wxyz_to_R(qo)
    t_loc = R_o.T @ (t_h - t_o)
    R_loc = R_o.T @ R_h
    R6 = np.concatenate([R_loc[:,0], R_loc[:,1]]).astype(float)  # 6D ortho rep
    return t_loc.astype(float), R6


# --- helpers to classify init→final relation and bucket name ---
def _face_id_from_Rw(Rw: np.ndarray) -> int:
    """
    Map which object axis points most along world +Z.
    IDs: 1:+Z(top), 2:+X, 3:-Z(bottom), 4:-X, 5:-Y, 6:+Y   (your convention)
    """
    z_world = np.array([0., 0., 1.], dtype=np.float32)
    cand = {
        1: Rw[:, 2],        # +Z_obj
        2: Rw[:, 0],        # +X_obj
        3: -Rw[:, 2],       # -Z_obj
        4: -Rw[:, 0],       # -X_obj
        5: -Rw[:, 1],       # -Y_obj
        6: Rw[:, 1],        # +Y_obj
    }
    best_k = max(cand, key=lambda k: float(np.dot(cand[k], z_world)))
    return int(best_k)

def classify_bucket(initial_q_wxyz, final_q_wxyz, small_deg=7.0, med_deg=45.0) -> str:
    """
    Returns one of: 'SMALL', 'MEDIUM', 'ADJACENT', 'OPPOSITE'
    - If faces are same: angle <= small_deg -> SMALL, else -> MEDIUM
    - Else if opposite faces -> OPPOSITE, else -> ADJACENT
    """
    Ri = R.from_quat([initial_q_wxyz[1], initial_q_wxyz[2], initial_q_wxyz[3], initial_q_wxyz[0]]).as_matrix()
    Rf = R.from_quat([final_q_wxyz[1],   final_q_wxyz[2],   final_q_wxyz[3],   final_q_wxyz[0]]).as_matrix()
    fi = _face_id_from_Rw(Ri)
    ff = _face_id_from_Rw(Rf)

    if fi == ff:
        # relative rotation magnitude (should be yaw-only for stable faces)
        ang_deg = np.rad2deg((R.from_matrix(Ri.T @ Rf)).magnitude())
        return "SMALL" if ang_deg <= small_deg else "MEDIUM"
    # opposite mapping: 1↔3, 2↔4, 5↔6
    if (fi == 1 and ff == 3) or (fi == 3 and ff == 1) or \
       (fi == 2 and ff == 4) or (fi == 4 and ff == 2) or \
       (fi == 5 and ff == 6) or (fi == 6 and ff == 5):
        return "OPPOSITE"
    return "ADJACENT"

import os
def last_completed_index(path: str) -> int:
    """
    Reads the JSONL results file and returns the last 'index' seen.
    If file doesn't exist or is empty, returns -1.
    """
    if not os.path.exists(path):
        return -1
    last = -1
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "index" in obj:
                    last = max(last, int(obj["index"]))
            except Exception:
                # ignore malformed line; you said your writer is correct
                pass
    return last

if __name__ == "__main__":
    result = sample_dims()
    print(len(result))

 


    
