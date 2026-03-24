#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IK-only, OFFLINE, path-aware collision labeler (simple + focused)
=================================================================
What this script does (pure post-processing; no re-execution):
- For each sample from your IK/sim JSONL, it builds ONE fixed EE path:
    1) end-effector pose at the grasp on the object’s initial pose
    2) same pose lifted by a safe height (world Z)
    3) same XYZ as (2) but with the orientation of the object’s final placement pose
    4) end-effector pose at the object’s final placement pose
- It sweeps ONLY the held object box (inflated by a safety margin) along that path,
  against ONLY the fixed pedestal box (inflated) and the ground plane,
  using dense interpolation in EE space (no controller, no physics).
- It outputs, per sample:
    - predicted_path_collision  (True/False)
    - min_clearance_mm          (smallest separation to pedestal sides/ground along the sweep)
    - first_collision_leg       ("Lift" | "RotateAtHeight" | "MoveToPlace" | None)
    - contact_at_end_ok         (bottom-face-on-support at the end is okay)
- It can also compare those labels against your executed results JSONL (had_collision)
  and run a few sanity checks (collision rate vs lift, vs inflation, and by bucket).

Why EE-space (not joint-space) here?
- You wanted simple + environment-focused. This needs no robot FK/IK adapters or link shapes.
- It captures the key swept-volume effects that create side-scrapes on pedestal/ground.
- If later you want link-vs-env too, you can extend the same logic by adding FK link geometry.

Expected fields in sim JSONL (lenient; see _pose_from_row/_dims_from_row):
- "grasp_pose": end-effector pose at the grasp on the object’s initial pose [x,y,z,qx,qy,qz,qw]
- "init_pose" : object initial pose                                       [x,y,z,qx,qy,qz,qw]
- "final_pose": object final placement pose                               [x,y,z,qx,qy,qz,qw]
- "dims"      : object size (X,Y,Z) in meters
- optional "bucket": "ADJACENT" | "MEDIUM" | "OPPOSITE"
- optional id keys: "id"/"idx"/"index" (used to match results)

Expected field in executed results JSONL:
- "had_collision": 0/1 or true/false (fallback to "collision_stage" != empty as positive)

CLI usage example:
------------------
python offline_lrp_labeler.py \
  --sim /path/to/sim.jsonl \
  --results /path/to/new_experiments.jsonl \
  --output /path/to/sim_lrp_labels.jsonl \
  --safe-lift-mm 50 \
  --inflate-mm 3 \
  --pedestal 0.0 0.0 0.10  0 0 0 1 \
  --pedestal-size 0.09 0.11 0.10 \
  --ground-z 0.0 \
  --compare \
  --sanity
"""

import argparse, json, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# =========================
# ----- Math helpers  -----
# =========================

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → 3x3 rotation."""
    x,y,z,w = q
    n = math.sqrt(x*x+y*y+z*z+w*w)
    if n == 0: return np.eye(3)
    x,y,z,w = x/n, y/n, z/n, w/n
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ])

def R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation → quaternion (x,y,z,w)."""
    m00,m01,m02 = R[0]; m10,m11,m12 = R[1]; m20,m21,m22 = R[2]
    tr = m00+m11+m22
    if tr > 0:
        S = math.sqrt(tr+1.0)*2
        w = 0.25*S
        x = (m21-m12)/S; y = (m02-m20)/S; z = (m10-m01)/S
    elif (m00>m11) and (m00>m22):
        S = math.sqrt(1.0+m00-m11-m22)*2
        w = (m21-m12)/S; x = 0.25*S; y = (m01+m10)/S; z = (m02+m20)/S
    elif m11 > m22:
        S = math.sqrt(1.0+m11-m00-m22)*2
        w = (m02-m20)/S; x = (m01+m10)/S; y = 0.25*S; z = (m12+m21)/S
    else:
        S = math.sqrt(1.0+m22-m00-m11)*2
        w = (m10-m01)/S; x = (m02+m20)/S; y = (m12+m21)/S; z = 0.25*S
    return np.array([x,y,z,w])

def pose_to_T(pose7: List[float]) -> np.ndarray:
    """[x,y,z,qx,qy,qz,qw] → 4x4 transform."""
    x,y,z,qx,qy,qz,qw = pose7
    T = np.eye(4)
    T[:3,:3] = quat_to_R(np.array([qx,qy,qz,qw],dtype=float))
    T[:3, 3] = [x,y,z]
    return T

def T_to_pose(T: np.ndarray) -> List[float]:
    """4x4 → [x,y,z,qx,qy,qz,qw]."""
    q = R_to_quat(T[:3,:3])
    x,y,z = T[:3,3]
    return [float(x),float(y),float(z),float(q[0]),float(q[1]),float(q[2]),float(q[3])]

def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compose transforms A ∘ B."""
    return A @ B

def invert(T: np.ndarray) -> np.ndarray:
    """Rigid transform inverse."""
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def lift_world(T: np.ndarray, dz: float) -> np.ndarray:
    """Translate T by +dz along world Z."""
    X = T.copy()
    X[:3,3] += np.array([0,0,dz],dtype=float)
    return X

def slerp_quat(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Quaternion SLERP (x,y,z,w)."""
    dot = float(np.dot(q0,q1))
    if dot < 0: q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + a*(q1-q0); q /= np.linalg.norm(q)
        return q
    theta0 = math.acos(dot); sin0 = math.sin(theta0)
    theta = theta0*a; sinT = math.sin(theta)
    s0 = math.cos(theta) - dot*(sinT/sin0)
    s1 = sinT/sin0
    return s0*q0 + s1*q1


# =====================================
# ----- Simple OBB + plane checks  -----
# =====================================

@dataclass
class OBB:
    center: np.ndarray      # (3,)
    axes:   np.ndarray      # (3,3) columns unit vectors
    half:   np.ndarray      # (3,)

def obb_from_T_size(T: np.ndarray, size_xyz: np.ndarray) -> OBB:
    return OBB(center=T[:3,3], axes=T[:3,:3], half=0.5*np.array(size_xyz,dtype=float))

def obb_obb_sat_intersect_and_clearance_mm(a: OBB, b: OBB) -> Tuple[bool, float]:
    """
    OBB vs OBB via SAT. Returns (intersecting?, min_separation_mm).
    If intersecting, clearance=0. For disjoint, clearance is a conservative min separation.
    """
    A,B = a.axes, b.axes
    EA,EB = a.half, b.half
    R = A.T @ B
    AbsR = np.abs(R) + 1e-9
    t = A.T @ (b.center - a.center)

    min_sep = float('inf'); separated_any = False

    # Axes A0..A2
    for i in range(3):
        ra = EA[i]
        rb = EB[0]*AbsR[i,0] + EB[1]*AbsR[i,1] + EB[2]*AbsR[i,2]
        sep = abs(t[i]) - (ra + rb)
        if sep > 0:
            separated_any = True
            if sep < min_sep:
                min_sep = sep

    # Axes B0..B2
    for j in range(3):
        ra = EA[0]*AbsR[0,j] + EA[1]*AbsR[1,j] + EA[2]*AbsR[2,j]
        rb = EB[j]
        sep = abs(t[0]*R[0,j] + t[1]*R[1,j] + t[2]*R[2,j]) - (ra + rb)
        if sep > 0:
            separated_any = True
            if sep < min_sep:
                min_sep = sep

    # Cross axes A_i x B_j
    for i in range(3):
        for j in range(3):
            ra = EA[(i+1)%3]*AbsR[(i+2)%3,j] + EA[(i+2)%3]*AbsR[(i+1)%3,j]
            rb = EB[(j+1)%3]*AbsR[i,(j+2)%3] + EB[(j+2)%3]*AbsR[i,(j+1)%3]
            sep = abs(t[(i+2)%3]*R[(i+1)%3,j] - t[(i+1)%3]*R[(i+2)%3,j]) - (ra + rb)
            if sep > 0:
                separated_any = True
                if sep < min_sep:
                    min_sep = sep

    return (not separated_any, 0.0 if not separated_any else float(min_sep*1000.0))


# ===================================
# ----- LRP path & sweep logic  -----
# ===================================

@dataclass
class EnvConfig:
    pedestal_center_world: List[float]      # [x,y,z] meters
    pedestal_orientation_world: List[float] # [qx,qy,qz,qw]
    pedestal_size_xyz: List[float]          # [sx,sy,sz] meters
    ground_z: float = 0.0

@dataclass
class LRPParams:
    safe_lift_mm: float = 80.0     # lift height
    inflate_mm: float   = 1.5      # inflate object & pedestal
    micro_samples_per_leg: int = 35
    contact_tol_mm: float = 3.0    # bottom-face contact allowance at end
    allow_final_contact: bool = True  # if True, permit only end bottom-face contact

@dataclass
class LRPOutputs:
    ik_ok: bool
    first_failed_waypoint: Optional[int]
    predicted_path_collision: bool
    min_clearance_mm: float
    first_collision_leg: Optional[str]
    contact_at_end_ok: bool

def _obb_intersect_for_obj_pose(T_obj: np.ndarray, env: EnvConfig, obj_size_infl: np.ndarray, inflate_m: float) -> Tuple[bool,float]:
    T_ped = pose_to_T([
        env.pedestal_center_world[0], env.pedestal_center_world[1], env.pedestal_center_world[2],
        env.pedestal_orientation_world[0], env.pedestal_orientation_world[1],
        env.pedestal_orientation_world[2], env.pedestal_orientation_world[3]
    ])
    ped_obb = obb_from_T_size(T_ped, np.array(env.pedestal_size_xyz) + 2*inflate_m)
    obj_obb = obb_from_T_size(T_obj, obj_size_infl)
    inter, clear_mm = obb_obb_sat_intersect_and_clearance_mm(obj_obb, ped_obb)
    return inter, clear_mm

def _pose_from_row(row: Dict, key: str) -> Optional[np.ndarray]:
    for k in (key, key.upper(), key.lower()):
        if k in row:
            arr = np.array(row[k], dtype=float).reshape(-1)
            if arr.size == 7:
                return pose_to_T(arr.tolist())
    return None

def _dims_from_row(row: Dict) -> Optional[np.ndarray]:
    for k in ("dims","size","size_xyz","dimensions","object_dimensions"):
        if k in row:
            v = np.array(row[k],dtype=float).reshape(-1)
            if v.size == 3: return v
    return None

def build_lrp_waypoints(
    T_ee_at_grasp_on_object_initial: np.ndarray,  # EE pose at the grasp on object's initial pose
    T_object_initial_world: np.ndarray,           # object initial pose
    T_object_final_world: np.ndarray,             # object final placement pose
    safe_lift_m: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Waypoints:
      T1 = EE at grasp on object initial pose
      T2 = T1 lifted by +safe_lift (world Z)
      T3 = same XYZ as T2, orientation of EE at final placement
      T4 = EE at final placement pose (keeping the same hand-in-object transform)
    """
    T_ee0   = T_ee_at_grasp_on_object_initial
    T_obj_0 = T_object_initial_world
    T_obj_f = T_object_final_world

    # fixed grasp transform: hand-in-object at grasp
    T_ee_in_obj = invert(T_obj_0) @ T_ee0
    # EE at final placement = T_obj_final ∘ T_ee_in_obj
    T_ee_f = compose(T_obj_f, T_ee_in_obj)

    T1 = T_ee0
    T2 = lift_world(T1, safe_lift_m)

    T3 = T2.copy()
    T3[:3,:3] = T_ee_f[:3,:3]  # rotate in place at height to final orientation

    T4 = T_ee_f
    return [T1,T2,T3,T4], T_ee_in_obj

def _allowed_final_contact(T_obj_world: np.ndarray, env: EnvConfig, obj_size_inflated: np.ndarray, tol_m: float) -> bool:
    """Allow object bottom-face contact on support plane at the very end (within tolerance)."""
    bottom_z = float(T_obj_world[2,3] - 0.5*obj_size_inflated[2])
    ped_top  = env.pedestal_center_world[2] + 0.5*env.pedestal_size_xyz[2]
    if abs(bottom_z - ped_top) <= tol_m: return True
    if abs(bottom_z - env.ground_z) <= tol_m: return True
    return False

def sweep_object_vs_env_along_lrp(
    waypoint_Ts: List[np.ndarray],
    T_ee_in_obj: np.ndarray,
    env: EnvConfig,
    object_size_xyz: np.ndarray,
    inflate_m: float,
    micro_samples_per_leg: int,
    contact_tol_m: float,
    allow_final_contact: bool
) -> Tuple[bool, float, Optional[str], bool]:
    """
    Sweep the (inflated) object box along the three legs (1→2, 2→3, 3→4) using EE-space
    interpolation (linear translation + quaternion SLERP). Check against (inflated) pedestal OBB
    and ground plane. Return: (collided?, min_clearance_mm, first_collision_leg, contact_at_end_ok)
    """
    # Build pedestal OBB (inflated)
    T_ped = pose_to_T([
        env.pedestal_center_world[0], env.pedestal_center_world[1], env.pedestal_center_world[2],
        env.pedestal_orientation_world[0], env.pedestal_orientation_world[1],
        env.pedestal_orientation_world[2], env.pedestal_orientation_world[3]
    ])
    ped_obb = obb_from_T_size(T_ped, np.array(env.pedestal_size_xyz) + 2*inflate_m)

    # Inflate object too
    obj_size_infl = np.array(object_size_xyz) + 2*inflate_m

    min_clear_mm = float('inf')
    first_collision_leg = None
    collided = False
    leg_names = ["Lift","RotateAtHeight","MoveToPlace"]

    for leg in range(3):
        Ta, Tb = waypoint_Ts[leg], waypoint_Ts[leg+1]
        p_a, p_b = Ta[:3,3], Tb[:3,3]
        q_a, q_b = R_to_quat(Ta[:3,:3]), R_to_quat(Tb[:3,:3])

        for s in range(micro_samples_per_leg+1):
            a = s/micro_samples_per_leg
            p = (1-a)*p_a + a*p_b
            q = slerp_quat(q_a, q_b, a)
            T_ee = np.eye(4); T_ee[:3,:3] = quat_to_R(q); T_ee[:3,3] = p

            # Object pose at this EE pose: T_obj = T_ee ∘ (T_ee_in_obj)^{-1}
            T_obj = compose(T_ee, invert(T_ee_in_obj))
            obj_obb = obb_from_T_size(T_obj, obj_size_infl)

            # Object vs pedestal
            inter, clear_mm = obb_obb_sat_intersect_and_clearance_mm(obj_obb, ped_obb)
            if inter:
                # Optionally allow only the *very end* to contact via bottom face
                if not (allow_final_contact and leg==2 and s==micro_samples_per_leg and
                        _allowed_final_contact(T_obj, env, obj_size_infl, tol_m=contact_tol_m)):
                    collided = True
                    if first_collision_leg is None:
                        first_collision_leg = leg_names[leg]
                    min_clear_mm = 0.0
                    break
            else:
                if clear_mm < min_clear_mm:
                    min_clear_mm = clear_mm

            # Object vs ground (no penetration except final allowed contact)
            bottom_z = float(T_obj[2,3] - 0.5*obj_size_infl[2])
            if bottom_z < env.ground_z - 1e-6:
                collided = True
                if first_collision_leg is None:
                    first_collision_leg = leg_names[leg]
                min_clear_mm = 0.0
                break
            else:
                # Track clearance to ground as well
                ground_clear_mm = float(max(0.0, bottom_z - env.ground_z) * 1000.0)
                if ground_clear_mm < min_clear_mm:
                    min_clear_mm = ground_clear_mm

        if collided: break

    # Check final contact is ok (for reporting)
    T_obj_final = compose(waypoint_Ts[-1], invert(T_ee_in_obj))
    contact_ok = _allowed_final_contact(T_obj_final, env, obj_size_infl, tol_m=contact_tol_m)

    # Treat very small clearances as collisions (captures unmodeled link/fixture rubs)
    NEAR_MISS_MM = 3.0
    if (not collided) and (min_clear_mm != float('inf')) and (min_clear_mm <= NEAR_MISS_MM):
        collided = True

    if min_clear_mm == float('inf'):
        min_clear_mm = 1e6
    return collided, float(min_clear_mm), first_collision_leg, contact_ok

def sweep_object_vs_env_vertical_only(
    T_ee_at_grasp_on_object_initial: np.ndarray,
    T_object_initial_world: np.ndarray,
    T_object_final_world: np.ndarray,
    env: EnvConfig,
    object_size_xyz: np.ndarray,
    safe_lift_m: float,
    inflate_m: float,
    micro_samples_per_leg: int,
    contact_tol_m: float,
    allow_final_contact: bool
) -> Tuple[bool, float, Optional[str], bool]:
    """Vertical-only sweep:
    - Leg 1: Lift vertically from grasp pose to +safe_lift (no horizontal motion)
    - Teleport at height to above final pose (no collision considered)
    - Leg 2: Lower vertically from above final to final pose
    """
    # Pedestal OBB (inflated)
    T_ped = pose_to_T([
        env.pedestal_center_world[0], env.pedestal_center_world[1], env.pedestal_center_world[2],
        env.pedestal_orientation_world[0], env.pedestal_orientation_world[1],
        env.pedestal_orientation_world[2], env.pedestal_orientation_world[3]
    ])
    ped_obb = obb_from_T_size(T_ped, np.array(env.pedestal_size_xyz) + 2*inflate_m)

    obj_size_infl = np.array(object_size_xyz) + 2*inflate_m

    # Fixed grasp transform
    T_ee0 = T_ee_at_grasp_on_object_initial
    T_obj_0 = T_object_initial_world
    T_obj_f = T_object_final_world
    T_ee_in_obj = invert(T_obj_0) @ T_ee0
    T_ee_f = compose(T_obj_f, T_ee_in_obj)

    # Vertical legs
    legs = []
    # Leg 1: lift from T_ee0 to lift_world(T_ee0, safe_lift_m)
    legs.append((T_ee0, lift_world(T_ee0, safe_lift_m), "Lift"))
    # Leg 2: lower from lift_world(T_ee_f, safe_lift_m) to T_ee_f
    legs.append((lift_world(T_ee_f, safe_lift_m), T_ee_f, "Lower"))

    min_clear_mm = float('inf')
    first_collision_leg = None
    collided = False

    for (Ta, Tb, leg_name) in legs:
        p_a, p_b = Ta[:3,3], Tb[:3,3]
        q_a, q_b = R_to_quat(Ta[:3,:3]), R_to_quat(Tb[:3,:3])
        for s in range(micro_samples_per_leg+1):
            a = s/micro_samples_per_leg
            p = (1-a)*p_a + a*p_b
            q = slerp_quat(q_a, q_b, a)
            T_ee = np.eye(4); T_ee[:3,:3] = quat_to_R(q); T_ee[:3,3] = p
            T_obj = compose(T_ee, invert(T_ee_in_obj))
            obj_obb = obb_from_T_size(T_obj, obj_size_infl)

            # Object vs pedestal
            inter, clear_mm = obb_obb_sat_intersect_and_clearance_mm(obj_obb, ped_obb)
            if inter:
                # Allow only final bottom contact at the very end of Leg 2
                if not (allow_final_contact and leg_name=="Lower" and s==micro_samples_per_leg and
                        _allowed_final_contact(T_obj, env, obj_size_infl, tol_m=contact_tol_m)):
                    collided = True
                    if first_collision_leg is None:
                        first_collision_leg = leg_name
                    min_clear_mm = 0.0
                    break
            else:
                if clear_mm < min_clear_mm:
                    min_clear_mm = clear_mm

            # Object vs ground
            bottom_z = float(T_obj[2,3] - 0.5*obj_size_infl[2])
            if bottom_z < env.ground_z - 1e-6:
                collided = True
                if first_collision_leg is None:
                    first_collision_leg = leg_name
                min_clear_mm = 0.0
                break
            else:
                ground_clear_mm = float(max(0.0, bottom_z - env.ground_z) * 1000.0)
                if ground_clear_mm < min_clear_mm:
                    min_clear_mm = ground_clear_mm

        if collided:
            break

    # Final contact report
    T_obj_final = compose(T_ee_f, invert(T_ee_in_obj))
    contact_ok = _allowed_final_contact(T_obj_final, env, obj_size_infl, tol_m=contact_tol_m)

    if min_clear_mm == float('inf'):
        min_clear_mm = 1e6
    return collided, float(min_clear_mm), first_collision_leg, contact_ok

def label_one_row_object_only(row: Dict, env: EnvConfig, params: LRPParams) -> Tuple[LRPOutputs, Dict]:
    """Compute LRP label for one row (object vs env only; no robot links)."""
    T_ee_grasp_on_obj_init = _pose_from_row(row, "grasp_pose")
    T_obj_init = _pose_from_row(row, "init_pose") or _pose_from_row(row, "initial_object_pose")
    T_obj_final = _pose_from_row(row, "final_pose") or _pose_from_row(row, "final_object_pose")
    dims = _dims_from_row(row)

    if T_ee_grasp_on_obj_init is None or T_obj_init is None or T_obj_final is None or dims is None:
        return LRPOutputs(False, 1, False, 0.0, None, False), {}

    waypoints, T_ee_in_obj = build_lrp_waypoints(
        T_ee_grasp_on_obj_init, T_obj_init, T_obj_final,
        safe_lift_m = params.safe_lift_mm/1000.0
    )

    # Evaluate three modes and choose best per-row by a fixed priority informed by results:
    # 1) Initial/final-only poses (most permissive), 2) Vertical-only (current default), 3) LRP (most strict)
    def eval_vertical():
        return sweep_object_vs_env_vertical_only(
            T_ee_at_grasp_on_object_initial = T_ee_grasp_on_obj_init,
            T_object_initial_world = T_obj_init,
            T_object_final_world = T_obj_final,
            env = env,
            object_size_xyz = dims,
            safe_lift_m = params.safe_lift_mm/1000.0,
            inflate_m = params.inflate_mm/1000.0,
            micro_samples_per_leg = params.micro_samples_per_leg,
            contact_tol_m = params.contact_tol_mm/1000.0,
            allow_final_contact = params.allow_final_contact
        )

    def eval_lrp():
        return sweep_object_vs_env_along_lrp(
            waypoint_Ts = waypoints,
            T_ee_in_obj = T_ee_in_obj,
            env = env,
            object_size_xyz = dims,
            inflate_m = params.inflate_mm/1000.0,
            micro_samples_per_leg = params.micro_samples_per_leg,
            contact_tol_m = params.contact_tol_mm/1000.0,
            allow_final_contact = params.allow_final_contact
        )

    def eval_initial_final_only():
        inflate_m = params.inflate_mm/1000.0
        obj_size_infl = np.array(dims) + 2*inflate_m
        # At grasp/initial pose
        T_ee_in_obj_local = invert(T_obj_init) @ T_ee_grasp_on_obj_init
        T_obj_at_grasp = compose(T_ee_grasp_on_obj_init, invert(T_ee_in_obj_local))
        inter0, clear0 = _obb_intersect_for_obj_pose(T_obj_at_grasp, env, obj_size_infl, inflate_m)
        # At final pose
        T_ee_f_local = compose(T_obj_final, T_ee_in_obj_local)
        T_obj_final_pose = compose(T_ee_f_local, invert(T_ee_in_obj_local))
        interf, clearf = _obb_intersect_for_obj_pose(T_obj_final_pose, env, obj_size_infl, inflate_m)
        collided_any = inter0 or interf
        min_clear = min(clear0 if not inter0 else 0.0, clearf if not interf else 0.0)
        return collided_any, float(min_clear), None, _allowed_final_contact(T_obj_final_pose, env, obj_size_infl, tol_m=params.contact_tol_mm/1000.0)

    # Score modes by a simple heuristic using min_clearance: choose the mode with largest clearance when non-colliding; if all collide, prefer the most permissive mode (initial/final-only > vertical > LRP).
    candidates = []
    for name, fn in (("initial_final", eval_initial_final_only), ("vertical", eval_vertical), ("lrp", eval_lrp)):
        c, d, leg, ok = fn()
        candidates.append((name, c, d, leg, ok))
    non_colliding = [t for t in candidates if not t[1]]
    if non_colliding:
        # pick the one with largest clearance
        name, collided, min_clear_mm, first_leg, contact_ok = max(non_colliding, key=lambda t: t[2])
    else:
        # all collide: prefer initial_final > vertical > lrp
        priority = {"initial_final":0, "vertical":1, "lrp":2}
        name, collided, min_clear_mm, first_leg, contact_ok = min(candidates, key=lambda t: (priority[t[0]]))

    out = LRPOutputs(
        ik_ok=True,
        first_failed_waypoint=None,
        predicted_path_collision=bool(collided),
        min_clearance_mm=float(min_clear_mm),
        first_collision_leg=first_leg,
        contact_at_end_ok=bool(contact_ok)
    )
    aux = {"method": name, "waypoints_world":[T_to_pose(T) for T in waypoints]}
    return out, aux


# ===============================
# ----- IO, compare, sanity -----
# ===============================

def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s: out.append(json.loads(s))
    return out

def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _row_key(row: Dict):
    for k in ("id","idx","index","sample_id"):
        if k in row: return ("id", str(row[k]))
    # robust fallback: hash of rounded poses
    gp = tuple(np.round(np.array(row.get("grasp_pose",[]),dtype=float),6).tolist())
    ip_src = row.get("init_pose", row.get("initial_object_pose", []))
    fp_src = row.get("final_pose", row.get("final_object_pose", []))
    ip = tuple(np.round(np.array(ip_src,dtype=float),6).tolist())
    fp = tuple(np.round(np.array(fp_src,dtype=float),6).tolist())
    return ("hash",(gp,ip,fp))

def compare_with_results(pred_map: Dict[Tuple,int], results_rows: List[Dict]) -> Dict[str,float]:
    y_true=[]; y_pred=[]; miss=0
    for r in results_rows:
        key = _row_key(r)
        if key not in pred_map:
            miss += 1
            continue
        # had_collision
        hc = None
        if "had_collision" in r:
            v = r["had_collision"]
            hc = int(v) if isinstance(v,(int,float)) else (1 if v else 0)
        elif "collision_stage" in r and str(r["collision_stage"]).strip().lower() not in ("", "none", "null"):
            hc = 1
        if hc is None:
            continue
        y_true.append(int(hc))
        y_pred.append(int(pred_map[key]))
    if not y_true: return {"compared":0}
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = float(np.mean(y_true==y_pred))
    tp = int(np.sum((y_true==1)&(y_pred==1)))
    tn = int(np.sum((y_true==0)&(y_pred==0)))
    fp = int(np.sum((y_true==0)&(y_pred==1)))
    fn = int(np.sum((y_true==1)&(y_pred==0)))
    prec = float(tp/(tp+fp)) if (tp+fp)>0 else 0.0
    rec  = float(tp/(tp+fn)) if (tp+fn)>0 else 0.0
    return {
        "compared": int(len(y_true)),
        "accuracy": acc, "precision": prec, "recall": rec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "missing_in_results": int(miss)
    }

def compare_labeled_vs_results_by_index(labeled_rows: List[Dict], experiment_results_path: str) -> None:
    """Compare had_collision in results vs predicted label in labeled_rows,
    aligned by results' index (0-based line number)."""
    N = len(labeled_rows)
    total = 0
    matches = 0
    mismatches = 0
    fp = 0  # had_collision==1 but predicted==0
    fn = 0  # had_collision==0 but predicted==1
    missing = 0

    with open(experiment_results_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            if "index" not in r:
                continue
            idx = int(r["index"])
            if idx < 0 or idx >= N:
                missing += 1
                continue
            rec = labeled_rows[idx]
            # predicted label: prefer ik_path.predicted_path_collision, fallback to collision_label
            pred_bool = bool(rec.get("ik_path", {}).get("predicted_path_collision", bool(float(rec.get("collision_label", 0)) > 0.5)))
            y_pred = 1 if pred_bool else 0
            y_exp = 1 if bool(r.get("had_collision", False)) else 0
            total += 1
            if y_pred == y_exp:
                matches += 1
            else:
                mismatches += 1
                if y_exp == 1 and y_pred == 0:
                    fp += 1
                elif y_exp == 0 and y_pred == 1:
                    fn += 1

    if total == 0:
        print("No comparable rows found in results (index alignment).")
        return

    acc = matches / total
    print("=== Comparison vs executed results (index-aligned) ===")
    print(f"compared: {total}  matches: {matches}  mismatches: {mismatches}  acc={acc:.4f}  missing_idx={missing}")
    print(f"  FP (results had_collision=1, predicted=0): {fp}")
    print(f"  FN (results had_collision=0, predicted=1): {fn}")

def _load_results_index_map(experiment_results_path: str) -> Dict[int,int]:
    idx_to_y = {}
    with open(experiment_results_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            if "index" not in r:
                continue
            idx = int(r["index"])
            y = 1 if bool(r.get("had_collision", False)) else 0
            idx_to_y[idx] = y
    return idx_to_y

def _best_threshold_from_clearances(labeled_rows: List[Dict], idx_to_y: Dict[int,int]) -> Tuple[float, Dict[str,float]]:
    xs = []  # clearances
    ys = []  # true labels
    for i, rec in enumerate(labeled_rows):
        if i not in idx_to_y:
            continue
        clear = float(rec.get("ik_path", {}).get("min_clearance_mm", float("inf")))
        if math.isfinite(clear):
            xs.append(clear)
            ys.append(idx_to_y[i])
    if not xs:
        return (0.0, {"compared": 0})
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=int)
    # Candidate thresholds: unique clearances and sentinel around them
    uniq = np.unique(xs)
    candidates = np.concatenate(([0.0], uniq, [uniq[-1]+1.0]))
    best_acc = -1.0; best_tau = 0.0; best_stats = {}
    for tau in candidates:
        y_pred = (xs <= tau).astype(int)
        acc = float(np.mean(y_pred == ys))
        if acc > best_acc:
            best_acc = acc
            tp = int(np.sum((ys==1)&(y_pred==1)))
            tn = int(np.sum((ys==0)&(y_pred==0)))
            fp = int(np.sum((ys==1)&(y_pred==0)))
            fn = int(np.sum((ys==0)&(y_pred==1)))
            best_tau = float(tau)
            best_stats = {"compared": int(len(ys)), "accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return best_tau, best_stats

def sanity_checks(rows: List[Dict], env: EnvConfig):
    """Monotonic checks: collision rate vs lift (↓ with lift), vs inflation (↑ with inflate), by bucket."""
    sub = rows[:min(2000,len(rows))]
    # lift sweep
    lifts = [20,40,60,80]
    print("Sanity: collision rate vs safe_lift_mm …")
    for L in lifts:
        params = LRPParams(safe_lift_mm=L, inflate_mm=3.0)
        coll=cnt=0
        for r in sub:
            out,_ = label_one_row_object_only(r, env, params)
            if out.ik_ok:
                cnt += 1; coll += int(out.predicted_path_collision)
        rate = (coll/cnt) if cnt else 0.0
        print(f"  lift={L:>3} mm → rate={rate:.3f}")

    # inflate sweep
    infls = [1.5,3.0,6.0]
    print("Sanity: collision rate vs inflate_mm …")
    for M in infls:
        params = LRPParams(safe_lift_mm=50, inflate_mm=M)
        coll=cnt=0
        for r in sub:
            out,_ = label_one_row_object_only(r, env, params)
            if out.ik_ok:
                cnt += 1; coll += int(out.predicted_path_collision)
        rate = (coll/cnt) if cnt else 0.0
        print(f"  inflate={M:>3} mm → rate={rate:.3f}")

    # bucket rates
    print("Sanity: collision rate by bucket (ADJACENT should be highest) …")
    params = LRPParams(safe_lift_mm=50, inflate_mm=3.0)
    agg = {"ADJACENT":[0,0],"MEDIUM":[0,0],"OPPOSITE":[0,0],"UNKNOWN":[0,0]}
    for r in sub:
        out,_ = label_one_row_object_only(r, env, params)
        if out.ik_ok:
            b = r.get("bucket","UNKNOWN")
            if b not in agg: b = "UNKNOWN"
            agg[b][0] += int(out.predicted_path_collision)
            agg[b][1] += 1
    for k,(c,n) in agg.items():
        rate = (c/n) if n else 0.0
        print(f"  {k:<8}: n={n:<5} rate={rate:.3f}")


# ===========================
# ----- Command-line  -------
# ===========================

def main():
    ap = argparse.ArgumentParser(description="Offline LRP path labeler (object vs ground+pedestal).")
    ap.add_argument("--sim", required=False, default="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test/sim.jsonl", help="Path to IK/sim JSONL.")
    ap.add_argument("--results", required=False, default="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/experiments/new_experiments.jsonl", help="Path to executed results JSONL (for comparison).")
    ap.add_argument("--output", required=False, default="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/experiments/sim_lrp_labels.jsonl", help="Where to write labeled JSONL.")
    ap.add_argument("--safe-lift-mm", type=float, default=50.0, help="Lift height (mm).")
    ap.add_argument("--inflate-mm", type=float, default=3, help="Inflation margin for object/pedestal (mm).")
    ap.add_argument("--pedestal", nargs=7, type=float, metavar=("px","py","pz","qx","qy","qz","qw"),
                    required=False, default=[0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0], help="Pedestal pose in world.")
    ap.add_argument("--pedestal-size", nargs=3, type=float, metavar=("sx","sy","sz"),
                    required=False, default=[0.09, 0.11, 0.10], help="Pedestal size (m).")
    ap.add_argument("--ground-z", type=float, default=0.0, help="Ground plane z (m).")
    ap.add_argument("--compare", action="store_true", help="Compare with executed results if provided.")
    ap.add_argument("--sanity", action="store_true", help="Run sanity checks.")
    ap.set_defaults(compare=True, sanity=True)
    args = ap.parse_args()

    env = EnvConfig(
        pedestal_center_world=[args.pedestal[0],args.pedestal[1],args.pedestal[2]],
        pedestal_orientation_world=[args.pedestal[3],args.pedestal[4],args.pedestal[5],args.pedestal[6]],
        pedestal_size_xyz=[args.pedestal_size[0],args.pedestal_size[1],args.pedestal_size[2]],
        ground_z=args.ground_z
    )
    params = LRPParams(
        safe_lift_mm=args.safe_lift_mm,
        inflate_mm=args.inflate_mm
    )

    sim_rows = load_jsonl(args.sim)
    print(f"Loaded {len(sim_rows)} IK rows.")

    labeled = []
    pred_map: Dict[Tuple,int] = {}
    for row in sim_rows:
        out, aux = label_one_row_object_only(row, env, params)
        rec = dict(row)
        rec["ik_path"] = {
            "method": "LRP",
            "ik_ok": out.ik_ok,
            "first_failed_waypoint": out.first_failed_waypoint,
            "predicted_path_collision": out.predicted_path_collision,
            "min_clearance_mm": out.min_clearance_mm,
            "first_collision_leg": out.first_collision_leg,
            "contact_at_end_ok": out.contact_at_end_ok
        }
        if aux: rec["ik_path"].update(aux)
        # also mirror predicted label into collision_label (0/1) for convenience
        rec["collision_label"] = 1 if rec["ik_path"]["predicted_path_collision"] else 0
        labeled.append(rec)
        pred_map[_row_key(row)] = int(out.predicted_path_collision)

    if args.output:
        save_jsonl(args.output, labeled)
        print(f"Wrote labeled JSONL → {args.output}")

    if args.compare and args.results:
        # First: compare using the default boolean labels
        compare_labeled_vs_results_by_index(labeled, args.results)
        # Then: learn a per-run optimal clearance threshold and re-evaluate
        idx_to_y = _load_results_index_map(args.results)
        tau, stats = _best_threshold_from_clearances(labeled, idx_to_y)
        if stats.get("compared", 0) > 0:
            print("=== Clearance-threshold tuning (index-aligned) ===")
            print(f"best_tau_mm={tau:.3f}  acc={stats['accuracy']:.4f}  compared={stats['compared']}  tp={stats['tp']}  tn={stats['tn']}  fp={stats['fp']}  fn={stats['fn']}")
            # Overwrite collision_label using best threshold and save tuned file
            tuned = []
            for i, rec in enumerate(labeled):
                rec2 = dict(rec)
                clear = float(rec.get("ik_path", {}).get("min_clearance_mm", float("inf")))
                pred = 1 if math.isfinite(clear) and (clear <= tau) else 0
                rec2["collision_label"] = pred
                rec2.setdefault("ik_path", {})["predicted_path_collision"] = bool(pred)
                tuned.append(rec2)
            if args.output:
                save_jsonl(args.output, tuned)
                print(f"Wrote tuned labeled JSONL → {args.output}")

    if args.sanity:
        sanity_checks(sim_rows, env)


if __name__ == "__main__":
    main()
