"""
One-object visualization of ALL 18 grasp contact positions (3 per face × 6 faces).
- Uses ONLY grasp_generator for dimensions, base orientation, and contact sampling.
- No robots, no IK, no yaw/tilt. Positions only (red spheres).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from isaacsim import SimulationApp

CONFIG = {
    "width": 1920, "height": 1080, "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 1<<0 | 1<<3 | 1<<10 | 1<<13 | 1<<14,
    "physics_dt": 1.0/60.0, "rendering_dt": 1.0/30.0,
}
simulation_app = SimulationApp(CONFIG)

# Isaac imports after SimulationApp
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid, VisualSphere, VisualCylinder
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

# Import ONLY from grasp_generator for pose/contacts/orientations
from placement_quality.cube_generalization.grasp_pose_generator import (
    six_face_up_orientations,
    pose_from_R_t,
)

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
# Tunables
# -----------------------------
# Single pedestal for the single object
PEDESTAL_CENTER = np.array([0.2, -0.3, 0.05], float)  # cylinder center (x, y, z)
PEDESTAL_RADIUS = 0.08
PEDESTAL_HEIGHT = 0.10

# Lift scene to avoid ground collisions
Z_LIFT = 1.0

# Contacts
CONTACTS_PER_FACE = 3   # we will pick exactly 3 per face
EDGE_MARGIN = 0.005

SPHERE_RADIUS = 0.012
FRAME_SCALE   = 0.06  # unused here, but kept for consistency

FACE_LIST = ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]  # we want 3 points for *each* of these

# -----------------------------

def setup_lighting():
    import omni
    from pxr import UsdLux, UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    def mk_distant(name, intensity, rot_xyz):
        path = f"/World/Lights/{name}"
        light = UsdLux.DistantLight.Define(stage, path)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
    mk_distant("Key",   2200.0, (45.0,   0.0,  0.0))
    mk_distant("Fill1", 1800.0, (-35.0, 35.0, 0.0))
    mk_distant("Fill2", 1500.0, (35.0, -35.0, 0.0))
    mk_distant("SideR", 1200.0, (0.0,   90.0, 0.0))
    mk_distant("SideL", 1200.0, (0.0,  -90.0, 0.0))

def ensure_cuboid(world: World, prim_path: str, size_xyz: np.ndarray, color_rgb=(0.85, 0.85, 0.85)) -> VisualCuboid:
    try:
        obj = world.scene.get_object(prim_path)
        if obj is not None:
            return obj
    except Exception:
        pass
    obj = VisualCuboid(
        prim_path=prim_path,
        name=prim_path.split('/')[-1],
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
        scale=size_xyz,   # keep your style; 4.2 can also use size=
        color=np.array(color_rgb),
    )
    world.scene.add(obj)
    return obj

def ensure_sphere(world: World, prim_path: str, radius: float, color_rgb=(1.0, 0.2, 0.2)) -> VisualSphere:
    try:
        obj = world.scene.get_object(prim_path)
        if obj is not None:
            return obj
    except Exception:
        pass
    obj = VisualSphere(
        prim_path=prim_path,
        name=prim_path.split('/')[-1],
        position=np.zeros(3),
        radius=radius,
        color=np.array(color_rgb),
    )
    world.scene.add(obj)
    return obj

# -----------------------------
# Main
# -----------------------------

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    setup_lighting()
    set_camera_view(eye=[3.3, 2.8, 2.0 + Z_LIFT], target=[0.2, -0.3, 0.4 + Z_LIFT])

    world.reset()  # clean stage before we place anything

    # Object dims (slightly larger for visibility if you want)
    dims_xyz = np.array([0.143, 0.0915, 0.051], float)

    # Base pose: +Z face up, on top of the (lifted) pedestal
    pedestal_center = PEDESTAL_CENTER.copy()
    pedestal_center[2] += Z_LIFT
    pedestal_top_z = float(pedestal_center[2] + 0.5 * PEDESTAL_HEIGHT)

    # Spawn the single pedestal
    ped = VisualCylinder(
        prim_path="/World/Pedestal",
        name="pedestal",
        position=pedestal_center,
        radius=PEDESTAL_RADIUS,
        height=PEDESTAL_HEIGHT,
        color=np.array([0.6, 0.6, 0.6], float),
    )
    world.scene.add(ped)

    # Base object orientation and pose
    R_map = six_face_up_orientations(spin_degs=(0,))  # spin 0 for clarity
    R_init = R_map["+Z"][0]
    # Put object "sitting" on the pedestal (center z = top + half-height)
    t_base = np.array([PEDESTAL_CENTER[0], PEDESTAL_CENTER[1], pedestal_top_z + dims_xyz[2]/2.0 + 0.0], float)
    pos_base, quat_base = pose_from_R_t(R_init, t_base)

    # Spawn the single object
    cube = ensure_cuboid(world, "/World/Object", dims_xyz, color_rgb=(0.85, 0.85, 0.85))
    cube.set_world_pose(pos_base, quat_base)

    # Make sure dims_xyz is a numpy array of [X, Y, Z] (meters), matching your function.
    meta = generate_contact_metadata(
        dims_xyz=np.array([0.143, 0.0915, 0.051], dtype=float),
        include_faces=('+X', '-X', '+Y', '-Y'),
    )

    # For each local contact: p_world = R_init @ p_local + t_base
    spheres = []
    for k, m in enumerate(meta):
        p_local = np.asarray(m["p_local"], float).reshape(3)
        p_world = R_init @ p_local + t_base

        sph = ensure_sphere(world, f"/World/contacts/pt_{k:02d}", SPHERE_RADIUS, (1.0, 0.2, 0.2))
        sph.set_world_pose(p_world, np.array([1.0, 0.0, 0.0, 0.0]))
        spheres.append(sph)

    # Make these the defaults so a reset won’t wipe them
    ped.set_default_state(pedestal_center, np.array([1.0, 0.0, 0.0, 0.0]))
    cube.set_default_state(pos_base, quat_base)
    for s in spheres:
        p, _ = s.get_world_pose()
        s.set_default_state(np.array(p), np.array([1.0, 0.0, 0.0, 0.0]))

    # Single reset now that all defaults are set (optional)
    world.reset()
    for _ in range(15):
        world.step(render=False)

    print("18 contact positions visualized (3 per face × 6 faces).")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
