"""
Object pose visualizer for Isaac Sim that groups FINAL poses under each INITIAL pose
using ONLY pose utilities from grasp_generator.py, with hardcoded spacing so cubes are clearly separated.
"""

import os
import sys
from typing import Dict, List

from isaacsim import SimulationApp
CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 1<<0 | 1<<3 | 1<<10 | 1<<13 | 1<<14,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
}

simulation_app = SimulationApp(CONFIG)

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from placement_quality.cube_generalization.grasp_pose_generator import sample_dims, six_face_up_orientations, pose_from_R_t


PEDESTAL_TOP_Z = 0.10
INIT_SPINS_DEG = (0, 90, 180)
N_FINAL_SPINS_PER_FACE = 2
FINAL_SPIN_CANDIDATES_DEG = (0, 90, 180, 270)

# Hard separation distances
INIT_X_SPACING = 3.0     # space between each initial block in X
FINAL_X_SPACING = 1.0    # space between finals in X within a block
FINAL_Y_SPACING = 1.0    # space between rows of finals


def ensure_cuboid(world: World, prim_path: str, size_xyz: np.ndarray, color_rgb=(0.8, 0.8, 0.8)) -> VisualCuboid:
    try:
        obj = world.scene.get_object(prim_path)
        if obj is not None:
            return obj
    except Exception:
        pass
    obj = VisualCuboid(
        prim_path=prim_path,
        name=os.path.basename(prim_path),
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=size_xyz,
        color=np.array(color_rgb),
    )
    world.scene.add(obj)
    return obj

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


def main():
    rng = np.random.default_rng(77)
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    dims_xyz = np.array(sample_dims(1, seed=77)[0], dtype=float)

    for init_idx, init_spin_deg in enumerate(INIT_SPINS_DEG):
        R_init_map = six_face_up_orientations(spin_degs=(init_spin_deg,))
        R_init = R_init_map["+Z"][0]
        init_x = init_idx * INIT_X_SPACING
        t_init = np.array([init_x, 0.0, PEDESTAL_TOP_Z])
        pos_i, quat_i = pose_from_R_t(R_init, t_init)
        cube_init = ensure_cuboid(world, f"/World/Initial_{init_idx:02d}", dims_xyz, color_rgb=(0.85, 0.85, 0.85))
        cube_init.set_world_pose(np.array(pos_i), np.array(quat_i))
        print(f"Initial pose {init_idx:02d} at {pos_i} {quat_i}")

        spins_this_block = tuple(sorted(rng.choice(FINAL_SPIN_CANDIDATES_DEG, size=N_FINAL_SPINS_PER_FACE, replace=False)))
        R_final_map: Dict[str, List[np.ndarray]] = six_face_up_orientations(spin_degs=spins_this_block)
        face_list = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        for face_idx, face in enumerate(face_list):
            R_list = R_final_map[face]
            row = face_idx // 3
            col = face_idx % 3
            base_x = init_x + (col - 1) * FINAL_X_SPACING
            base_y = -(row + 1) * FINAL_Y_SPACING

            for spin_idx, R_fin in enumerate(R_list):
                offset_x = base_x + (spin_idx - (len(R_list)-1)/2.0) * 0.3
                t_fin = np.array([offset_x, base_y, PEDESTAL_TOP_Z])
                pos_f, quat_f = pose_from_R_t(R_fin, t_fin)
                face_safe = face.replace('+', 'pos').replace('-', 'neg')
                prim_path = f"/World/Final_{init_idx:02d}_{face_safe}_{spin_idx:02d}"
                face_color = {
                    "+Z": (1.0, 0.6, 0.6), "-Z": (0.6, 1.0, 0.6),
                    "+X": (0.6, 0.6, 1.0), "-X": (1.0, 1.0, 0.6),
                    "+Y": (1.0, 0.6, 1.0), "-Y": (0.6, 1.0, 1.0),
                }.get(face, (0.8, 0.8, 0.8))
                cube_fin = ensure_cuboid(world, prim_path, dims_xyz, color_rgb=face_color)
                cube_fin.set_world_pose(np.array(pos_f), np.array(quat_f))
                print(f"Final pose {init_idx:02d}_{face_safe}_{spin_idx:02d} at {pos_f} {quat_f}, it's face ID is {face_id_from_R(R_fin)}")

    try:
        while simulation_app.is_running():
            world.step(render=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
