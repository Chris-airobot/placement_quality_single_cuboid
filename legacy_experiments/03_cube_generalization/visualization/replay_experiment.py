import json
from typing import List, Dict, Tuple

import numpy as np
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


# Layout
GRID_COLS = 5
GRID_ROWS = 3
CELL_SPACING_X = 1.6
CELL_SPACING_Y = 1.4

# Pedestal
PEDESTAL_RADIUS = 0.08
PEDESTAL_HEIGHT = 0.10
PEDESTAL_CENTER_Z = 0.05
PEDESTAL_TOP_Z = PEDESTAL_CENTER_Z + 0.5 * PEDESTAL_HEIGHT

ROBOT_BASE_OFFSET = np.array([-0.35, 0.0, 0.0])
FRAME_SCALE = 0.06
SPHERE_RADIUS = 0.010

# ---- Simple inline configuration (edit here; no CLI needed) ----
DEFAULT_JSON_PATH = \
    "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_generation_v2.json"
# Mode: if True, display all grasps sampled by STRIDE; if False, group by REF_INDEX
REPLAY_ALL = True
REF_INDEX = 0
LIMIT = 0          # 0 = no cap
STRIDE = 5         # sample every k-th record when REPLAY_ALL is True
SPAWN_ROBOTS = False  # set True to spawn robots (heavier)


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
    return ArticulationKinematicsSolver(articulation, lula, "panda_hand")


def load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def group_by_init_final(records: List[Dict], ref_idx: int, atol: float = 1e-6) -> List[int]:
    ref = records[ref_idx]
    init_ref = np.array(ref["initial_object_pose"], dtype=float)
    final_ref = np.array(ref["final_object_pose"], dtype=float)
    out: List[int] = []
    for i, r in enumerate(records):
        same_init = np.allclose(init_ref, np.array(r["initial_object_pose"], float), atol=atol)
        same_final = np.allclose(final_ref, np.array(r["final_object_pose"], float), atol=atol)
        if same_init and same_final:
            out.append(i)
    return out


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    setup_lighting()
    set_camera_view(eye=[16.0, 13.0, 8.5], target=[0.0, 0.0, 0.4])
    world.reset()

    records = load_json(DEFAULT_JSON_PATH)
    if REPLAY_ALL:
        idxs = list(range(0, len(records), max(1, int(STRIDE))))
        if LIMIT and LIMIT > 0:
            idxs = idxs[: LIMIT]
        layout_cols = int(np.ceil(np.sqrt(len(idxs)))) if idxs else 0
        layout_rows = int(np.ceil(len(idxs) / max(1, layout_cols))) if idxs else 0
    else:
        idxs = group_by_init_final(records, REF_INDEX)
        if LIMIT and LIMIT > 0:
            idxs = idxs[: LIMIT]
        layout_cols = GRID_COLS
        layout_rows = GRID_ROWS

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

    for k, idx in enumerate(idxs):
        rec = records[idx]
        dims_xyz = np.array(rec["object_dimensions"], dtype=float)
        init_pose = np.array(rec["initial_object_pose"], dtype=float)
        grasp_pose = np.array(rec["grasp_pose"], dtype=float)

        # Determine layout position
        if REPLAY_ALL:
            col = k % max(1, layout_cols)
            row = k // max(1, layout_cols)
        else:
            row = k // GRID_COLS
            col = k % GRID_COLS
        cell_xy = np.array([col * CELL_SPACING_X, -row * CELL_SPACING_Y], dtype=float)

        ped_center = np.array([cell_xy[0], cell_xy[1], PEDESTAL_CENTER_Z], dtype=float)
        ped_top_z = float(ped_center[2] + 0.5 * PEDESTAL_HEIGHT)

        # Place object at its initial pose, but shifted into the cell
        pos_i = init_pose[:3]
        quat_i = init_pose[3:]
        pos_i_cell = np.array([pos_i[0] + cell_xy[0], pos_i[1] + cell_xy[1], pos_i[2]], dtype=float)

        ped = VisualCylinder(
            prim_path=f"/World/Pedestal_{k:02d}",
            name=f"pedestal_{k:02d}",
            position=ped_center,
            radius=float(PEDESTAL_RADIUS),
            height=float(PEDESTAL_HEIGHT),
            color=np.array([0.6, 0.6, 0.6], dtype=float),
        )
        world.scene.add(ped)

        cube = ensure_cuboid(world, f"/World/Object_{k:02d}", dims_xyz, color_rgb=colors[k % len(colors)])
        cube.set_world_pose(pos_i_cell, quat_i)

        # panda_hand grasp target directly (already corrected in JSON), shifted to the cell
        tgt = add_frame_prim(f"/World/target_{k:02d}")
        grasp_pos_cell = np.array([
            grasp_pose[0] + cell_xy[0],
            grasp_pose[1] + cell_xy[1],
            grasp_pose[2],
        ], dtype=float)
        tgt.set_world_pose(grasp_pos_cell, grasp_pose[3:])

        # Robot setup
        if SPAWN_ROBOTS:
            robot = add_franka(world, f"/World/panda_{k:02d}", name=f"franka_robot_{k:02d}")
            base_pose = np.array([pos_i_cell[0], pos_i_cell[1], 0.0], dtype=float) + ROBOT_BASE_OFFSET
            robot.set_world_pose(base_pose, np.array([1.0, 0.0, 0.0, 0.0]))
            kin = setup_kinematics(robot)

            robots.append(robot)
            ik_solvers.append(kin)
            targets.append(tgt)

            # Defaults
            XFormPrim(robot.prim_path).set_default_state(base_pose, np.array([1.0, 0.0, 0.0, 0.0]))

        ped.set_default_state(ped_center, np.array([1.0, 0.0, 0.0, 0.0]))
        cube.set_default_state(pos_i_cell, quat_i)
        tgt.set_default_state(grasp_pos_cell, grasp_pose[3:])

    world.reset()
    if REPLAY_ALL:
        print(f"Replaying {len(idxs)} grasps (all mode){' with robots' if SPAWN_ROBOTS else ''}")
    else:
        print(f"Replaying {len(idxs)} grasps grouped by trial {REF_INDEX}{' with robots' if SPAWN_ROBOTS else ''}")

    try:
        while simulation_app.is_running():
            if SPAWN_ROBOTS:
                for robot, kin, tgt in zip(robots, ik_solvers, targets):
                    tgt_pos, tgt_quat = tgt.get_world_pose()
                    grasp_position = np.array(tgt_pos, dtype=float)
                    grasp_orientation = np.array(tgt_quat, dtype=float)
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

