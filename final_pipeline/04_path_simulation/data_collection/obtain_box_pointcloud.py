import numpy as np
from scipy.spatial.transform import Rotation as R

# Isaac Sim imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # Set headless=False for GUI!

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

def sample_box_surface(dx, dy, dz, n_points_per_side=100):
    xs = np.linspace(-dx/2, dx/2, n_points_per_side)
    ys = np.linspace(-dy/2, dy/2, n_points_per_side)
    zs = np.linspace(-dz/2, dz/2, n_points_per_side)
    points = []

    # +X face
    Y, Z = np.meshgrid(ys, zs)
    X = np.full_like(Y, dx/2)
    points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    # -X face
    X = np.full_like(Y, -dx/2)
    points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    # +Y face
    X, Z = np.meshgrid(xs, zs)
    Y = np.full_like(X, dy/2)
    points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    # -Y face
    Y = np.full_like(X, -dy/2)
    points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    # +Z face
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, dz/2)
    points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    # -Z face
    Z = np.full_like(X, -dz/2)
    points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))

    return np.vstack(points)  # shape (N, 3)

def save_pcd(filename, points):
    """Saves points (N,3) as a PCD file (ASCII)"""
    N = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA ascii\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved point cloud to {filename}, shape: {points.shape}")

def main():
    # 1. Start Isaac Sim world
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
    world.scene.add_default_ground_plane()

    # Randomly sample box dimensions ensuring at least one graspable dimension (≤8cm)
    while True:
        dx = np.random.uniform(0.03, 0.20)
        dy = np.random.uniform(0.03, 0.20)
        dz = np.random.uniform(0.03, 0.20)
        if min(dx, dy, dz) <= 0.08:
            break


    # 3. Create the cube at the origin (no rotation)
    box = DynamicCuboid(
        prim_path="/World/PerfectBox",
        name="PerfectBox",
        position=np.array([0.0, 0.0, dz/2]),  # sit on ground
        scale=np.array([dx, dy, dz])
    )
    world.scene.add(box)

    # 4. Step sim a few times to ensure placement
    for _ in range(5):
        world.step(render=True)

    # 5. Get the cube's pose (position, orientation as quaternion)
    position, orientation = box.get_world_pose()
    quat_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])

    # 6. Sample surface points in local frame
    n_points_per_side = 30  # adjust for density
    points_local = sample_box_surface(dx, dy, dz, n_points_per_side=n_points_per_side)
    points_local_centered = points_local - points_local.mean(axis=0)

    # 7. Transform to world frame
    rot = R.from_quat(quat_xyzw)
    points_world = rot.apply(points_local_centered) + position

    # 8. Save as PCD
    save_pcd(f"box_cube_{dx:.3f}_{dy:.3f}_{dz:.3f}.pcd", points_world)

    print("Cube created and PCD saved. Isaac Sim GUI will remain open for inspection.")
    print("Press Ctrl+C or close the Isaac Sim window to end the script.")

    # 9. Keep the simulation open for visual inspection
    while simulation_app.is_running():
        world.step(render=True)

if __name__ == "__main__":
    main()
