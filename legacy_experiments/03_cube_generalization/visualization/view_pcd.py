import numpy as np
import open3d as o3d




def generate_box_pcd(dimensions, target_points=1024, seed=None):
    """
    Fast uniform sampling of ~target_points on a cuboid’s six faces.
    - points are in the object–local frame (centre at origin)
    - no post‑FPS needed unless you want exact count
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    L, W, H = map(float, dimensions)
    # areas of each pair of faces
    areas = np.array([L*W, L*W, L*H, L*H, W*H, W*H])
    probs = areas / areas.sum()          # sampling proportion per face
    counts = np.floor(probs * target_points).astype(int)
    counts[-1] = target_points - counts[:-1].sum()   # fix rounding

    pts = []
    # helpers: sample n points on a 2‑D rectangle
    def xy(n, z):
        u, v = rng.uniform(-L/2, L/2, n), rng.uniform(-W/2, W/2, n)
        return np.column_stack([u, v, np.full(n, z)])
    def xz(n, y):
        u, v = rng.uniform(-L/2, L/2, n), rng.uniform(-H/2, H/2, n)
        return np.column_stack([u, np.full(n, y), v])
    def yz(n, x):
        u, v = rng.uniform(-W/2, W/2, n), rng.uniform(-H/2, H/2, n)
        return np.column_stack([np.full(n, x), u, v])

    # +Z, −Z, +Y, −Y, +X, −X faces
    pts.extend(xy(counts[0],  H/2))
    pts.extend(xy(counts[1], -H/2))
    pts.extend(xz(counts[2],  W/2))
    pts.extend(xz(counts[3], -W/2))
    pts.extend(yz(counts[4],  L/2))
    pts.extend(yz(counts[5], -L/2))

    return np.asarray(pts, dtype=np.float32)






points = generate_box_pcd([0.08, 0.111, 0.149], 4096)


# === 2. Convert to Open3D PointCloud ===
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(points)

# === 3. Copy for visualization, color by Z ===
pcd_visual = pcd_original
points = np.asarray(pcd_visual.points)
colors = np.zeros((len(points), 3))
z_min, z_max = points[:, 2].min(), points[:, 2].max()
z_normalized = (points[:, 2] - z_min) / (z_max - z_min)
colors[:, 0] = 0.0           # No red
colors[:, 1] = z_normalized  # Green
colors[:, 2] = 1.0 - z_normalized  # Blue
pcd_visual.colors = o3d.utility.Vector3dVector(colors)

# === 4. Local coordinate frame at cuboid center ===
box_center = np.mean(points, axis=0)
frame_size = 0.05
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=frame_size, origin=box_center
)

# === 5. Visualize ===
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Cube PCD with Local Coordinate Frame')
vis.add_geometry(pcd_visual)
vis.add_geometry(coordinate_frame)
ctr = vis.get_view_control()
ctr.set_front([0.5, -0.5, -0.7])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, 0, 1])
ctr.set_zoom(0.8)
vis.run()
vis.destroy_window()

print("Original PCD has", len(pcd_original.points), "points and is unmodified")
print(f"Local coordinate frame displayed at: {box_center}")
print("Red arrow = X-axis, Green arrow = Y-axis, Blue arrow = Z-axis")
