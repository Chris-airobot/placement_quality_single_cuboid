import open3d as o3d
import numpy as np

# Load the PCD (original remains unchanged)
pcd_original = o3d.io.read_point_cloud("/home/chris/Chris/placement_ws/src/box_cube_0.031_0.096_0.190.pcd")
print("Loaded", pcd_original)

# Create a copy for visualization (original stays unchanged)
pcd_visual = pcd_original

# Create smooth blue-to-green gradient based on height (Z coordinate)
points = np.asarray(pcd_visual.points)
colors = np.zeros((len(points), 3))

# Normalize Z coordinates to [0, 1]
z_min, z_max = points[:, 2].min(), points[:, 2].max()
z_normalized = (points[:, 2] - z_min) / (z_max - z_min)

# Create blue-to-green gradient
colors[:, 0] = 0.0  # No red component
colors[:, 1] = z_normalized  # Green increases with height
colors[:, 2] = 1.0 - z_normalized  # Blue decreases with height

# Apply colors only to the copy for visualization
pcd_visual.colors = o3d.utility.Vector3dVector(colors)

# Create coordinate frame to show local axes
# Calculate the center of the point cloud dynamically
points = np.asarray(pcd_original.points)
box_center = np.mean(points, axis=0)  # Center of the point cloud

# Create coordinate frame (scale based on box size for visibility)
frame_size = 0.05  # 5cm axes for visibility
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=frame_size, origin=box_center
)

# Create visualizer with custom view
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Cube PCD with Local Coordinate Frame')
vis.add_geometry(pcd_visual)  # Use the colored copy
vis.add_geometry(coordinate_frame)  # Add the coordinate frame

# Set a good viewing angle (isometric-like view)
ctr = vis.get_view_control()
ctr.set_front([0.5, -0.5, -0.7])  # View direction
ctr.set_lookat([0, 0, 0])  # Look at origin
ctr.set_up([0, 0, 1])  # Z-axis up
ctr.set_zoom(0.8)  # Zoom level

# Run the visualizer
vis.run()
vis.destroy_window()

# Original PCD remains completely unchanged
print("Original PCD has", len(pcd_original.points), "points and is unmodified")

# Print coordinate frame information
print(f"Local coordinate frame displayed at: {box_center}")
print("Red arrow = X-axis, Green arrow = Y-axis, Blue arrow = Z-axis")

# Uncomment below to see point cloud statistics
# points = np.asarray(pcd_original.points)
# print("X: min = %.6f, max = %.6f, range = %.6f" % (points[:,0].min(), points[:,0].max(), points[:,0].max()-points[:,0].min()))
# print("Y: min = %.6f, max = %.6f, range = %.6f" % (points[:,1].min(), points[:,1].max(), points[:,1].max()-points[:,1].min()))
# print("Z: min = %.6f, max = %.6f, range = %.6f" % (points[:,2].min(), points[:,2].max(), points[:,2].max()-points[:,2].min()))
