import open3d as o3d

# Replace 'your_file.pcd' with the path to your PCD file
# pcd = o3d.io.read_point_cloud("/home/chris/Chris/placement_ws/src/collected.pcd")
pcd = o3d.io.read_point_cloud("/home/chris/Chris/placement_ws/src/pcds/pcd_0.pcd")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])