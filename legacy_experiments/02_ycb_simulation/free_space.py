import numpy as np

def quaternion_to_rot_matrix(q):
    """Convert quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R

def aperture_filter_numpy(corners, pos, quat, gripper_max_aperture):
    """
    corners: Nx3 array of object AABB corner positions in world coords
    pos:     [x, y, z] grasp frame origin in world coords
    quat:    [w, x, y, z] quaternion of grasp frame in world coords
    aperture: scalar max opening width of the gripper
    """
    R = quaternion_to_rot_matrix(quat)
    # Invert the transform: world → gripper local
    R_inv = R.T
    t = np.array(pos)
    
    # Transform corners into gripper-local frame
    local_pts = (R_inv @ (corners - t).T).T
    xs = local_pts[:, 0]
    width = xs.max() - xs.min()
    
    return width <= gripper_max_aperture, width

# Define a long thin rod: length=1.0 in X, thickness=0.04 in Y and Z
corners = np.array([
    [x, y, z]
    for x in (-0.5, 0.5)
    for y in (-0.02, 0.02)
    for z in (-0.02, 0.02)
])

gripper_max_aperture = 0.05  # 5 cm

# Test 1: Identity grasp frame (no rotation)
pos = [0, 0, 0]
quat_id = [1, 0, 0, 0]  # w, x, y, z
ok1, w1 = aperture_filter_numpy(corners, pos, quat_id, gripper_max_aperture)
print(f"Test 1 (identity): width = {w1:.3f} m → fits? {ok1}")

# Test 2: Grasp frame rotated 90° about Z (so gripper X aligns with world Y)
angle = np.pi / 2
quat_z = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
ok2, w2 = aperture_filter_numpy(corners, pos, quat_z, gripper_max_aperture)
print(f"Test 2 (Z-rot):    width = {w2:.3f} m → fits? {ok2}")
