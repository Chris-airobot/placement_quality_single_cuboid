import numpy as np
from scipy.spatial.transform import Rotation as R
# from placement_quality.path_simulation.model_testing.helper import *








def get_grasp_points_top_left_right(box_pose, box_dims, up_axis):
    dx, dy, dz = box_dims
    # Define box local axes
    local_axes = np.eye(3)  # x, y, z
    t = np.array(box_pose[:3])
    q = np.array(box_pose[3:])
    r = R.from_quat([q[0], q[1], q[2], q[3]])  # [x, y, z, w]
    world_axes = r.apply(local_axes)

    # Find which local axis is most aligned with world z
    # axis_scores = np.abs(world_axes @ np.array(world_z_axis))
    # up_axis = np.argmax(axis_scores)
    
    # Calculate the sign based on the z-component of the up_axis in world coordinates
    # world_axes[up_axis, 2] gives the z-component of the local up_axis in world frame
    up_sign = np.sign(world_axes[up_axis, 2])
    
    # Handle the case where the z-component is exactly 0 (identity orientation)
    # In this case, we need to determine the sign based on the intended orientation
    if up_sign == 0:
        # For identity orientation, if up_axis is 0 (x-axis), we want positive x to be "up"
        # So we set up_sign to 1 to make the top face point in the positive x direction
        up_sign = 1
    
    # Map index to axis name for reference
    axis_names = ['x', 'y', 'z']
    up_axis_name = axis_names[up_axis]
    if up_axis_name == 'x':
        top_sign = up_sign
        left_axis = 'y'
        right_axis = 'y'
        left_sign = 1
        right_sign = -1
        # Build local positions for the three faces
        top_local_pts = [
            [top_sign*dx/2, 0, 0],                    # top center
            [top_sign*dx/2, dy/4, 0],                 # top left edge
            [top_sign*dx/2, -dy/4, 0],                # top right edge
        ]
        left_local_pts = [
            [0, left_sign*dy/2, 0],
            [dx/4, left_sign*dy/2, 0],
            [-dx/4, left_sign*dy/2, 0],
        ]
        right_local_pts = [
            [0, right_sign*dy/2, 0],
            [dx/4, right_sign*dy/2, 0],
            [-dx/4, right_sign*dy/2, 0],
        ]
        top_corners = [
            [top_sign*dx/2, left_sign*dy/2, 0],
            [top_sign*dx/2, right_sign*dy/2, 0]
        ]
    elif up_axis_name == 'z':
        top_sign = up_sign
        left_axis = 'x'
        right_axis = 'x'
        left_sign = -1
        right_sign = 1
        # Build local positions for the three faces
        top_local_pts = [
            [0, 0, top_sign*dz/2],                    # top center
            [dx/4, 0, top_sign*dz/2],                 # top left edge
            [-dx/4, 0, top_sign*dz/2],                # top right edge
        ]
        right_local_pts = [
            [left_sign*dx/2, 0, 0],
            [left_sign*dx/2, dy/4, 0],
            [left_sign*dx/2, -dy/4, 0],
        ]
        left_local_pts = [
            [right_sign*dx/2, 0, 0],
            [right_sign*dx/2, dy/4, 0],
            [right_sign*dx/2, -dy/4, 0],
        ]
        top_corners = [
            [left_sign*dx/2, 0, top_sign*dz/2],
            [right_sign*dx/2, 0, top_sign*dz/2]
        ]
    else:
        raise NotImplementedError("Only x or z as top face supported in this implementation.")

    # Combine all points and transform to world
    local_pts = np.array(top_local_pts + left_local_pts + right_local_pts + top_corners)
    world_pts = r.apply(local_pts) + t
    return world_pts




def get_grasp_pose(box_pose, box_dims):
    """
    box_pose: [x, y, z, qx, qy, qz, qw]  (world)
    box_dims: [dx, dy, dz]
    Returns:
        poses: (33, 7) array of [x, y, z, qx, qy, qz, qw]
        metadata: list of (point_type, angle) for each pose
    """
    up_axis = 0
    # 1. Positions (assumed to be correct)
    pts = get_grasp_points_top_left_right(box_pose, box_dims, up_axis)
    point_types = [
        'top_center', 'top_left', 'top_right',
        'left_center', 'left_top', 'left_bottom',
        'right_center', 'right_top', 'right_bottom',
        'corner1', 'corner2'
    ]

    # 2. Find base orientation for each point
    face_groups = {
        'top':    [0, 1, 2],
        'left':   [3, 4, 5],
        'right':  [6, 7, 8],
        'corner': [9, 10],
    }

    box_center = np.array(box_pose[:3])
    r_base_list = [None] * len(pts)

    # a) SURFACE POINTS
    for face in ['top', 'left', 'right']:
        idxs = face_groups[face]
        # You should define the *surface normal* for each face
        if face == 'top':
            normal_local = np.array([1, 0, 0])
        elif face == 'left':
            normal_local = np.array([0, 1, 0])
        elif face == 'right':
            normal_local = np.array([0, -1, 0])
        # Convert normal to world frame
        q = np.array(box_pose[3:])
        box_rot = R.from_quat(q)  # [x, y, z, w]
        normal_world = box_rot.apply(normal_local)
        approach_dir = -normal_world  # Gripper Z points opposite the normal
        # Now construct rotation: gripper Z aligns with approach_dir
        # We'll set gripper Y arbitrarily (as long as it is orthogonal), then fix roll via ±45° later.
        z_axis = approach_dir / np.linalg.norm(approach_dir)
        # Choose Y as "up" in world, project out Z component
        y_axis = np.array([0, 0, 1])
        if np.abs(np.dot(z_axis, y_axis)) > 0.99:
            y_axis = np.array([1, 0, 0])
        y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot_mtx = np.column_stack((x_axis, y_axis, z_axis))
        r_base = R.from_matrix(rot_mtx)
        # if face == "top":
        #     r_base = r_base * R.from_euler('z', 90, degrees=True)   # <--- THE MISSING LINE!
        # if face == "left":
        #     r_base = r_base * R.from_euler('z', 90, degrees=True)   # <--- THE MISSING LINE!
        # if face == "right":
        #     r_base = r_base * R.from_euler('z', 90, degrees=True)   # <--- THE MISSING LINE!
        # r_base = r_base * R.from_euler('z', 90, degrees=True)   # <--- for [0.0915, 0.051,  0.143]
        # r_base = r_base * R.from_euler('x', 90, degrees=True)   # <--- for [0.143, 0.0915,  0.051]
        # Assign to all three points on that face
        for idx in idxs:
            r_base_list[idx] = r_base

    # b) CORNER POINTS
    for i, idx in enumerate(face_groups['corner']):
        point = pts[idx]
        # Approach: from point to box center (gripper Z)
        approach_dir = box_center - point
        z_axis = approach_dir / np.linalg.norm(approach_dir)
        # Y: any orthogonal vector, here up-world projected out Z
        y_axis = np.array([0, 0, 1])
        if np.abs(np.dot(z_axis, y_axis)) > 0.99:
            y_axis = np.array([1, 0, 0])
        y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot_mtx = np.column_stack((x_axis, y_axis, z_axis))
        r_base = R.from_matrix(rot_mtx)
        # print("Before:", r_base.as_euler('xyz', degrees=True))
        r_base = r_base * R.from_euler('x', 90, degrees=True)
        # print("After:", r_base.as_euler('xyz', degrees=True))

        r_base_list[idx] = r_base

    # 3. Generate 3 orientations for each point by rotating about gripper Y axis
    all_poses, all_metadata = [], []
    y_angles = [0, -45, 45]  # degrees; negative is left, positive is right (in right-handed frame)
    for idx, (point, r_base) in enumerate(zip(pts, r_base_list)):
        for angle in y_angles:
            r_y = R.from_euler('y', angle, degrees=True)
            r_final = r_base * r_y  # Apply rotation about gripper Y
            quat = r_final.as_quat()  # [x, y, z, w]
            pose = [point[0], point[1], point[2], quat[0], quat[1], quat[2], quat[3]]  # xyzw
            all_poses.append(pose)
            all_metadata.append((point_types[idx], angle))
    
    return np.array(all_poses), all_metadata






def get_prepose(grasp_pos, grasp_quat, offset=0.15):
    # grasp_quat: [w, x, y, z]
    # scipy uses [x, y, z, w]!
    grasp_quat_xyzw = [grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]
    rot = R.from_quat(grasp_quat_xyzw)
    # Get approach direction (z-axis of gripper in world frame)
    approach_dir = rot.apply([0, 0, 1])  # [0, 0, 1] is z-axis
    # Compute pregrasp position (move BACK along approach vector)
    pregrasp_pos = np.array(grasp_pos) - offset * approach_dir
    return pregrasp_pos, grasp_quat  # Same orientation
    


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


def get_flipped_object_pose(initial_pose, flip_angle_deg, axis='z'):
    """Return new pose after flipping initial_pose by flip_angle_deg about axis."""
    position = np.array(initial_pose[:3])
    quat = np.array(initial_pose[3:])
    r_initial = R.from_quat(quat)
    r_flip = R.from_euler(axis, flip_angle_deg, degrees=True)
    r_target = r_flip * r_initial  # or r_initial * r_flip, depending on convention
    quat_target = r_target.as_quat()
    return np.concatenate([position, quat_target])  # [x, y, z, qx, qy, qz, qw]

if __name__ == "__main__":
    import json
    box_pose = [0, 0, 0, 0, 0, 0, 1]
    # box_dims = [0.0915, 0.051,  0.143]
    box_dims = [0.143, 0.0915,  0.051]
    grasps, _ = get_grasp_pose(box_pose, box_dims)
    output = {}
    for i, pose in enumerate(grasps):
        x, y, z, qx, qy, qz, qw = pose
        output[str(i+1)] = {
            "position": [x, y, z],
            "orientation_wxyz": [qw, qx, qy, qz]  # Convert to WXYZ order
        }

    # To pretty-print or save as JSON:
    with open("/home/chris/Chris/placement_ws/src/placement_quality/path_simulation/model_testing/actual_box_grasp_test.json", "w") as f:
        json.dump(output, f, indent=4)

    from placement_quality.ycb_simulation.utils.helper import draw_frame, transform_relative_pose, local_transform

    test_data = []
    object_initial_pose = [0.2, -0.3, 0.1715, 0.5, 0.5, -0.5, 0.5]
    flip_angles = [180]
    for flip_angle in flip_angles:
        final_pose = get_flipped_object_pose(object_initial_pose, flip_angle, axis='y')
        for i in range(0,27):
            x, y, z, qx, qy, qz, qw = grasps[i]
            grasp_pose = [x, y, z, qw, qx, qy, qz]
            grasp_offset = [0, 0, -0.1023]
            grasp_pose_local = [grasp_pose[:3], grasp_pose[3:]]
            world_grasp_pose = transform_relative_pose(grasp_pose_local, object_initial_pose[:3], object_initial_pose[3:])
            transformed_grasp_pose = local_transform(world_grasp_pose, grasp_offset)
            # if flip_angle == 90 or flip_angle == 270:
            #     final_pose[2] = 0.14575
            
            current_data = {
                "initial_object_pose": object_initial_pose[2:],
                "final_object_pose": final_pose.tolist()[2:],
                "grasp_pose": np.concatenate([transformed_grasp_pose[0], transformed_grasp_pose[1]]).tolist()
            }
            test_data.append(current_data)
    with open("/home/chris/Chris/placement_ws/src/placement_quality/path_simulation/model_testing/actual_box_experiment_test.json", "w") as f:
        json.dump(test_data, f, indent=4)
