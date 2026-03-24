from scipy.spatial.transform import Rotation as R
import os
import open3d as o3d
import json
import tf2_ros
from rclpy.time import Time
import struct
import sys
import numpy as np

# Add the cube_simulation directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Original import should now work
from network.network_client import GraspClient
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, quat2euler
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import random
import math
from rclpy.node import Node
class TFSubscriber(Node):
    def __init__(self, pcd_topic):
        super().__init__("tf_subscriber")
        self.latest_tf = None  # Store the latest TFMessage here
        self.latest_pcd = None # Store the latest Pointcloud here

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        # Create the subscription
        self.tf_subscription = self.create_subscription(
            TFMessage,           # Message type
            "/tf",               # Topic name
            self.tf_callback,    # Callback function
            10                   # QoS
        )

        self.pcd_subscription = self.create_subscription(
            PointCloud2,         # Message type
            pcd_topic,        # Topic name
            self.pcd_callback,   # Callback function
            10                   # QoS
        )

    def pcd_callback(self, msg):
        self.latest_pcd = msg


    def tf_callback(self, msg):
        # This callback is triggered for every new TF message on /tf
        self.latest_tf = msg

def convert_wxyz_to_xyzw(q_wxyz):
    """Convert a quaternion from [w, x, y, z] format to [x, y, z, w] format."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def cube_orientation_to_ee_orientation(q_cube_init, q_ee_init, q_cube_target):
    """
    Compute the target end-effector orientation in quaternion form 
    to preserve the same relative orientation (offset).
    
    Parameters:
    -----------
    q_cube_init  : (4,) array_like
        Quaternion (x, y, z, w) for the initial cube orientation (world frame).
    q_ee_init    : (4,) array_like
        Quaternion for the initial end-effector orientation.
    q_cube_target: (4,) array_like
        Quaternion for the target cube orientation.
        
    Returns:
    --------
    q_ee_target : (4,) np.ndarray
        Quaternion (x, y, z, w) for the target end-effector orientation.
    """


    R_cube_init  = R.from_quat(convert_wxyz_to_xyzw(q_cube_init))
    R_ee_init    = R.from_quat(convert_wxyz_to_xyzw(q_ee_init))
    R_cube_target= R.from_quat(convert_wxyz_to_xyzw(q_cube_target))
    
    # Offset is the rotation taking the cube frame to the ee frame
    # offset = R_cube_init⁻¹ * R_ee_init
    offset = R_cube_init.inv() * R_ee_init
    
    # Target end-effector orientation = R_cube_target * offset
    R_ee_target = R_cube_target * offset
    
    return R_ee_target.as_quat()


def cube_feasible_orientation():
    random_orientations = {
        "top_surface":      np.array([   0.0,  0.0,   np.random.uniform(-180, 180)]),     # top face up  z diff
        "bottom_surface":   np.array([-180.0,  0.0,   np.random.uniform(-180, 180)]),     # bottom face up z diff
        "left_surface":     np.array([  90.0,  np.random.uniform(-180, 180),   0.0]),     # left face up y diff
        "right_surface":    np.array([ -90.0,  np.random.uniform(-180, 180),   0.0]),     # right face up y diff
        "front_surface":    np.array([  90.0,  np.random.uniform(-180, 180),  90.0]),     # front face up y diff
        "back_surface":     np.array([   0.0,  np.random.uniform(-180, 180), -90.0]),     # back face up y diff
    }

    # Randomly select one of the keys
    random_surface = random.choice(list(random_orientations.keys()))

    # Get the corresponding orientation
    random_orientation = np.deg2rad(random_orientations[random_surface])

    return euler2quat(*random_orientation)

def surface_detection(rpy):
    local_normals = {
        "1": np.array([0, 0, 1]),   # +z going up (0, 0, 0)
        "2": np.array([1, 0, 0]),   # +x going up (-90, -90, -90)
        "3": np.array([0, 0, -1]),  # -z going up (180, 0, -180)
        "4": np.array([-1, 0, 0]),  # -x going up (90, 90, -90)
        "5": np.array([0, -1, 0]),  # -y going up (-90, 0, 0)
        "6": np.array([0, 1, 0]),   # +y going up (90, 0, 0)
        }
    
    global_up = np.array([0, 0, 1]) 

      # Replace with your actual quaternion x,y,z,w
    rotation = R.from_euler('xyz', rpy)

    # Transform normals to the world frame
    world_normals = {face: rotation.apply(local_normal) for face, local_normal in local_normals.items()}

    # Find the face with the highest dot product with the global up direction
    upward_face = max(world_normals, key=lambda face: np.dot(world_normals[face], global_up))
    
    return int(upward_face)


def extract_grasping(input_file_path):
    # Load the original JSON data
  output_file_path = os.path.dirname(input_file_path) + '/Grasping.json'  # Path to save the filtered file

  with open(input_file_path, 'r') as file:
      data = json.load(file)

  # Extract data until the stage number hits 4
  filtered_data = []
  for entry in data.get("Isaac Sim Data", []):
      if entry["data"]["stage"] == 4:
          break
      filtered_data.append(entry)

  output_data = {"Isaac Sim Data": filtered_data}

  # Save the filtered data into a new JSON file
  with open(output_file_path, 'w') as output_file:
      json.dump(output_data, output_file)


def orientation_creation():

    """
    Generate a list of random orientations for the grasping pose.
    Each orientation is represented as an array of three random angles (Euler angles)
    in the range [-pi, pi].
    
    Parameters:
        num_orientations (int): Number of random orientations to generate.
        
    Returns:
        List[np.ndarray]: A list containing `num_orientations` arrays, each with 3 random angles.
    """
    result = []
    for _ in range(1000):
        # Generate three random angles in the range [-pi, pi]
        orientation = np.random.uniform(-np.pi, np.pi, size=3)
        result.append(orientation)

    return result





def projection(q_current_cube, q_current_ee, q_desired_ee):
    """
    Args:
        q_current_cube: orientation of current cube
        q_current_ee: orientation of current end effector
        q_desired_ee: orientation of desired end effector

        all in w,x,y,z format
    """
    # --- Step 1: Compute the relative rotation from EE to cube ---
    q_rel = qmult(qinverse(q_current_ee), q_current_cube)

    # --- Step 2: Compute the initial desired cube orientation based on desired EE ---
    q_desired_cube = qmult(q_desired_ee, q_rel)

    # Convert this quaternion to a rotation matrix for projection
    R_cube = quat2mat(q_desired_cube)

    # --- Step 3: Project the orientation so the cube's designated face is up ---
    # Define world up direction
    u = np.array([0, 0, 1])

    # Assuming the cube's local z-axis should point up:
    v = R_cube[:, 2]  # Current "up" direction of the cube according to its orientation

    # Compute rotation axis and angle to align v with u
    axis = np.cross(v, u)
    axis_norm = np.linalg.norm(axis)

    # Handle special cases: if axis is nearly zero, v is aligned or anti-aligned with u
    if axis_norm < 1e-6:
        # If anti-aligned, choose an arbitrary perpendicular axis
        if np.dot(v, u) < 0:
            axis = np.cross(v, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v, np.array([0, 1, 0]))
        else:
            # v is already aligned with u
            axis = np.array([0, 0, 1])
        axis_norm = np.linalg.norm(axis)

    axis = axis / axis_norm  # Normalize the axis
    angle = np.arccos(np.clip(np.dot(v, u), -1.0, 1.0))  # Angle between v and u

    # Compute corrective rotation matrix
    R_align = axangle2mat(axis, angle)

    # Apply the corrective rotation to project the cube's orientation
    R_cube_projected = np.dot(R_align, R_cube)

    # Convert the projected rotation matrix back to quaternion form
    q_cube_projected = mat2quat(R_cube_projected)

    return q_cube_projected





def tf_graph_generation(object_frame_path):
    import omni.graph.core as og
    from omni.isaac.core.utils.prims import is_prim_path_valid

    keys = og.Controller.Keys

    robot_frame_path= "/World/Franka"
    graph_path = "/Graphs/TF"
    # test_cube = "/World/Franka/test_cube"
    if is_prim_path_valid(graph_path):
        return
    (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
        {
            "graph_path": graph_path, 
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacClock", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("RosContext", "omni.isaac.ros2_bridge.ROS2Context"),
                ("TF_Tree", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                
            ],

            keys.SET_VALUES: [
                ("TF_Tree.inputs:topicName", "/tf"),
                ("TF_Tree.inputs:targetPrims", [robot_frame_path, object_frame_path]),
                # ("TF_Tree.inputs:targetPrims", cube_frame),
                ("TF_Tree.inputs:queueSize", 10),
 
            ],

            keys.CONNECT: [
                ("OnTick.outputs:tick", "TF_Tree.inputs:execIn"),
                ("IsaacClock.outputs:simulationTime", "TF_Tree.inputs:timeStamp"),
                ("RosContext.outputs:context", "TF_Tree.inputs:context"),
            ]
        }
    )

def compute_viewpoint(object_center, radius, azimuth_deg, elevation_deg):
    """
    Compute a viewpoint on a sphere given an object center, radius, azimuth, and elevation.
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    x = object_center[0] + radius * math.cos(elevation) * math.cos(azimuth)
    y = object_center[1] + radius * math.cos(elevation) * math.sin(azimuth)
    z = object_center[2] + radius * math.sin(elevation)
    return np.array([x, y, z])

def compute_lookat_orientation(camera_position, target_position, up_vector=np.array([0, 0, 1])):
    """
    Compute a quaternion (in [w, x, y, z] order) so that a camera at camera_position
    will look at target_position. If the forward vector is nearly parallel to up_vector,
    a fallback up vector is used to avoid degenerate cross products.
    """
    forward = target_position - camera_position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return np.array([1, 0, 0, 0])  # No valid orientation if camera == target

    forward = forward / forward_norm
    # If forward is nearly parallel to up_vector, pick a different up.
    if abs(np.dot(forward, up_vector)) > 0.99:
        up_vector = np.array([0, 1, 0])  # fallback up

    right = np.cross(up_vector, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    true_up = np.cross(forward, right)
    R_mat = np.column_stack((right, true_up, forward))
    
    # as_quat() returns [x, y, z, w]
    quat_xyzw = R.from_matrix(R_mat).as_quat()
    # Convert to [w, x, y, z]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz

def pcd_movements(cube_position, ee_pos, ee_ori):
    """
    Compute a series of viewpoints around an object, including an overhead view.
    Args:
        cube_position: (3,) np.ndarray: [x, y, z] position of the cube.
        ee_pos: (3,) np.ndarray: [x, y, z] position of the end-effector.
        ee_ori: (4,) np.ndarray: [w, x, y, z] orientation of the end-effector.
    Returns:
        target_positions: List of (3,) np.ndarray: [x, y, z] positions of the viewpoints.
        target_orientations: List of (4,) np.ndarray: [w, x, y, z] orientations of the viewpoints.
    """
    object_center = cube_position
    radius = 0.4  # Distance from the object for the viewpoints
    initial_position = np.array(ee_pos)
    initial_orientation = np.array(ee_ori)

    

    azimuth_samples = 4
    elevation_angles = [45]

    waypoints_positions = []
    waypoints_orientations = []


    for elevation_deg in elevation_angles:
        for i in range(azimuth_samples):
            az = (360.0 / azimuth_samples) * i
            cam_pos = compute_viewpoint(object_center, radius, az, elevation_deg)
            cam_orient = compute_lookat_orientation(cam_pos, object_center)

            waypoints_positions.append(cam_pos)
            waypoints_orientations.append(cam_orient)

    # After all scanning viewpoints, return to the initial pose
    waypoints_positions.append(initial_position)
    waypoints_orientations.append(initial_orientation)

    return waypoints_positions, waypoints_orientations

def get_current_end_effector_pose() -> np.ndarray:
    """
    Return the current end-effector (grasp center, i.e. outside the robot) pose.
    return: (3,) np.ndarray: [x, y, z] position of the grasp center.
            (4,) np.ndarray: [w, x, y, z] orientation of the grasp center.
    """

    offset = np.array([0.0, 0.0, 0.1034])
    from omni.isaac.dynamic_control import _dynamic_control
    """Return the current end-effector (grasp center) position."""
    # Acquire dynamic control interface
    dc = _dynamic_control.acquire_dynamic_control_interface()
    # Get rigid body handles for the two gripper fingers
    ee_body = dc.get_rigid_body("/World/Franka/panda_hand")
    # Query the world poses of each finger
    ee_pose = dc.get_rigid_body_pose(ee_body)
    panda_hand_translation = ee_pose.p 
    panda_hand_quat = ee_pose.r

    # Create a Rotation object from the panda_hand quaternion.
    hand_rot = R.from_quat(panda_hand_quat)
    
    # Rotate the local offset into world coordinates.
    offset_world = hand_rot.apply(offset)
    
    # Compute the tool_center position.
    tool_center_translation = panda_hand_translation + offset_world
    
    # Since the relative orientation is identity ([0,0,0]), the tool_center's orientation
    # remains the same as the panda_hand's.
    tool_center_quat = [panda_hand_quat[3], panda_hand_quat[0], panda_hand_quat[1], panda_hand_quat[2]]
    # print(f"Tool center translation: {tool_center_translation}")
    # print(f"Tool center quaternion: {tool_center_quat}")
    
    return tool_center_translation, tool_center_quat



from omni.isaac.core import World
def draw_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    world: World = None,
    scale: float = 0.1,
):
    # Isaac Sim's debug draw interface
    from omni.isaac.debug_draw import _debug_draw
    from carb import Float3, ColorRgba
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    # Acquire the debug draw interface once at startup or script init
    draw = _debug_draw.acquire_debug_draw_interface()
    # Clear previous lines so we have only one, moving frame.
    draw.clear_lines()
    """
    Draws a coordinate frame with colored X, Y, Z axes at the specified pose.
    
    Args:
        frame_name: A unique name for this drawing (used by Debug Draw).
        position:   (3,) array-like of [x, y, z] for frame origin in world coordinates.
        orientation:(4,) array-like quaternion [x, y, z, w].
        scale:      Length of each axis line.
        duration:   How long (in seconds) the lines remain in the viewport
                    before disappearing. If you keep calling draw_frame each
                    physics step, it will appear continuously.
    """
    # Convert the quaternion into a 3x3 rotation matrix
    rot_mat = R.from_quat(convert_wxyz_to_xyzw(orientation)).as_matrix()
    
    # Extract the basis vectors for x, y, z from the rotation matrix
    # and scale them to draw lines
    x_axis = rot_mat[:, 0] * scale
    y_axis = rot_mat[:, 1] * scale
    z_axis = rot_mat[:, 2] * scale
    
    # Convert position to a numpy array if needed
    origin = np.array(position, dtype=float)
    
    # Create carb.Float3 objects for start and end points.
    start_points = [
        Float3(origin[0], origin[1], origin[2]),  # for x-axis
        Float3(origin[0], origin[1], origin[2]),  # for y-axis
        Float3(origin[0], origin[1], origin[2])   # for z-axis
    ]
    end_points = [
        Float3(*(origin + x_axis)),
        Float3(*(origin + y_axis)),
        Float3(*(origin + z_axis))
    ]

    # Create carb.ColorRgba objects for each axis.
    colors = [
        ColorRgba(1.0, 0.0, 0.0, 1.0),  # red for x-axis
        ColorRgba(0.0, 1.0, 0.0, 1.0),  # green for y-axis
        ColorRgba(0.0, 0.0, 1.0, 1.0)   # blue for z-axis
    ]

    # Specify line thicknesses as a list of floats.
    sizes = [2.0, 2.0, 2.0]

    # Draw the three axes.
    draw.draw_lines(start_points, end_points, colors, sizes)

    




def process_tf_message(tf_message: TFMessage):
    allowed_frames = {"world", "panda_link0", "panda_hand", "Cube"}
    # Extract the frames and transformations
    tf_data = []
    for transform in tf_message.transforms:
        parent_frame = transform.header.frame_id
        child_frame = transform.child_frame_id
        if parent_frame in allowed_frames and child_frame in allowed_frames:
            frame_data = {
                "parent_frame": parent_frame,
                "child_frame": child_frame,
                "translation": {
                    "x": transform.transform.translation.x,
                    "y": transform.transform.translation.y,
                    "z": transform.transform.translation.z
                },
                "rotation": {
                    "x": transform.transform.rotation.x,
                    "y": transform.transform.rotation.y,
                    "z": transform.transform.rotation.z,
                    "w": transform.transform.rotation.w
                }
            }
            tf_data.append(frame_data)

    return tf_data


def downsample_points(points: np.ndarray, voxel_size: float = 0.0025) -> np.ndarray:
    """
    Downsample a Nx4 or Nx3 point cloud using Open3D's voxel grid filter.
    This reduces the file size and solves many 'voxelized cloud: 0' issues in PCL.
    """
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    if points.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    else:  # Nx4, ignoring last column for geometry
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Downsample
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # Convert back to NumPy (only x,y,z)
    down_xyz = np.asarray(down_pcd.points)

    # If original data had 4 columns, reattach the 'rgb' (0.0f or something):
    # We'll do a simple approach: everything gets the same 4th column
    if points.shape[1] == 4:
        # For now set all 'rgb' to 0.0. If you want to preserve color,
        # you'd need a more advanced approach, e.g. approximate nearest neighbors
        # in the original data.
        downsampled = np.zeros((down_xyz.shape[0], 4), dtype=np.float32)
        downsampled[:, :3] = down_xyz
        return downsampled
    else:
        return down_xyz


def filter_points_in_bounding_box(points, box_center, box_size):
    """
    Removes points that lie **inside** a specified axis-aligned bounding box.

    Args:
        points (np.ndarray): Nx4 array of [x, y, z, rgb].
        box_center (list/tuple): [cx, cy, cz] for the center of the box.
        box_size (list/tuple): [sx, sy, sz] for the full box dimensions.
                               (orientation = 0,0,0, so axis-aligned)

    Returns:
        np.ndarray: Filtered Nx4 array, with points inside the box removed.
    """
    c = np.array(box_center, dtype=np.float32)
    s = np.array(box_size, dtype=np.float32) * 0.5  # half-extends
    box_min = c - s
    box_max = c + s

    pts_xyz = points[:, :3]  # Nx3
    inside_mask = np.all((pts_xyz >= box_min) & (pts_xyz <= box_max), axis=1)

    # Keep points that are **outside** the box
    return points[~inside_mask]


def process_pointcloud(msg: PointCloud2, height_offset: float = 0.0) -> np.ndarray:
    """
    Saves a ROS PointCloud2 (with x,y,z,[rgb]) to an ASCII PCD file.
    The file includes a voxel-based downsampling step to reduce size.

    The output path is:  root/pointcloud.pcd
    """
    # Identify offsets for x,y,z,rgb
    offset_x = offset_y = offset_z = None
    offset_rgb = None
    for field in msg.fields:
        if field.name == "x":
            offset_x = field.offset
        elif field.name == "y":
            offset_y = field.offset
        elif field.name == "z":
            offset_z = field.offset
        elif field.name in ("rgb", "rgba"):
            offset_rgb = field.offset

    if offset_x is None or offset_y is None or offset_z is None:
        raise ValueError("PointCloud2 does not contain x, y, z fields!")

    # Convert data to (N,4) float32 array: [x y z rgb]
    num_points = len(msg.data) // msg.point_step
    points = np.zeros((num_points, 4), dtype=np.float32)

    for i in range(num_points):
        start = i * msg.point_step
        x = struct.unpack_from("f", msg.data, start + offset_x)[0]
        y = struct.unpack_from("f", msg.data, start + offset_y)[0]
        z = struct.unpack_from("f", msg.data, start + offset_z)[0]

        if offset_rgb is not None:
            # read raw 32 bits as float or int
            rgb_uint = struct.unpack_from("I", msg.data, start + offset_rgb)[0]
            # store as float so we can put it in 'rgb' column
            points[i] = [x, y, z, float(rgb_uint)]
        else:
            points[i] = [x, y, z, 0.0]


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # [A] FILTER OUT POINTS INSIDE THE ROBOT'S BOUNDING BOX
    # NOTE: Make sure points are in the same coordinate frame as the bounding box
    box_center = [-0.04, 0.0, 0.0]  # center of the robot bounding box in "world" frame
    box_size   = [0.24, 0.20, 2] # size in x,y,z
    points_filtered = filter_points_in_bounding_box(points, box_center, box_size)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # --- NEW: remove ground plane at z≈0 ---
    ground_height    = 0.0 + height_offset     # floor plane height in world frame
    ground_tolerance = 0.01    # 1cm above the floor
    # keep only points strictly above the floor plane + tolerance
    mask = points_filtered[:, 2] > (ground_height + ground_tolerance)
    points_noground = points_filtered[mask]


    # 5) Downsample to reduce size (voxel_size=0.01 => 1cm grid; adjust as needed)
    points_down = downsample_points(points_noground, voxel_size=0.0005)

    return points_down

def save_pointcloud(data, file_path):
    """
    Save pointcloud data to a PCD file. Accepts either a numpy array or PointCloud2 message.
    
    Args:
        data: Either a numpy array of shape (N,4) with [x,y,z,rgb] or a PointCloud2 message
        file_path: Directory where the pointcloud.pcd file will be saved
        
    Returns:
        str: Path to the saved PCD file
    """
    # Ensure the root directory exists
    os.makedirs(file_path, exist_ok=True)
    
    # Fixed filename: "pointcloud.pcd"
    file_path = os.path.join(file_path, "pointcloud.pcd")
    
    # If data is a PointCloud2 message, convert it to numpy array
    if isinstance(data, PointCloud2):
        # Convert PointCloud2 to numpy array
        points = pointcloud2_to_numpy(data)
    else:
        # Assume it's a numpy array
        points = data
    
    # Build ASCII PCD header
    header = (
        "# .PCD v.7 - Point Cloud Data file format\n"
        "VERSION .7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"  # <--- we use float type for all fields
        "COUNT 1 1 1 1\n"
        f"WIDTH {points.shape[0]}\n"
        "HEIGHT 1\n"
        f"POINTS {points.shape[0]}\n"
        "DATA ascii\n"
    )

    # Write ASCII data: x, y, z, rgb
    with open(file_path, "w") as f:
        f.write(header)
        for i in range(points.shape[0]):
            x, y, z, rgbf = points[i]
            # Here we treat the last column as float
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgbf:.6f}\n")

    print(f"[INFO] Saved ASCII PCD to {file_path}: {points.shape[0]} points).")
    return file_path

def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """
    Convert a ROS PointCloud2 message to a numpy array without any processing.
    
    Args:
        msg: PointCloud2 message
        
    Returns:
        np.ndarray: Nx4 array of [x, y, z, rgb]
    """
    # Identify offsets for x,y,z,rgb
    offset_x = offset_y = offset_z = None
    offset_rgb = None
    for field in msg.fields:
        if field.name == "x":
            offset_x = field.offset
        elif field.name == "y":
            offset_y = field.offset
        elif field.name == "z":
            offset_z = field.offset
        elif field.name in ("rgb", "rgba"):
            offset_rgb = field.offset

    if offset_x is None or offset_y is None or offset_z is None:
        raise ValueError("PointCloud2 does not contain x, y, z fields!")

    # Convert data to (N,4) float32 array: [x y z rgb]
    num_points = len(msg.data) // msg.point_step
    points = np.zeros((num_points, 4), dtype=np.float32)

    for i in range(num_points):
        start = i * msg.point_step
        x = struct.unpack_from("f", msg.data, start + offset_x)[0]
        y = struct.unpack_from("f", msg.data, start + offset_y)[0]
        z = struct.unpack_from("f", msg.data, start + offset_z)[0]

        if offset_rgb is not None:
            # read raw 32 bits as float or int
            rgb_uint = struct.unpack_from("I", msg.data, start + offset_rgb)[0]
            # store as float so we can put it in 'rgb' column
            points[i] = [x, y, z, float(rgb_uint)]
        else:
            points[i] = [x, y, z, 0.0]
            
    return points

def pointcloud2_to_o3d(pcd_ros):
    import sensor_msgs_py.point_cloud2 as pc2
    # Extract the structured points as a list of tuples (x, y, z)
    points_list = list(pc2.read_points(pcd_ros, field_names=("x", "y", "z"), skip_nans=True))
    
    # Convert the list of tuples to a standard NumPy array of floats
    points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float64)
    
    if points.shape[0] == 0:
        raise ValueError("Empty point cloud received!")

    # Create an Open3D PointCloud object and assign the points
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    
    return pcd_o3d

def transform_pointcloud_to_frame(
    cloud_in: PointCloud2,
    tf_buffer: tf2_ros.Buffer,
    target_frame: str="panda_link0"
) -> PointCloud2:
    """
    Transforms a PointCloud2 from its current frame (cloud_in.header.frame_id)
    into 'target_frame', using the transform data in tf_msg.

    Args:
        cloud_in (PointCloud2): The input point cloud.
        tf_msg (TFMessage): A TFMessage containing one or more transforms.
        target_frame (str): Name of the desired target frame, e.g. "panda_link0".

    Returns:
        PointCloud2: A new point cloud in the target frame.
    """
    # ---------------------------------------------------
    # 1. Find the Transform from cloud_in's frame to target_frame
    # ---------------------------------------------------
    source_frame = cloud_in.header.frame_id

    # We assume tf_msg.transforms has the needed transform directly
    transform_stamped: TransformStamped = tf_buffer.lookup_transform(
        target_frame=target_frame,
        source_frame=source_frame,
        time=Time())  # "latest" transform

    if transform_stamped is None:
        raise ValueError(f"No direct transform from '{source_frame}' to '{target_frame}' found in TFMessage.")

    # Extract translation
    tx = transform_stamped.transform.translation.x
    ty = transform_stamped.transform.translation.y
    tz = transform_stamped.transform.translation.z

    # Extract rotation (quaternion)
    qx = transform_stamped.transform.rotation.x
    qy = transform_stamped.transform.rotation.y
    qz = transform_stamped.transform.rotation.z
    qw = transform_stamped.transform.rotation.w

    # ---------------------------------------------------
    # 2. Build a 4x4 transform matrix
    # ---------------------------------------------------
    # Rotation from quaternion
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # Construct the rotation part of the matrix:
    rx = 1 - 2*(qy**2 + qz**2)
    ry = 2*(qx*qy - qz*qw)
    rz = 2*(qx*qz + qy*qw)

    ux = 2*(qx*qy + qz*qw)
    uy = 1 - 2*(qx**2 + qz**2)
    uz = 2*(qy*qz - qx*qw)

    fx = 2*(qx*qz - qy*qw)
    fy = 2*(qy*qz + qx*qw)
    fz = 1 - 2*(qx**2 + qy**2)

    transform_mat = np.array([
        [rx, ry, rz, tx],
        [ux, uy, uz, ty],
        [fx, fy, fz, tz],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)

    # ---------------------------------------------------
    # 3. Parse points from cloud_in into a Nx? array
    #    We'll handle x,y,z at least, and keep other fields as-is.
    # ---------------------------------------------------
    # Identify offsets for x, y, z
    offset_x = offset_y = offset_z = None
    # We keep track of other fields too, so we can copy them unmodified
    fields_dict = {}  # {field_name: (offset, datatype, count)}

    for f in cloud_in.fields:
        fields_dict[f.name] = (f.offset, f.datatype, f.count)
        if f.name == 'x':
            offset_x = f.offset
        elif f.name == 'y':
            offset_y = f.offset
        elif f.name == 'z':
            offset_z = f.offset

    if offset_x is None or offset_y is None or offset_z is None:
        raise ValueError("PointCloud2 is missing x, y, or z fields.")

    point_step = cloud_in.point_step
    row_step = cloud_in.row_step
    data_bytes = cloud_in.data

    num_points = cloud_in.width * cloud_in.height

    # We'll build a new bytearray for the output data
    out_data = bytearray(len(data_bytes))

    # For each point, transform x,y,z
    for i in range(num_points):
        point_offset = i * point_step

        # Read x,y,z from the input
        x = struct.unpack_from('f', data_bytes, point_offset + offset_x)[0]
        y = struct.unpack_from('f', data_bytes, point_offset + offset_y)[0]
        z = struct.unpack_from('f', data_bytes, point_offset + offset_z)[0]

        # Make it homogeneous
        pt_hom = np.array([x, y, z, 1.0], dtype=np.float64)

        # Apply the transform
        pt_trans = transform_mat @ pt_hom
        x_out = pt_trans[0]
        y_out = pt_trans[1]
        z_out = pt_trans[2]

        # Store x_out,y_out,z_out into the new data
        struct.pack_into('f', out_data, point_offset + offset_x, x_out)
        struct.pack_into('f', out_data, point_offset + offset_y, y_out)
        struct.pack_into('f', out_data, point_offset + offset_z, z_out)

    # We'll do a small fix: first copy everything, then transform x,y,z in the loop above
    out_data[:] = data_bytes[:]  # copy entire buffer
    # Then the loop modifies x,y,z in place

    for i in range(num_points):
        point_offset = i * point_step
        x = struct.unpack_from('f', data_bytes, point_offset + offset_x)[0]
        y = struct.unpack_from('f', data_bytes, point_offset + offset_y)[0]
        z = struct.unpack_from('f', data_bytes, point_offset + offset_z)[0]

        pt_hom = np.array([x, y, z, 1.0], dtype=np.float64)
        pt_trans = transform_mat @ pt_hom
        x_out, y_out, z_out = pt_trans[0], pt_trans[1], pt_trans[2]

        struct.pack_into('f', out_data, point_offset + offset_x, x_out)
        struct.pack_into('f', out_data, point_offset + offset_y, y_out)
        struct.pack_into('f', out_data, point_offset + offset_z, z_out)

    # ---------------------------------------------------
    # 4. Construct a new PointCloud2 with the same metadata
    #    but updated frame & data
    # ---------------------------------------------------
    cloud_out = PointCloud2()
    cloud_out.header.stamp = cloud_in.header.stamp
    cloud_out.header.frame_id = target_frame  # new frame
    cloud_out.height = cloud_in.height
    cloud_out.width = cloud_in.width
    cloud_out.fields = cloud_in.fields
    cloud_out.is_bigendian = cloud_in.is_bigendian
    cloud_out.point_step = cloud_in.point_step
    cloud_out.row_step = cloud_in.row_step
    cloud_out.is_dense = cloud_in.is_dense

    # set the new data
    cloud_out.data = bytes(out_data)
    print("Transformation complete")
    return cloud_out






def get_upward_facing_marker(cube_prim_path):
    import omni
    """
    Determines which marker (attached to a cube) is the highest in world space.
    This marker indicates which face of the cube is currently facing upward.
    
    Args:
        cube_prim_path (str): The prim path of the cube.
    
    Returns:
        A tuple (marker_name, world_position) for the marker with the highest Z.
    """
    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    if not cube_prim:
        print(f"Cube prim not found at {cube_prim_path}")
        return None


    upward_marker = None
    max_z = -float('inf')
    
    # Assume the markers are children of the cube with names starting with "marker_"
    for child in cube_prim.GetChildren():
        if not child.GetName().startswith("marker_"):
            continue

        # Use XformCommonAPI to get the marker's world translation.
        world_trans = get_world_translation(child)
        if world_trans[2] > max_z:
            max_z = world_trans[2]
            upward_marker = (child.GetName(), world_trans)

    
    return upward_marker


def get_world_translation(prim):
    """
    Computes the world translation of a prim by computing its local-to-world
    transformation matrix and extracting its translation.
    """
    from pxr import UsdGeom, Gf, Usd

    time=Usd.TimeCode(0)
    xformable = UsdGeom.Xformable(prim)
    # Compute the local-to-world transform matrix at the default time.
    world_matrix = xformable.ComputeLocalToWorldTransform(time)
    return world_matrix.ExtractTranslation()


def task_randomization():
    """
    Generate random initial and target positions for the cube.
    Output: cube_initial_position, cube_target_position, cube_initial_orientation, cube_target_orientation
    """
    ranges = [(-0.5, -0.15), (0.15, 0.5)]
    range_choice = ranges[np.random.choice(len(ranges))]
    
    # Generate x and y as random values between -π and π
    x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
    p, q = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])

    cube_initial_position = np.array([x, y, 0.3])
    cube_target_position = np.array([p, q, 0.075])
    cube_initial_orientation = cube_feasible_orientation()
    cube_target_orientation = cube_feasible_orientation()

    return cube_initial_position, cube_target_position, cube_initial_orientation, cube_target_orientation

def pose_init():
    ranges = [(-0.5, -0.15), (0.15, 0.5)]
    range_choice = ranges[np.random.choice(len(ranges))]
    
    # Generate x and y as random values between -π and π
    x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
    z = np.random.uniform(1.0, 2.0)

    # Random Euler angles in degrees and convert to radians
    euler_angles_deg = [random.uniform(0, 360) for _ in range(3)]
    euler_angles_rad = np.deg2rad(euler_angles_deg)
    
    # Convert Euler angles to quaternion using your preferred function,
    # e.g., if you have a function euler2quat that takes 3 angles:
    quat = euler2quat(*euler_angles_rad)  # Make sure euler2quat returns in the correct order
    pos =  np.array([x, y, z])

    return pos, quat
    


def spawn_random_cube(prim_path: str):
    from omni.isaac.core.utils import prims
    ranges = [(-0.5, -0.15), (0.15, 0.5)]
    range_choice = ranges[np.random.choice(len(ranges))]
    
    # Generate x and y as random values between -π and π
    x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
    
    pos = [x,y, np.random.uniform(1.0, 2.0)]  # Random z between 0.1 and 0.5
    # Random Euler angles in degrees and convert to radians
    euler_angles_deg = [random.uniform(0, 360) for _ in range(3)]
    euler_angles_rad = np.deg2rad(euler_angles_deg)
    
    # Convert Euler angles to quaternion using your preferred function,
    # e.g., if you have a function euler2quat that takes 3 angles:
    quat = euler2quat(*euler_angles_rad)  # Make sure euler2quat returns in the correct order

    # Create the cube prim with a given scale (adjust as needed)
    prims.create_prim(
        prim_path=prim_path,
        prim_type="Cube",
        position=pos,
        orientation=quat,
        scale=[0.1, 0.1, 0.1]
    )
    print(f"Spawned cube at {pos} with Euler angles (rad): {euler_angles_rad}")

def obtain_grasps(pcd_path, port):
    node = GraspClient()
    raw_grasps = node.request_grasps(port, pcd_path)

    grasps = []
    for key in sorted(raw_grasps.keys(), key=int):
        item = raw_grasps[key]
        position = item["position"]
        orientation = item["orientation_wxyz"]
        # Each grasp: [ [position], [orientation] ]
        grasps.append([position, orientation])

    return grasps



def view_pcd(file_path):
    # Load and visualize the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])


def transform_relative_pose(grasp_pose, relative_translation, relative_rotation=None):
    from pyquaternion import Quaternion
    """
    Transforms a grasp pose using a relative transformation.
    input: grasp_pose: [position, orientation]
    input: relative_translation: [x, y, z]
    input: relative_rotation: [w, x, y, z]
    output: [position, orientation]
    """
    # Helper: Convert a pose (position, quaternion) to a 4x4 homogeneous transformation matrix.
    def pose_to_matrix(position, orientation):
        T = np.eye(4)
        q = Quaternion(orientation)  # expects [w, x, y, z]
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = position
        return T

    # Helper: Convert a 4x4 homogeneous transformation matrix back to a pose.
    def matrix_to_pose(T):
        position = T[:3, 3].tolist()
        q = Quaternion(matrix=T[:3, :3])
        orientation = q.elements.tolist()  # [w, x, y, z]
        return position, orientation

    # Convert the input grasp pose to a homogeneous matrix.
    T_current = pose_to_matrix(grasp_pose[0], grasp_pose[1])

    # Build the relative transformation matrix.
    T_relative = np.eye(4)
    if relative_rotation is None:
        q_relative = Quaternion()  # Identity rotation.
    else:
        q_relative = Quaternion(relative_rotation)
    T_relative[:3, :3] = q_relative.rotation_matrix
    T_relative[:3, 3] = relative_translation

    # Apply the transformation - for local to global, we need:
    # T_target = T_relative * T_current (object_world * grasp_local)
    T_target = np.dot(T_relative, T_current)

    # Convert back to position and quaternion.
    new_position, new_orientation = matrix_to_pose(T_target)
    
    return [new_position, new_orientation]


def local_transform(pose, offset):
    """Apply offset in the local frame of the pose"""
    from pyquaternion import Quaternion
    # Convert to matrices
    T_pose = np.eye(4)
    q = Quaternion(pose[1])  # [w, x, y, z]
    T_pose[:3, :3] = q.rotation_matrix
    T_pose[:3, 3] = pose[0]
    
    # Create offset matrix (identity rotation)
    T_offset = np.eye(4)
    T_offset[:3, 3] = offset
    
    # Multiply in correct order: pose * offset (applies offset in local frame)
    T_result = np.dot(T_pose, T_offset)
    
    # Convert back to position, orientation
    new_position = T_result[:3, 3].tolist()
    q_new = Quaternion(matrix=T_result[:3, :3])
    new_orientation = q_new.elements.tolist()  # [w, x, y, z]
    
    return [new_position, new_orientation]


def compute_camera_pose(object_position: np.ndarray,
                        offset: np.ndarray = np.array([0.0, -0.4, 0.2]),
                        up: np.ndarray     = np.array([0.0, 0.0, 1.0])):
    """
    Given the object's world position, return a camera position and quaternion
    such that the camera is offset by `offset` (in world frame) and
    oriented to look at the object.

    Args:
        object_position: (3,) world coordinates of your box center.
        offset:          (3,) vector from object to camera in world frame.
        up:              (3,) preferred "up" direction for the camera.

    Returns:
        camera_position: (3,) world position of the camera.
        quat:            (4,) [x, y, z, w] quaternion for camera orientation.
    """
    # 1) Position the camera
    camera_position = object_position + offset

    # 2) Build camera local axes so that local −Z points toward the object:
    #    target_vec = object − camera
    target_vec = object_position - camera_position
    #    world Z_cam (local +Z) must be opposite of target_vec
    z_cam = -target_vec / np.linalg.norm(target_vec)

    #    X_cam = up × Z_cam
    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    #    Y_cam = Z_cam × X_cam
    y_cam = np.cross(z_cam, x_cam)

    # 3) Make rotation matrix whose columns are [X_cam, Y_cam, Z_cam]
    rot_mat = np.column_stack((x_cam, y_cam, z_cam))

    # 4) Convert to quaternion [x, y, z, w] (or roll to [w,x,y,z] as before)
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)   # if you need wxyz

    return camera_position, quat_wxyz


def set_cameras(object_position):
    """
    Convenience function to set up and start multiple cameras at once.
    
    Args:
        cameras: List of Camera objects to start
        enable_pcd: Whether to enable pointcloud publishing
        topic_prefix: Prefix for ROS topics
    """
    import omni
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.sensor import Camera
    cam_prim_path = "/World/camera"

    cam_pos, rotation_quat = compute_camera_pose(
        object_position=np.array(object_position),
        offset=np.array([0.0, -0.4, 0.2]),     # tweak these as needed
        up=np.array([0.0, 0.0, 1.0])
    )

    # Improved camera parameters - increased radius for better field of view
    radius = 0.4       # Increased distance from 0.6 to 1.0
    cam = Camera(
                prim_path=cam_prim_path,
                name="camera",
                position=cam_pos.tolist(),
                orientation=rotation_quat.tolist(),
                resolution=[640, 480],
            )
            
    # Set the camera's world pose
    cam.set_world_pose(position=cam_pos.tolist(), 
                       orientation=rotation_quat.tolist(), 
                       camera_axes="usd")
    
    # Set clipping range
    cam.set_clipping_range(0.01, 1000.0)
    
    # Set horizontal aperture for wider field of view
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(cam.prim_path)
    if camera_prim:
        camera_prim.GetAttribute("horizontalAperture").Set(36.0)  # 36mm is a wide-angle setting
        camera_prim.GetAttribute("verticalAperture").Set(24.0)    # Maintain aspect ratio
        camera_prim.GetAttribute("focalLength").Set(24.0)         # Standard focal length
        camera_prim.GetAttribute("focusDistance").Set(radius)     # Focus at object distance
    
    
    # Set up the multi-camera graph
    setup_multi_camera_graph([cam])
    
    print(f"Started the camera")
    return



def setup_multi_camera_graph(
    cameras: list,
    graph_path: str = "/Graphs/MultiCameraROS",
    tf_graph_path: str = "/Graphs/CameraTFActionGraph",
    node_namespace: str = "",
    topic_prefix: str = "",
):
    """
    Creates a comprehensive OmniGraph that handles multiple cameras and publishes:
    1) TF transforms for each camera
    2) Camera Info for each camera
    3) RGB Image for each camera
    4) Depth Pointcloud for each camera
    
    Args:
        cameras: List of Camera objects to add to the graph
        graph_path: Path for the main camera data graph
        tf_graph_path: Path for the transform graph
        node_namespace: ROS namespace for the nodes
        topic_prefix: Prefix to add to topic names
        
    Returns:
        None
    """
    import time
    import omni.graph.core as og
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core_nodes.scripts.utils import set_target_prims
    if len(cameras) == 0:
        raise ValueError("No cameras provided. Please provide at least one camera.")
    
    # First, set up the TF graph if it doesn't exist yet
    try:
        # Check if the TF graph already exists
        if not is_prim_path_valid(tf_graph_path):
            # Create the base TF graph with clock
            (ros_camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": tf_graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("IsaacClock", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("RosPublisher", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                        ("IsaacClock.outputs:simulationTime", "RosPublisher.inputs:timeStamp"),
                    ]
                }
            )
        
        # Add each camera to the TF graph
        for camera in cameras:
            camera_prim = camera.prim_path
            if not is_prim_path_valid(camera_prim):
                print(f"Warning: Camera path '{camera_prim}' is invalid. Skipping this camera for TF.")
                continue
                
            camera_frame_id = camera_prim.split("/")[-1]
            
            # Check if this camera's TF nodes already exist
            if not is_prim_path_valid(f"{tf_graph_path}/PublishTF_{camera_frame_id}"):
                # Add camera-specific TF nodes
                og.Controller.edit(
                    tf_graph_path,
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            (f"PublishTF_{camera_frame_id}", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            (f"PublishTF_{camera_frame_id}.inputs:topicName", "/tf"),
                        ],
                        og.Controller.Keys.CONNECT: [
                            (f"{tf_graph_path}/OnTick.outputs:tick",
                                f"PublishTF_{camera_frame_id}.inputs:execIn"),
                            (f"{tf_graph_path}/IsaacClock.outputs:simulationTime",
                                f"PublishTF_{camera_frame_id}.inputs:timeStamp"),
                        ],
                    },
                )
                
                # Add target prims for the USD pose
                set_target_prims(
                    primPath=f"{tf_graph_path}/PublishTF_{camera_frame_id}",
                    inputName="inputs:targetPrims",
                    targetPrimPaths=[camera_prim],
                )
    except Exception as e:
        print(f"Error setting up camera TF graph: {e}")
    
    # Now set up the main camera data graph if it doesn't exist
    if not is_prim_path_valid(graph_path):
        try:
            # First create the base graph with common nodes
            og.Controller.edit(
                {
                    "graph_path": graph_path, 
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RunOnce", "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame"),
                        ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                    ],
                }
            )
            
            # Now add camera-specific nodes for each camera
            for i, camera in enumerate(cameras):
                camera_prim = camera.prim_path
                if not is_prim_path_valid(camera_prim):
                    print(f"Warning: Camera path '{camera_prim}' is invalid. Skipping this camera for data graph.")
                    continue
                
                camera_frame_id = camera_prim.split("/")[-1]
                camera_prefix = f"cam{i}"
                
                # Format topic names with prefix and camera index
                rgb_topic = f"{topic_prefix}/{camera_prefix}/rgb"
                depth_topic = f"{topic_prefix}/{camera_prefix}/depth"
                depth_pcl_topic = f"{topic_prefix}/{camera_prefix}/depth_pcl"
                camera_info_topic = f"{topic_prefix}/{camera_prefix}/camera_info"
                
                # Add camera-specific nodes
                og.Controller.edit(
                    graph_path,
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            (f"RenderProduct_{camera_prefix}", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                            (f"CameraInfo_{camera_prefix}", "omni.isaac.ros2_bridge.ROS2CameraInfoHelper"),
                            (f"RGB_{camera_prefix}", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                            (f"Depth_{camera_prefix}", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                            (f"DepthPCL_{camera_prefix}", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            (f"RenderProduct_{camera_prefix}.inputs:cameraPrim", camera_prim),
                            
                            (f"CameraInfo_{camera_prefix}.inputs:topicName", camera_info_topic),
                            (f"CameraInfo_{camera_prefix}.inputs:frameId", camera_frame_id),
                            (f"CameraInfo_{camera_prefix}.inputs:nodeNamespace", node_namespace),
                            (f"CameraInfo_{camera_prefix}.inputs:resetSimulationTimeOnStop", True),
                            
                            (f"RGB_{camera_prefix}.inputs:topicName", rgb_topic),
                            (f"RGB_{camera_prefix}.inputs:type", "rgb"),
                            (f"RGB_{camera_prefix}.inputs:frameId", camera_frame_id),
                            (f"RGB_{camera_prefix}.inputs:nodeNamespace", node_namespace),
                            (f"RGB_{camera_prefix}.inputs:resetSimulationTimeOnStop", True),
                            
                            (f"Depth_{camera_prefix}.inputs:topicName", depth_topic),
                            (f"Depth_{camera_prefix}.inputs:type", "depth"),
                            (f"Depth_{camera_prefix}.inputs:frameId", camera_frame_id),
                            (f"Depth_{camera_prefix}.inputs:nodeNamespace", node_namespace),
                            (f"Depth_{camera_prefix}.inputs:resetSimulationTimeOnStop", True),
                            
                            (f"DepthPCL_{camera_prefix}.inputs:topicName", depth_pcl_topic),
                            (f"DepthPCL_{camera_prefix}.inputs:type", "depth_pcl"),
                            (f"DepthPCL_{camera_prefix}.inputs:frameId", camera_frame_id),
                            (f"DepthPCL_{camera_prefix}.inputs:nodeNamespace", node_namespace),
                            (f"DepthPCL_{camera_prefix}.inputs:resetSimulationTimeOnStop", True),
                        ],
                    }
                )
                
                # Small delay to ensure nodes are fully created
                time.sleep(0.1)
                
                # Verify that nodes exist before connecting
                render_product_path = f"{graph_path}/RenderProduct_{camera_prefix}"
                if not is_prim_path_valid(render_product_path):
                    print(f"Warning: Node {render_product_path} was not created properly. Skipping connections for this camera.")
                    continue
                
                # Add connections in a separate edit to ensure nodes exist first
                og.Controller.edit(
                    graph_path,
                    {
                        og.Controller.Keys.CONNECT: [
                            # RunOnce -> RenderProduct
                            (f"{graph_path}/RunOnce.outputs:step", f"{graph_path}/RenderProduct_{camera_prefix}.inputs:execIn"),
                            
                            # RenderProduct -> Helpers
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:execOut", f"{graph_path}/CameraInfo_{camera_prefix}.inputs:execIn"),
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:renderProductPath", f"{graph_path}/CameraInfo_{camera_prefix}.inputs:renderProductPath"),
                            
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:execOut", f"{graph_path}/RGB_{camera_prefix}.inputs:execIn"),
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:renderProductPath", f"{graph_path}/RGB_{camera_prefix}.inputs:renderProductPath"),
                            
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:execOut", f"{graph_path}/Depth_{camera_prefix}.inputs:execIn"),
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:renderProductPath", f"{graph_path}/Depth_{camera_prefix}.inputs:renderProductPath"),
                            
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:execOut", f"{graph_path}/DepthPCL_{camera_prefix}.inputs:execIn"),
                            (f"{graph_path}/RenderProduct_{camera_prefix}.outputs:renderProductPath", f"{graph_path}/DepthPCL_{camera_prefix}.inputs:renderProductPath"),
                            
                            # Context -> Helpers
                            (f"{graph_path}/Context.outputs:context", f"{graph_path}/CameraInfo_{camera_prefix}.inputs:context"),
                            (f"{graph_path}/Context.outputs:context", f"{graph_path}/RGB_{camera_prefix}.inputs:context"),
                            (f"{graph_path}/Context.outputs:context", f"{graph_path}/Depth_{camera_prefix}.inputs:context"),
                            (f"{graph_path}/Context.outputs:context", f"{graph_path}/DepthPCL_{camera_prefix}.inputs:context"),
                        ],
                    }
                )
                
        except Exception as e:
            print(f"Error setting up camera data graph: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Graph at {graph_path} already exists. Using existing graph.")
    
    print(f"Successfully set up multi-camera graph for {len(cameras)} cameras")
    return

def numpy_to_pointcloud2(points_array, frame_id="panda_link0"):
    """
    Convert a numpy array of shape (N,4) with [x,y,z,rgb] to a ROS PointCloud2 message.
    
    Args:
        points_array: Nx4 numpy array containing [x,y,z,rgb] for each point
        frame_id: Frame ID for the pointcloud message
        
    Returns:
        PointCloud2: ROS PointCloud2 message
    """
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import struct
    
    # Create header
    header = Header()
    header.frame_id = frame_id
    
    # Define the fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    
    # Create the PointCloud2 message
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = len(points_array)
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 16  # 4 bytes per float, 4 fields
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    cloud_msg.is_dense = True
    
    # Convert numpy array to bytes for message data
    cloud_msg.data = points_array.astype(np.float32).tobytes()
    
    return cloud_msg

def o3d_to_pointcloud2(pcd_o3d, frame_id="panda_link0"):
    """
    Convert an Open3D point cloud to a ROS PointCloud2 message.
    
    Args:
        pcd_o3d: Open3D point cloud object
        frame_id: Frame ID for the output message
        
    Returns:
        PointCloud2: ROS PointCloud2 message
    """
    import sensor_msgs_py.point_cloud2 as pc2
    from std_msgs.msg import Header
    
    # Get the points from the Open3D pointcloud
    points = np.asarray(pcd_o3d.points)
    
    # Check if we have colors in the pointcloud
    has_colors = hasattr(pcd_o3d, 'colors') and len(pcd_o3d.colors) > 0
    
    if has_colors:
        # Convert colors from [0,1] to [0,255] and pack into a single float
        colors = np.asarray(pcd_o3d.colors)
        colors_uint32 = (colors * 255.0).astype(np.uint8)
        # Create RGB packed values
        rgb_packed = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            rgb_packed[i] = struct.unpack('f', struct.pack('BBBB', 
                                         colors_uint32[i, 0], 
                                         colors_uint32[i, 1], 
                                         colors_uint32[i, 2], 
                                         0))[0]
        
        # Create array with both points and colors
        point_data = np.column_stack((points, rgb_packed))
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
    else:
        # Just points, no color
        point_data = points
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
    
    # Create header
    header = Header()
    header.frame_id = frame_id
    
    # Create PointCloud2 message
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = len(point_data)
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 16 if has_colors else 12
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    cloud_msg.is_dense = True
    
    # Add the data
    cloud_msg.data = point_data.tobytes()
    
    return cloud_msg


if __name__ == "__main__":
    # # # Example Usage
    # file_path = "/home/chris/Chris/placement_ws/src/data/pcd_0/pointcloud.pcd"
    # view_pcd(file_path)   
    # data_analysis("/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json")
    print(len(orientation_creation()))
    
    # Suppose you have an Nx3 cloud (world frame), e.g.:
    # cloud_world = np.random.rand(1000, 3) - 0.5  # random points in [-0.5, 0.5]^3

    # # Your robot bounding box info:
    # robot_center = [-0.04, 0.0, 0.0]
    # robot_size   = [0.24, 0.20, 0.01]

    # filtered_cloud = filter_robot_bounding_box(cloud_world, robot_center, robot_size)
    # print("Original cloud size:", len(cloud_world))
    # print("Filtered cloud size:", len(filtered_cloud))

