import numpy as np
from scipy.spatial.transform import Rotation as R
import os

import open3d as o3d
import json
import tf2_ros
from rclpy.time import Time
import struct
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, quat2euler
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import random
import math
# from gpd_container import GraspClient
from rclpy.node import Node
import rclpy
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

class SimSubscriber(Node):
    def __init__(self, buffer_size=100.0, visualization=False):
        super().__init__("Sim_subscriber")
        self.latest_tf = None
        self.latest_pcd1 = None
        self.latest_pcd2 = None
        self.latest_pcd3 = None

        self.buffer = Buffer(rclpy.duration.Duration(seconds=buffer_size))
        self.listener = TransformListener(self.buffer, self)

        self.tf_subscription = self.create_subscription(
            TFMessage,
            "/tf",
            self.tf_callback,
            10
        )

        import message_filters
        self.pcd_sub1 = message_filters.Subscriber(self, PointCloud2, "/cam0/depth_pcl")
        self.pcd_sub2 = message_filters.Subscriber(self, PointCloud2, "/cam1/depth_pcl")
        self.pcd_sub3 = message_filters.Subscriber(self, PointCloud2, "/cam2/depth_pcl")
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcd_sub1, self.pcd_sub2, self.pcd_sub3], 
            queue_size=10, 
            slop=0.1
        )
        
        self.ts.registerCallback(self.synchronized_pcd_callback)
        self.pcd_callback_enabled = True
        if visualization:
            # Add publishers for visualization
            self.pcd_pub = self.create_publisher(PointCloud2, '/visualization/raw_pcd', 10)
            self.grasp_pub = self.create_publisher(MarkerArray, '/visualization/grasp_pose', 10)
        
            # Timer for continuous publishing
            self.viz_timer = self.create_timer(0.1, self.publish_visualization)
            
            # Store raw point cloud
            self.raw_pcd = None
            self.current_grasp_pose = None

    def synchronized_pcd_callback(self, pcd1_msg, pcd2_msg, pcd3_msg):
        if not self.pcd_callback_enabled:
            return
        self.latest_pcd1 = pcd1_msg
        self.latest_pcd2 = pcd2_msg
        self.latest_pcd3 = pcd3_msg
        
    def tf_callback(self, msg):
        self.latest_tf = msg
        if self.latest_tf is not None and len(self.latest_tf.transforms) > 0:
            if not hasattr(self, 'all_transforms'):
                self.all_transforms = {}
            for transform in self.latest_tf.transforms:
                try:
                    self.buffer.set_transform(transform, "default_authority")
                    self.all_transforms[transform.child_frame_id] = transform
                except Exception:
                    pass
        
    def get_latest_pcds(self):
        return {
            "pcd1": self.latest_pcd1,
            "pcd2": self.latest_pcd2,
            "pcd3": self.latest_pcd3
        }
        
    def check_transform_exists(self, target_frame, source_frame):
        try:
            self.buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return True
        except Exception:
            return False

    def set_raw_pcd(self, raw_pcd):
        """Set the raw point cloud data"""
        self.raw_pcd = raw_pcd
    
    def set_grasp_pose(self, grasp_pose):
        """Set the current grasp pose"""
        self.current_grasp_pose = grasp_pose
    
    def publish_visualization(self):
        """Publish point cloud and grasp pose to RViz"""
        # Publish point cloud if available
        if self.raw_pcd is not None:
            print("Publishing point cloud to RViz")
            import numpy as np
            from sensor_msgs.msg import PointCloud2, PointField
            import std_msgs.msg
            
            # Convert Open3D point cloud to ROS PointCloud2 message
            points = np.asarray(self.raw_pcd.points)
            
            # Create point cloud message
            msg = PointCloud2()
            msg.header = std_msgs.msg.Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"  # Set appropriate frame
            
            # Set point cloud fields
            msg.height = 1
            msg.width = points.shape[0]
            
            # Define fields (x, y, z)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            msg.fields = fields
            
            # Set other properties
            msg.is_bigendian = False
            msg.point_step = 12  # 3 * float32 (4 bytes)
            msg.row_step = msg.point_step * points.shape[0]
            msg.is_dense = True
            
            # Convert points to byte array
            msg.data = points.astype(np.float32).tobytes()
            
            self.pcd_pub.publish(msg)
        
        # Publish grasp pose as axis if available
        if self.current_grasp_pose is not None:
            from visualization_msgs.msg import Marker, MarkerArray
            from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
            import numpy as np
            print("Publishing markerarray to RViz")
            
            marker_array = MarkerArray()
            
            # Convert list format to Pose message
            # current_grasp_pose is [position[x,y,z], orientation[w,x,y,z]]
            position = self.current_grasp_pose[0]
            orientation = self.current_grasp_pose[1]
            
            # Create base pose
            pose_msg = Pose()
            pose_msg.position = Point(x=position[0], y=position[1], z=position[2])
            pose_msg.orientation = Quaternion(w=orientation[0], x=orientation[1], y=orientation[2], z=orientation[3])
            
            # Colors for each axis (RGB)
            colors = [
                (1.0, 0.0, 0.0),  # X-axis: Red
                (0.0, 1.0, 0.0),  # Y-axis: Green
                (0.0, 0.0, 1.0)   # Z-axis: Blue
            ]
            
            # Axis directions
            directions = [
                Vector3(x=1.0, y=0.0, z=0.0),  # X-axis
                Vector3(x=0.0, y=1.0, z=0.0),  # Y-axis
                Vector3(x=0.0, y=0.0, z=1.0)   # Z-axis
            ]
            
            # Create a marker for each axis
            for i in range(3):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"axis_{i}"
                marker.id = i
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                
                # Set scale - length and width of the arrow
                marker.scale.x = 0.005  # Length
                marker.scale.y = 0.01  # Width
                marker.scale.z = 0.0  # Height
                
                # Set color for this axis
                marker.color.r = colors[i][0]
                marker.color.g = colors[i][1]
                marker.color.b = colors[i][2]
                marker.color.a = 1.0
                
                # Set starting pose
                marker.pose = pose_msg
                
                # Set direction for this axis
                marker.points = []
                start_point = Point(x=0.0, y=0.0, z=0.0)
                
                # End point in the direction of the axis
                end_point = Point(
                    x=directions[i].x * 0.2,
                    y=directions[i].y * 0.2,
                    z=directions[i].z * 0.2
                )
                
                marker.points.append(start_point)
                marker.points.append(end_point)
                
                marker_array.markers.append(marker)
            
            self.grasp_pub.publish(marker_array)


def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

def obtain_grasps(tcp_msg, port):
    from gpd_container import GraspClient
    node = GraspClient()
    serializable_msg = convert_numpy_to_python(tcp_msg)
    raw_grasps = node.request_grasps(serializable_msg, port)

    grasps = []
    for key in sorted(raw_grasps.keys(), key=int):
        item = raw_grasps[key]
        position = item["position"]
        orientation = item["orientation_wxyz"]
        # Each grasp: [ [position], [orientation] ]
        grasps.append([position, orientation])

    return grasps


def generate_grasp_poses(object_pose, object_dims, num_poses):
    """
    Generate a list of random grasp poses for an object.
    
    Parameters:
      object_pose: List with two elements:
                   [0] -> position: [x, y, z] (object center in world coordinates)
                   [1] -> orientation: [x, y, z, w] (object rotation as a quaternion)
      object_dims: list with measured dimensions [0.085, -0.07, 0.035].
                   We take the absolute values to define the box's extents.
      num_poses:   Number of random grasp poses to generate.
                   
    Returns:
      A list of grasp poses. Each grasp pose is a list with two elements:
        [0] -> position: Grasp position in world coordinates.
        [1] -> orientation: Grasp orientation as a quaternion (random).
    """
    # Use absolute values for dimensions (width, depth, height)
    width, depth, height = abs(object_dims[0]), abs(object_dims[1]), abs(object_dims[2])
    
    poses = []
    for _ in range(num_poses):
        # Sample a random point on the top surface of the box (in the object's local frame)
        local_x = random.uniform(-width/2, width/2)
        local_y = random.uniform(-depth/2, depth/2)
        local_z = height / 2.0  # top surface
        local_grasp = np.array([local_x, local_y, local_z])
        
        # Transform the local grasp position to world coordinates
        object_position = np.array(object_pose[0])
        object_rot = R.from_quat(object_pose[1])
        world_position = object_position + object_rot.apply(local_grasp)
        
        # Generate a random grasp orientation (completely random rotation)
        euler_angles = [random.uniform(0, 360) for _ in range(3)]
        grasp_orientation = R.from_euler('xyz', euler_angles, degrees=True).as_quat()
        
        poses.append([world_position, grasp_orientation])
    
    return poses

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


def extract_replay_data(input_file_path):
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
  print(f"Filtered data saved to: {output_file_path}")


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





def tf_graph_generation():
    import omni.graph.core as og
    from omni.isaac.core.utils.prims import is_prim_path_valid

    keys = og.Controller.Keys
    robot_frame_path= "/World/Franka"
    ycb_frame = "/World/Ycb_object"
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
                ("TF_Tree.inputs:targetPrims", [robot_frame_path, ycb_frame]),
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






def draw_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    scale: float = 0.1,
):
    # Isaac Sim's debug draw interface
    from omni.isaac.debug_draw import _debug_draw
    from carb import Float3, ColorRgba
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



def process_tf_message(tf_dict: dict):
    allowed_frames = {"world", "panda_link0", "panda_hand", "Ycb_object"}
    # Extract the frames and transformations
    tf_data = []
    for transform in tf_dict.values():
        parent_frame = transform.header.frame_id
        child_frame = transform.child_frame_id
        # print(f'parent_frame: {parent_frame}, child_frame: {child_frame}')
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

def transform_pointcloud(
    cloud_in: PointCloud2,
    transform_stamped: TransformStamped
) -> PointCloud2:
    """
    Transforms a PointCloud2 using a pre-computed transform.
    
    Args:
        cloud_in (PointCloud2): The input point cloud.
        transform_stamped (TransformStamped): The transform to apply, which should be 
                                              looked up at the same timestamp as the point cloud.
    
    Returns:
        PointCloud2: A new point cloud in the target frame.
    """
    if transform_stamped is None:
        raise ValueError("No transform provided.")

    # Extract transform components
    tx = transform_stamped.transform.translation.x
    ty = transform_stamped.transform.translation.y
    tz = transform_stamped.transform.translation.z

    qx = transform_stamped.transform.rotation.x
    qy = transform_stamped.transform.rotation.y
    qz = transform_stamped.transform.rotation.z
    qw = transform_stamped.transform.rotation.w

    # Build transform matrix
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

    # Parse point cloud data
    fields_dict = {}
    offset_x = offset_y = offset_z = None
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
    data_bytes = cloud_in.data
    num_points = cloud_in.width * cloud_in.height
    out_data = bytearray(len(data_bytes))
    out_data[:] = data_bytes[:]

    # Transform each point
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

    # Create output message
    cloud_out = PointCloud2()
    cloud_out.header.stamp = cloud_in.header.stamp  # Preserve original timestamp
    cloud_out.header.frame_id = transform_stamped.header.frame_id  # Use target frame from transform
    cloud_out.height = cloud_in.height
    cloud_out.width = cloud_in.width
    cloud_out.fields = cloud_in.fields
    cloud_out.is_bigendian = cloud_in.is_bigendian
    cloud_out.point_step = cloud_in.point_step
    cloud_out.row_step = cloud_in.row_step
    cloud_out.is_dense = cloud_in.is_dense
    cloud_out.data = bytes(out_data)

    return cloud_out



def get_current_end_effector_pose() -> np.ndarray:
    """
    Return the current end-effector (grasp center, i.e. outside the robot) pose.
    return: (3,) np.ndarray: [x, y, z] position of the grasp center.
            (4,) np.ndarray: [w, x, y, z] orientation of the grasp center.
    """

    offset = np.array([0.0, 0.0, 0.1034])
    # offset = np.array([0.0, 0.0, 0.0])
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
    
    return tool_center_translation, tool_center_quat


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


def view_pcd(file_path, show_normals=False):
    """
    Load and visualize a point cloud file.
    
    Args:
        file_path (str): Path to the point cloud file
        show_normals (bool): Whether to visualize normals
        
    Returns:
        o3d.geometry.PointCloud: The loaded point cloud
    """
    import open3d as o3d
    
    try:
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        
        if len(pcd.points) == 0:
            print(f"Point cloud at {file_path} is empty")
            return None
            
        # Create visualization objects
        vis_objects = [pcd]
        
        # Add coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis_objects.append(coordinate_frame)
        
        # Visualize normals if requested and available
        if show_normals and pcd.has_normals():
            # Create a new point cloud with fewer points for normal visualization
            normals_pcd = o3d.geometry.PointCloud()
            # Downsample for clearer normal visualization
            downsampled = pcd.voxel_down_sample(voxel_size=0.02)
            normals_pcd.points = downsampled.points
            normals_pcd.normals = downsampled.normals
            
            # Visualize the point cloud with normals
            o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                             point_show_normal=True)
        else:
            # Visualize the point cloud
            o3d.visualization.draw_geometries(vis_objects)
            
        return pcd
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")
        return None



def process_pointcloud(pcd, remove_plane=True):
    """
    Process a point cloud with advanced filtering, normal estimation, and segmentation.
    
    This function performs the following steps:
      1. Removes the dominant plane (e.g., table surface).
      2. Removes statistical outliers.
      3. Estimates normals with translation adjustments.
      4. Segments the processed point cloud using DBSCAN clustering to obtain
         the indices corresponding to the largest cluster (assumed to be the object).
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        remove_plane (bool): Whether to remove the dominant plane.
    
    Returns:
        tuple: (processed_pcd, object_indices)
            processed_pcd: o3d.geometry.PointCloud after processing.
            object_indices: List of indices corresponding to the segmented object.
    """
    if pcd is None or len(pcd.points) == 0:
        return pcd
    
    # 1. Remove the dominant plane (e.g., table surface)
    if remove_plane:
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.015, 
                ransac_n=100, 
                num_iterations=1000
            )
            if len(inliers) > 0:
                # Remove plane inliers (keep only points not belonging to the plane)
                pcd = pcd.select_by_index(inliers, invert=True)
        except Exception as e:
            print(f"Plane segmentation failed: {e}")
    
    # 2. Remove statistical outliers
    try:
        filtered_pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, 
            std_ratio=2
        )
        if len(filtered_pcd.points) > 0:
            pcd = filtered_pcd
    except Exception as e:
        print(f"Statistical outlier removal failed: {e}")
    
    # 3. Normal estimation with translation adjustments
    try:
        # Compute bounding box and translate to center the object
        obj_bbox = pcd.get_axis_aligned_bounding_box()
        center = obj_bbox.get_center()
        pcd.translate(-center)
        
        # Apply a small upward translation to improve normal estimation
        pcd.translate(np.array([0, 0, 0.05]))
        
        # Estimate normals
        pcd.estimate_normals(
            fast_normal_computation=True,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
        
        # Flip normals if needed
        normals = np.asarray(pcd.normals)
        pcd.normals = o3d.utility.Vector3dVector(-normals)
        
        # Reverse the translation
        pcd.translate(np.array([0, 0, -0.05]))
        pcd.translate(center)   
    except Exception as e:
        print(f"Error processing point cloud: {e}")
    
    # 4. Segment the object using DBSCAN clustering
    object_indices = []
    try:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        if labels.size == 0 or labels.max() < 0:
            # No valid clusters found, return all indices.
            object_indices = list(range(len(pcd.points)))
        else:
            # Exclude noise (label -1) and find the largest cluster.
            valid_labels = labels[labels >= 0]
            if valid_labels.size == 0:
                object_indices = list(range(len(pcd.points)))
            else:
                largest_cluster_label = np.bincount(valid_labels).argmax()
                object_indices = np.where(labels == largest_cluster_label)[0].tolist()
    except Exception as e:
        print(f"Segmentation using DBSCAN failed: {e}")
        object_indices = list(range(len(pcd.points)))
    
    return pcd, object_indices

def merge_and_save_pointclouds(pcds_dict: dict, tf_buffer: Buffer, output_path="/home/chris/Chris/placement_ws/src/pcds/pointcloud.pcd"):
    """
    Merges point clouds from multiple cameras after transforming them to the world frame.
    Saves both the raw merged point cloud and a processed version.
    
    Args:
        pcds_dict (dict): Dictionary containing point clouds from different cameras
        tf_buffer (tf2_ros.Buffer): TF buffer containing transform information
        output_path (str): Path where to save the merged point cloud
        
    Returns:
        tuple: (raw_pcd, processed_pcd) The raw and processed point clouds
    """
    transformed_pcds = []
    camera_source_list = []
    
    # We'll assign camera indices based on sorted keys to ensure consistency
    sorted_keys = sorted(pcds_dict.keys())

    for cam_idx, name in enumerate(sorted_keys):
        pcd_msg = pcds_dict[name]
        if pcd_msg is None:
            print(f"Camera {name} has no point cloud data")
            continue
            
        try:
            # Get transform at the exact time of the point cloud
            transform = tf_buffer.lookup_transform("Ycb_object", 
                                                   pcd_msg.header.frame_id, 
                                                   rclpy.time.Time(), 
                                                   timeout=rclpy.duration.Duration(seconds=1.0))
            # Use a function that accepts the pre-computed transform

            print(f"Transform: {transform}")
            print()
            print()
            print()
            print()
            print()
            world_pcd_msg = transform_pointcloud(pcd_msg, transform)
            o3d_pcd = pointcloud2_to_o3d(world_pcd_msg)
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.005)
            transformed_pcds.append(o3d_pcd)

            # Create an array of the same length as this point cloud filled with the camera index.
            num_points = len(o3d_pcd.points)
            camera_source_list.append(np.full((num_points,), cam_idx))

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    if not transformed_pcds:
        empty_pcd = o3d.geometry.PointCloud()
        o3d.io.write_point_cloud(output_path, empty_pcd)
        return {"cloud_sources": {"cloud": None, "camera_source": [], "view_points": []},
                "indices": []}
    
    # Merge point clouds
    merged_pcd = transformed_pcds[0]
    merged_camera_source = camera_source_list[0]
    for pcd, cam_source in zip(transformed_pcds[1:], camera_source_list[1:]):
        merged_pcd += pcd
        merged_camera_source = np.concatenate((merged_camera_source, cam_source))
    

    # Save the raw merged point cloud
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = merged_pcd.points  # already merged
    raw_output_path = output_path.replace(".pcd", "_raw.pcd")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        o3d.io.write_point_cloud(raw_output_path, raw_pcd)
        print("Point clouds merged successfully!")
    except Exception:
        pass
    
    # Process the point cloud
    processed_pcd, object_indices = process_pointcloud(raw_pcd)
    # Save the processed point cloud
    try:
        o3d.io.write_point_cloud(output_path, processed_pcd)
        print("Processed point cloud processed successfully!")
    except Exception:
        pass
    
    transform_object = tf_buffer.lookup_transform("world", 
                                                  "Ycb_object", 
                                                  rclpy.time.Time(), 
                                                  timeout=rclpy.duration.Duration(seconds=1.0))
    
    print(f"Transform object: {transform_object}")

    tcp_msg = {
                "cloud_sources": {
                    "cloud": format_o3d_pcd(raw_pcd),
                    "camera_source": merged_camera_source.tolist(),  # if available
                    "view_points": []     # if available
                },
                "indices": object_indices
            }
    return tcp_msg

def format_o3d_pcd(o3d_pcd):
    """
    Converts an Open3D point cloud into a JSON-friendly format.
    Each point is represented as a dictionary with x, y, z, and rgb fields.
    """
    points = np.asarray(o3d_pcd.points)
    data = []
    
    # Check if point cloud has colors
    if o3d_pcd.has_colors():
        colors = np.asarray(o3d_pcd.colors)
        for i in range(len(points)):
            # Convert RGB [0-1] to int
            r, g, b = colors[i]
            rgb_int = int((r*255) << 16 | (g*255) << 8 | (b*255))
            data.append({
                "x": float(points[i][0]),
                "y": float(points[i][1]),
                "z": float(points[i][2]),
                "rgb": rgb_int
            })
    else:
        # If no colors, use a default RGB value
        default_rgb = 0xFFFFFF  # white
        for i in range(len(points)):
            data.append({
                "x": float(points[i][0]),
                "y": float(points[i][1]),
                "z": float(points[i][2]),
                "rgb": default_rgb
            })
    
    return data





def pcd_processing_visualization(input_path="/home/chris/Chris/placement_ws/src/pcds/pointcloud_raw.pcd"):
    """
    Process an existing point cloud file with advanced filtering and normal estimation.
    Visualizes both the raw and processed point clouds.
    
    Args:
        input_path (str): Path to the input point cloud file
        
    Returns:
        o3d.geometry.PointCloud: Processed point cloud
    """
    import open3d as o3d
    
    try:
        # Load the point cloud
        raw_pcd = o3d.io.read_point_cloud(input_path)
        
        if len(raw_pcd.points) == 0:
            print("Empty point cloud file")
            return None
        
        # Visualize the raw point cloud
        print("Visualizing raw point cloud...")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([raw_pcd, coordinate_frame])
            
        # Process the point cloud
        processed_pcd, object_indices = process_pointcloud(raw_pcd)

        print(f"Object indices: {object_indices}")
        # Create a point cloud for the segmented object (using the indices)
        object_pcd = processed_pcd.select_by_index(object_indices)
        # Color the segmented object points in red
        object_pcd.paint_uniform_color([1, 0, 0])  # Red

        # Create a point cloud for the remaining points (background)
        background_pcd = processed_pcd.select_by_index(object_indices, invert=True)
        
        print("Visualizing processed point cloud with segmentation...")
        if len(background_pcd.points) > 0:
            background_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for background
            o3d.visualization.draw_geometries([background_pcd, object_pcd, coordinate_frame])
        else:
            # Only object points exist, so just draw them.
            o3d.visualization.draw_geometries([object_pcd, coordinate_frame])
        
        return processed_pcd
    except Exception as e:
        print(f"Error processing point cloud: {e}")
        return None
    


def transform_relative_pose(grasp_pose, relative_translation, relative_rotation=None):

    from pyquaternion import Quaternion
    """
    Transforms a grasp pose using a relative transformation.
    
    Parameters:
        grasp_pose (list): 
            - "position": list of [x, y, z]
            - "orientation_wxyz": list of quaternion components [w, x, y, z]
        relative_translation (list): The relative translation [x, y, z] from the current frame to the target frame.
        relative_rotation (list, optional): The relative rotation as a quaternion [w, x, y, z]. 
            If None, the identity rotation is used.
    
    Returns:
        dict: A dictionary representing the transformed pose with keys:
            - "position": list of [x, y, z]
            - "orientation_wxyz": list of quaternion components [w, x, y, z]
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
    print(f"Pose: {pose}, Offset: {offset}")
    from pyquaternion import Quaternion
    position = pose[:3]
    orientation = pose[3:]

    # Convert to matrices
    T_pose = np.eye(4)
    q = Quaternion(orientation)  # [w, x, y, z]
    T_pose[:3, :3] = q.rotation_matrix
    T_pose[:3, 3] = position
    
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









if __name__ == "__main__":
    # # # Example Usage
    file_path = "/home/chris/Chris/placement_ws/src/data/YCB_data/run_20250326_094701/Pcd_2/pointcloud_raw.pcd"
    pcd_processing_visualization(file_path)   
    # process_existing_pointcloud()
    # data_analysis("/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json")
    # print(len(orientation_creation()))
    
    # Suppose you have an Nx3 cloud (world frame), e.g.:
    # cloud_world = np.random.rand(1000, 3) - 0.5  # random points in [-0.5, 0.5]^3

    # # Your robot bounding box info:
    # robot_center = [-0.04, 0.0, 0.0]
    # robot_size   = [0.24, 0.20, 0.01]

    # filtered_cloud = filter_robot_bounding_box(cloud_world, robot_center, robot_size)
    # print("Original cloud size:", len(cloud_world))
    # print("Filtered cloud size:", len(filtered_cloud))