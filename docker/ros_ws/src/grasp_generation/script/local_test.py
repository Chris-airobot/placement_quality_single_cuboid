#!/usr/bin/python3
# Add this to change the namespace, otherwise the robot model cannot be found
 
from typing import List, Union
import socket
import json
import struct
import rospy
from geometry_msgs.msg import PoseArray, Vector3, Point, Pose, Quaternion
from gpd_ros.msg import GraspConfig, GraspConfigList, CloudIndexed, CloudSources
 
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Int64
import open3d as o3d
# from pathlib import Path
import numpy as np
import pyquaternion
from visualization_msgs.msg import Marker, MarkerArray
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point



 
HOST = ''        # Typically bind to all interfaces inside the container
PORT = 12345     # Choose any free port
 
# For gpd_ros, the topic that outputs the grasp poses are "clustered_grasps"
 
class Grasp:
    def __init__(self):
        rospy.init_node("grasp_generation")
        
        # Call the point cloud data service
        # self.left_scan = rospy.ServiceProxy("/left_scan", AssembleScans2)
 
        self.grasp_process_timeout = rospy.Duration(5)
        
        # Publish the point cloud data into topic that GPD package which receives the input
        self.grasps_pub = rospy.Publisher("/gpd_input", CloudIndexed, queue_size=1)
        
        # Add a publisher for point cloud visualization in RViz
        self.cloud_viz_pub = rospy.Publisher("/point_cloud_viz", PointCloud2, queue_size=1)
        
        # Subscribe to the gpd topic and get the grasp gesture data, save it into the grasp_list through call_back function
        self.gpd_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.save_grasps)
        self.grasp_list: Union[List[GraspConfig], None] = None
        
        # Publisher for MarkerArray (e.g., in your __init__ method)
        self.marker_pub = rospy.Publisher("/pose_viz", MarkerArray, queue_size=1)
        
        # Store the calculated pose array
        self.saved_pose_array = None

        rospy.sleep(1)
 
    def transform_relative_pose(self, grasp_pose, relative_translation, relative_rotation=None):
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

        # Apply the relative transformation.
        # The new (target) pose is computed as:
        #   T_target = T_current * T_relative
        T_target = T_current.dot(T_relative)

        # Convert back to position and quaternion.
        new_position, new_orientation = matrix_to_pose(T_target)
        
        return [new_position, new_orientation]
 
    def save_grasps(self, data: GraspConfigList):
        self.grasp_list = data.grasps
 
        
    def vector3ToNumpy(self, vec: Vector3) -> np.ndarray:
        return np.array([vec.x, vec.y, vec.z])        
        
        
    def find_grasps(self, start_time):
        grasp_dict = dict()
        i = 0
        while (self.grasp_list is None and
               rospy.Time.now() - start_time < self.grasp_process_timeout):
            rospy.loginfo("waiting for left grasps")
            rospy.sleep(0.5)
 
            if self.grasp_list is None:
                rospy.logwarn("No grasps detected on pcl, restarting!!!")
                continue
        # Sort all grasps based on the gpd_ros's msg: GraspConfigList.grasps.score.data            
        self.grasp_list.sort(key=lambda x: x.score.data, reverse=True)

         # Only process the first grasp
        first_grasp = self.grasp_list[0]

        # Set the header frame to "map" (to match RViz default)
        header_frame = "world"
        current_time = rospy.Time.now()

        # Length of the arrows representing each axis
        arrow_length = 0.01         # or even smaller if needed
        arrow_shaft_diameter = 0.0002
        arrow_head_diameter  = 0.0004

        # Build the rotation matrix from the grasp vectors
        rot = np.zeros((3, 3))
        # Grasp approach direction becomes the z-axis
        rot[:, 2] = self.vector3ToNumpy(first_grasp.approach)
        # Hand closing direction becomes the x-axis
        rot[:, 0] = self.vector3ToNumpy(first_grasp.binormal)
        # Hand axis becomes the y-axis
        rot[:, 1] = self.vector3ToNumpy(first_grasp.axis)

        # Convert rotation matrix to a quaternion (for record keeping if needed)
        quat = pyquaternion.Quaternion(matrix=rot)
        pos = first_grasp.position

        grasp_dict[1] = {
            "position": [pos.x, pos.y, pos.z],
            "orientation_wxyz": [quat[0], quat[1], quat[2], quat[3]]
        }

        # Create a MarkerArray for visualization in RViz
        marker_array = MarkerArray()

        # Define the start point from the grasp position
        start_pt = pos

        pos_list = [[pos.x, pos.y, pos.z],[quat[0], quat[1], quat[2], quat[3]]]
        transformed_pos = self.transform_relative_pose(pos_list, [0, 0, 0.065])
        new_pos = Point()
        new_pos.x = transformed_pos[0][0]
        new_pos.y = transformed_pos[0][1]
        new_pos.z = transformed_pos[0][2]
        # Define the start point from the grasp position
        start_pt = new_pos
        pos = new_pos



        # --- X-axis marker (Red) ---
        marker_x = Marker()
        marker_x.header.frame_id = header_frame
        marker_x.header.stamp = current_time
        marker_x.ns = "grasp_axes"
        marker_x.id = 0
        marker_x.type = Marker.ARROW
        marker_x.action = Marker.ADD
        marker_x.scale.x = arrow_length      # Arrow length
        marker_x.scale.y = arrow_shaft_diameter   # Shaft diameter
        marker_x.scale.z = arrow_head_diameter   # Head diameter
        marker_x.color.r = 1.0
        marker_x.color.g = 0.0
        marker_x.color.b = 0.0
        marker_x.color.a = 1.0
        end_x = Point()
        end_x.x = pos.x + arrow_length * rot[0, 0]
        end_x.y = pos.y + arrow_length * rot[1, 0]
        end_x.z = pos.z + arrow_length * rot[2, 0]
        marker_x.points = [start_pt, end_x]

        # --- Y-axis marker (Green) ---
        marker_y = Marker()
        marker_y.header.frame_id = header_frame
        marker_y.header.stamp = current_time
        marker_y.ns = "grasp_axes"
        marker_y.id = 1
        marker_y.type = Marker.ARROW
        marker_y.action = Marker.ADD
        marker_y.scale.x = arrow_length
        marker_y.scale.y = arrow_shaft_diameter  
        marker_y.scale.z = arrow_head_diameter
        marker_y.color.r = 0.0
        marker_y.color.g = 1.0
        marker_y.color.b = 0.0
        marker_y.color.a = 1.0
        end_y = Point()
        end_y.x = pos.x + arrow_length * rot[0, 1]
        end_y.y = pos.y + arrow_length * rot[1, 1]
        end_y.z = pos.z + arrow_length * rot[2, 1]
        marker_y.points = [start_pt, end_y]

        # --- Z-axis marker (Blue) ---
        marker_z = Marker()
        marker_z.header.frame_id = header_frame
        marker_z.header.stamp = current_time
        marker_z.ns = "grasp_axes"
        marker_z.id = 2
        marker_z.type = Marker.ARROW
        marker_z.action = Marker.ADD
        marker_z.scale.x = arrow_length
        marker_z.scale.y = arrow_shaft_diameter
        marker_z.scale.z = arrow_head_diameter
        marker_z.color.r = 0.0
        marker_z.color.g = 0.0
        marker_z.color.b = 1.0
        marker_z.color.a = 1.0
        end_z = Point()
        end_z.x = pos.x + arrow_length * rot[0, 2]
        end_z.y = pos.y + arrow_length * rot[1, 2]
        end_z.z = pos.z + arrow_length * rot[2, 2]
        marker_z.points = [start_pt, end_z]

        # Add all three markers to the MarkerArray
        marker_array.markers.extend([marker_x, marker_y, marker_z])
        
        # Save the marker array for continuous publishing
        self.saved_pose_array = marker_array

        
        return grasp_dict
 
            
 
    def read_pcd_file(self, file_path):
        """
        Reads a PCD file using Open3D and converts it to a ROS PointCloud2 message.

        Parameters:
            file_path (str): Path to the PCD file.

        Returns:
            PointCloud2: The converted ROS PointCloud2 message.
        """
        # Read the PCD file using Open3D
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Extract points and colors if available
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Check if we have colors
        if pcd.has_colors():
            # Convert RGB [0-1] to uint32 format
            colors = np.asarray(pcd.colors, dtype=np.float32)
            colors_uint32 = (colors * 255).astype(np.uint8)
            rgb_uint32 = colors_uint32[:, 0] << 16 | colors_uint32[:, 1] << 8 | colors_uint32[:, 2]
            rgb_uint32 = rgb_uint32.astype(np.uint32)
            
            # Combine points and colors
            cloud_data = np.zeros(points.shape[0], dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.uint32)
            ])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            cloud_data['rgb'] = rgb_uint32
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1)
            ]
        else:
            # Points only
            cloud_data = np.zeros(points.shape[0], dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32)
            ])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]

        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"  # Changed from "world" to "map" to match RViz default

        # Create PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)
        
        return cloud_msg
    
 
    def json_to_cloud_indexed(self, data: dict) -> CloudIndexed:
        """
        Converts JSON-friendly point cloud data (in CloudIndexed format) to a CloudIndexed ROS message.
        Expects data to have the following structure:
          {
              "cloud_sources": {
                  "cloud": [ { "x": ..., "y": ..., "z": ..., "rgb": ... }, ... ],
                  "camera_source": [ int, int, ... ],
                  "view_points": [ { "x": ..., "y": ..., "z": ... }, ... ]
              },
              "indices": [ int, int, ... ]
          }
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        
        # Build PointCloud2 from the "cloud" field in cloud_sources
        cs_data = data["cloud_sources"]
        points = [
            [pt["x"], pt["y"], pt["z"], pt["rgb"]]
            for pt in cs_data["cloud"]
        ]
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]
        pc2_msg = point_cloud2.create_cloud(header, fields, points)
        
        # Build CloudSources message
        cs_msg = CloudSources()
        cs_msg.cloud = pc2_msg
        cs_msg.camera_source = [Int64(data=int(val)) for val in cs_data.get("camera_source", [])]
        
        # Build view_points list
        cs_msg.view_points = []
        for vp in cs_data.get("view_points", []):
            pt = Point()
            pt.x = vp["x"]
            pt.y = vp["y"]
            pt.z = vp["z"]
            cs_msg.view_points.append(pt)
        
        # Build CloudIndexed message
        ci_msg = CloudIndexed()
        ci_msg.cloud_sources = cs_msg
        ci_msg.indices = [Int64(data=int(i)) for i in data.get("indices", [])]
        
        return ci_msg
        
    def main(self):
 
        rospy.loginfo("ROS 1 Container Server: Listening on port %d", PORT)
        
        file_path = "/home/ros_ws/src/grasp_generation/incoming_message.json"
        # file_path = "/home/gpd/tutorials/krylon.pcd"
        
        # file_path = "/home/pointcloud.pcd"
        # pcd = self.read_pcd_file(file_path)
        data = json.load(open(file_path))

        ci_msg: CloudIndexed = self.json_to_cloud_indexed(data)

        start_time = rospy.Time.now()
        # Process point cloud data to generate grasps
        self.grasp_list = None

        # Publish point cloud for GPD processing
        self.grasps_pub.publish(ci_msg)
        
        # Also publish the point cloud for visualization in RViz
        self.cloud_viz_pub.publish(ci_msg.cloud_sources.cloud)
        
        rospy.loginfo("Published point cloud to /gpd_input and /point_cloud_viz")
        
        # Set up rate for continuous publishing
        rate = rospy.Rate(10)  # 10Hz refresh rate
        
        grasp_detected = False
        
        while not rospy.is_shutdown():
            # Always publish the point cloud for visualization
            self.cloud_viz_pub.publish(ci_msg.cloud_sources.cloud)
            
            # Check if we have grasp poses and calculate them once
            if self.grasp_list is not None and not grasp_detected:
                rospy.loginfo(f"Received {len(self.grasp_list)} grasp poses")
                # Calculate grasp poses and store them
                response_dict = self.find_grasps(start_time)
                grasp_detected = True
                rospy.loginfo("Grasp poses published to /pose_viz for RViz visualization")
            
            # Continue to publish already calculated grasp poses
            if grasp_detected and self.saved_pose_array is not None:
                print("publishing grasp poses")
                # Update the timestamp for visualization
                # self.saved_pose_array.header.stamp = rospy.Time.now()
                # Republish the saved pose array
                self.marker_pub.publish(self.saved_pose_array)
            
            # Make sure to sleep to maintain the publishing rate
            rate.sleep()
        
        rospy.loginfo("Node is shutting down")

                    

 
   
if __name__ == '__main__':
    # params_path = "/home/jason/catkin_ws/src/ur_grasping/src/box_grasping/configs/base_config.yaml"
    
    try:
        grasper = Grasp()
        grasper.main()
        # grasper.robot.move_gripper(0.3)
        #[51 47 20]
    except KeyboardInterrupt:
        pass