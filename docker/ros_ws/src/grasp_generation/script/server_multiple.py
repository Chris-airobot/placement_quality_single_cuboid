#!/usr/bin/python3
# Add this to change the namespace, otherwise the robot model cannot be found
 
from typing import List, Union
import socket
import json
import struct
import rospy
from geometry_msgs.msg import PoseArray, Vector3, Point
from gpd_ros.msg import GraspConfig, GraspConfigList, CloudIndexed, CloudSources
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Int64
 
import numpy as np
import pyquaternion
import subprocess
import os
import signal
import time
 
HOST = ''        # Typically bind to all interfaces inside the container
PORT = 12345     # Choose any free port
 
# For gpd_ros, the topic that outputs the grasp poses are "clustered_grasps"
 
import subprocess
import os
import signal
import time
import sys
def start_detect_grasps():
    """
    Launch the gpd_ros detect_grasps node via rosrun with the desired parameters.
    This does not require you to modify any launch file.
    """
    # Build the rosrun command with remapped parameters:
    # Note: The underscore notation (e.g., _cloud_type:=0) passes private parameters.
    cmd = [
        "rosrun", "gpd_ros", "detect_grasps",
        "_cloud_type:=1",
        "_cloud_topic:=/gpd_input",
        "_config_file:=/home/ros_ws/src/grasp_generation/config/ros_eigen_params.cfg",
        "_rviz_topic:=plot_grasps"
    ]
    # Launch the process in its own process group so it can be killed later.
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    # Give the node some time to start up (adjust as needed)
    time.sleep(5)
    return proc

def stop_detect_grasps(proc):
    """
    Terminate the detect_grasps node by killing its process group.
    """
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()





class Grasp:
    def __init__(self):
        rospy.init_node("grasp_generation")

        # Publish the point cloud data into topic that GPD package which receives the input
        self.grasps_pub = rospy.Publisher("/gpd_input", CloudIndexed, queue_size=1)
        
        # Subscribe to the gpd topic and get the grasp gesture data, save it into the grasp_list through call_back function
        self.gpd_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.save_grasps)
        self.grasp_list: Union[List[GraspConfig], None] = None
        
        # Topic for visualization of the grasp pose
        self.pose_pub = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)
 
        # Below are TCP connections
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind((HOST, PORT))
        self.server_sock.listen(1)
        rospy.sleep(1)
 
    
 
    def save_grasps(self, data: GraspConfigList):
        self.grasp_list = data.grasps
 
        
    def vector3ToNumpy(self, vec: Vector3) -> np.ndarray:
        return np.array([vec.x, vec.y, vec.z])        
        
        
    def find_grasps(self, start_time):
        grasp_dict = dict()
        i = 0
        while (self.grasp_list is None and
               rospy.Time.now() - start_time < rospy.Duration(20)):
            rospy.loginfo("waiting for grasps")
            rospy.sleep(0.5)
 
            if self.grasp_list is None:
                rospy.logwarn("No grasps detected on pcl within 20 seconds, continuing...")
                continue
        

        if self.grasp_list is None:
            print("No grasps detected on pcl, returning None")
            return None
        # Sort all grasps based on the gpd_ros's msg: GraspConfigList.grasps.score.data            
        self.grasp_list.sort(key=lambda x: x.score.data, reverse=True)
 
 
        ### Pay attention here, the axis of the gripper
        for grasp in self.grasp_list:            
            i+=1 
            rot = np.zeros((3, 3))
            # grasp approach direction
            rot[:, 2] = self.vector3ToNumpy(grasp.approach)
            # hand closing direction
            rot[:, 0]= self.vector3ToNumpy(grasp.binormal)
            # hand axis
            rot[:, 1] = self.vector3ToNumpy(grasp.axis)
                        
            # Turn the roll pitch yaw thing into the quaternion axis
            quat = pyquaternion.Quaternion(matrix=rot)
            pos: Point = grasp.position
            
            grasp_dict[i] = {
                "position":[pos.x, pos.y, pos.z],
                "orientation_wxyz": [quat[0], quat[1], quat[2], quat[3]]
            }
            
            # grasp_pose = Pose(
            #     position=pos,
            #     orientation=Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0]),
            # )
        
        return grasp_dict
 
            
 
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
        try:
            while not rospy.is_shutdown():

                # Launch a fresh instance of the detect_grasps node for this cycle.
                gpd_proc = start_detect_grasps()
                rospy.loginfo("detect_grasps node launched.")
                
                conn, addr = self.server_sock.accept()
                rospy.loginfo("Connected by %s", addr)
                with conn:
                    data_len = struct.unpack('>I', conn.recv(4))[0]
                    data = b""
                    while len(data) < data_len:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
 
                    
                    # Decode and parse the incoming JSON
                    incoming_data = json.loads(data.decode('utf-8'))

                    # Open a file in write mode and save the JSON data into it
                    # with open("/home/ros_ws/src/grasp_generation/incoming_message.json", "w") as json_file:
                    #     json.dump(incoming_data, json_file, indent=4)  # indent=4 makes it pretty-printed
 
                    ci_msg = self.json_to_cloud_indexed(incoming_data)
 
                    start_time = rospy.Time.now()
                    # Process point cloud data to generate grasps
                    self.grasp_list = None
                    print("publishing CloudIndexed message to /gpd_input")
                    self.grasps_pub.publish(ci_msg)
                    
                    # Build the response
                    response_dict = self.find_grasps(start_time)
                    response_str = json.dumps(response_dict)
                    send_data = response_str.encode('utf-8')
                    send_len = struct.pack('>I', len(send_data))
 
                    # Send back the response
                    conn.sendall(send_len + send_data)
                    rospy.loginfo("Sent grasp data to client.")
                
                    # Terminate the detect_grasps node for this cycle.
                    stop_detect_grasps(gpd_proc)
                    rospy.loginfo("detect_grasps node terminated for this cycle.")
        except KeyboardInterrupt:
            pass
        finally:
            self.server_sock.close()
 
   
if __name__ == '__main__':
    # params_path = "/home/jason/catkin_ws/src/ur_grasping/src/box_grasping/configs/base_config.yaml"
    
    try:
        grasper = Grasp()
        grasper.main()
        # grasper.robot.move_gripper(0.3)
        #[51 47 20]
    except KeyboardInterrupt:
        pass