#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
import socket
import json
import struct
 
class GraspClient(Node):
    def __init__(self):
        super().__init__('grasp_client')
 
    def request_grasps(self, port=12346, file_path="/home/chris/Desktop/krylon.pcd"):
        # The IP/hostname should match how you can access the container
        # For Docker on the same machine, you might use 'localhost' + port-mapping
        # Or you might have a specific container IP address (e.g. 172.17.x.x)
        HOST = '127.0.0.1'  # or container IP / hostname
        PORT = port
 
        # Construct some sample point cloud info (JSON-serializable dict)
        pointcloud_data = self.format_pcd(file_path)
 
        self.get_logger().info(f"Connecting to container server at {HOST}:{PORT}")
 
        # Set up a TCP client socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
 
            # Send the pointcloud data in JSON
            send_str = json.dumps({"pointcloud": pointcloud_data})
            send_data = send_str.encode('utf-8')
            send_len = struct.pack('>I', len(send_data))
            s.sendall(send_len + send_data)
 
            # Receive the response (list of grasps)
            response_len = struct.unpack('>I', s.recv(4))[0]
            response_bytes = s.recv(response_len)  # adjust as needed
            response_str = response_bytes.decode('utf-8')
            # print(f"received data:{response_str}")
            response_data = json.loads(response_str)
 
        


        return response_data
        # Example of parsing the grasps
        # grasps = response_data.get('grasps', [])
        # for i, g in enumerate(grasps):
        #     pos = g.get('position', [0,0,0])
        #     ori = g.get('orientation', [0,0,0,1])
        #     self.get_logger().info(f"Grasp #{i}: position={pos}, orientation={ori}")
 
 
    def format_pcd(self, file_path):
        """
        Reads a PCD file and converts its contents into a JSON-friendly format.
        Each point is represented as a dictionary with x, y, z, and rgb fields.
        """

        with open(file_path, 'r') as file:
            lines = file.readlines()
 
        # Parse header
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('DATA'):
                data_start_idx = i + 1
                break
 
        # Parse data
        data = []
        for line in lines[data_start_idx:]:
            values = line.strip().split()
            x, y, z = map(float, values[:3])
            rgb = int(float(values[3]))
            data.append({"x": x, "y": y, "z": z, "rgb": rgb})
 
        return data
 
 
 
def main(args=None):
    file_path = "/home/chris/Chris/placement_ws/src/data/pcd_0/pointcloud.pcd"
    # file_path = "/home/chris/Chris/placement_ws/src/krylon.pcd"
    rclpy.init(args=args)
    node = GraspClient()
    node.request_grasps(file_path)
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
 