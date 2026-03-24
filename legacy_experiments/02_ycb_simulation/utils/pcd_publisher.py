#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
import open3d as o3d

class PCDPublisher(Node):
    def __init__(self):
        super().__init__('pcd_publisher')
        
        self.declare_parameter('pcd_file', '/home/chris/Chris/placement_ws/src/gpd/tutorials/krylon.pcd')
        self.pcd_file = self.get_parameter('pcd_file').get_parameter_value().string_value
        
        self.publisher_ = self.create_publisher(PointCloud2, '/camera/pointcloud', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz
        
        self.get_logger().info(f'Publishing PCD file: {self.pcd_file}')
        
        # Load the PCD file using Open3D
        self.pcd = o3d.io.read_point_cloud(self.pcd_file)
        self.points = np.asarray(self.pcd.points)
        
    def timer_callback(self):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        # Set up the message
        msg.height = 1
        msg.width = len(self.points)
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * float (4 bytes)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Pack the points into a byte array
        buffer = bytearray()
        for point in self.points:
            buffer.extend(struct.pack('fff', point[0], point[1], point[2]))
        
        msg.data = buffer
        
        self.publisher_.publish(msg)
        self.get_logger().info('Published point cloud')

def main(args=None):
    rclpy.init(args=args)
    pcd_publisher = PCDPublisher()
    rclpy.spin(pcd_publisher)
    pcd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()