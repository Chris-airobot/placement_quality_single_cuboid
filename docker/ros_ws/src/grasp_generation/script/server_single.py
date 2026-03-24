#!/usr/bin/python3

from typing import List, Union
import socket
import json
import struct
import rospy
from geometry_msgs.msg import PoseArray, Vector3, Point
from gpd_ros.msg import GraspConfig, GraspConfigList

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

import numpy as np
import pyquaternion
import open3d as o3d
import os
import sys
import subprocess
import time

HOST = ''        # Typically bind to all interfaces inside the container
PORT = 12345     # Choose any free port

LAUNCH_CMD = ["roslaunch", "grasp_generation", "grasp_single.launch"]

class Grasp:
    def __init__(self):
        rospy.init_node("grasp_generation")
        self.grasp_process_timeout = rospy.Duration(100)
        self.grasps_pub = rospy.Publisher("/gpd_input", PointCloud2, queue_size=1)
        self.gpd_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.save_grasps)
        self.grasp_list: Union[List[GraspConfig], None] = None
        self.pose_pub = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)

        # --- Allow address reuse for socket ---
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((HOST, PORT))
        self.server_sock.listen(1)
        rospy.sleep(1)

    def generate_box_pcd(self, dimensions, target_points=4096, seed=None):
        """
        Fast uniform sampling of ~target_points on a cuboid’s six faces.
        - points are in the object–local frame (centre at origin)
        - no post‑FPS needed unless you want exact count
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        L, W, H = map(float, dimensions)
        # areas of each pair of faces
        areas = np.array([L*W, L*W, L*H, L*H, W*H, W*H])
        probs = areas / areas.sum()          # sampling proportion per face
        counts = np.floor(probs * target_points).astype(int)
        counts[-1] = target_points - counts[:-1].sum()   # fix rounding

        pts = []
        # helpers: sample n points on a 2‑D rectangle
        def xy(n, z):
            u, v = rng.uniform(-L/2, L/2, n), rng.uniform(-W/2, W/2, n)
            return np.column_stack([u, v, np.full(n, z)])
        def xz(n, y):
            u, v = rng.uniform(-L/2, L/2, n), rng.uniform(-H/2, H/2, n)
            return np.column_stack([u, np.full(n, y), v])
        def yz(n, x):
            u, v = rng.uniform(-W/2, W/2, n), rng.uniform(-H/2, H/2, n)
            return np.column_stack([np.full(n, x), u, v])

        # +Z, −Z, +Y, −Y, +X, −X faces
        pts.extend(xy(counts[0],  H/2))
        pts.extend(xy(counts[1], -H/2))
        pts.extend(xz(counts[2],  W/2))
        pts.extend(xz(counts[3], -W/2))
        pts.extend(yz(counts[4],  L/2))
        pts.extend(yz(counts[5], -L/2))

        return np.asarray(pts, dtype=np.float32)

    def save_points_as_pcd(self, points, file_path):
        """Save a numpy (N,3) float32 array `points` to a .pcd file."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        try:
            o3d.io.write_point_cloud(file_path, pcd, write_ascii=False)
            rospy.loginfo(f"Saved point cloud to {file_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save point cloud: {e}")

    def points_to_pointcloud2(self, points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        rgb_uint32 = int((255 << 16) | (255 << 8) | 255)
        pcl = [[p[0], p[1], p[2], rgb_uint32] for p in points]
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pc2_msg = point_cloud2.create_cloud(header, fields, pcl)
        return pc2_msg

    def save_grasps(self, data: GraspConfigList):
        self.grasp_list = data.grasps

    def vector3ToNumpy(self, vec: Vector3) -> np.ndarray:
        return np.array([vec.x, vec.y, vec.z])        

    def find_grasps(self, start_time, target=100, sim_threshold=0.95):
        """
        Return up to `target` grasps:
        1) sort by GPD score (high→low)
        2) keep grasps whose approach vectors differ by > acos(sim_threshold)
        3) if still short, fill with the next‑best grasps (no diversity test)

        sim_threshold : cosine similarity cutoff (0.95 ≈ 18°)
        """
        # ── wait for GPD output ────────────────────────────────────────────────
        while self.grasp_list is None and \
            rospy.Time.now() - start_time < self.grasp_process_timeout:
            rospy.sleep(0.2)
        if not self.grasp_list:
            rospy.logwarn("No grasps detected")
            return {}

        # ── sort by grasp quality ─────────────────────────────────────────────
        self.grasp_list.sort(key=lambda g: g.score.data, reverse=True)

        grasp_dict, chosen_dirs, used_ids = {}, [], set()

        def unit(v):                   # helper → unit vector
            v = v / np.linalg.norm(v)
            return v

        def add_grasp(g):
            """Pack grasp `g` into dict and record its approach dir."""
            idx = len(grasp_dict)
            rot = np.column_stack((self.vector3ToNumpy(g.axis),
                                -self.vector3ToNumpy(g.binormal),
                                self.vector3ToNumpy(g.approach)))
            q = pyquaternion.Quaternion(matrix=rot)
            p = g.position
            grasp_dict[idx] = {
                "position":          [p.x, p.y, p.z],
                "orientation_wxyz":  [q[0], q[1], q[2], q[3]],
            }
            chosen_dirs.append(unit(self.vector3ToNumpy(g.approach)))
            used_ids.add(id(g))

        # ── pass 1: diversity‑aware selection ────────────────────────────────
        for g in self.grasp_list:
            if len(grasp_dict) == target:
                break
            v = unit(self.vector3ToNumpy(g.approach))
            if any(v @ u > sim_threshold for u in chosen_dirs):
                continue
            add_grasp(g)

        # ── pass 2: back‑fill if we’re short ─────────────────────────────────
        if len(grasp_dict) < target:
            for g in self.grasp_list:
                if len(grasp_dict) == target:
                    break
                if id(g) in used_ids:
                    continue
                add_grasp(g)

        rospy.loginfo(f"Selected {len(grasp_dict)} grasps (target {target})")
        return grasp_dict

    def main(self):
        rospy.loginfo("ROS 1 Container Server: Listening on port %d", PORT)
        try:
            # Only accept ONE request, then restart
            conn, addr = self.server_sock.accept()
            rospy.loginfo("Connected by %s", addr)
            with conn:
                while not rospy.is_shutdown():
                    try:
                        header = conn.recv(4)
                        if not header or len(header) < 4:
                            rospy.loginfo("Client disconnected")
                            break
                        data_len = struct.unpack('>I', header)[0]
                        data = b""
                        while len(data) < data_len:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data += chunk
                        if not data:
                            break
                        incoming_data = json.loads(data.decode('utf-8'))
                        box_dims = incoming_data["dimensions"]
                        points = self.generate_box_pcd(box_dims)
                        # ── save the generated point cloud to a .pcd file ──
                        dir_path = "/home/ros_ws/src/grasp_generation/pcd_folder/"
                        pcd_filename = f"{box_dims[0]:.3f}_{box_dims[1]:.3f}_{box_dims[2]:.3f}.pcd"
                        file_path = os.path.join(dir_path, pcd_filename)
                        self.save_points_as_pcd(points, file_path)
                        pcd_msg = self.points_to_pointcloud2(points)
                        start_time = rospy.Time.now()
                        self.grasp_list = None
                        self.grasps_pub.publish(pcd_msg)
                        response_dict = self.find_grasps(start_time)
                        response_str = json.dumps(response_dict)
                        send_data = response_str.encode('utf-8')
                        send_len = struct.pack('>I', len(send_data))
                        conn.sendall(send_len + send_data)
                        rospy.loginfo("Sent grasp data to client.")
                        break  # Handle ONE request, then restart!
                    except Exception as e:
                        rospy.logerr(f"Connection error: {e}")
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.server_sock.close()
            rospy.loginfo("Exiting script so relaunch will take over.")
            sys.exit(0)

if __name__ == '__main__':
    try:
        grasper = Grasp()
        grasper.main()
    except KeyboardInterrupt:
        pass
