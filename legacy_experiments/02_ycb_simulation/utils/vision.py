import omni.graph.core as og
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core_nodes.scripts.utils import set_target_prims

import time
import carb.settings
import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def start_cameras(cameras: list, enable_pcd=False, topic_prefix=""):
    """
    Convenience function to set up and start multiple cameras at once.
    
    Args:
        cameras: List of Camera objects to start
        enable_pcd: Whether to enable pointcloud publishing
        topic_prefix: Prefix for ROS topics
    """
    
    # Set up the multi-camera graph
    setup_multi_camera_graph(cameras, topic_prefix=topic_prefix)
    
    print(f"Started {len(cameras)} cameras")
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


def compute_grasp_indices(pcd, voxel_size=0.01, normal_radius=0.03, min_neighbors=10):
    """
    Compute indices in the point cloud that are likely grasp candidates.
    
    Args:
        pcd: Open3D PointCloud object
        voxel_size: Size for downsampling
        normal_radius: Radius for normal estimation
        min_neighbors: Minimum number of points in local neighborhood
        
    Returns:
        List of indices in the original point cloud suitable for grasp sampling
    """
    # Ensure we have normals
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
    
    # Downsample point cloud to get candidate points
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Get points and normals
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    down_points = np.asarray(downsampled_pcd.points)
    
    # Build KD-tree for original points
    tree = KDTree(points)
    
    # For each downsampled point, find its index in the original cloud
    # and check if it has enough neighbors
    indices = []
    for i, point in enumerate(down_points):
        # Find the closest point in the original cloud (should be exact or very close)
        dist, idx = tree.query(point, k=1)
        orig_idx = idx
        
        # Count neighbors within a radius
        indices_nn = tree.query_ball_point(point, r=normal_radius)
        
        # Check if there are enough neighbors and if normal points somewhat upward
        if len(indices_nn) >= min_neighbors and normals[orig_idx][2] > -0.3:
            indices.append(orig_idx)
    
    # If we have too many indices, further downsample
    if len(indices) > 500:  # Adjust based on your needs and gpd requirements
        step = len(indices) // 500
        indices = indices[::step]
    
    return indices

