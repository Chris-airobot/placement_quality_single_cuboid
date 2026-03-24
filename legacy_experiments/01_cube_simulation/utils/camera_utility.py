import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core_nodes.scripts.utils import set_target_prims
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import os


def process_collected_pointclouds(collected_msgs: list):
    """
    Process the collected point cloud messages.
    The pipeline is:
        1. Convert ROS2 messages to Open3D point clouds and (optionally) downsample.
        2. Merge them using registration.
        3. Crop to a workspace, remove a dominant plane and outliers.
        4. Estimate normals with translation adjustments.
        5. Save the final processed point cloud to disk.
    """
    print("Now starting to process collected point clouds...")
    if not collected_msgs:
        print("Warning: No collected messages to process.")
        return

    pcd_folder = "/home/chris/Chris/placement_ws/src/pcds/"
     # Make sure the folder exists; if not, create it.
    if not os.path.exists(pcd_folder):
        os.makedirs(pcd_folder)
        print(f"Created folder: {pcd_folder}")

    # 1. Convert collected messages.
    print(f"There are {len(collected_msgs)} collected messages")
    counter = 0
    pcds = []
    collected_msgs.pop()
    for msg in collected_msgs:
        pcd = convert_pointcloud2_to_open3d(msg)
        # Create a complete file path for the point cloud.
        file_path = os.path.join(pcd_folder, f"pcd_{counter}.pcd")

        o3d.io.write_point_cloud(file_path, pcd)
        print("Processed point cloud saved to:", file_path)
        counter += 1
        if pcd is not None:
            # Optionally downsample for speed.
            pcd = downsample_pointcloud(pcd, voxel_size=0.005)
            pcds.append(pcd)
            print(f"{len(pcds)}/{len(collected_msgs)} pcds have been processed.")

    if not pcds:
        print("Warning: No valid point clouds after conversion.")
        return
    
    # 2. Merge all point clouds.
    merged_pcd = pcds[0]
    for pcd in pcds[1:]:
        try:
            print("Registering and merging point clouds...")
            merged_pcd = register_and_merge(merged_pcd, pcd, voxel_size=0.01)
        except Exception as e:
            print("Error during registration/merge:", e)
            continue

    if merged_pcd is None:
        print("Error: Merging resulted in None.")
        return
    

    # # 3. Process the merged point cloud.
    # # 3b. Plane segmentation (using RANSAC).
    # print("Processing merged point cloud...")
    # print(f"This is your merged point cloud: {merged_pcd}")
    # try:
    #     plane_model, inliers = merged_pcd.segment_plane(
    #         distance_threshold=0.0075, ransac_n=100, num_iterations=1000
    #     )
    # except Exception as e:
    #     print("Error during plane segmentation:", e)
    #     return
    
    # if len(inliers) == 0:
    #     print("Warning: No plane found. Skipping plane removal.")
    #     non_plane_cloud = merged_pcd
    # else:
    #     # Remove plane inliers (i.e. keep only points not belonging to the plane).
    #     non_plane_cloud = merged_pcd.select_by_index(inliers, invert=True)

    # 3c. Remove statistical outliers.
    try:

        # filtered_cloud, ind_filt = non_plane_cloud.remove_statistical_outlier(
        filtered_cloud, ind_filt = merged_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )
    except Exception as e:
        print("Error during outlier removal:", e)
        return


    # 4. Normal estimation with translation adjustments.
    print("Estimating normals and adjusting translations...")
    try:
        # Compute the bounding box and translate to center the object.
        obj_bbox = filtered_cloud.get_axis_aligned_bounding_box()
        center = obj_bbox.get_center()
        filtered_cloud.translate(-center)
        
        # Apply a small upward translation to improve normal estimation.
        filtered_cloud.translate(np.array([0, 0, 0.05]))
        filtered_cloud.estimate_normals(
            fast_normal_computation=True,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        filtered_cloud.normalize_normals()
        filtered_cloud.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
        
        # Flip normals.
        normals = np.asarray(filtered_cloud.normals)
        filtered_cloud.normals = o3d.utility.Vector3dVector(-normals)
        
        # Reverse the translation.
        filtered_cloud.translate(np.array([0, 0, -0.05]))
        filtered_cloud.translate(center)
    except Exception as e:
        print("Error during normal estimation:", e)
        return
    

    # 5. Save the processed point cloud.
    print("Saving processed point cloud...")
    try:
        full_save_path = "/home/chris/Chris/placement_ws/src/collected.pcd"
        o3d.io.write_point_cloud(full_save_path, filtered_cloud)
        print("Processed point cloud saved to:", full_save_path)
    except Exception as e:
        print("Error saving processed point cloud:", e)











def convert_pointcloud2_to_open3d(msg: PointCloud2):
    """
    Convert a sensor_msgs/PointCloud2 message to an Open3D point cloud.
    This example uses sensor_msgs_py.point_cloud2 to extract the (x,y,z) points.
    """
    # Extract field names (e.g. 'x','y','z')
    field_names = [field.name for field in msg.fields]
    cloud_data = list(pc2.read_points(msg, skip_nans=True, field_names=field_names))
    if len(cloud_data) == 0:
        return None
    # print(f"here is the cloud data: {cloud_data}")

    points = np.array(cloud_data)
    # Extract x, y, z fields explicitly and stack them into a 2D array.
    xyz = np.stack([points['x'], points['y'], points['z']], axis=-1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def downsample_pointcloud(pcd, voxel_size=0.01):
    """
    Downsample an Open3D point cloud using a voxel grid filter.
    """
    return pcd.voxel_down_sample(voxel_size)

def register_and_merge(accumulated_pcd, new_pcd, voxel_size=0.01):
    """
    Use ICP to register the new point cloud to the accumulated point cloud,
    then merge and downsample the result.
    """
    threshold = voxel_size * 1.5  # distance threshold for ICP
    trans_init = np.identity(4)
    reg = o3d.pipelines.registration.registration_icp(
        new_pcd, accumulated_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # Transform the new point cloud to align with the accumulated cloud.
    new_pcd.transform(reg.transformation)
    
    # Merge the point clouds and downsample the merged result.
    merged_pcd = accumulated_pcd + new_pcd
    # merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    return merged_pcd


def publish_camera_info(camera: Camera, freq):
    from omni.isaac.ros2_bridge import read_camera_info
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_camera_info"
    queue_size = 1
    node_namespace = f""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    writer = rep.writers.get("ROS2PublishCameraInfo")
    camera_info = read_camera_info(render_product_path=render_product)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info["width"],
        height=camera_info["height"],
        projectionType=camera_info["projectionType"],
        k=camera_info["k"].reshape([1, 9]),
        r=camera_info["r"].reshape([1, 9]),
        p=camera_info["p"].reshape([1, 12]),
        physicalDistortionModel=camera_info["physicalDistortionModel"],
        physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
    )
    writer.attach([render_product])
    omni.syntheticdata
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_pointcloud_from_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_pointcloud" # Set topic name to the camera's name
    queue_size = 1
    node_namespace = f""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    # Note, this pointcloud publisher will simply convert the Depth image to a pointcloud using the Camera intrinsics.
    # This pointcloud generation method does not support semantic labelled objects.
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        omni.syntheticdata._syntheticdata.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return

def publish_rgb(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(omni.syntheticdata._syntheticdata.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_depth"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                            omni.syntheticdata._syntheticdata.SensorType.DistanceToImagePlane.name
                        )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_camera_tf(camera: Camera):
    camera_prim = camera.prim_path

    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    try:
        # Generate the camera_frame_id. OmniActionGraph will use the last part of
        # the full camera prim path as the frame name, so we will extract it here
        # and use it for the pointcloud frame_id.
        camera_frame_id=camera_prim.split("/")[-1]

        # Generate an action graph associated with camera TF publishing.
        ros_camera_graph_path = "/Graphs/CameraTFActionGraph"
        if is_prim_path_valid(ros_camera_graph_path):
            return  # Graph already exists, so skip re-creation
        # If a camera graph is not found, create a new one.
        if not is_prim_path_valid(ros_camera_graph_path):
            (ros_camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": ros_camera_graph_path,
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

        # Generate 2 nodes associated with each camera: TF from world to ROS camera convention, and world frame.
        og.Controller.edit(
            ros_camera_graph_path,
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("PublishTF_"+camera_frame_id, "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                    ("PublishRawTF_"+camera_frame_id+"_world", "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF_"+camera_frame_id+".inputs:topicName", "/tf"),
                    # Note if topic_name is changed to something else besides "/tf",
                    # it will not be captured by the ROS tf broadcaster.
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:topicName", "/tf"),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:parentFrameId", camera_frame_id),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:childFrameId", camera_frame_id+"_world"),
                    # Static transform from ROS camera convention to world (+Z up, +X forward) convention:
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
                ],
                og.Controller.Keys.CONNECT: [
                    (ros_camera_graph_path+"/OnTick.outputs:tick",
                        "PublishTF_"+camera_frame_id+".inputs:execIn"),
                    (ros_camera_graph_path+"/OnTick.outputs:tick",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:execIn"),
                    (ros_camera_graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishTF_"+camera_frame_id+".inputs:timeStamp"),
                    (ros_camera_graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:timeStamp"),
                ],
            },
        )
    except Exception as e:
        print(e)

    # Add target prims for the USD pose. All other frames are static.
    set_target_prims(
        primPath=ros_camera_graph_path+"/PublishTF_"+camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )
    return

def merge_graphs(camera: Camera):
    import omni.graph.core as og
    keys = og.Controller.Keys

    robot_prim = "/World/Franka"
    cube_prim = "/World/Cube"
    camera_prim = camera.prim_path
    graph_path = "/Graphs/TF"


    if is_prim_path_valid(graph_path):
        return
    try:
        # Generate the camera_frame_id. OmniActionGraph will use the last part of
        # the full camera prim path as the frame name, so we will extract it here
        # and use it for the pointcloud frame_id.
        camera_frame_id=camera_prim.split("/")[-1]

        # If a camera graph is not found, create a new one.
        if not is_prim_path_valid(graph_path):
            (ros_camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("IsaacClock", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("RosPublisher", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                        ("RosContext", "omni.isaac.ros2_bridge.ROS2Context"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                        ("IsaacClock.outputs:simulationTime", "RosPublisher.inputs:timeStamp"),
                    ]
                }
            )

        # Generate 2 nodes associated with each camera: TF from world to ROS camera convention, and world frame.
        og.Controller.edit(
            graph_path,
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("PublishTF_"+camera_frame_id, "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                    ("PublishRawTF_"+camera_frame_id+"_world", "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF_"+camera_frame_id+".inputs:topicName", "/tf"),
                    # Note if topic_name is changed to something else besides "/tf",
                    # it will not be captured by the ROS tf broadcaster.
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:topicName", "/tf"),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:parentFrameId", camera_frame_id),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:childFrameId", camera_frame_id+"_world"),
                    # Static transform from ROS camera convention to world (+Z up, +X forward) convention:
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
                ],
                og.Controller.Keys.CONNECT: [
                    (graph_path+"/OnTick.outputs:tick",
                        "PublishTF_"+camera_frame_id+".inputs:execIn"),
                    (graph_path+"/RosContext.outputs:context",
                        "PublishTF_"+camera_frame_id+".inputs:context"),
                    (graph_path+"/OnTick.outputs:tick",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:execIn"),
                    (graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishTF_"+camera_frame_id+".inputs:timeStamp"),
                    (graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:timeStamp"),
                    
                ],
            },
        )
    except Exception as e:
        print(e)

    # Add target prims for the USD pose. All other frames are static.
    set_target_prims(
        primPath=graph_path+"/PublishTF_"+camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim, robot_prim, cube_prim],
    )
    return


























def camera_graph_generation(
    camera: Camera,
    graph_path: str = "/Graphs/ROS_Camera",
    node_namespace: str = "",
    rgb_topic: str = "/rgb",
    depth_topic: str = "/depth",
    depth_pcl_topic: str = "/depth_pcl",
):
    """
    Creates an OmniGraph that publishes:
    1) Camera Info
    2) RGB Image
    3) Depth Image
    4) Depth Pointcloud
    from the specified camera prim via ROS2.

    No conditionals—always publishes the three streams plus camera info.
    """
    camera_prim = camera.prim_path
    frame_id = camera_prim.split("/")[-1]

    # # Stop the timeline so we can safely build the graph
    # timeline = omni.timeline.get_timeline_interface()
    # timeline.stop()
    if is_prim_path_valid(graph_path):
        return  # Graph already exists, so skip re-creation
    # Create a new graph with OnPlaybackTick, RunOnce, RenderProduct, ROS2Context
    # Always use "execution" evaluator
    graph_edit_result = og.Controller.edit(
        {
            "graph_path": graph_path, 
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            # Create nodes
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RunOnce", "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame"),
                ("RenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("CameraInfo", "omni.isaac.ros2_bridge.ROS2CameraInfoHelper"),
                ("RGB", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("Depth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("DepthPCL", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            # Set attribute values
            og.Controller.Keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", camera_prim),
                ("CameraInfo.inputs:topicName", "camera_info"),
                ("CameraInfo.inputs:frameId", frame_id),
                ("CameraInfo.inputs:nodeNamespace", node_namespace),
                ("CameraInfo.inputs:resetSimulationTimeOnStop", True),

                ("RGB.inputs:topicName", rgb_topic),
                ("RGB.inputs:type", "rgb"),
                ("RGB.inputs:frameId", frame_id),
                ("RGB.inputs:nodeNamespace", node_namespace),
                ("RGB.inputs:resetSimulationTimeOnStop", True),

                ("Depth.inputs:topicName", depth_topic),
                ("Depth.inputs:type", "depth"),
                ("Depth.inputs:frameId", frame_id),
                ("Depth.inputs:nodeNamespace", node_namespace),
                ("Depth.inputs:resetSimulationTimeOnStop", True),

                ("DepthPCL.inputs:topicName", depth_pcl_topic),
                ("DepthPCL.inputs:type", "depth_pcl"),
                ("DepthPCL.inputs:frameId", frame_id),
                ("DepthPCL.inputs:nodeNamespace", node_namespace),
                ("DepthPCL.inputs:resetSimulationTimeOnStop", True),
            ],
            # Connect execution flows and data
            og.Controller.Keys.CONNECT: [
                # Tick -> RunOnce
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),

                # RunOnce -> RenderProduct
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),

                # RenderProduct -> CameraInfo
                ("RenderProduct.outputs:execOut", "CameraInfo.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "CameraInfo.inputs:renderProductPath"),

                # RenderProduct -> RGB
                ("RenderProduct.outputs:execOut", "RGB.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RGB.inputs:renderProductPath"),

                # RenderProduct -> Depth
                # ("RenderProduct.outputs:execOut", "Depth.inputs:execIn"),
                # ("RenderProduct.outputs:renderProductPath", "Depth.inputs:renderProductPath"),

                # RenderProduct -> DepthPCL
                ("RenderProduct.outputs:execOut", "DepthPCL.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "DepthPCL.inputs:renderProductPath"),

                # Context -> CameraInfo
                ("Context.outputs:context", "CameraInfo.inputs:context"),
                # Context -> RGB
                ("Context.outputs:context", "RGB.inputs:context"),
                # Context -> Depth
                ("Context.outputs:context", "Depth.inputs:context"),
                # Context -> DepthPCL
                ("Context.outputs:context", "DepthPCL.inputs:context"),
            ],
        },
    )









def start_camera(camera: Camera, enable_pcd=False):
    # publish_camera_tf(camera)
    if enable_pcd:
        camera_graph_generation(camera)
    # publish_camera_info(camera, approx_freq)
    # publish_rgb(camera, approx_freq)
    # publish_depth(camera, approx_freq)
    # publish_pointcloud_from_depth(camera, approx_freq)
