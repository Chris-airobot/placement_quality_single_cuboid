import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from isaacsim import SimulationApp
import json
import time
from scipy.spatial.transform import Rotation as R

DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON   = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height":1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
    "physics_dt": 1.0/60.0,  # Physics timestep (60Hz)
    "rendering_dt": 1.0/30.0,  # Rendering timestep (30Hz)
}

simulation_app = SimulationApp(CONFIG)


import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from transforms3d.euler import euler2quat
from omni.isaac.nucleus import get_assets_root_path
import random
from pxr import UsdPhysics, Gf, UsdGeom
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.objects import DynamicCuboid

ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"

# --- Surface labeling helper functions ---
def get_surface_up_label(orientation_quat):
    """Determine which surface of the box is facing up using the correct approach"""
    # Define local normals for each face of the cube
    local_normals = {
        "z_up": np.array([0, 0, 1]),    # +z going up
        "x_up": np.array([1, 0, 0]),    # +x going up  
        "z_down": np.array([0, 0, -1]), # -z going up
        "x_down": np.array([-1, 0, 0]), # -x going up
        "y_down": np.array([0, -1, 0]), # -y going up
        "y_up": np.array([0, 1, 0]),    # +y going up
    }
    
    global_up = np.array([0, 0, 1])
    
    # Convert quaternion to rotation matrix and apply to normals
    quat_wxyz = orientation_quat
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rotation = R.from_quat(quat_xyzw)
    
    # Transform normals to the world frame
    world_normals = {face: rotation.apply(local_normal) for face, local_normal in local_normals.items()}
    
    # Find the face with the highest dot product with the global up direction
    upward_face = max(world_normals, key=lambda face: np.dot(world_normals[face], global_up))
    
    return upward_face


class PoseVisualization:
    def __init__(self, num_poses=1000, num_simultaneous=1000, output_file="/home/chris/Chris/placement_ws/src/object_poses.json"):
        # Basic variables
        self.world = None
        self.scene = None
        self.objects = []  # Changed to list of objects
        self.num_poses = num_poses
        self.num_simultaneous = min(num_simultaneous, num_poses)  # Number of objects to drop at once
        self.output_file = output_file
        self.object_name = "009_gelatin_box.usd"  # Fixed to one object

        self.box_dims = np.array([0.143, 0.0915, 0.051])
        

    def start(self, ycb=True):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.scene = self.world.scene
        self.scene.add_default_ground_plane()
        self.world.reset()
        self.create_objects(ycb)


        
    def create_objects(self, ycb=True):
        """Create multiple YCB objects as rigid bodies with physics enabled"""
        self.objects = []
        
        # Create the specified number of objects
        for i in range(self.num_simultaneous):
            prim_path = f"/World/Ycb_object_{i}"
            name = f"ycb_object_{i}"
            if ycb:
                unique_prim_path = find_unique_string_name(
                    initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
                unique_name = find_unique_string_name(
                    initial_name=name, is_unique_fn=lambda x: not self.scene.object_exists(x)
                )
                
                usd_path = ROOT_PATH + self.object_name
                
                # Add reference to stage and get the prim
                object_prim = add_reference_to_stage(usd_path=usd_path, prim_path=unique_prim_path)
                
                # # Apply physics properties directly to the prim
                UsdPhysics.RigidBodyAPI.Apply(object_prim)
                UsdPhysics.CollisionAPI.Apply(object_prim)
                
                # Create the RigidPrim object after applying physics properties
                obj = RigidPrim(
                    prim_path=unique_prim_path,
                    name=unique_name,
                    position=np.array([0, 0, 0]),
                    scale=[0.8, 0.8, 0.8]
                )
                
                # Make sure physics is enabled
                obj.enable_rigid_body_physics()
            else:
                obj = DynamicCuboid(
                        prim_path=prim_path,
                        name=name,
                        position=np.array([0, 0, 0]),
                        scale=self.box_dims.tolist(),  # [x, y, z]
                        color=np.array([0.8, 0.8, 0.8])  # Default gray
                    )
            
            self.scene.add(obj)
            self.objects.append(obj)
            
        return self.objects
    

def main():
    # Create 6 objects instead of just 1
    num_poses = 120
    num_simultaneous = 6
    env = PoseVisualization(num_poses=num_poses, num_simultaneous=num_simultaneous)
    env.start(ycb=False)

    # Load the pose file
    poses_all = json.load(open("/home/chris/Chris/placement_ws/src/object_poses_box.json"))
    
    # Use the last 216 poses to match a single run (36 per face × 6 faces)
    if len(poses_all) >= num_poses:
        poses = poses_all[-num_poses:]
    else:
        poses = poses_all
    
    total_poses = len(poses)
    
    # Number of poses per object (expect 36 when total_poses == 216)
    poses_per_object = total_poses // 6
    
    # Main simulation loop
    step = 0
    max_steps = poses_per_object  # We'll run for 72 steps
    
    print("\n=== Surface Label Visualization ===")
    print("Color Legend:")
    print("  Red = Z-up (top face pointing up)")
    print("  Green = Z-down (bottom face pointing up)")
    print("  Blue = X-up (side face pointing up)")
    print("  Yellow = X-down (opposite side pointing up)")
    print("  Magenta = Y-up (front face pointing up)")
    print("  Cyan = Y-down (back face pointing up)")
    print("=====================================\n")
    
    while simulation_app.is_running():
        if step < max_steps:
            # Update each object with its corresponding pose
            for obj_idx, obj in enumerate(env.objects):
                # Calculate the pose index for this object at this step
                pose_idx = obj_idx * poses_per_object + step
                
                # Get position and orientation from the pose file
                position = poses[pose_idx]["position"].copy()
                # Offset each object horizontally so they don't overlap
                position[0] = 0.4 + obj_idx * 0.5  # Space objects along x-axis
                position[1] = 0.0  # Keep y position the same
                
                # Set the object's pose
                obj.set_world_pose(
                    position=position,
                    orientation=poses[pose_idx]["orientation_quat"]
                )
                
                # Determine which surface is up and color the object
                surface_label_0 = get_surface_up_label(poses[0*poses_per_object + step]["orientation_quat"])
                surface_label_1 = get_surface_up_label(poses[1*poses_per_object + step]["orientation_quat"])
                surface_label_2 = get_surface_up_label(poses[2*poses_per_object + step]["orientation_quat"])
                surface_label_3 = get_surface_up_label(poses[3*poses_per_object + step]["orientation_quat"])
                surface_label_4 = get_surface_up_label(poses[4*poses_per_object + step]["orientation_quat"])
                surface_label_5 = get_surface_up_label(poses[5*poses_per_object + step]["orientation_quat"])
   
                
                # Print surface label for the first object (to avoid spam)
                print(f"Step {step}: Object 0 surface = {surface_label_0}")
                print(f"Step {step}: Object 1 surface = {surface_label_1}")
                print(f"Step {step}: Object 2 surface = {surface_label_2}")
                print(f"Step {step}: Object 3 surface = {surface_label_3}")
                print(f"Step {step}: Object 4 surface = {surface_label_4}")
                print(f"Step {step}: Object 5 surface = {surface_label_5}")
            
            # Advance simulation
            env.world.step(render=True)
            simulation_app.update()
            
            # Move to next step
            step += 1
            
            # Optional: add a small delay to make visualization clearer
            time.sleep(0.1)  # Increased delay to better see the surface changes
        else:
            # Reset the loop to start over
            step = 0
            print("\n=== Restarting visualization loop ===\n")
    
    print("Pose visualization complete")
    simulation_app.close()
    
if __name__ == "__main__":
    main()