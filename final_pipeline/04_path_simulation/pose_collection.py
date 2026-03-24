import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from isaacsim import SimulationApp
import json
import time

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
from pxr import UsdPhysics

ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"

class PoseCollection:
    def __init__(self, 
                 num_poses=1000, 
                 num_simultaneous=1000, 
                 output_file="/home/chris/Chris/placement_ws/src/object_poses_v2.json",
                 mode="ycb"):
        
        self.mode = mode
        self.box_dims = np.array([0.143, 0.0915, 0.051])
        # Basic variables
        self.world = None
        self.scene = None
        self.objects = []  # Changed to list of objects
        self.num_poses = num_poses
        self.num_simultaneous = min(num_simultaneous, num_poses)  # Number of objects to drop at once
        self.output_file = output_file
        self.collected_poses = []
        self.object_name = "009_gelatin_box.usd"  # Fixed to one object
        self.settlement_threshold = 0.001  # Position change threshold in meters
        self.settlement_time = 1.5  # Time object must be still to be considered settled
        self.drop_height = 0.05  # Height above ground to drop objects from (meters)
        # Area dimensions for distributing objects
        self.area_size = max(5.0, (self.num_simultaneous ** 0.5) * 0.3)  # Scale area based on object count
        # Add tracking for previous positions
        self.previous_positions = {}
        self.position_history = {}
        self.position_timestamps = {}
        self.save_buffer = []  # Buffer to collect poses before saving
        self.save_buffer_size = 5000  # Save every 5000 poses (adjust as needed)
        
        # New: Track which orientations have been used
        self.available_orientations = []
        self.orientation_index = 0
        
    def start(self):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.scene = self.world.scene
        self.scene.add_default_ground_plane()
        self.world.reset()
        # New: Initialize orientations before creating objects
        self.initialize_orientations()
        self.create_objects()
        
    def initialize_orientations(self):
        """Prepare all orientations to be used during drops"""
        # Get orientations from all surfaces
        all_orientations_dict = self.sample_uniform_orientations()
        
        # Flatten all orientations into a single list
        self.available_orientations = []
        for surface_name, quats in all_orientations_dict.items():
            self.available_orientations.extend(quats)
            
        # Shuffle to mix orientations from different surfaces
        # random.shuffle(self.available_orientations)
        
        # Reset the index
        self.orientation_index = 0
        
        print(f"Prepared {len(self.available_orientations)} orientations for testing")
    
    def create_objects(self):
        """Create multiple YCB objects as rigid bodies with physics enabled"""
        self.objects = []
        
        # Create the specified number of objects
        for i in range(self.num_simultaneous):
            prim_path = f"/World/Object_{i}"
            name = f"object_{i}"
            
            unique_prim_path = find_unique_string_name(
                initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            unique_name = find_unique_string_name(
                initial_name=name, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            if self.mode == "ycb":
                usd_path = ROOT_PATH + self.object_name
                
                # Add reference to stage and get the prim
                object_prim = add_reference_to_stage(usd_path=usd_path, prim_path=unique_prim_path)
                
                # Apply physics properties directly to the prim
                UsdPhysics.RigidBodyAPI.Apply(object_prim)
                UsdPhysics.CollisionAPI.Apply(object_prim)
                
                # Create the RigidPrim object after applying physics properties
                obj = RigidPrim(
                    prim_path=unique_prim_path,
                    name=unique_name,
                    position=np.array([0, 0, 0]),
                    scale=[0.8, 0.8, 0.8]
                )

            elif self.mode == "box":
                from omni.isaac.core.objects import DynamicCuboid
                obj = DynamicCuboid(
                    prim_path=unique_prim_path,
                    name=unique_name,
                    position=np.array([0, 0, 0]),
                    scale=self.box_dims.tolist(),  # [x, y, z]
                    color=np.random.rand(3)  # Give each a random color if you want
                )
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            # Make sure physics is enabled
            obj.enable_rigid_body_physics()
            
            self.scene.add(obj)
            self.objects.append(obj)
            
        return self.objects
    
    def set_drop_poses(self):
        """Position the objects above the ground with uniform orientations, each in its own area"""
        initial_poses = []
        
        # Calculate grid dimensions for areas per object
        grid_size = int(np.ceil(np.sqrt(self.num_simultaneous)))
        
        # Place objects in a grid, each at the center of its own area
        for i, obj in enumerate(self.objects):
            row = i // grid_size
            col = i % grid_size
            
            # Calculate position - exact center of each cell
            x = (col - (grid_size-1)/2)
            y = (row - (grid_size-1)/2)
            z = self.drop_height
            
            # Get the next orientation from our pre-computed list
            quat = self.available_orientations[self.orientation_index]
            self.orientation_index = (self.orientation_index + 1) % len(self.available_orientations)
            
            # Set the initial pose
            pos = np.array([x, y, z])
            obj.set_world_pose(position=pos, orientation=quat)
            initial_poses.append((pos, quat))
        
        # Wait a physics step to ensure all poses are set
        self.world.step(render=True)
        return initial_poses
    
    def are_objects_settled(self):
        """Check if all objects have settled based on position change"""
        current_time = time.time()
        all_settled = True
        
        for i, obj in enumerate(self.objects):
            # Get current position
            current_position, _ = obj.get_world_pose()
            obj_id = id(obj)
            
            # Initialize tracking for this object if not already done
            if obj_id not in self.previous_positions:
                self.previous_positions[obj_id] = current_position
                self.position_history[obj_id] = []
                self.position_timestamps[obj_id] = current_time
                all_settled = False
                continue
            
            # Calculate position change
            position_change = np.linalg.norm(current_position - self.previous_positions[obj_id])
            
            # Update history
            self.position_history[obj_id].append((current_time, position_change))
            # Remove old history entries (older than settlement_time)
            self.position_history[obj_id] = [entry for entry in self.position_history[obj_id] 
                                            if current_time - entry[0] < self.settlement_time]
            
            # Check if the object has moved more than the threshold
            if position_change > self.settlement_threshold:
                self.position_timestamps[obj_id] = current_time  # Reset settlement timer
                all_settled = False
            elif current_time - self.position_timestamps[obj_id] < self.settlement_time:
                # Not settled long enough
                all_settled = False
            
            # Update previous position
            self.previous_positions[obj_id] = current_position
        
        return all_settled
    
    def wait_for_settlement(self, max_wait_time=10.0):
        """Wait for all objects to settle on the ground"""
        start_time = time.time()
        
        # Step a few times before starting to check (let objects fall a bit)
        for _ in range(20):
            self.world.step(render=True)
        
        settled_count = 0
        last_report_time = start_time
        
        while time.time() - start_time < max_wait_time:
            self.world.step(render=True)
            
            # Check if all objects have settled
            if self.are_objects_settled():
                return True
                
            # Report progress every second
            current_time = time.time()
            if current_time - last_report_time > 1.0:
                print(f"Waiting for settlement... ({int(current_time - start_time)}s / {max_wait_time}s)")
                last_report_time = current_time
        
        # If we get here, objects didn't all settle within time limit
        print("Not all objects settled within time limit")
        return False
    
    def record_poses(self):
        """Record the current poses of all settled objects"""
        poses = []
        
        for i, obj in enumerate(self.objects):
            position, orientation_quat = obj.get_world_pose()
            
            # Check if object is on the ground (z close to 0)
            if position[2] > 0.5:  # Ignore objects that might be resting on top of others
                continue
                
            
            pose_data = {
                "position": position.tolist(),
                "orientation_quat": orientation_quat.tolist(),
            }
            
            poses.append(pose_data)
            self.collected_poses.append(pose_data)
            self.save_buffer.append(pose_data)
        
        # Check if buffer is large enough to trigger a save
        if len(self.save_buffer) >= self.save_buffer_size:
            self.save_poses()
        
        return poses
    
    def save_poses(self, force_save=False):
        """
        Save collected poses to file, either by appending or creating new file
        When force_save is True, saves regardless of buffer size
        """
        if not self.save_buffer and not force_save:
            return  # Nothing to save
        
        # If file doesn't exist, create it with an empty list
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                json.dump([], f)
        
        # Read existing data
        try:
            with open(self.output_file, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    # File exists but is empty or invalid
                    existing_data = []
        except FileNotFoundError:
            existing_data = []
        
        # Append new data and write back
        existing_data.extend(self.save_buffer)
        with open(self.output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Saved {len(self.save_buffer)} new poses (Total: {len(existing_data)}) to {self.output_file}")
        
        # Clear the buffer after saving
        self.save_buffer = []

    def sample_uniform_orientations(self):
        """
        Returns a dict mapping each surface name to a list of quaternions
        using per-surface equal-angle bins over [0, 360) and sampling one
        angle uniformly within each bin around the free axis.

        Mirrors the orientation sampling logic from
        `cube_generalization/utils.py::sample_object_poses` (orientation part),
        without encoding positional offsets, since poses are dropped via physics.
        """
        # Baseline Euler angles (deg) for each surface, and which index to vary:
        # roll (0), pitch (1), yaw (2)
        configs = {
            "top_surface":    (np.array([   0.0,   0.0,   0.0]), 2),
            "bottom_surface": (np.array([-180.0,   0.0,   0.0]), 2),
            "left_surface":   (np.array([  90.0,   0.0,   0.0]), 1),
            "right_surface":  (np.array([ -90.0,   0.0,   0.0]), 1),
            "front_surface":  (np.array([  90.0,   0.0,  90.0]), 1),
            "back_surface":   (np.array([  90.0,   0.0, -90.0]), 1),
        }

        # Decide number of samples per surface; match total desired poses across 6 faces
        num_samples_per_surface = max(1, int(self.num_poses // 6))

        # Evenly spaced angles over [0, 360) at bin edges (e.g., 0,10,...,350 for 36 samples)
        angles = np.linspace(0.0, 360.0, num_samples_per_surface, endpoint=False).tolist()

        all_orients = {}
        for name, (base_euler, var_idx) in configs.items():
            quats = []
            for a in angles:
                e = base_euler.copy()
                e[var_idx] = a
                # convert to radians and then quaternion (w, x, y, z)
                q = euler2quat(*np.deg2rad(e), axes='rxyz')
                quats.append(q)
            all_orients[name] = quats

        return all_orients
    


def main(save_path, mode="ycb"):
    # Adjust these parameters as needed
    simultaneous_objects = 20  # You can increase this based on your system's capabilities
    env = PoseCollection(num_poses=120, num_simultaneous=simultaneous_objects, output_file=save_path, mode=mode)
    env.start()
    
    poses_collected = 0
    batch_count = 0
    
    try:
        while simulation_app.is_running() and poses_collected < env.num_poses:
            batch_count += 1
            print(f"Batch {batch_count}: Dropping {len(env.objects)} objects simultaneously")
            
            # Set drop positions and orientations for all objects
            env.set_drop_poses()
            
            # Wait for objects to settle
            print("Waiting for objects to settle...")
            if env.wait_for_settlement(max_wait_time=10.0):
                # Record all stable poses
                batch_poses = env.record_poses()
                new_poses = len(batch_poses)
                poses_collected += new_poses
                print(f"Recorded {new_poses} stable poses (Total: {poses_collected}/{env.num_poses})")
                
                # Save after each successful batch
                env.save_poses(force_save=True)  # Force save even if buffer isn't full
                
                # If we've collected enough poses, break
                if poses_collected >= env.num_poses:
                    break
            else:
                print("Not all objects settled properly, but continuing...")
                # Still record what we can
                batch_poses = env.record_poses()
                new_poses = len(batch_poses)
                poses_collected += new_poses
                print(f"Recorded {new_poses} stable poses (Total: {poses_collected}/{env.num_poses})")
                
                # Save after each batch even if not all objects settled
                env.save_poses(force_save=True)  # Force save even if buffer isn't full
            
            # Small delay between batches
            time.sleep(0.5)
        
        # Final save to make sure everything is saved
        env.save_poses(force_save=True)
        
    except KeyboardInterrupt:
        print("Collection interrupted by user")
        # Save what we have if interrupted
        env.save_poses(force_save=True)
    
    print("Pose collection complete")
    simulation_app.close()
    
if __name__ == "__main__":
    # tmp_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/experiments/test_different_dimensions/object_poses_box.json"
    output_path = "/home/chris/Chris/placement_ws/src/object_poses_box.json"
    mode = "box"
    main(output_path, mode)