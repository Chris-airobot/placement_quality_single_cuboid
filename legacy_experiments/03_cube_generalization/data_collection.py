import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from isaacsim import SimulationApp

# Display options for the simulation viewport
DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

# Simulation configuration
CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    # "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

# Initialize the simulation application
simulation_app = SimulationApp(CONFIG)

import datetime 
import numpy as np
import json
import carb
from copy import deepcopy
import omni
from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux, Gf, UsdGeom
from collision_check import GroundCollisionDetector
from utils import sample_object_poses, sample_dims, quat_wxyz_to_R, corners_local, face_id_from_R, hand_object_local_features
from placement_quality.cube_simulation import helper
from collections import defaultdict
import json
from scipy.spatial.transform import Rotation as R
from placement_quality.cube_generalization.grasp_pose_generator import generate_grasp_poses_including_bottom



class DataCollection:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self.object = None
        self._pedestal = None
        self.world = None
        self.collision_detector = None
        self.base_path = "/World/panda"
        self.robot_parts_to_check = []
        self.collision_detected = False
        self.dimension_set = sample_dims()
        self.box_dims = self.dimension_set.pop(0)

        self.pedestal_height = 0.10  # meters

        # Offset of the grasp from the object center
        # Grasp poses are based on the gripper center, so we need to offset it to transform it to the tool_center, i.e. 
        # the middle of the gripper fingers
        self.grasp_offset = [0, 0, -0.065] 
        self.episode_count = 0
        self.grasp_count = 0
        self.object_count = 0
        self.grasps = None
        self.object_poses = None
        self.data_dict = defaultdict(list)
        self.data_folder = "/home/chris/Chris/placement_ws/src/data/cube_generalization/v1/data_collection/raw_data/"



    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()
        
        # Load robot and target
        self._articulation, self.object = self.load_assets()
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self.object)
        
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        # Set up the physics scene
        self.world.reset()

        # Set up the kinematics solver
        self.setup_kinematics()



    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])
        

    def load_assets(self):
        """Load the Franka robot and target frame"""
        # Add the Franka robot to the stage
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        articulation = Articulation(robot_prim_path)
        
        # Add a box object for collision testing (similar to the data collection script)
        from omni.isaac.core.objects import VisualCuboid
        object = VisualCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array([0.2, -0.3, 0.125]),  # Adjusted z to sit properly on pedestal (0.05 + 0.10 + 0.051/2)
            scale=self.box_dims,  # [x, y, z]
            color=np.array([0.8, 0.8, 0.8])  # Light gray color
        )
        return articulation, object
    

    def setup_kinematics(self):
        """Set up the kinematics solver for the Franka robot"""
        # Load kinematics configuration for the Franka robot
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)        
        # Configure the articulation kinematics solver with the end effector
        end_effector_name = "panda_hand"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, 
            self._kinematics_solver, 
            end_effector_name
        )

    def setup_collision_detection(self):
        """Set up the ground collision detector"""
        stage = omni.usd.get_context().get_stage()
        self.collision_detector = GroundCollisionDetector(stage, non_colliding_part="/World/panda/panda_link0")
        
        # Create a virtual ground plane at z=0 (ground level)
        self.collision_detector.create_virtual_ground(
            size_x=20.0, 
            size_y=20.0, 
            position=Gf.Vec3f(0, 0, 0)  # Ground level
        )

        self.collision_detector.create_virtual_pedestal(
            position=Gf.Vec3f(0.2, -0.3, 0.05)
        )
        
        # Define robot parts to check for collisions (explicitly excluding the base link0)
        self.robot_parts_to_check = [
            f"{self.base_path}/panda_link1",
            f"{self.base_path}/panda_link2",
            f"{self.base_path}/panda_link3",
            f"{self.base_path}/panda_link4",
            f"{self.base_path}/panda_link5",
            f"{self.base_path}/panda_link6",
            f"{self.base_path}/panda_link7",
            f"{self.base_path}/panda_hand",
            f"{self.base_path}/panda_leftfinger",
            f"{self.base_path}/panda_rightfinger"
        ]
        # Note: panda_link0 is deliberately excluded since it's expected to touch the ground


    def check_for_collisions(self):
        """Check if any robot parts are colliding with ground, pedestal, or box."""
        ground_hit = any(
            self.collision_detector.is_colliding_with_ground(part_path)
            for part_path in self.robot_parts_to_check
        )
        pedestal_hit = any(
            self.collision_detector.is_colliding_with_pedestal(part_path)
            for part_path in self.robot_parts_to_check
        )
        box_hit = any(
            self.collision_detector.is_colliding_with_box(part_path)
            for part_path in self.robot_parts_to_check
        )
        self.collision_detected = ground_hit or pedestal_hit or box_hit
        return self.collision_detected
    
    
    def set_cuboid_scale(self, new_scale):
        prim = self.object.prim
        xform = UsdGeom.Xformable(prim)
        # Look for existing scale op
        found = False
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3d(*new_scale))  # Use Vec3d for double3 precision
                found = True
                break
        if not found:
            # If none exists, create one (rare for VisualCuboid)
            xform.AddScaleOp().Set(Gf.Vec3d(*new_scale))

    def save_data(self):
        """Save the data to a file"""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_folder), exist_ok=True)
        
        # Create the file path
        file_path = self.data_folder + f"object_{self.box_dims[0]}_{self.box_dims[1]}_{self.box_dims[2]}.json"
        
        # Save the data to the file
        with open(file_path, "w") as f:
            json.dump(self.data_dict, f)

        self.data_dict = defaultdict(list)

    
    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()

        current_object_finished = True

        
        # Main simulation loop
        while simulation_app.is_running():
            if current_object_finished:
                if len(self.dimension_set) == 0:
                    break
                self.box_dims = self.dimension_set.pop(0)
                self.set_cuboid_scale(self.box_dims)
                self.object_poses = sample_object_poses(36, self.box_dims)

                # Precompute grasps once per object pose
                TILT_SET = (75.0, 90.0, 105.0)
                YAW_SET  = (-15.0, 0.0, 15.0)
                ROLL_SET = (-15.0, 0.0, 15.0)

                self.grasp_batches = []
                for object_pose in self.object_poses:
                    q_wxyz = np.array(object_pose[3:7], dtype=float)
                    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
                    R_obj = R.from_quat(q_xyzw).as_matrix()
                    t_obj = np.array(object_pose[:3], dtype=float)
                    t_obj[2] += self.pedestal_height  # use the same z as during execution

                    acc = []
                    for tilt in TILT_SET:
                        for yaw in YAW_SET:
                            for roll in ROLL_SET:
                                acc.extend(generate_grasp_poses_including_bottom(
                                    dims_xyz=np.array(self.box_dims, dtype=float),
                                    R_obj_to_world=R_obj,
                                    t_obj_world=t_obj,
                                    enable_tilt=True, tilt_deg=tilt,
                                    enable_yaw=True,  yaw_deg=yaw,
                                    enable_roll=True, roll_deg=roll,
                                    filter_by_gripper_open=False,
                                    apply_hand_to_tcp=True, hand_to_tcp_z=0.1034, extra_insert=-0.0334,
                                ))
                    # Cap to 300; no dedup
                    self.grasp_batches.append(acc)

                print(f"Prepared {len(self.object_poses)} poses with ~300 grasps each")
                current_object_finished = False
                
            else:
                # After: self.object_poses = sample_object_poses(...)
                for obj_idx, object_pose in enumerate(self.object_poses):
                    self.current_object_pose = object_pose.copy()
                    self.current_object_pose[2] += self.pedestal_height  # keep pedestal
                    self.object.set_world_pose(self.current_object_pose[:3], self.current_object_pose[3:])
                    print(f"The current {self.episode_count}th object pose {obj_idx+1}/{len(self.object_poses)}")

                    grasps = self.grasp_batches[obj_idx]
                    for g_idx, g in enumerate(grasps):
                        hand_pos = np.array(g['hand_position_world'], dtype=float)
                        hand_quat = np.array(g['hand_quaternion_wxyz'], dtype=float)

                        base_t, base_q = self._articulation.get_world_pose()
                        self._kinematics_solver.set_robot_base_pose(base_t, base_q)

                        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(hand_pos, hand_quat)
                        if success:
                            self._articulation.apply_action(action)

                        # step the sim for THIS grasp (so it’s not “all at once”)
                        self.world.step(render=True)

                        # now check collisions and log
                        collision = self.check_for_collisions()
                        ground_hit   = any(self.collision_detector.is_colliding_with_ground(p)   for p in self.robot_parts_to_check)
                        pedestal_hit = any(self.collision_detector.is_colliding_with_pedestal(p) for p in self.robot_parts_to_check)
                        box_hit      = any(self.collision_detector.is_colliding_with_box(p)      for p in self.robot_parts_to_check)

                        grasp_pose = hand_pos.tolist() + hand_quat.tolist()
                        # unpack dims for this object/orientation
                        dx, dy, dz = float(self.box_dims[0]), float(self.box_dims[1]), float(self.box_dims[2])

                        # compute local hand-object features
                        t_loc, R_loc6 = hand_object_local_features(grasp_pose, self.current_object_pose)

                        # corners → min Z at FINAL pose
                        R_obj = quat_wxyz_to_R(self.current_object_pose[3:7])
                        t_obj = np.array(self.current_object_pose[:3], float)
                        cw = corners_local(dx, dy, dz)
                        final_world = (R_obj @ cw.T).T + t_obj
                        min_z_corner_final = float(final_world[:,2].min())

                        # face id and hand z
                        final_face_id = face_id_from_R(R_obj)
                        z_hand_final  = float(grasp_pose[2])

                        # build extras (only must-have fields here)
                        extras = {
                            "dims": [dx, dy, dz],
                            "orientation_id": int(obj_idx),
                            "grasp_id_local": int(g_idx),           # whatever counter you use in this loop
                            "final_face_id": final_face_id,         # 1..6 as defined above
                            "t_loc": t_loc.tolist(),                # [3]
                            "R_loc6": R_loc6.tolist(),              # [6]
                            "min_z_corner_final": min_z_corner_final,
                            "z_hand_final": z_hand_final
                            # (optional) "gen_ver": "exp_v6", "dataset_ver": "2025-08-16"
                            # (optional) "jaw_span_m": ..., "lateral_offset_rel": ...
                        }

                        # append with extras as 8th element — order of the first 7 elements untouched
                        self.data_dict[f"grasp_{obj_idx}"].append(
                            [grasp_pose, self.current_object_pose, success, collision, ground_hit, pedestal_hit, box_hit, extras]
                        )

                        # optional: per-grasp progress
                        print(f"Pose {obj_idx+1}/{len(self.object_poses)} | Grasp {g_idx+1}/{len(grasps)}")

                self.save_data()
                self.episode_count += 1
                self.grasp_count = 0
                current_object_finished = True
                # rows = self.data_dict[f"grasp_{obj_idx}"]
                # faces = {}
                # colls = succs = 0
                # for r in rows:
                #     extras = r[7]
                #     faces[extras["final_face_id"]] = faces.get(extras["final_face_id"], 0) + 1
                #     succs += int(bool(r[2]))
                #     colls += int(bool(r[3]))
                # print(f"[obj_idx={obj_idx}] count={len(rows)} face_counts={faces} success={succs} collision={colls}")

        
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env = DataCollection()
    env.run()
    
    # Close the simulation application
    simulation_app.close()