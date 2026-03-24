import os
import sys
import json
import numpy as np
import datetime
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import random

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

CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

simulation_app = SimulationApp(CONFIG)

import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux
PEDESTAL_SIZE = np.array([0.09, 0.11, 0.1])

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

def surface_transition_type(init_label, final_label):
    if init_label == final_label:
        return 'same'
    axis = init_label[0]
    if final_label.startswith(axis) and init_label != final_label:
        return 'opposite'
    return 'adjacent'

class DifficultyLabeler:
    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None
        self.world = None
        self.base_path = "/World/panda"
        self.pedestal_height = 0.10  # meters

        self.test_data_path = '/home/chris/Chris/placement_ws/src/data/box_simulation/v2/combined_data/test.json'
        with open(self.test_data_path, 'r') as file:
            self.test_data = json.load(file)
        self.box_dims = np.array([0.143, 0.0915, 0.051])
        self.DIR_PATH = os.path.join("/home/chris/Chris/placement_ws/src/data/box_simulation/v2", f"combined_data/")

    def setup_scene(self):
        create_new_stage()
        self._add_light_to_stage()
        self.world: World = World()
        self._articulation, self._target = self.load_assets()
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        self.world.reset()
        self.setup_kinematics()

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def load_assets(self):
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        articulation = Articulation(robot_prim_path)
        from omni.isaac.core.objects import VisualCuboid
        box_dims = np.array([0.143, 0.0915, 0.051])
        target = VisualCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array([0.2, -0.3, 0.125]),
            scale=box_dims.tolist(),
            color=np.array([0.8, 0.8, 0.8])
        )
        return articulation, target

    def setup_kinematics(self):
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)
        end_effector_name = "panda_hand"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation,
            self._kinematics_solver,
            end_effector_name
        )

    def compute_ik(self, ee_position, ee_orientation):
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        ee_position = np.array(ee_position, dtype=np.float64)
        ee_orientation = np.array(ee_orientation, dtype=np.float64)
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(ee_position, ee_orientation)
        if success:
            return action.joint_positions
        else:
            print(f"IK failed")
            return None

    def compute_jacobian(self, joint_config, ee_link_name="panda_hand"):
        if len(joint_config) == 7:
            joint_config = np.concatenate([joint_config, [0.04, 0.04]])
        elif len(joint_config) == 9:
            joint_config = np.array(joint_config)
        self._articulation.set_joint_positions(joint_config)
        self.world.step(render=True)
        articulation_view = self._articulation._articulation_view
        jacobians = articulation_view.get_jacobians()
        ee_index = articulation_view.get_link_index(ee_link_name)
        jacobian = jacobians[0, ee_index, :, :]
        JJ_T = jacobian @ jacobian.T
        manipulability = np.sqrt(np.linalg.det(JJ_T))
        return manipulability

    def label_difficulty(self, value, thresholds):
        if value < thresholds[0]:
            return 'Easy'
        elif value < thresholds[1]:
            return 'Medium'
        else:
            return 'Hard'

    def calculate_placement_pose(self, grasp_pose, object_initial_pose, object_final_pose):
        # print(f"Calculating placement pose")
        # print(f"grasp_pose: {grasp_pose}")
        # print(f"object_initial_pose: {object_initial_pose}")
        # print(f"object_final_pose: {object_final_pose}")
        grasp_position_initial             = np.array(grasp_pose[0:3])
        grasp_orientation_wxyz_initial     = grasp_pose[3:]

        # Initial object pose in world: [z, qw, qx, qy, qz]
        object_height_initial              = object_initial_pose[2] 
        object_orientation_wxyz_initial    = object_initial_pose[3:]
        object_position_initial            = np.array([0.2, -0.3, object_height_initial])

        # Final object pose in world: [z, qw, qx, qy, qz]
        object_height_final                = object_final_pose[2]
        object_orientation_wxyz_final      = object_final_pose[3:]
        object_position_final              = np.array([0.2, -0.3, object_height_final])
    


        # scipy Rotation.from_quat expects [x, y, z, w]
        rotation_grasp_initial = R.from_quat([
            grasp_orientation_wxyz_initial[1],
            grasp_orientation_wxyz_initial[2],
            grasp_orientation_wxyz_initial[3],
            grasp_orientation_wxyz_initial[0]
        ]).as_matrix()

        rotation_object_initial = R.from_quat([
            object_orientation_wxyz_initial[1],
            object_orientation_wxyz_initial[2],
            object_orientation_wxyz_initial[3],
            object_orientation_wxyz_initial[0]
        ]).as_matrix()

        rotation_object_final = R.from_quat([
            object_orientation_wxyz_final[1],
            object_orientation_wxyz_final[2],
            object_orientation_wxyz_final[3],
            object_orientation_wxyz_final[0]
        ]).as_matrix()

        # --- 3) Compute the fixed hand‐to‐object transform ---------------

        # R_grasp_in_object = R_object_initial^T * R_grasp_initial
        rotation_grasp_in_object_frame = (
            rotation_object_initial.T @ rotation_grasp_initial
        )

        # p_grasp_in_object = R_object_initial^T * (p_grasp_initial – p_object_initial)
        translation_grasp_in_object_frame = (
            rotation_object_initial.T @ (grasp_position_initial - object_position_initial)
        )

        # --- 4) Re‐apply that to the new object pose --------------------

        # p_grasp_final = R_object_final * p_grasp_in_object + p_object_final
        final_grasp_position = (
            rotation_object_final @ translation_grasp_in_object_frame
            + object_position_final
        )

        # R_grasp_final = R_object_final * R_grasp_in_object
        rotation_final_grasp = (
            rotation_object_final @ rotation_grasp_in_object_frame
        )

        # --- 5) Convert back to quaternion [w, x, y, z] -----------------

        # scipy gives [x, y, z, w]
        quaternion_final_grasp_xyzw = R.from_matrix(
            rotation_final_grasp
        ).as_quat()

        # reorder to [w, x, y, z]
        quaternion_final_grasp_wxyz = np.array([
            quaternion_final_grasp_xyzw[3],
            quaternion_final_grasp_xyzw[0],
            quaternion_final_grasp_xyzw[1],
            quaternion_final_grasp_xyzw[2]
        ])

        # --- 6) Stack into your final grasp pose ------------------------

        # final_grasp_pose = np.concatenate([
        #     final_grasp_position,
        #     quaternion_final_grasp_wxyz
        # ])

        return [final_grasp_position, quaternion_final_grasp_wxyz]


    def calculate_angular_distance(self, initial_quat, final_quat):
        initial_quat_R = R.from_quat(initial_quat)
        final_quat_R = R.from_quat(final_quat)
        relative_rotation = initial_quat_R.inv() * final_quat_R
        angular_distance = relative_rotation.magnitude() * (180 / np.pi)
        return angular_distance

    def calculate_joint_distance(self, grasp_pose, initial_pose, final_pose):
        # initial_quat = initial_pose[3:]
        # final_quat = final_pose[3:]
        # initial_quat_R = R.from_quat(initial_quat)
        # final_quat_R = R.from_quat(final_quat)
        grasp_quat = np.array(grasp_pose[3:], dtype=np.float64)
        grasp_position = np.array(grasp_pose[:3], dtype=np.float64)

        # initial_ee_orientation = (initial_quat_R * R.from_quat(grasp_quat)).as_quat()
        # final_ee_orientation = (final_quat_R * R.from_quat(grasp_quat)).as_quat()
        # initial_ee_orientation = np.array(initial_ee_orientation, dtype=np.float64)
        # final_ee_orientation = np.array(final_ee_orientation, dtype=np.float64)

        placement_position, placement_orientation = self.calculate_placement_pose(grasp_pose, 
                                                                                  initial_pose, 
                                                                                  final_pose)

        initial_joints = self.compute_ik(grasp_position, grasp_quat)
        final_joints = self.compute_ik(placement_position, placement_orientation)
        if final_joints is None:
            print(f"IK failed")
            return None, None
        joint_distance = np.linalg.norm(final_joints - initial_joints)
        return joint_distance, final_joints

    def run(self):
        self.setup_scene()
        
        # Process all cases in segments
        total_cases = len(self.test_data)
        segment_size = 10000  # Process 10k cases at a time
        total_segments = (total_cases + segment_size - 1) // segment_size  # Ceiling division
        
        print(f"Processing {total_cases} cases in {total_segments} segments of {segment_size} each...")

        # Initialize output file
        output_path = os.path.join(self.DIR_PATH, 'labeled_test_data_full.json')
        os.makedirs(self.DIR_PATH, exist_ok=True)
        
        # Start writing to file
        with open(output_path, 'w') as file:
            file.write('[\n')  # Start JSON array
            
            for segment_idx in range(total_segments):
                start_idx = segment_idx * segment_size
                end_idx = min((segment_idx + 1) * segment_size, total_cases)
                segment_cases = self.test_data[start_idx:end_idx]
                
                print(f"\nProcessing segment {segment_idx + 1}/{total_segments} (cases {start_idx}-{end_idx-1})")
                
                # Process this segment
                segment_results = []
                for case in tqdm(segment_cases, desc=f"Segment {segment_idx + 1}", unit="case"):
                    grasp_pose = case['grasp_pose']
                    initial_pose = [0.2, -0.3] + case['initial_object_pose']
                    final_pose = [0.2, -0.3] + case['final_object_pose']
                    
                    initial_quat = initial_pose[3:]
                    final_quat = final_pose[3:]
                    angular_distance = self.calculate_angular_distance(initial_quat, final_quat)
                    joint_distance, final_joints = self.calculate_joint_distance(grasp_pose, initial_pose, final_pose)
                    
                    # Surface transition logic
                    surface_up_initial = get_surface_up_label(initial_pose[3:])
                    surface_up_final = get_surface_up_label(final_pose[3:])
                    transition_type = surface_transition_type(surface_up_initial, surface_up_final)
                    
                    # Calculate manipulability if IK was successful
                    manipulability = None
                    ik_success = False
                    if joint_distance is not None and final_joints is not None:
                        manipulability = self.compute_jacobian(final_joints)
                        ik_success = True
                    
                    # Create result case
                    result_case = {
                        'grasp_pose': case['grasp_pose'],
                        'initial_object_pose': case['initial_object_pose'],
                        'final_object_pose': case['final_object_pose'],
                        'angular_distance': float(angular_distance),
                        'joint_distance': float(joint_distance) if joint_distance is not None else None,
                        'manipulability': float(manipulability) if manipulability is not None else None,
                        'surface_up_initial': surface_up_initial,
                        'surface_up_final': surface_up_final,
                        'surface_transition_type': transition_type,
                        'success_label': case['success_label'],
                        'collision_label': case['collision_label'],
                        'ik_success': ik_success
                    }
                    
                    segment_results.append(result_case)
                
                # Write this segment to file
                for i, result in enumerate(segment_results):
                    json_str = json.dumps(result, indent=2)
                    if segment_idx == 0 and i == 0:
                        # First case - no comma needed
                        file.write(json_str)
                    else:
                        # Add comma before each case except the first
                        file.write(',\n' + json_str)
                
                # Force write to disk
                file.flush()
                
                # Print segment summary
                successful_ik = sum(1 for r in segment_results if r['ik_success'])
                print(f"Segment {segment_idx + 1} complete: {len(segment_results)} cases, {successful_ik} successful IK")
            
            file.write('\n]')  # End JSON array
        
        print(f"\n=== Processing Complete ===")
        print(f"Total cases processed: {total_cases}")
        print(f"Output saved to: {output_path}")






import os

def fix_broken_json_array(filename):
    # Read the file backwards in chunks, looking for the last valid object
    chunk_size = 8192  # Read 8 KB at a time
    end_marker = b'}'
    last_valid = None
    file_size = os.path.getsize(filename)

    with open(filename, 'rb') as f:
        # Start from the end, go backwards
        pos = file_size
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            idx = chunk.rfind(end_marker)
            if idx != -1:
                # Move forward to the end of this object
                last_valid = pos + idx + 1
                break

    if last_valid is None:
        print("Could not find a complete JSON object. Nothing changed.")
        return

    # Copy up to the last valid object
    fixed_file = filename.replace('.json', '_fixed.json')
    with open(filename, 'rb') as original, open(fixed_file, 'w+b') as fixed:
        fixed.write(original.read(last_valid))
        # Remove a possible trailing comma
        fixed.seek(-1, os.SEEK_END)
        if fixed.read(1) == b',':
            fixed.truncate(fixed.tell() - 1)
        # Add closing bracket
        fixed.write(b'\n]')
    print(f"Written fixed file: {fixed_file}")
# Usage

if __name__ == "__main__":
    # env = DifficultyLabeler()
    # env.run()
    # simulation_app.close()
    fix_broken_json_array('/home/chris/Chris/placement_ws/src/data/box_simulation/v2/combined_data/labeled_test_data_full.json')
