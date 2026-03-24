import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from isaacsim import SimulationApp
from scipy.spatial.transform import Rotation as R
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
    "headless": False,
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
from pxr import Sdf, UsdLux
# removed invalid import of ArticulationAction

# Import the collision detection functionality
from collision_check import GroundCollisionDetector
from pxr import UsdGeom, Gf, Usd
from omni.physx import get_physx_scene_query_interface


base_dir = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")

# --- Pilot toggles ---
# Medium pilot (~96,000): 4 pedestals × 400 grasps × 60 object poses
RUN_96K_PILOT = False

# Mini pilot (~9,600): 2 pedestals × 120 grasps × 40 object poses
RUN_9K_PILOT = False


# --- Multi-machine pedestal split ---
# Set this on each machine to shard pedestals without downsampling objects/grasps.
# Example assignments for 3 machines (contiguous 3/3/4 split):
#   "A": [0, 1, 2]
#   "B": [3, 4, 5]
#   "C": [6, 7, 8, 9]
MACHINE_CASE = "A"  # set to "A", "B", or "C" on each machine; leave None to run all pedestals
PEDESTAL_SPLITS = {
    "A": [0, 1, 2],
    "B": [3, 4, 5],
    "C": [6, 7, 8, 9],
}



def transform_relative_pose(grasp_pose, relative_translation, relative_rotation=None):
    from pyquaternion import Quaternion
    """
    Transforms a grasp pose using a relative transformation.
    input: grasp_pose: [position, orientation]
    input: relative_translation: [x, y, z]
    input: relative_rotation: [w, x, y, z]
    output: [position, orientation]
    """
    # Helper: Convert a pose (position, quaternion) to a 4x4 homogeneous transformation matrix.
    def pose_to_matrix(position, orientation):
        T = np.eye(4)
        q = Quaternion(orientation)  # expects [w, x, y, z]
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = position
        return T

    # Helper: Convert a 4x4 homogeneous transformation matrix back to a pose.
    def matrix_to_pose(T):
        position = T[:3, 3].tolist()
        q = Quaternion(matrix=T[:3, :3])
        orientation = q.elements.tolist()  # [w, x, y, z]
        return position, orientation

    # Convert the input grasp pose to a homogeneous matrix.
    T_current = pose_to_matrix(grasp_pose[0], grasp_pose[1])

    # Build the relative transformation matrix.
    T_relative = np.eye(4)
    if relative_rotation is None:
        q_relative = Quaternion()  # Identity rotation.
    else:
        q_relative = Quaternion(relative_rotation)
    T_relative[:3, :3] = q_relative.rotation_matrix
    T_relative[:3, 3] = relative_translation

    # Apply the transformation - for local to global, we need:
    # T_target = T_relative * T_current (object_world * grasp_local)
    T_target = np.dot(T_relative, T_current)

    # Convert back to position and quaternion.
    new_position, new_orientation = matrix_to_pose(T_target)
    
    return [new_position, new_orientation]


def local_transform(pose, offset):
    """Apply offset in the local frame of the pose"""
    from pyquaternion import Quaternion
    # Convert to matrices
    T_pose = np.eye(4)
    q = Quaternion(pose[1])  # [w, x, y, z]
    T_pose[:3, :3] = q.rotation_matrix
    T_pose[:3, 3] = pose[0]
    
    # Create offset matrix (identity rotation)
    T_offset = np.eye(4)
    T_offset[:3, 3] = offset
    
    # Multiply in correct order: pose * offset (applies offset in local frame)
    T_result = np.dot(T_pose, T_offset)
    
    # Convert back to position, orientation
    new_position = T_result[:3, 3].tolist()
    q_new = Quaternion(matrix=T_result[:3, :3])
    new_orientation = q_new.elements.tolist()  # [w, x, y, z]
    
    return [new_position, new_orientation]




class StandaloneIK:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None
        self._pedestal = None
        self.world = None
        self.collision_detector = None
        self.base_path = "/World/panda"
        self.robot_parts_to_check = []
        self.collision_detected = False
        self.pedestal_height = 0.10  # meters
        self.pedestal_dims = np.array([0.27, 0.22, 0.10], dtype=float)



        # File reads
        object_poses_file = "/home/chris/Chris/placement_ws/src/object_poses_box.json"
        self.all_object_poses = json.load(open(object_poses_file))
        
        self.grasp_poses_file = "/home/chris/Chris/placement_ws/src/grasps.json"
        self.grasp_poses = json.load(open(self.grasp_poses_file))

        pedestal_poses_file = "/home/chris/Chris/placement_ws/src/pedestal_poses.json"
        self.all_pedestal_poses = json.load(open(pedestal_poses_file))


        # Iteration indices and ordered grasp keys (do not pop)
        self.pedestal_cursor = 0   # skip first pedestal (index 0 already done)
        self.object_idx = 0
        self.grasp_keys = sorted(self.grasp_poses.keys(), key=lambda k: int(k))
        self.grasp_idx = 0

        # --- Pilot subset wiring ---
        num_all_ped = len(self.all_pedestal_poses)
        num_all_obj = len(self.all_object_poses)
        num_all_grasps = len(self.grasp_keys)

        # Defaults = full run
        self.pedestal_indices = list(range(num_all_ped))
        self.object_indices = list(range(num_all_obj))
        # keep grasp_keys as-is by default

        # Apply pilots (96k takes priority if both True)
        if RUN_96K_PILOT:
            # 4 pedestals: [0, 3, 6, 9] (assumes you have 10)
            self.pedestal_indices = [0, 3, 6, 9]
            # 60 objects, 400 grasps
            obj_idx = self._subsample_indices(num_all_obj, 60)
            grasp_idx = self._subsample_indices(num_all_grasps, 400)
            self.object_indices = obj_idx
            self.grasp_keys = [self.grasp_keys[i] for i in grasp_idx]

        elif RUN_9K_PILOT:
            # 2 pedestals: [0, 5]
            self.pedestal_indices = [0, 5]
            # 40 objects, 120 grasps
            obj_idx = self._subsample_indices(num_all_obj, 40)
            grasp_idx = self._subsample_indices(num_all_grasps, 120)
            self.object_indices = obj_idx
            self.grasp_keys = [self.grasp_keys[i] for i in grasp_idx]

        # Apply machine-specific pedestal sharding (no downsampling of objects/grasps)
        if MACHINE_CASE in PEDESTAL_SPLITS:
            self.pedestal_indices = PEDESTAL_SPLITS[MACHINE_CASE]
            self.object_indices = list(range(num_all_obj))  # ensure full objects
            # keep self.grasp_keys as-is for full grasps

        # Tracks the real pedestal index used in this episode (for saving)
        self.active_pedestal_index = int(self.pedestal_indices[self.pedestal_cursor])


        self.data_folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection"
        # New: base folder for raw data per pedestal index
        self.data_root = os.path.join(self.data_folder, "raw_data")
        # New: start from pedestal index 1 (skip the first done already)
        # already set above

        # Offset of the grasp from the object center
        # Grasp poses are based on the gripper center, so we need to offset it to transform it to the tool_center, i.e. 
        # the middle of the gripper fingers
        self.grasp_offset = [0, 0, -0.065] 
        self.data_dict = {}
        self.count = 0
        self.episode_count = 0

        self.box_dims = np.array([0.143, 0.0915, 0.051])

    @staticmethod
    def _subsample_indices(total: int, desired: int):
        """
        Deterministically pick `desired` indices spread over range(total).
        If desired >= total, return all indices.
        """
        if desired >= total:
            return list(range(total))
        if desired <= 0:
            return []
        # Evenly-spaced via linspace rounding; ensure uniqueness & sorted order.
        xs = np.linspace(0, total - 1, num=desired)
        idxs = sorted({int(round(x)) for x in xs})
        # If rounding caused a shortfall, fill greedily.
        while len(idxs) < desired:
            for k in range(total):
                if k not in idxs:
                    idxs.append(k)
                    if len(idxs) == desired:
                        break
        return sorted(idxs)

    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()
        
        # Load robot and target
        self._articulation, self._target = self.load_assets()
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        # self.world.scene.add(self._pedestal)
        
        # self._articulation.set_enabled_self_collisions(True)
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        # Set up the physics scene
        self.world.reset()

        # self._target.set_world_pose(self.object_pose_at_grasp["position"], 
        #                             self.object_pose_at_grasp["orientation_quat"])
        
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
        box_dims = self.box_dims  # Same dimensions as in data collection
        target = VisualCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array([0.2, -0.3, 0.125]),  # Adjusted z to sit properly on pedestal (0.05 + 0.10 + 0.051/2)
            scale=box_dims.tolist(),  # [x, y, z]
            color=np.array([0.8, 0.8, 0.8])  # Light gray color
        )

        
        
        return articulation, target
    def setup_kinematics(self):
        """Set up the kinematics solver for the Franka robot"""
        # Load kinematics configuration for the Franka robot
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)
        
        # Print valid frame names for debugging
        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
        
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
            position=Gf.Vec3f(0.2, -0.3, 0.05),
            size_x=float(self.pedestal_dims[0]),
            size_y=float(self.pedestal_dims[1]),
            size_z=float(self.pedestal_dims[2])
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
    
    def _compose_pregrasp_lift(self, contact_pos, contact_quat):
        """Given contact TCP pose (world), return pregrasp P and lift L poses."""
        # tool Z (approach) from contact_quat
        Rcw = R.from_quat([contact_quat[1], contact_quat[2], contact_quat[3], contact_quat[0]]).as_matrix()
        z_tool = Rcw[:, 2]  # world direction of +Z_tool
        P_pos = contact_pos - self.d_a * z_tool
        P_quat = contact_quat
        L_pos = contact_pos + np.array([0.0, 0.0, self.h_lift], dtype=float)
        L_quat = contact_quat
        return (P_pos, P_quat), (L_pos, L_quat)

    def _sweep_joints_collision(self, q_start, q_end):
        """Joint-space sweep from q_start to q_end with collision checks.
        Returns a dict: {
            "ok": bool,        # True if no ground/pedestal/box were hit
            "ok_env": bool,    # True if no ground/pedestal were hit
            "hit_ground": bool,
            "hit_pedestal": bool,
            "hit_box": bool,
            "fail_step": Optional[int]  # first step where ground or pedestal is hit; None if no env hit
        }
        """
        q_start = np.asarray(q_start, dtype=float)
        q_end   = np.asarray(q_end, dtype=float)
        dq      = np.abs(q_end - q_start)
        n_steps = int(max(1, np.ceil(np.max(dq) / float(self.dq_step))))

        hit_ground = False
        hit_pedestal = False
        hit_box = False
        env_fail_step = None

        for s in range(1, n_steps + 1):
            alpha = s / n_steps
            q_s_arm = (1.0 - alpha) * q_start + alpha * q_end  # typically 7-DOF arm vector

            curr_full = np.array(self._articulation.get_joint_positions(), dtype=float)
            arm_len = min(len(q_s_arm), len(curr_full))
            curr_full[:arm_len] = q_s_arm[:arm_len]
            self._articulation.set_joint_positions(curr_full)
            self.world.step(render=False)

            # Check collisions
            ground_now = any(self.collision_detector.is_colliding_with_ground(p)   for p in self.robot_parts_to_check)
            ped_now    = any(self.collision_detector.is_colliding_with_pedestal(p) for p in self.robot_parts_to_check)
            box_now    = any(self.collision_detector.is_colliding_with_box(p)      for p in self.robot_parts_to_check)

            # Accumulate flags
            if box_now:
                hit_box = True
            if ground_now:
                hit_ground = True
            if ped_now:
                hit_pedestal = True

            # Early-exit ONLY on environment hit (ground or pedestal)
            if (ground_now or ped_now) and env_fail_step is None:
                env_fail_step = s
                break

        ok_env = not (hit_ground or hit_pedestal)
        ok_any = ok_env and (not hit_box)

        return {
            "ok": bool(ok_any),
            "ok_env": bool(ok_env),
            "hit_ground": bool(hit_ground),
            "hit_pedestal": bool(hit_pedestal),
            "hit_box": bool(hit_box),
            "fail_step": (int(env_fail_step) if env_fail_step is not None else None)
        }

    
    def _store_sample(self, okC, okP, okL, sweep_PC, sweep_CL, P, C, L, qC=None, qP=None, qL=None):
        (P_pos, P_quat), (C_pos, C_quat), (L_pos, L_quat) = P, C, L

        # Endpoint collision snapshot at the CURRENT robot state.
        # Callers now move to C before storing when C-IK is feasible.
        collision = self.check_for_collisions()
        ground_hit = any(self.collision_detector.is_colliding_with_ground(p)   for p in self.robot_parts_to_check)
        pedestal_hit = any(self.collision_detector.is_colliding_with_pedestal(p) for p in self.robot_parts_to_check)
        box_hit = any(self.collision_detector.is_colliding_with_box(p)         for p in self.robot_parts_to_check)

        default_sweep = {
            "ok": False,
            "ok_env": False,
            "hit_ground": False,
            "hit_pedestal": False,
            "hit_box": False,
            "fail_step": None
        }

        self.data_dict[self.count] = {
            "endpoint_collision_at_C": {
                "any": bool(collision),
                "ground": bool(ground_hit),
                "pedestal": bool(pedestal_hit),
                "box": bool(box_hit)
            },
            "object_pose_world": {
                "position": self.current_object_pose["position"],
                "orientation_quat": self.current_object_pose["orientation_quat"]
            },
            "grasp_pose_contact_world": {
                "position": np.asarray(C_pos).tolist(),
                "orientation_quat": np.asarray(C_quat).tolist()
            },
            "pregrasp_world": {
                "position": np.asarray(P_pos).tolist(),
                "orientation_quat": np.asarray(P_quat).tolist()
            },
            "lift_world": {
                "position": np.asarray(L_pos).tolist(),
                "orientation_quat": np.asarray(L_quat).tolist()
            },
            "ik_endpoints": {
                "C": {"ok": bool(okC)},
                "P": {"ok": bool(okP)},
                "L": {"ok": bool(okL)}
            },
            "local_segments": {
                "P_to_C": (sweep_PC if sweep_PC is not None else default_sweep),
                "C_to_L": (sweep_CL if sweep_CL is not None else default_sweep)
            }
        }

    def update(self, step):
        """Update the robot's position based on the target's position"""
        # object_position, object_orientation = self._target.get_world_pose()
        
        # Local grasp pose in gripper frame  
        grasp_pose_local = [self.grasp_pose["position"], self.grasp_pose["orientation_wxyz"]]
        # World grasp pose in gripper frame
        grasp_pose_world = transform_relative_pose(grasp_pose_local, 
                                                          self.current_object_pose["position"], 
                                                          self.current_object_pose["orientation_quat"])
        # World grasp pose in tool center frame
        grasp_pose_center = local_transform(grasp_pose_world, self.grasp_offset)
        C_pos, C_quat = np.array(grasp_pose_center[0], dtype=float), np.array(grasp_pose_center[1], dtype=float)

        # Path micro-skeleton (PREGRASP 'P', LIFT 'L')
        (P_pos, P_quat), (L_pos, L_quat) = self._compose_pregrasp_lift(C_pos, C_quat)
        
        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # --- IK at endpoints: C, then P, then L (seeded) ---
        actionC, okC = self._articulation_kinematics_solver.compute_inverse_kinematics(np.array(C_pos), np.array(C_quat))
        if not okC:
            # store minimal record and bail early
            # leave robot where it is; record and continue
            self._store_sample(okC=False, okP=False, okL=False,
                               sweep_PC=None, sweep_CL=None,
                               P=(P_pos, P_quat), C=(C_pos, C_quat), L=(L_pos, L_quat))
            return
        
        # Move to C and tick once so the next IK seeds from qC
        self._articulation.apply_action(actionC)
        self.world.step(render=False)
        try:
            qC_vec = np.array(actionC.joint_positions, dtype=float)
        except Exception:
            qC_vec = np.array(self._articulation.get_joint_positions(), dtype=float)

        # --- IK at endpoints: P, then L (seeded) ---
        # Seed P from C
        actionP, okP = self._articulation_kinematics_solver.compute_inverse_kinematics(np.array(P_pos), np.array(P_quat))
        if not okP:
            self._store_sample(okC=True, okP=False, okL=False,
                               sweep_PC=None, sweep_CL=None,
                               P=(P_pos, P_quat), C=(C_pos, C_quat), L=(L_pos, L_quat),
                               qC=qC_vec)

            return
        
        # Move to P and tick once so the next IK seeds from qP
        self._articulation.apply_action(actionP)
        self.world.step(render=False)   
        try:
            qP_vec = np.array(actionP.joint_positions, dtype=float)
        except Exception:
            qP_vec = np.array(self._articulation.get_joint_positions(), dtype=float)
        
        # Seed L from C
        actionL, okL = self._articulation_kinematics_solver.compute_inverse_kinematics(np.array(L_pos), np.array(L_quat))
        if not okL:
            # Move back to C so the endpoint snapshot is really at C
            self._articulation.apply_action(actionC)
            self.world.step(render=False)
            self._store_sample(okC=True, okP=True, okL=False,
                            sweep_PC=None, sweep_CL=None,
                            P=(P_pos, P_quat), C=(C_pos, C_quat), L=(L_pos, L_quat),
                            qC=None, qP=None)
            return
        
        # Apply L and capture qL vector
        self._articulation.apply_action(actionL)
        try:
            qL_vec = np.array(actionL.joint_positions, dtype=float)
        except Exception:
            qL_vec = np.array(self._articulation.get_joint_positions(), dtype=float)

        # --- Local path sweeps (joint-space for raw speed) ---
        if self.use_joint_sweep:
            sweep_PC = self._sweep_joints_collision(qP_vec, qC_vec)
            sweep_CL = self._sweep_joints_collision(qC_vec, qL_vec)
        else:
            sweep_PC = {
                "ok": True,
                "ok_env": True,
                "hit_ground": False,
                "hit_pedestal": False,
                "hit_box": False,
                "fail_step": None
            }
            sweep_CL = {
                "ok": True,
                "ok_env": True,
                "hit_ground": False,
                "hit_pedestal": False,
                "hit_box": False,
                "fail_step": None
            }

        # Optionally leave robot at contact pose for visualization
        self._articulation.apply_action(actionC)

        # Store record
        self._store_sample(okC=True, okP=True, okL=True,
                        sweep_PC=sweep_PC,
                        sweep_CL=sweep_CL,
                        P=(P_pos, P_quat), C=(C_pos, C_quat), L=(L_pos, L_quat),
                        qC=None, qP=None, qL=None)
        

    def save_data(self):
        """Save the data to a file"""
        # Create pedestal-specific directory: .../raw_data/p{pedestal_idx}/
        ped_dir = os.path.join(self.data_root, f"p{int(self.active_pedestal_index)}")
        os.makedirs(ped_dir, exist_ok=True)
        # File path: data_{indexid}.json
        file_path = os.path.join(ped_dir, f"data_{int(self.episode_count)}.json")
        
        # Save the data to the file
        with open(file_path, "w") as f:
            json.dump(self.data_dict, f)

        self.data_dict = {}

    def _describe_pose(self, name, pos, quat):
        # quat is WXYZ; convert to matrix and pull tool Z for checks
        from scipy.spatial.transform import Rotation as R
        Rcw = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        z_tool = Rcw[:, 2]
        print(f"[{name}] p={np.round(pos, 4).tolist()}  z_tool={np.round(z_tool, 4).tolist()}")
        return z_tool

    def sanity_check_one(self):
        """
        Runs ONE sample through: builds C/P/L, does IK with warm-starts, runs both sweeps,
        and prints invariants you can eyeball.
        """
        # 1) Pick the *current* object pose & grasp the way update() would
        grasp_pose_local = [self.grasp_pose["position"], self.grasp_pose["orientation_wxyz"]]
        C_world = transform_relative_pose(grasp_pose_local,
                                                self.current_object_pose["position"],
                                                self.current_object_pose["orientation_quat"])
        C_pos, C_quat = local_transform(C_world, self.grasp_offset)

        # 2) Compose P/L and print frame info
        (P_pos, P_quat), (L_pos, L_quat) = self._compose_pregrasp_lift(np.array(C_pos), np.array(C_quat))
        zC = self._describe_pose("C", np.array(C_pos), np.array(C_quat))
        zP = self._describe_pose("P", np.array(P_pos), np.array(P_quat))
        zL = self._describe_pose("L", np.array(L_pos), np.array(L_quat))

        # 3) Quick invariants
        vPC = (np.array(C_pos) - np.array(P_pos))
        vPC /= (np.linalg.norm(vPC) + 1e-12)
        print("[INV] dot(zC, n_face) ~ -1 is implied by your build_tool(); we check pregrasp direction instead.")
        print("[INV] cos(angle(zC, P->C)) = ", float(np.dot(zC, vPC)))  # should be close to +1
        print("[INV] L.z - C.z (m)       = ", float(L_pos[2] - C_pos[2]), " ~ h_lift=", self.h_lift)

        # 4) IK with real warm starts (your seeding patch)
        qC, okC = self._articulation_kinematics_solver.compute_inverse_kinematics(np.array(C_pos), np.array(C_quat))
        print("[IK] C:", okC); 
        if not okC: return
        self._articulation.apply_action(qC); self.world.step(render=False)

        qP, okP = self._articulation_kinematics_solver.compute_inverse_kinematics(np.array(P_pos), np.array(P_quat))
        print("[IK] P:", okP); 
        if not okP: return
        self._articulation.apply_action(qP); self.world.step(render=False)

        qL, okL = self._articulation_kinematics_solver.compute_inverse_kinematics(np.array(L_pos), np.array(L_quat))
        print("[IK] L:", okL); 
        if not okL: return

        # 5) Sweeps (joint-space)
        ok_PC, fail_PC = self._sweep_joints_collision(qP, qC)
        ok_CL, fail_CL = self._sweep_joints_collision(qC, qL)
        print("[SWEEP] P->C:", ok_PC, "fail_step:", fail_PC)
        print("[SWEEP] C->L:", ok_CL, "fail_step:", fail_CL)

        # 6) Endpoint collision snapshot at C (for alignment with update())
        self._articulation.apply_action(qC); self.world.step(render=False)
        any_col = self.check_for_collisions()
        print("[COL @C] any:", bool(any_col))

    
    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()

        # --- path label defaults (set once) ---
        if not hasattr(self, "d_a"):          self.d_a = 0.04    # pregrasp offset (m)
        if not hasattr(self, "h_lift"):       self.h_lift = 0.20 # lift/retreat height (m)
        if not hasattr(self, "dq_step"):      self.dq_step = 0.02 # joint sweep step (rad)
        if not hasattr(self, "use_joint_sweep"): self.use_joint_sweep = True  # joint sweep vs cartesian IK (raw stage)


        
        # Prepare counts (no popping; we iterate via indices)
        self.object_poses = [self.all_object_poses[i] for i in self.object_indices]
        num_objects = len(self.object_poses)
        num_grasps = len(self.grasp_keys)

        # Main simulation loop
        while simulation_app.is_running():
            print(f"The current progress is: ped {self.pedestal_cursor}/{len(self.pedestal_indices)-1}, grasp {self.grasp_idx+1}/{num_grasps}, obj {self.object_idx+1}/{num_objects}")

            # Place current pedestal and object orientation (via subset index)
            ped_idx = int(self.pedestal_indices[self.pedestal_cursor])
            self.active_pedestal_index = ped_idx  # used by save_data()
            ped = self.all_pedestal_poses[ped_idx]

            ped_x = float(ped["position"][0])
            ped_y = float(ped["position"][1])
            ped_z = float(ped["position"][2])
            # Update the collision pedestal to this pose (cuboid)
            self.collision_detector.create_virtual_pedestal(
                position=Gf.Vec3f(ped_x, ped_y, ped_z),
                size_x=float(self.pedestal_dims[0]),
                size_y=float(self.pedestal_dims[1]),
                size_z=float(self.pedestal_dims[2])
            )
            ped_top_z = float(ped_z + 0.5 * self.pedestal_dims[2])
            obj_pose = self.object_poses[self.object_idx]
            self.current_object_pose = {
                "position": [ped_x, ped_y, ped_top_z + float(obj_pose["position"][2])],
                "orientation_quat": obj_pose["orientation_quat"],
            }
            self._target.set_world_pose(self.current_object_pose["position"], self.current_object_pose["orientation_quat"])

            self.world.step(render=True)

            # Update the robot's position
            # Current grasp pose from ordered keys
            current_grasp_key = self.grasp_keys[self.grasp_idx]
            self.grasp_pose = self.grasp_poses[current_grasp_key]
            self.update(step=1.0/60.0)
            self.count += 1

            # Advance orientation index; on completion, save and advance grasp/pedestal
            self.object_idx += 1
            if self.object_idx >= num_objects:
                # finished all orientations for this grasp -> save and advance grasp index
                self.save_data()
                self.episode_count += 1
                self.count = 0
                self.data_dict = {}
                self.object_idx = 0
                self.grasp_idx += 1
                if self.grasp_idx >= num_grasps:
                    # finished all grasps for this pedestal -> move to next pedestal
                    self.grasp_idx = 0
                    self.episode_count = 0
                    self.pedestal_cursor += 1
                    if self.pedestal_cursor >= len(self.pedestal_indices):
                        print("All pedestals completed.")
                        break

        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env = StandaloneIK()
    env.run()
    
    # Close the simulation application
    simulation_app.close()